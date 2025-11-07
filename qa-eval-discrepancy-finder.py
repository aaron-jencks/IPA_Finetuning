import argparse
import logging
import os
import pathlib
from typing import Tuple, List

from datasets import load_dataset, concatenate_datasets, Dataset, Value
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

import config
from hf_wrapper import GPTForQuestionAnswering
from tokenizer import load_tokenizer
import utils
from utils import load_pretrained_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrained_trainer_model(path: pathlib.Path, base_model: GPTForQuestionAnswering, device: str = 'cuda') -> GPTForQuestionAnswering:
    state_dict = torch.load(path, map_location=device)
    base_model.load_state_dict(state_dict, strict=True)
    return base_model


def get_checkpoint_path(cfg: dict, type: str) -> pathlib.Path:
    checkpoint_settings = cfg["checkpoints"]
    prefix_path = pathlib.Path(checkpoint_settings["prefix"]) / checkpoint_settings[type]
    logger.info(f"Checkpoint path: {prefix_path}")
    return prefix_path / "ckpt.pt"


def get_tokenizer_paths(cfg: dict, type: str) -> Tuple[pathlib.Path, pathlib.Path]:
    tokenizer_settings = cfg["tokenizers"]
    tokenizer_name = tokenizer_settings[f'{type}_prefix']
    logger.info(f"Loading tokenizer '{tokenizer_name}' from '{tokenizer_settings['prefix']}'")
    return (
        pathlib.Path(tokenizer_settings["prefix"]) / f'{tokenizer_name}-vocab.json',
        pathlib.Path(tokenizer_settings["prefix"]) / f'{tokenizer_name}-merges.txt',
    )


def get_fields(settings: dict, model_type: str) -> List[str]:
    feats = settings["train_features"]
    if model_type == "ipa":
        return list(map(lambda f: f'{f}-phoneme', feats))
    logger.info(f'Features used: {feats}')
    return feats


def get_eval_fields(settings: dict, model_type: str) -> str:
    feat = settings["eval_feature"]
    if model_type == "ipa":
        feat += '-phoneme'
    logger.info(f'Using eval feature: {feat}')
    return feat


def format_qa_string(q: str, c: str, sep: str) -> Tuple[str, int]:
    return f'{sep} {q} {sep} {c} {sep}', len(q) + len(sep) * 2 + 3


def load_and_preprocess(cfg: dict, db: dict, lang, split, tokenizer, model_type, cpus: int = os.cpu_count()) -> Dataset:
    dataset_settings = db[lang][cfg["task"]][cfg["datasets"][lang]]
    dataset_name = dataset_settings["dataset"]

    logger.info(f'Loading dataset "{dataset_name}"')
    logger.info(f'Label feature: {dataset_settings["eval_feature"]}')

    ds = load_dataset(dataset_name, split=split, cache_dir=cfg["hf_cache"])
    fields = get_fields(dataset_settings, model_type)
    q_feat = fields[0]
    c_feat = fields[1]
    eval_feat = get_eval_fields(dataset_settings, model_type)

    def preprocess(examples):
        strings = []
        answers = []
        for q, c, a in zip(examples[q_feat], examples[c_feat], examples[eval_feat]):
            s, off = format_qa_string(q, c, tokenizer.eos_token)
            strings.append(s)
            a['answer_start'][0] += off
            answers.append(a)
        return {
            'formatted_strings': strings,
            eval_feat: answers,
        }

    ds_pre = ds.map(preprocess, batched=True, num_proc=cpus)

    if dataset_settings['filter_length']:
        def preprocess(examples):
            encoded = tokenizer(examples['formatted_strings'])
            return {
                'encoding_length': [len(row) for row in encoded['input_ids']],
            }

        ds_pre = ds_pre.map(preprocess, batched=True, num_proc=cpus)
        ds_pre = ds_pre.filter(lambda r: r['encoding_length'] <= 1024)

    # source: https://huggingface.co/docs/transformers/en/tasks/question_answering
    def preprocess(examples):
        inputs = tokenizer(
            examples['formatted_strings'],
            max_length=1024,
            truncation=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        offset_mapping = inputs["offset_mapping"]
        answers = examples[eval_feat]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer['answer_start'][0]
            end_char = answer['answer_start'][0] + len(answer['text'][0])

            # Find the start and end of the context
            pad_token_count = 0
            context_start = -1
            context_end = -1
            for ti, token in enumerate(inputs['input_ids'][i]):
                if token == tokenizer.eos_token_id:
                    pad_token_count += 1
                    if pad_token_count == 2:
                        context_start = ti + 1
                    elif pad_token_count == 3:
                        context_end = ti - 1
                        break
            if context_start < 0 or context_end < 0:
                raise ValueError("context not found")

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return ds_pre.map(preprocess, batched=True, num_proc=cpus)


def numpy_topk(arr, k):
    ind = np.argpartition(arr, -k)[-k:]
    return ind[np.argsort(arr[ind])]


# 1) Minimal postprocess: logits -> span text
def postprocess_qa_predictions(cfg, examples, features, raw_predictions):
    hyperparameters = cfg['hyperparameters']
    n_best_size = hyperparameters['top_k']
    max_answer_length = hyperparameters['max_answer_length']

    logger.info(f'evaluating top {n_best_size} indices')

    logger.info('extracting contexts and offsets')

    contexts = examples["formatted_strings"]
    offset_maps = features["offset_mapping"]

    if len(raw_predictions) == 2:
        start_logits, end_logits = raw_predictions
    elif len(raw_predictions) == 3:
        start_logits, end_logits, _ = raw_predictions
    else:
        raise ValueError('unrecognized raw_predictions value')
    assert len(features) == len(examples)

    logger.info('starting evaluation loop')

    preds = {}
    use_ids = "id" in examples.column_names

    for i in range(len(features)):
        context = contexts[i]
        offsets = offset_maps[i]

        s_log = start_logits[i]
        e_log = end_logits[i]

        first_score = True
        best_score, best_idx = -1e9, -1
        tried_answers = {}

        # top-k search that enforces e >= s and max_answer_length
        start_idxes = numpy_topk(s_log, n_best_size)
        end_idxes = numpy_topk(e_log, n_best_size)

        for s in start_idxes:
            tried_answers[s] = None
            for e in end_idxes:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                s_off, e_off = offsets[s], offsets[e]
                if s_off == (0, 0) or e_off == (0, 0):
                    continue  # skip non-context/special/pad
                score = s_log[s] + e_log[e]
                text = context[s_off[0]:e_off[1]]
                answer_dict = {
                    'start': s_off[0],
                    'end': e_off[1],
                    'text': text,
                    'score': score,
                    'logits': (s_log, e_log),
                    'logit_indices': (s, e)
                }
                if tried_answers[s] is None or score > tried_answers[s]['score']:
                    tried_answers[s] = answer_dict
                if first_score or score > best_score:
                    first_score = False
                    best_score = score
                    best_idx = s

        ex_key = examples["id"][i] if use_ids else str(i)
        preds[ex_key] = {
            'answers': tried_answers,
            'best_idx': best_idx,
        }

    return preds


def truncate_list_output(l: list) -> list:
    current = -1
    if l[current] != -65504.:
        return l
    while l[current] == -65504.:
        current -= 1
    return l[:current+1]


def char_to_token_offset(c: int, mappings: List[Tuple[int, int]]) -> int:
    for mi, (ms, me) in enumerate(mappings):
        if me >= c >= ms:
            return mi
    return -1


# 2) Factory that returns a compute_metrics compatible with Trainer
def make_qa_compute_metrics(cfg, db, lang, model_type: str, examples, features):
    """
    examples: the *original* eval split (with 'id', 'context', 'answers')
    features: the tokenized eval features you pass to Trainer (must include 'example_id' and 'offset_mapping')
    squad_v2: set True if you have unanswerables and want 'squad_v2' metric
    normalizer: optional callable to normalize strings (e.g., your IPA normalizer)
    """
    dataset_settings = db[lang][cfg["task"]][cfg["datasets"][lang]]
    efeat = get_eval_fields(dataset_settings, model_type)
    metric = evaluate.load("squad")
    id_to_row = {ex["id"]: (ex, feat) for ex, feat in zip(examples, features)}

    def compute_metrics(eval_pred):
        # eval_pred.predictions is (start_logits, end_logits)
        # eval_pred.label_ids is usually (start_pos, end_pos), but we don't need it here
        logger.info("starting metric computation")
        logger.info('starting postprocessing')
        predictions = postprocess_qa_predictions(
            cfg, examples, features, eval_pred.predictions,
        )

        gold_texts_arr = examples[efeat]

        logger.info('building metric arrays')
        # Build HF metric inputs
        preds = []
        refs  = []
        mistakes = []
        for i, eid in enumerate(tqdm(examples["id"], desc='building metric arrays')):
            pred_answers = predictions.get(
                eid, None
            )
            pred_answer = pred_answers['answers'][pred_answers['best_idx']]
            pred_text = pred_answer['text']

            answer = gold_texts_arr[i]
            gold_texts = answer["text"]

            ex_row, ex_feat = id_to_row[eid]
            ans_token_start = ex_feat['start_positions']
            ans_token_end = ex_feat['end_positions']
            pred_start, pred_end = pred_answer['logit_indices']

            answer_deviation = abs(ans_token_start - pred_start) + abs(ans_token_end - pred_end)

            preds.append({"id": str(eid), "prediction_text": pred_text})
            refs.append({"id": str(eid), "answers": {
                "text": gold_texts,
                "answer_start": answer["answer_start"],
            }})

            if answer_deviation > 3:
                mistakes.append({
                    'id': str(eid),
                    'row': ex_row,
                    'features': ex_feat,
                    'answer': pred_answer,
                })

        logger.info('computing metrics')
        built_in_metrics = metric.compute(predictions=preds, references=refs)

        return {
            **built_in_metrics,
            'mistakes': mistakes,
        }

    return compute_metrics


def concatenate_datasets_reenumerate_ids(
        datasets: List[Dataset], id_feature: str = 'id',
        cpus: int = os.cpu_count()
) -> Dataset:
    mixed = concatenate_datasets(list(datasets))
    def _add_id(_, idx): return {id_feature: int(idx)}
    mixed = mixed.map(_add_id, with_indices=True, num_proc=cpus)
    mixed = mixed.cast_column(id_feature, Value("int64"))
    return mixed


def do_eval_run(
        cfg: dict, db: dict,
        model_path: pathlib.Path,
        eval_langs: List[str], model_type: str,
        cpus: int = os.cpu_count(),
):
    device = 'cpu' if not torch.cuda.is_available() or cfg['cpu_only'] else 'cuda'
    logger.info(f'Using device "{device}"')

    logger.info(f'Evaluation on: {eval_langs}')
    logger.info(f'Model: {model_type}')

    # load the model
    vocab_path, merges_path = get_tokenizer_paths(cfg, model_type)
    tokenizer = load_tokenizer(vocab_path, merges_path)
    checkpoint_path = get_checkpoint_path(cfg, model_type)
    base_model = load_pretrained_model(checkpoint_path, device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.padding_side = tokenizer.padding_side

    logger.info('loading eval datasets')

    eval_datasets = {}
    for eval_lang in eval_langs:
        dataset_settings = db[eval_lang][cfg["task"]][cfg["datasets"][eval_lang]]
        if dataset_settings["task_type"] != "question-answering":
            raise NotImplementedError("non-qa tasks are not supported")
        ds = load_and_preprocess(cfg, db, eval_lang, dataset_settings["splits"][1], tokenizer, model_type, cpus)
        eval_datasets[eval_lang] = ds

    logger.info('setting up model wrapper')

    model = GPTForQuestionAnswering(base_model).to(device)
    model = load_pretrained_trainer_model(model_path, model, device)

    logger.info('configuring training args')

    # configure trainer
    hyperparameters = cfg["hyperparameters"]
    training_args = TrainingArguments(
        output_dir='why-is-this-required?',
        save_strategy='no',
        per_device_train_batch_size=hyperparameters["batch_size"],
        per_device_eval_batch_size=hyperparameters["batch_size"],
        no_cuda=cfg["cpu_only"],
        report_to=None,  # disable wandb
    )

    # evaluate on each output language
    results = {}
    for eval_lang, eval_dataset in eval_datasets.items():
        metrics = make_qa_compute_metrics(
            cfg, db, eval_lang,
            model_type,
            eval_dataset, eval_dataset,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=metrics,
        )
        lang_results = trainer.evaluate(
            eval_dataset=eval_dataset,
        )
        logger.info(f'found {len(lang_results["eval_mistakes"])} errors')
        results[eval_lang] = lang_results['eval_mistakes']

    return results


def evaluate_run_results(run_results: dict) -> dict:
    current_errors = {}
    for mt in run_results.keys():
        for eval_lang in run_results[mt].keys():
            id_set = set([d['id'] for d in run_results[mt][eval_lang]])
            if eval_lang not in current_errors:
                current_errors[eval_lang] = id_set
            else:
                current_errors[eval_lang] = current_errors[eval_lang] ^ id_set
    error_data = {}
    for mt in run_results.keys():
        for eval_lang in run_results[mt].keys():
            if eval_lang not in error_data:
                error_data[eval_lang] = {}
            error_set = current_errors[eval_lang]
            for row in run_results[mt][eval_lang]:
                if row['id'] not in error_set:
                    continue
                if row['id'] not in error_data[eval_lang]:
                    error_data[eval_lang][row['id']] = {}
                error_data[eval_lang][row['id']][mt] = row
    return error_data


def generate_csv_rows(found_errors: dict) -> list:
    output_rows = [
        [
            'eval_lang', 'model_type', 'rowid',
            'gold_indices', 'gold_input_string', 'gold_answer_text', 'gold_answer_character_start',
            'character_index_mapping',
            'answer_indices', 'answer_character_start', 'answer_confidence_score', 'answer_text',
            'answer_start_logits', 'answer_end_logits',
        ]
    ]
    for eval_lang in found_errors.keys():
        for rowid in found_errors[eval_lang].keys():
            for model_type in found_errors[eval_lang][rowid].keys():
                row = found_errors[eval_lang][rowid][model_type]
                row_feat = row['features']
                row_values = row['row']
                answer = row['answer']
                output_rows.append([
                    eval_lang, model_type, rowid,
                    (row_feat['start_positions'], row_feat['end_positions']),               # logit indices
                    row_values['formatted_strings'].replace('\t', '<tab_place_holder>'),
                    row_values['answers']['text'][0].replace('\t', '<tab_place_holder>'),
                    row_values['answers']['answer_start'][0],                               # character position
                    row_feat['offset_mapping'],                                             # character to index mapping
                    answer['logit_indices'],
                    answer['start'],                                                        # character position
                    answer['score'],
                    answer['text'].replace('\t', '<tab_place_holder>'),
                    truncate_list_output(answer["logits"][0].tolist()),                     # start logits
                    truncate_list_output(answer["logits"][0].tolist())                      # end logits
                ])
    return output_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--normal-model-path', type=pathlib.Path, required=True, help='path to the normal model checkpoint')
    ap.add_argument('--ipa-model-path', type=pathlib.Path, required=True, help='path to the ipa model checkpoint')
    ap.add_argument('--output-file', type=pathlib.Path, required=True, help='path to output tsv file')
    ap.add_argument('--eval-langs', nargs='+', type=str, help='The languages to evaluate on')
    args = ap.parse_args()
    cfg, db = config.load_config(args.config, args.default_config, args.language_database)

    model_dict = {
        'normal': args.normal_model_path,
        'ipa': args.ipa_model_path,
    }

    results = {}
    for mt in ['normal', 'ipa']:
        run_results = do_eval_run(
            cfg, db,
            model_dict[mt],
            args.eval_langs, mt,
            args.cpus,
        )
        results[mt] = run_results

    found_errors = evaluate_run_results(results)
    tsv_rows = generate_csv_rows(found_errors)

    logger.info(f'found {len(tsv_rows) >> 1} errors')

    with open(args.output_file, 'w+') as fp:
        for row in tqdm(tsv_rows, desc='writing tsv rows to file'):
            s_arr = list(map(str, row))
            line = '\t'.join(s_arr) + '\n'
            fp.write(line)
