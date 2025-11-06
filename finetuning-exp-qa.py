import argparse
import logging
import os
import pathlib
import random
from typing import Tuple, List, Optional

from datasets import load_dataset, concatenate_datasets, Dataset, Value
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import wandb

import config
from hf_wrapper import GPTForQuestionAnswering
from tokenizer import load_tokenizer
import utils
from utils import load_pretrained_model, create_downsampled_dataset

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
def make_qa_compute_metrics(cfg, db, lang, model_type: str, examples, features,
                            sample_rows: List[int], display_incorrect: bool,
                            n_best_size=20, max_answer_length=30,
                            debug: bool = False):
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

        if debug:
            sample_preds = set(random.sample(list(predictions.keys()), 5)) if len(sample_rows) == 0 else set(sample_rows)

        gold_texts_arr = examples[efeat]

        logger.info('building metric arrays')
        # Build HF metric inputs
        preds = []
        refs  = []
        pbar = None
        if debug:
            pbar = tqdm(total=len(examples['id']), desc='building metric arrays')
        for i, eid in enumerate(examples["id"]):
            pred_answers = predictions.get(
                eid, None
            )
            pred_answer = pred_answers['answers'][pred_answers['best_idx']]
            pred_text = pred_answer['text']

            answer = gold_texts_arr[i]
            gold_texts = answer["text"]

            if (debug and eid in sample_preds) or (display_incorrect and (abs(ans_token_start - pred_start) > 2 or abs(ans_token_end - pred_end) > 2)):
                ex_row, ex_feat = id_to_row[eid]
                ans_token_start = ex_feat['start_positions']
                ans_token_end = ex_feat['end_positions']
                pred_start, pred_end = pred_answer['logit_indices']
                logger.info(f'{str(eid)} gold tokens: ({ans_token_start}, {ans_token_end}), gold: {ex_row["formatted_strings"]}')
                logger.info(f'{str(eid)} predicted positions: ({pred_start}, {pred_end})')
                logger.info(f'{str(eid)} character accuracy: ({pred_answer["start"]} vs {answer["answer_start"][0]}) score: {pred_answer["score"]}')
                logger.info(f'{str(eid)}: "{pred_text}" vs "{gold_texts[0]}"')
                # torch.set_printoptions(threshold=float('inf'))
                logger.info(f'{str(eid)}: Offsets: {ex_feat["offset_mapping"]}')
                logger.info(f'{str(eid)}: start logits: {truncate_list_output(pred_answer["logits"][0].tolist())}')
                logger.info(f'{str(eid)}: end logits: {truncate_list_output(pred_answer["logits"][1].tolist())}')
                # torch.set_printoptions(profile="default")
                # logger.info('tried answers:')
                # for s_index in pred_answers['answers'].keys():
                #     ans = pred_answers['answers'][s_index]
                #     if ans is None:
                #         logger.info(f'no answer found for {s_index}')
                #         continue
                #     logger.info(f'\t"{ans["text"]}" {ans["start"]} score: {ans["score"]}')

            preds.append({"id": str(eid), "prediction_text": pred_text})
            refs.append({"id": str(eid), "answers": {
                "text": gold_texts,
                "answer_start": answer["answer_start"],
            }})

            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        logger.info('computing metrics')
        return metric.compute(predictions=preds, references=refs)

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


def do_train_run(
        job_number: str,
        cfg: dict, db: dict,
        train_langs: List[str], eval_langs: List[str], model_type: str,
        eval_samples: int, eval_rows: List[int], display_incorrect: bool,
        cpus: int = os.cpu_count(), debug: bool = False,
        eval_only: Optional[pathlib.Path] = None,
) -> dict:
    device = 'cpu' if not torch.cuda.is_available() or cfg['cpu_only'] else 'cuda'
    logger.info(f'Using device "{device}"')

    logger.info(f'Trainig on: {train_langs}')
    logger.info(f'Evaluation on: {eval_langs}')
    logger.info(f'Model: {model_type}')

    # load the model
    vocab_path, merges_path = get_tokenizer_paths(cfg, model_type)
    tokenizer = load_tokenizer(vocab_path, merges_path)
    checkpoint_path = get_checkpoint_path(cfg, model_type)
    base_model = load_pretrained_model(checkpoint_path, device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.padding_side = tokenizer.padding_side

    # load the datasets
    # merge train datasets
    # keep validation separate
    logger.info('loading training datasets')

    if eval_only is None:
        train_datasets = []
        for train_lang in train_langs:
            dataset_settings = db[train_lang][cfg["task"]][cfg["datasets"][train_lang]]
            if dataset_settings["task_type"] != "question-answering":
                raise NotImplementedError("non-qa tasks are not supported")
            ds = load_and_preprocess(cfg, db, train_lang, dataset_settings["splits"][0], tokenizer, model_type, cpus)
            train_datasets.append(ds)
        if len(train_datasets) > 0:
            train_dataset = concatenate_datasets_reenumerate_ids(train_datasets, "id", cpus)
        else:
            train_dataset = train_datasets[0]
    else:
        train_dataset = None

    logger.info('loading eval datasets')

    eval_datasets = {}
    for eval_lang in eval_langs:
        dataset_settings = db[eval_lang][cfg["task"]][cfg["datasets"][eval_lang]]
        if dataset_settings["task_type"] != "question-answering":
            raise NotImplementedError("non-qa tasks are not supported")
        ds = load_and_preprocess(cfg, db, eval_lang, dataset_settings["splits"][1], tokenizer, model_type, cpus)
        eval_datasets[eval_lang] = ds

    if eval_only is None:
        train_eval_dataset_name = sorted(list(eval_datasets.keys()), key=lambda k: len(eval_datasets[k]))[0]
        logger.info(f'using "{train_eval_dataset_name}" for trainning evaluation because it\'s the shortest')
        train_eval_dataset = eval_datasets[train_eval_dataset_name]
        train_eval_dataset = create_downsampled_dataset(train_eval_dataset, eval_samples)

        logger.info('creating metrics function')

        metrics = make_qa_compute_metrics(
            cfg, db, train_eval_dataset_name,
            model_type,
            train_eval_dataset,
            train_eval_dataset,
            eval_rows, display_incorrect,
            debug=debug,
        )
    else:
        train_eval_dataset = None
        metrics = None

    logger.info('setting up model wrapper')

    model = GPTForQuestionAnswering(base_model).to(device)
    if eval_only is not None:
        model = load_pretrained_trainer_model(eval_only, model, device)

    logger.info('configuring training args')

    # configure trainer
    run_name = f'{model_type}-{"-".join(train_langs)}'
    temporary_output_dir = pathlib.Path(cfg["checkpoints"]["training"]) / f"{job_number}-{cfg['wandb']['project']}-{run_name}/"
    temporary_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'saving checkpoints to "{temporary_output_dir}"')
    hyperparameters = cfg["hyperparameters"]
    training_args = TrainingArguments(
        eval_strategy="steps",
        eval_steps=0.01,
        output_dir=str(temporary_output_dir),
        save_strategy='steps',
        save_steps=0.01,
        metric_for_best_model="f1",
        greater_is_better=True,  # Remember to update this if you update the metric used
        load_best_model_at_end=True,
        learning_rate=hyperparameters["learning_rate"],
        # lr_scheduler_type=SchedulerType.COSINE,
        per_device_train_batch_size=hyperparameters["batch_size"],
        per_device_eval_batch_size=hyperparameters["batch_size"],
        num_train_epochs=hyperparameters["epochs"],
        weight_decay=hyperparameters["weight_decay"],
        max_grad_norm=hyperparameters["gradient_clipping"],  # gradient clipping
        logging_steps=100,
        fp16=True,
        warmup_ratio=hyperparameters["warmup_ratio"],
        save_safetensors=False,
        disable_tqdm=not debug,
        no_cuda=cfg["cpu_only"],
    )

    logger.info('starting wandb')

    hyperparameters['job_number'] = job_number
    hyperparameters['checkpoint_location'] = temporary_output_dir

    # run training
    if eval_only is None:
        wandb_settings = cfg["wandb"]
        wrun = wandb.init(
            entity=wandb_settings["entity"],
            project=wandb_settings["project"],
            name=run_name,
            config=hyperparameters,
        )
    else:
        wrun = None

    if eval_only is None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=metrics,
        )

        logger.info("starting training")
        results = trainer.train()
        logger.info("finished training")
        logger.info(f'Results: {results}')
    else:
        trainer = None

    # evaluate on each output language
    f1_results = {}
    for eval_lang, eval_dataset in eval_datasets.items():
        metrics = make_qa_compute_metrics(
            cfg, db, eval_lang,
            model_type,
            eval_dataset, eval_dataset,
            eval_rows, display_incorrect,
            debug=debug,
        )
        if trainer is None:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=metrics,
            )
        else:
            trainer.compute_metrics = metrics
        metric_prefix = f'eval_{eval_lang}'
        lang_results = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=metric_prefix,
        )
        logger.info(f'Evaluation for {eval_lang}:')
        logger.info(str(lang_results))
        f1_results[eval_lang] = lang_results[f'{metric_prefix}_f1']

    if wrun is not None:
        wrun.finish()

    # return best f1 score
    return f1_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--train-langs', nargs='+', type=str, help='The languages to train on')
    ap.add_argument('--eval-langs', nargs='+', type=str, help='The languages to evaluate on')
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], help='The model type')
    ap.add_argument('--training-eval-size', type=int, default=1000,
                    help='The number of records to sample from the eval dataset to use while training')
    ap.add_argument('--eval-only', type=pathlib.Path, default=None, help='If supplied, specifies a checkpoint to evaluate, training is skipped, assumes that it is a trainer checkpoint')
    ap.add_argument('--sample-examples', type=int, nargs='*', default=[], help='The specific rows to sample examples from, defaults to random')
    ap.add_argument('--display-incorrect', action='store_true', help='Display incorrect predictions')
    args = ap.parse_args()
    cfg, db = config.load_config(args.config, args.default_config, args.language_database)

    for mt in args.model_type:
        do_train_run(
            args.job_number, cfg, db,
            args.train_langs, args.eval_langs, mt,
            args.training_eval_size,
            args.sample_examples, args.display_incorrect,
            args.cpus,
            args.debug, args.eval_only
        )

