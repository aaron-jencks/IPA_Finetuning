import argparse
import logging
import os
import pathlib
from typing import Tuple, List

from datasets import load_dataset, concatenate_datasets, Dataset, Value
import evaluate
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.modeling_outputs import QuestionAnsweringModelOutput

import wandb

import config
from hf_wrapper import GPTForQuestionAnswering
from tokenizer import load_tokenizer
import utils
from utils import load_pretrained_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def format_qa_string(q: str, c: str, sep: str) -> str:
    return f'{sep} {q} {sep} {c} {sep}'


def load_and_preprocess(cfg: dict, db: dict, lang, split, tokenizer, model_type, cpus: int = os.cpu_count()) -> Dataset:
    dataset_settings = db[lang][cfg["task"]][cfg["datasets"][lang]]
    dataset_name = dataset_settings["dataset"]

    logger.info(f'Loading dataset "{dataset_name}"')
    logger.info(f'Label feature: {dataset_settings["eval_feature"]}')

    ds = load_dataset(dataset_name, split=split, cache_dir=cfg["hf_cache"])
    fields = get_fields(dataset_settings, model_type)
    q_feat = fields[0]
    c_feat = fields[1]

    def preprocess(examples):
        strings = [format_qa_string(q, c, tokenizer.eos_token) for q, c in zip(examples[q_feat], examples[c_feat])]
        return {
            'formatted_strings': strings,
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
        answers = examples[dataset_settings["eval_feature"]]
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


# 1) Minimal postprocess: logits -> span text
def postprocess_qa_predictions(cfg, examples, features, raw_predictions):
    hyperparameters = cfg['hyperparameters']
    n_best_size = hyperparameters['top_k']
    max_answer_length = hyperparameters['max_answer_length']

    if len(raw_predictions) == 2:
        start_logits, end_logits = raw_predictions
    elif len(raw_predictions) == 3:
        start_logits, end_logits, _ = raw_predictions
    else:
        raise ValueError('unrecognized raw_predictions value')
    assert len(features) == len(examples)

    preds = {}
    use_ids = "id" in examples.column_names

    for i in range(len(features)):
        context = examples["context"][i]
        offsets = features["offset_mapping"][i]

        s_log = start_logits[i]
        e_log = end_logits[i]

        best_text, best_score = "", -1e9

        # top-k search that enforces e >= s and max_answer_length
        start_idxes = np.argsort(s_log)[-n_best_size:][::-1]
        end_idxes = np.argsort(e_log)[-n_best_size:][::-1]

        for s in start_idxes:
            for e in end_idxes:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                s_off, e_off = offsets[s], offsets[e]
                if s_off == (0, 0) or e_off == (0, 0):
                    continue  # skip non-context/special/pad
                score = s_log[s] + e_log[e]
                if score > best_score:
                    best_score = score
                    best_text = context[s_off[0]:e_off[1]]

        ex_key = examples["id"][i] if use_ids else str(i)
        preds[ex_key] = best_text if best_text is not None else ""

    return preds


# 2) Factory that returns a compute_metrics compatible with Trainer
def make_qa_compute_metrics(cfg, db, lang, examples, features,
                            n_best_size=20, max_answer_length=30):
    """
    examples: the *original* eval split (with 'id', 'context', 'answers')
    features: the tokenized eval features you pass to Trainer (must include 'example_id' and 'offset_mapping')
    squad_v2: set True if you have unanswerables and want 'squad_v2' metric
    normalizer: optional callable to normalize strings (e.g., your IPA normalizer)
    """
    dataset_settings = db[lang][cfg["task"]][cfg["datasets"][lang]]
    efeat = dataset_settings['eval_feature']
    metric = evaluate.load("squad")

    def compute_metrics(eval_pred):
        # eval_pred.predictions is (start_logits, end_logits)
        # eval_pred.label_ids is usually (start_pos, end_pos), but we don't need it here
        predictions = postprocess_qa_predictions(
            cfg, examples, features, eval_pred.predictions,
        )

        # Build HF metric inputs
        preds = []
        refs  = []
        for i, eid in enumerate(examples["id"]):
            pred_text = predictions.get(eid, "")

            gold_texts = examples[efeat][i]["text"]

            preds.append({"id": eid, "prediction_text": pred_text})
            refs.append({"id": eid, "answers": {
                "text": gold_texts,
                "answer_start": examples[efeat][i]["answer_start"],
            }})

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
        cfg: dict, db: dict,
        train_langs: List[str], eval_langs: List[str], model_type: str,
        cpus: int = os.cpu_count(), debug: bool = False,
) -> dict:
    device = 'cpu' if not torch.cuda.is_available() or cfg['cpu_only'] else 'cuda'
    logger.info(f'Using device "{device}"')

    # load the model
    vocab_path, merges_path = get_tokenizer_paths(cfg, model_type)
    tokenizer = load_tokenizer(vocab_path, merges_path)
    base_model = load_pretrained_model(get_checkpoint_path(cfg, model_type), device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.padding_side = tokenizer.padding_side

    # load the datasets
    # merge train datasets
    # keep validation separate
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

    eval_datasets = {}
    for eval_lang in eval_langs:
        dataset_settings = db[eval_lang][cfg["task"]][cfg["datasets"][eval_lang]]
        if dataset_settings["task_type"] != "question-answering":
            raise NotImplementedError("non-qa tasks are not supported")
        ds = load_and_preprocess(cfg, db, eval_lang, dataset_settings["splits"][1], tokenizer, model_type, cpus)
        eval_datasets[eval_lang] = ds

    train_eval_dataset_name = sorted(list(eval_datasets.keys()), key=lambda k: len(eval_datasets[k]))[0]
    logger.info(f'using "{train_eval_dataset_name}" for trainning evaluation because it\'s the shortest')
    train_eval_dataset = eval_datasets[train_eval_dataset_name]

    metrics = make_qa_compute_metrics(
        cfg, db, train_eval_dataset_name,
        train_eval_dataset,
        train_eval_dataset,
    )

    model = GPTForQuestionAnswering(base_model).to(device)

    # configure trainer
    run_name = f'{model_type}-{"-".join(train_langs)}'
    temporary_output_dir = pathlib.Path(cfg["checkpoints"]["training"]) / f"{cfg['wandb']['project']}-{run_name}/"
    temporary_output_dir.mkdir(parents=True, exist_ok=True)
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

    # run training
    wandb_settings = cfg["wandb"]
    wrun = wandb.init(
        entity=wandb_settings["entity"],
        project=wandb_settings["project"],
        name=run_name,
        config=hyperparameters,
    )

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

    # evaluate on each output language
    f1_results = {}
    for eval_lang, eval_dataset in eval_datasets.items():
        metrics = make_qa_compute_metrics(
            cfg, db, train_eval_dataset_name,
            eval_dataset, eval_dataset,
        )
        trainer.compute_metrics = metrics
        metric_prefix = f'eval_{eval_lang}'
        lang_results = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=metric_prefix,
        )
        logger.info(f'Evaluation for {eval_lang}:')
        logger.info(str(lang_results))
        f1_results[eval_lang] = lang_results[f'{metric_prefix}_f1']

    wrun.finish()

    # return best f1 score
    return f1_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--train-langs', nargs='+', type=str, help='The languages to train on')
    ap.add_argument('--eval-langs', nargs='+', type=str, help='The languages to evaluate on')
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], help='The model type')
    args = ap.parse_args()
    cfg, db = config.load_config(args.config, args.default_config, args.language_database)

    for mt in args.model_type:
        do_train_run(cfg, db, args.train_langs, args.eval_langs, mt, args.cpus, args.debug)

