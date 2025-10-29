import argparse
import logging
import os
import pathlib
from typing import Tuple, List

from datasets import load_dataset, concatenate_datasets, Dataset
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, SchedulerType
import wandb

import config
from hf_wrapper import GPTForSequenceClassification
from tokenizer import load_tokenizer
import utils
from utils import flatten_multi_features, load_pretrained_model, compute_metrics


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


def load_and_preprocess(cfg: dict, db: dict, lang, split, tokenizer, model_type, cpus: int = os.cpu_count()) -> Dataset:
    dataset_settings = db[lang][cfg["task"]][cfg["datasets"][lang]]
    dataset_name = dataset_settings["dataset"]

    logger.info(f'Loading dataset "{dataset_name}"')
    logger.info(f'Label feature: {dataset_settings["eval_feature"]}')

    ds = load_dataset(dataset_name, split=split, cache_dir=cfg["hf_cache"])
    fields = get_fields(dataset_settings, model_type)

    if dataset_settings['filter_length']:
        def preprocess(examples):
            features = flatten_multi_features(examples, fields)
            encoded = tokenizer(features)
            encoded['label'] = examples[dataset_settings["eval_feature"]]
            return encoded

        result = ds.map(preprocess, batched=True, num_proc=cpus)
        return result.filter(lambda r: len(r['input_ids']) <= 1024)

    def preprocess(examples):
        features = flatten_multi_features(examples, fields)
        encoded = tokenizer(features, truncation=True, max_length=1024)
        encoded['label'] = examples[dataset_settings["eval_feature"]]
        return encoded

    return ds.map(preprocess, batched=True, num_proc=cpus)


def do_train_run(
        cfg: dict, db: dict,
        train_langs: List[str], eval_langs: List[str], model_type: str,
        cpus: int = os.cpu_count()
) -> dict:
    device = 'cpu' if not torch.cuda.is_available() or cfg['cpu_only'] else 'cuda'
    logger.info(f'Using device "{device}"')

    logger.info(f'Trainig on: {train_langs}')
    logger.info(f'Evaluation on: {eval_langs}')
    logger.info(f'Model: {model_type}')

    # load the model
    vocab_path, merges_path = get_tokenizer_paths(cfg, model_type)
    tokenizer = load_tokenizer(vocab_path, merges_path)
    base_model = load_pretrained_model(get_checkpoint_path(cfg, model_type), device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.padding_side = tokenizer.padding_side

    # load the datasets
    # merge train datasets
    # keep validation separate
    class_count = -1
    train_datasets = []
    for train_lang in train_langs:
        dataset_settings = db[train_lang][cfg["task"]][cfg["datasets"][train_lang]]
        if dataset_settings["task_type"] != "classification":
            raise NotImplementedError("non-classification tasks are not supported")
        dclasses = dataset_settings["classes"]
        if class_count < 0:
            class_count = dclasses
        elif class_count != dclasses:
            raise ValueError(f'dataset output class count mismatch ({dclasses} vs {class_count})')
        ds = load_and_preprocess(cfg, db, train_lang, dataset_settings["splits"][0], tokenizer, model_type, cpus)
        train_datasets.append(ds)
    if len(train_datasets) > 0:
        train_dataset = concatenate_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    eval_datasets = {}
    for eval_lang in eval_langs:
        dataset_settings = db[eval_lang][cfg["task"]][cfg["datasets"][eval_lang]]
        if dataset_settings["task_type"] != "classification":
            raise NotImplementedError("non-classification tasks are not supported")
        dclasses = dataset_settings["classes"]
        if class_count != dclasses:
            raise ValueError(f'dataset output class count mismatch ({dclasses} vs {class_count})')
        ds = load_and_preprocess(cfg, db, eval_lang, dataset_settings["splits"][1], tokenizer, model_type, cpus)
        eval_datasets[eval_lang] = ds

    train_eval_dataset_name = sorted(list(eval_datasets.keys()), key=lambda k: len(eval_datasets[k]))[0]
    logger.info(f'using "{train_eval_dataset_name}" for trainning evaluation because it\'s the shortest')
    train_eval_dataset = eval_datasets[train_eval_dataset_name]

    # Configure model now that we know class count
    if class_count <= 0:
        raise ValueError('dataset has no output classes')
    model = GPTForSequenceClassification(base_model, num_classes=class_count).to(device)

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
        disable_tqdm=True,
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
        compute_metrics=compute_metrics,
    )

    logger.info("starting training")
    results = trainer.train()
    logger.info("finished training")
    logger.info(f'Results: {results}')

    # evaluate on each output language
    f1_results = {}
    for eval_lang, eval_dataset in eval_datasets.items():
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
        do_train_run(cfg, db, args.train_langs, args.eval_langs, mt, args.cpus)

