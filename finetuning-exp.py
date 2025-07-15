import argparse
import os
import pathlib
from typing import List

from datasets import load_dataset, concatenate_datasets, ClassLabel
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from hf_wrapper import GPTForSequenceClassification
from model import GPT, GPTConfig
from tokenizer import load_tokenizer, eod_token


def load_pretrained_model(path: pathlib.Path, device: str = 'cuda') -> GPT:
    checkpoint = torch.load(path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    filtered = {k: v for k, v in state_dict.items()
                if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict({**model.state_dict(), **filtered})
    return model.to(device)


def flatten_multi_features(examples, features: List[str]) -> List[str]:
    sep = f'\n\n{eod_token}\n\n'
    return [sep.join([x or '' for x in items]) for items in zip(*[examples[f] for f in features])]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.from_numpy(logits).argmax(dim=-1)
    labels = torch.from_numpy(labels)

    correct = (preds == labels).sum().item()
    total = len(labels)
    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('ipa_model', type=str, help="Name of the ipa model")
    ap.add_argument('normal_model', type=str, help="Name of the normal model")
    ap.add_argument('ipa_tokenizer_prefix', type=str, help="Name of the ipa tokenizer prefix")
    ap.add_argument('normal_tokenizer_prefix', type=str, help="Name of the normal tokenizer prefix")
    ap.add_argument('languages', type=str, nargs=2, help="List of languages")
    ap.add_argument('language_datasets', type=str, nargs=2, help="Language datasets")
    ap.add_argument('--train-lang', type=str, default='both', help='The training language')
    ap.add_argument('--eval-lang', type=str, default='both', help='The evaluation language')
    ap.add_argument('--checkpoint-prefix', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/checkpoints'), help='the prefix of the checkpoints folder')
    ap.add_argument('--tokenizer-prefix', type=pathlib.Path, default=pathlib.Path('/fs/ess/PAS2836/ipa_gpt/tokenizers'), help='the prefix of the tokenizers folder')
    ap.add_argument('--is-medium', action='store_true', help='indicates that the model is a medium model')
    ap.add_argument('--random-seed', type=int, default=42, help='random seed')
    hp = ap.add_argument_group('hyperparameters')
    hp.add_argument('--epochs', type=int, default=3, help='number of training epochs')
    hp.add_argument('--context-size', type=int, default=1024, help='The context size of the model')
    hp.add_argument('--learning-rate', type=float, default=2e-5, help='The learning rate of the model')
    hp.add_argument('--batch-size', type=int, default=16, help='The batch size of the model')
    hp.add_argument('--hf-cache', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/cache'), help='The huggingface cache folder')
    hp.add_argument('--training-checkpoint-prefix', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/checkpoints'), help='The prefix of the temporary checkpoints folder')
    dp = ap.add_argument_group('dataset')
    dp.add_argument('--train-features', type=str, nargs='+', required=True, help='The training features')
    dp.add_argument('--eval-features', type=str, nargs='+', required=True, help='The validation features')
    dp.add_argument('--num-classes', type=int, default=3, help='The number of classes')
    dp.add_argument('--class-labels', type=str, nargs='+', default=['entailment', 'neutral', 'contradiction'], help='The class labels')
    args = ap.parse_args()

    CHECKPOINTS = {
        "ipa": args.checkpoint_prefix / f"{args.ipa_model}/ckpt.pt",
        "normal": args.checkpoint_prefix / f"{args.normal_model}/ckpt.pt",
    }

    TOKENIZERS = {
        "ipa": (
            args.tokenizer_prefix / f"{args.ipa_tokenizer_prefix}-vocab.json",
            args.tokenizer_prefix / f"{args.ipa_tokenizer_prefix}-merges.txt",
        ),
        "normal": (
            args.tokenizer_prefix / f"{args.normal_tokenizer_prefix}-vocab.json",
            args.tokenizer_prefix / f"{args.normal_tokenizer_prefix}-merges.txt",
        ),
    }

    LANG_TO_DATASET = {
        lang: dataset
        for lang, dataset in zip(args.languages, args.language_datasets)
    }

    LANG_TO_FEATURES = {
        lang: features
        for lang, features in zip(args.languages, [args.train_features, args.eval_features])
    }

    project_name = f"{'-'.join(args.languages)}{'-medium' if args.is_medium else '-small'}-{args.train_lang}-{args.eval_lang}-finetuning"
    temporary_output_dir = args.training_checkpoint_prefix / f"{project_name}-{args.train_lang}-{args.eval_lang}/"
    temporary_output_dir.mkdir(parents=True, exist_ok=True)

    def get_fields(lang, model_type):
        feats = LANG_TO_FEATURES[lang]
        if model_type == "ipa":
            return list(map(lambda f: f'{f}-phoneme', feats))
        return feats

    def load_and_preprocess(lang, split, tokenizer, model_type, cache):
        dataset_name = LANG_TO_DATASET[lang]
        ds = load_dataset(dataset_name, split=split, cache_dir=str(cache))
        if 'label' in ds.features and not isinstance(ds.features['label'], ClassLabel):
            ds = ds.cast_column("label", ClassLabel(names=args.class_labels))
        fields = get_fields(lang, model_type)

        def preprocess(examples):
            features = flatten_multi_features(examples, fields)
            return tokenizer(features, truncation=True, max_length=args.context_size)

        return ds.map(preprocess, batched=True, num_proc=os.cpu_count())

    # === Run both IPA and NORMAL models ===
    for model_type in ['ipa', 'normal']:
        print(f"\n🔧 Running setup for {model_type.upper()} model")
        vocab_path, merges_path = TOKENIZERS[model_type]
        tokenizer = load_tokenizer(vocab_path, merges_path)
        base_model = load_pretrained_model(CHECKPOINTS[model_type], 'cuda')
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.padding_side = tokenizer.padding_side
        model = GPTForSequenceClassification(base_model, num_classes=args.num_classes).to('cuda')

        lam_load_and_preprocess = lambda l, s: load_and_preprocess(l, s, tokenizer, model_type, cache=args.hf_cache)

        if args.train_lang == 'both':
            datasets = [
                lam_load_and_preprocess(lang, 'train')
                for lang in args.languages
            ]
            train_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        else:
            train_dataset = lam_load_and_preprocess(args.train_lang, 'train')

        if args.eval_lang == 'both':
            datasets = [
                lam_load_and_preprocess(lang, 'validation')
                for lang in args.languages
            ]
            eval_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        else:
            eval_dataset = lam_load_and_preprocess(args.eval_lang, 'validation')   #change to pl[20%] when required

        training_args = TrainingArguments(
            eval_strategy="steps",
            eval_steps=1000,
            output_dir=str(temporary_output_dir),
            save_strategy='steps',
            save_steps=1000,
            metric_for_best_model="precision",
            load_best_model_at_end=True,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=100,
            fp16=True,
            warmup_ratio=0.3,
            save_safetensors=False,
            disable_tqdm=True,
        )

        wrun = wandb.init(entity='aaronjencks-the-ohio-state-university', project=project_name, name=f'{model_type}-{args.train_lang}-{args.eval_lang}')

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        print(f"Training {model_type.upper()} model on {args.train_lang.upper()} → Evaluating on {args.eval_lang.upper()}")
        trainer.train()

        print(f"Final evaluation on {args.eval_lang.upper()} for model {model_type.upper()}")
        results = trainer.evaluate()
        print(results)

        wrun.finish()
