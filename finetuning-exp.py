import argparse
import os
import pathlib

from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, SchedulerType
import wandb

from hf_wrapper import GPTForSequenceClassification
from tokenizer import load_tokenizer
from utils import flatten_multi_features, load_pretrained_model, compute_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('job_number', type=str, help='slurm job number')
    ap.add_argument('task', type=str, help='the classification task used for wandb job name')
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
    hp.add_argument('--learning-rate', type=float, default=1e-4, help='The learning rate of the model')
    hp.add_argument('--batch-size', type=int, default=16, help='The batch size of the model')
    hp.add_argument('--hf-cache', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/cache'), help='The huggingface cache folder')
    hp.add_argument('--training-checkpoint-prefix', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/checkpoints'), help='The prefix of the temporary checkpoints folder')
    dp = ap.add_argument_group('dataset')
    dp.add_argument('--lang-1-features', type=str, nargs='+', required=True, help='The training features of language 1')
    dp.add_argument('--lang-2-features', type=str, nargs='+', required=True, help='The training features of language 2')
    dp.add_argument('--eval-feature', type=str, nargs='+', required=True, help='The validation feature for each dataset')
    dp.add_argument('--num-classes', type=int, default=3, help='The number of classes')
    dp.add_argument('--lang-1-splits', type=str, nargs=2, default=['train', 'validation'], help='The splits of language 1, must be "train eval"')
    dp.add_argument('--lang-2-splits', type=str, nargs=2, default=['train', 'validation'], help='The splits of language 2, must be "train eval"')
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
        for lang, features in zip(args.languages, [args.lang_1_features, args.lang_2_features])
    }

    LANG_TO_LABELS = {
        lang: label_feature
        for lang, label_feature in zip(args.languages, args.eval_feature)
    }

    LANG_TO_TRAIN_SPLITS = {
        lang: splits[0]
        for lang, splits in zip(args.languages, [args.lang_1_splits, args.lang_2_splits])
    }

    LANG_TO_TEST_SPLITS = {
        lang: splits[1]
        for lang, splits in zip(args.languages, [args.lang_1_splits, args.lang_2_splits])
    }

    project_name = f"{args.job_number}-{'-'.join(args.languages)}{'-medium' if args.is_medium else '-small'}-{args.train_lang}-{args.eval_lang}-finetuning-{args.task}"
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
        fields = get_fields(lang, model_type)

        def preprocess(examples):
            features = flatten_multi_features(examples, fields)
            encoded = tokenizer(features, truncation=True, max_length=args.context_size)
            encoded['label'] = examples[LANG_TO_LABELS[lang]]
            return encoded

        return ds.map(preprocess, batched=True, num_proc=os.cpu_count())

    # === Run both IPA and NORMAL models ===
    for model_type in ['ipa', 'normal']:
        print(f"\n🔧 Running setup for {model_type.upper()} model")
        vocab_path, merges_path = TOKENIZERS[model_type]
        print(f'loading tokenizer from {vocab_path} and {merges_path}')
        tokenizer = load_tokenizer(vocab_path, merges_path)
        base_model = load_pretrained_model(CHECKPOINTS[model_type], 'cuda')
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.padding_side = tokenizer.padding_side
        model = GPTForSequenceClassification(base_model, num_classes=args.num_classes).to('cuda')

        lam_load_and_preprocess = lambda l, s: load_and_preprocess(l, s, tokenizer, model_type, cache=args.hf_cache)

        if args.train_lang == 'both':
            datasets = [
                lam_load_and_preprocess(lang, LANG_TO_TRAIN_SPLITS[lang])
                for lang in args.languages
            ]
            train_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        else:
            train_dataset = lam_load_and_preprocess(args.train_lang, LANG_TO_TRAIN_SPLITS[args.train_lang])

        if args.eval_lang == 'both':
            datasets = [
                lam_load_and_preprocess(lang, LANG_TO_TEST_SPLITS[lang])
                for lang in args.languages
            ]
            eval_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        else:
            eval_dataset = lam_load_and_preprocess(args.eval_lang, LANG_TO_TEST_SPLITS[args.eval_lang])   #change to pl[20%] when required

        training_args = TrainingArguments(
            eval_strategy="steps",
            eval_steps=1000,
            output_dir=str(temporary_output_dir),
            save_strategy='steps',
            save_steps=1000,
            metric_for_best_model="f1",
            greater_is_better=True,  # Remember to update this if you update the metric used
            load_best_model_at_end=True,
            learning_rate=args.learning_rate,
            # lr_scheduler_type=SchedulerType.COSINE,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            max_grad_norm=1.0,  # gradient clipping
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
