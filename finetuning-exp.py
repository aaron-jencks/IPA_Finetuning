import argparse
import os
import pathlib

from datasets import load_dataset, concatenate_datasets
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, SchedulerType
import wandb

from hf_wrapper import GPTForSequenceClassification
from tokenizer import load_tokenizer
import utils
from utils import flatten_multi_features, load_pretrained_model, compute_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    hp = ap.add_argument_group('hyperparameters')
    hp.add_argument('--epochs', type=int, default=3, help='number of training epochs')
    hp.add_argument('--learning-rate', type=float, default=1e-5, help='The learning rate of the model')
    hp.add_argument('--warmup-ratio', type=float, default=0.05, help='The warmup ratio to use')
    hp.add_argument('--batch-size', type=int, default=16, help='The batch size of the model')
    hp.add_argument('--training-checkpoint-prefix', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/checkpoints'), help='The prefix of the temporary checkpoints folder')
    args = ap.parse_args()

    device = 'cpu' if not torch.cuda.is_available() or args.force_cpu else 'cuda'
    print('Using device:', device)

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

    project_name = f"{args.job_number}-{'-'.join(args.languages)}{'-medium' if args.is_medium else '-small'}-{args.train_lang}-{args.train_lang}-finetuning-{args.task}"
    temporary_output_dir = args.training_checkpoint_prefix / f"{project_name}-{args.train_lang}-both/"
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
            encoded = tokenizer(features, truncation=True, max_length=1024)
            encoded['label'] = examples[LANG_TO_LABELS[lang]]
            return encoded

        return ds.map(preprocess, batched=True, num_proc=os.cpu_count())

    # === Run both IPA and NORMAL models ===
    for model_type in ['ipa', 'normal']:
        print(f"🔧 Running setup for {model_type.upper()} model")
        vocab_path, merges_path = TOKENIZERS[model_type]
        print(f'loading tokenizer from {vocab_path} and {merges_path}')
        tokenizer = load_tokenizer(vocab_path, merges_path)
        base_model = load_pretrained_model(CHECKPOINTS[model_type], device)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.padding_side = tokenizer.padding_side
        model = GPTForSequenceClassification(base_model, num_classes=args.num_classes).to(device)

        lam_load_and_preprocess = lambda l, s: load_and_preprocess(l, s, tokenizer, model_type, cache=args.hf_cache)

        if args.train_lang == 'both':
            datasets = [
                lam_load_and_preprocess(lang, LANG_TO_TRAIN_SPLITS[lang])
                for lang in args.languages
            ]
            train_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        else:
            train_dataset = lam_load_and_preprocess(args.train_lang, LANG_TO_TRAIN_SPLITS[args.train_lang])

        if args.train_lang == 'both':
            datasets = [
                lam_load_and_preprocess(lang, LANG_TO_TEST_SPLITS[lang])
                for lang in args.languages
            ]
            eval_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        else:
            eval_dataset = lam_load_and_preprocess(args.train_lang, LANG_TO_TEST_SPLITS[args.train_lang])

        # if args.eval_lang == 'both':
        #    datasets = [
        #        lam_load_and_preprocess(lang, LANG_TO_TEST_SPLITS[lang])
        #        for lang in args.languages
        #    ]
        #    eval_dataset = concatenate_datasets(datasets).shuffle(seed=args.random_seed)
        #else:
        #    eval_dataset = lam_load_and_preprocess(args.eval_lang, LANG_TO_TEST_SPLITS[args.eval_lang])   #change to pl[20%] when required

        training_args = TrainingArguments(
            eval_strategy="steps",
            eval_steps=0.01,
            output_dir=str(temporary_output_dir),
            save_strategy='steps',
            save_steps=0.01,
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
            warmup_ratio=args.warmup_ratio,
            save_safetensors=False,
            disable_tqdm=True,
            no_cuda=args.force_cpu,
        )

        wrun = wandb.init(
          entity='aaronjencks-the-ohio-state-university', 
          project=project_name, 
          name=f'{model_type}-{args.train_lang}-{args.train_lang}',
          config={
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'warmup_ratio': args.warmup_ratio
          }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        print(f"Training {model_type.upper()} model on {args.train_lang.upper()} → Evaluating on {args.train_lang.upper()}")
        trainer.train()

        print(f"Final evaluation on {args.train_lang.upper()} for model {model_type.upper()}")
        results = trainer.evaluate()
        print(results)

        if args.train_lang == 'both':
            for eval_lang in args.languages:
                eval_dataset = lam_load_and_preprocess(eval_lang, LANG_TO_TEST_SPLITS[eval_lang])
                lang_results = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f'eval_{eval_lang}',
                )
                print(f'Evaluation for {eval_lang}:')
                print(lang_results)

        wrun.finish()
