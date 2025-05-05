import argparse
import pathlib
from typing import List

from datasets import load_dataset, load_from_disk
import evaluate
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import wandb

from hf_wrapper import GPTForSequenceClassification
from model import GPT, GPTConfig
from tokenizer import load_tokenizer, eod_token


def load_pretrained_model(path: pathlib.Path, device: str = 'cuda') -> GPT:
    # Load the pretrained model
    print(f"Loading pretrained model from {path}")
    checkpoint = torch.load(path, map_location=device)

    # Create the nanoGPT instance to load in saved weights
    gptconf = GPTConfig(**checkpoint['model_args'])
    pretrained_model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Clean up the saved state
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Only load the parameters that match the checkpoint weights
    model_dict = pretrained_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_state_dict)
    pretrained_model.load_state_dict(model_dict)
    pretrained_model.to(device)

    return pretrained_model


if __name__ == "__main__":
    # ---- Config ----
    ap = argparse.ArgumentParser(description='performs finetuning on a model on glue')
    ap.add_argument('--vocab', type=pathlib.Path, required=True)
    ap.add_argument('--merges', type=pathlib.Path, required=True)
    ap.add_argument('--model', type=pathlib.Path, required=True)
    ap.add_argument('--task', type=str, required=True,
                    choices=["sst2", "mrpc", "rte", "qnli", "qqp", "cola", "wnli"])
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--eval-interval', type=float, default=0.01)
    ap.add_argument('--context-size', type=int, default=1024)
    ap.add_argument('--learning-rate', type=float, default=2e-5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--hf-cache-dir', type=pathlib.Path, default=pathlib.Path('cache'))
    ap.add_argument('--dataset', type=str, default='nyu-mll/glue')
    ap.add_argument('--from-disk', action='store_true')
    ap.add_argument('--no-subset', action='store_true')
    ap.add_argument('--log-interval', type=float, default=0.01)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--wandb-project', type=str, default='ipa-finetune-english')
    ap.add_argument('--no-progress-bar', action='store_true')
    args = ap.parse_args()

    wandb.init(project=f'{args.wandb_project}-{args.task}', name=f'lr{args.learning_rate}-bs{args.batch_size}')


    # ---- Load Tokenizer ----
    tokenizer = load_tokenizer(args.vocab, args.merges)

    # ---- Load model ----
    base_model = load_pretrained_model(args.model, args.device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = GPTForSequenceClassification(base_model).to(args.device)

    # ---- Load dataset ----
    if args.no_subset:
        if args.from_disk:
            dataset = load_from_disk(args.dataset)
        else:
            dataset = load_dataset(args.dataset)
    else:
        if args.from_disk:
            raise Exception('can\'t load from disk with subset.')
        dataset = load_dataset(args.dataset, args.task, cache_dir=str(args.hf_cache_dir))

    def flatten_multi_features(examples, features: List[str]) -> List[str]:
        separator = f'\n\n{eod_token}\n\n'
        return [separator.join(example) for example in zip(*[examples[f] for f in features])]

    # ---- Preprocessing ----
    def preprocess_function(examples):
        if 'premise' in examples:
            feature = flatten_multi_features(examples, ['premise', 'hypothesis'])
        elif 'question' in examples:
            if 'sentence' in examples:
                feature = flatten_multi_features(examples, ['question', 'sentence'])
            else:
                feature = flatten_multi_features(examples, ['question', 'hypothesis'])
        elif 'sentence1' in examples:
            feature = flatten_multi_features(examples, ['sentence1', 'sentence2'])
        elif 'question1' in examples:
            feature = flatten_multi_features(examples, ['question1', 'question2'])
        else:
            feature = examples['sentence']

        return tokenizer(feature, truncation=True, max_length=args.context_size)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # ---- Data collator ----
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---- Metrics ----
    metric = evaluate.load("glue", args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.from_numpy(logits).argmax(dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        save_strategy="steps",
        save_steps=args.eval_interval,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=args.log_interval,
        report_to='wandb',
        disable_tqdm=True,
        warmup_ratio=0.3  # also try 0.2 up to 0.3
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    trainer.train()
