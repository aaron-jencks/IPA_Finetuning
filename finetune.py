import argparse
import pathlib

from datasets import load_dataset, load_from_disk
import evaluate
import torch
from transformers import (
    GPT2TokenizerFast,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ---- Config ----
ap = argparse.ArgumentParser(description='performs finetuning on a model on glue')
ap.add_argument('--vocab', type=pathlib.Path, required=True)
ap.add_argument('--merges', type=pathlib.Path, required=True)
ap.add_argument('--model', type=pathlib.Path, required=True)
ap.add_argument('--task', type=str, required=True,
                choices=["sst2", "mrpc", "rte", "qnli", "qqp", "cola", "wnli", "stsb"])
ap.add_argument('--output', type=pathlib.Path, required=True)
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--context-size', type=int, default=1024)
ap.add_argument('--batch-size', type=int, default=8)
ap.add_argument('--hf-cache-dir', type=pathlib.Path, default=pathlib.Path('cache'))
ap.add_argument('--dataset', type=str, default='nyu-mll/glue')
ap.add_argument('--from-disk', action='store_true')
ap.add_argument('--no-subset', action='store_true')
ap.add_argument('--log-dir', type=pathlib.Path, default=pathlib.Path('logs'))
args = ap.parse_args()

eod_token = "<|endoftext|>"

# ---- Load tokenizer ----
tokenizer = GPT2TokenizerFast(
    vocab_file=str(args.vocab_fname),
    merges_file=str(args.merges_fname),
    add_prefix_space=True,
)
tokenizer.add_special_tokens({'additional_special_tokens': [eod_token]})

# Set pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- Load model ----
model = AutoModelForSequenceClassification.from_pretrained(
    "./your_gpt2_model",
    num_labels=2,
)
model.resize_token_embeddings(len(tokenizer))

# ---- Load dataset ----
if args.no_subset:
    if args.from_disk:
        dataset = load_from_disk(args.parent_dataset)
    else:
        dataset = load_dataset(args.parent_dataset)
else:
    if args.from_disk:
        raise Exception('can\'t load from disk with subset.')
    dataset = load_dataset(args.parent_dataset, args.dataset, cache_dir=str(args.hf_cache))


# ---- Preprocessing ----
def preprocess_function(examples):
    feature = examples['sentence']
    if 'premise' in examples:
        feature = examples['premise'] + eod_token + examples['hypothesis']
    elif 'question' in examples:
        feature = examples['question'] + eod_token + examples['sentence']
    elif 'sentence1' in examples:
        feature = examples['sentence1'] + eod_token + examples['sentence2']

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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
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
