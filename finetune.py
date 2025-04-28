from transformers import (
    GPT2TokenizerFast,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import torch

# ---- Config ----
vocab_fname = "./path_to_vocab.json"
merges_fname = "./path_to_merges.txt"
eod_token = "<|endoftext|>"
glue_task = "sst2"  # or "mrpc", "rte", "qnli", "qqp", "cola", "wnli", "stsb"
output_dir = "./finetuned_gpt2"
num_train_epochs = 3
batch_size = 8

# ---- Load tokenizer ----
tokenizer = GPT2TokenizerFast(
    vocab_file=str(vocab_fname),
    merges_file=str(merges_fname),
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
dataset = load_dataset("glue", glue_task)

# ---- Preprocessing ----
def preprocess_function(examples):
    if "sentence1" in examples and "sentence2" in examples:
        return tokenizer(
            examples['sentence1'], examples['sentence2'],
            truncation=True, padding='max_length', max_length=128
        )
    else:
        return tokenizer(
            examples['sentence'],
            truncation=True, padding='max_length', max_length=128
        )

encoded_dataset = dataset.map(preprocess_function, batched=True)

# ---- Data collator ----
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---- Metrics ----
import evaluate
metric = evaluate.load("glue", glue_task)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.from_numpy(logits).argmax(dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# ---- Training arguments ----
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
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
