import pathlib
from typing import List

import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from model import GPT, GPTConfig
from tokenizer import eod_token


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


def load_random_from_pretrained_model(path: pathlib.Path, device: str = 'cuda') -> GPT:
    checkpoint = torch.load(path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
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
