import os
import pathlib
import random
from argparse import ArgumentParser
from typing import List
import logging

import torch
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

from model import GPT, GPTConfig
from tokenizer import eod_token


logger = logging.getLogger(__name__)


def create_downsampled_dataset(ds: Dataset, samples: int) -> Dataset:
    if samples > len(ds):
        logger.warning(f'sample size > population size: {samples} > {len(ds)}')
    if samples == len(ds):
        return ds
    idxes = list(range(len(ds)))
    sampled_idxs = random.sample(idxes, samples)
    downsample = ds.select(sampled_idxs)
    return downsample


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


def flatten_multi_features(examples, features: List[str], sequence_token: str = eod_token) -> List[str]:
    sep = f'\n\n{sequence_token}\n\n'
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
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0)
    }


def setup_default_args(ap: ArgumentParser) -> ArgumentParser:
    ap.add_argument('job_number', type=str, help='slurm job number')
    ap.add_argument('config', type=pathlib.Path, nargs='+', help='paths to config files')
    ap.add_argument('--default-config', type=pathlib.Path, default=pathlib.Path('config/default.json'),
                    help='path to the default config file')
    ap.add_argument('--language-database', type=pathlib.Path, default=pathlib.Path('config/language-database.json'),
                    help='path to the default config file')
    ap.add_argument('--cpus', type=int, default=os.cpu_count(), help='number of cpus')
    ap.add_argument('--debug', action='store_true', help='enable debug mode')
    return ap
