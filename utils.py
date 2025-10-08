import pathlib
from argparse import ArgumentParser
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
    ap.add_argument('task', type=str, help='the classification task used for wandb job name')
    ap.add_argument('ipa_model', type=str, help="Name of the ipa model")
    ap.add_argument('normal_model', type=str, help="Name of the normal model")
    ap.add_argument('ipa_tokenizer_prefix', type=str, help="Name of the ipa tokenizer prefix")
    ap.add_argument('normal_tokenizer_prefix', type=str, help="Name of the normal tokenizer prefix")
    ap.add_argument('languages', type=str, nargs=2, help="List of languages")
    ap.add_argument('language_datasets', type=str, nargs=2, help="Language datasets")
    ap.add_argument('--train-lang', type=str, default='both', help='The training language')
    ap.add_argument('--checkpoint-prefix', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/checkpoints'),
                    help='the prefix of the checkpoints folder')
    ap.add_argument('--tokenizer-prefix', type=pathlib.Path, 
                    default=pathlib.Path('/fs/ess/PAS2836/ipa_gpt/tokenizers'),
                    help='the prefix of the tokenizers folder')
    ap.add_argument('--is-medium', action='store_true', help='indicates that the model is a medium model')
    ap.add_argument('--random-seed', type=int, default=42, help='random seed')
    ap.add_argument('--force-cpu', action='store_true', help='force cpu only')
    ap.add_argument('--hf-cache', type=pathlib.Path, default=pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/cache'),
                    help='The huggingface cache folder')
    dp = ap.add_argument_group('dataset')
    dp.add_argument('--lang-1-features', type=str, nargs='+', required=True, help='The training features of language 1')
    dp.add_argument('--lang-2-features', type=str, nargs='+', required=True, help='The training features of language 2')
    dp.add_argument('--eval-feature', type=str, nargs='+', required=True,
                    help='The validation feature for each dataset')
    dp.add_argument('--num-classes', type=int, default=3, help='The number of classes')
    dp.add_argument('--lang-1-splits', type=str, nargs=2, default=['train', 'validation'],
                    help='The splits of language 1, must be "train eval"')
    dp.add_argument('--lang-2-splits', type=str, nargs=2, default=['train', 'validation'],
                    help='The splits of language 2, must be "train eval"')
    return ap


class DatasetInfo:
    def __init__(
            self,
            name: str,
            input_features: List[str], eval_features: str, train_split: str, eval_split: str,
    ):
        self.name = name
        self.input_features = input_features
        self.eval_features = eval_features
        self.train_split = train_split
        self.eval_split = eval_split


class LanguageMapper:
    def __init__(
            self, 
            checkpoint_prefix: pathlib.Path, tokenizer_prefix: pathlib.Path,
            ipa_model_prefix: str, normal_model_prefix: str,
            ipa_tokenizer_prefix: str, normal_tokenizer_prefix: str,
            languages: List[str], 
            datasets: List[str],
            input_features: List[List[str]], eval_features: List[str],
            train_splits: List[str], eval_splits: List[str],
            
    ):
        self.checkpoints = {
            "ipa": checkpoint_prefix / f"{ipa_model_prefix}/ckpt.pt",
            "normal": checkpoint_prefix / f"{normal_model_prefix}/ckpt.pt",
        }

        self.tokenizers = {
            "ipa": (
                tokenizer_prefix / f"{ipa_tokenizer_prefix}-vocab.json",
                tokenizer_prefix / f"{ipa_tokenizer_prefix}-merges.txt",
            ),
            "normal": (
                tokenizer_prefix / f"{normal_tokenizer_prefix}-vocab.json",
                tokenizer_prefix / f"{normal_tokenizer_prefix}-merges.txt",
            ),
        }

        self.datasets = {}
        for lang, d, fi, ef, ts, es in zip(languages, datasets, input_features, eval_features, train_splits, eval_splits):
            self.datasets[lang] = DatasetInfo(
                d, fi, ef, ts, es,
            )

    def get_dataset_info(self, language: str) -> DatasetInfo:
        return self.datasets[language]
