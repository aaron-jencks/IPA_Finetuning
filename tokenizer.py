import json
import pathlib

import torch

from transformers import GPT2TokenizerFast


eod_token = "<|endoftext|>"


def load_tokenizer(vocab: pathlib.Path, merges: pathlib.Path) -> GPT2TokenizerFast:
    # ---- Load tokenizer ----
    tokenizer = GPT2TokenizerFast(
        vocab_file=str(vocab),
        merges_file=str(merges),
        add_prefix_space=True,
    )

    # # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return tokenizer


class CharTokenizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab['character_map']
        self.pad_token = vocab['pad_token']
        self.unk_token = vocab['unknown_token']
        self.eos_token = vocab['sequence_token']
        self.inv_vocab = vocab['inverse_character_map']
        self.padding_side = 'right'

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    def encode(self, text: str) -> list:
        return [self.vocab.get(char, self.vocab[self.unk_token]) for char in text]

    def decode(self, ids: list) -> str:
        return ''.join(self.inv_vocab.get(i, self.unk_token) for i in ids)

    def __call__(self, texts, truncation=False, padding=False, max_length=None):
        # Support single string or list of strings
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_masks = []

        for text in texts:
            ids = self.encode(text)

            # Truncate
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]

            # Pad
            attn_mask = [1] * len(ids)
            if padding and max_length is not None and len(ids) < max_length:
                pad_id = self.vocab[self.pad_token]
                pad_len = max_length - len(ids)
                ids += [pad_id] * pad_len
                attn_mask += [0] * pad_len

            input_ids.append(ids)
            attention_masks.append(attn_mask)

        # Return in Hugging Face-compatible format
        return {
            "input_ids": input_ids if len(input_ids) > 1 else input_ids[0],
            "attention_mask": attention_masks if len(attention_masks) > 1 else attention_masks[0]
        }

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
        # List of dicts → keys: input_ids, attention_mask, (optionally label)
        if not isinstance(encoded_inputs, list):
            raise ValueError("pad() expects a list of encoded input dicts")

        # Figure out max length to pad to
        if max_length is None:
            max_length = max(len(x['input_ids']) for x in encoded_inputs)

        if pad_to_multiple_of is not None:
            if max_length % pad_to_multiple_of != 0:
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        batch = {k: [] for k in encoded_inputs[0].keys()}

        for item in encoded_inputs:
            length = len(item['input_ids'])
            pad_len = max_length - length

            # Pad input_ids and attention_mask
            padded_ids = item['input_ids'] + [self.pad_token_id] * pad_len
            padded_mask = item.get('attention_mask', [1] * length) + [0] * pad_len

            batch['input_ids'].append(padded_ids)
            batch['attention_mask'].append(padded_mask)

            # Copy over any other fields as-is (e.g., labels)
            for key, value in item.items():
                if key not in ('input_ids', 'attention_mask'):
                    batch[key].append(value)

        # Optionally return tensors
        if return_tensors == "pt":
            import torch
            for k in batch:
                batch[k] = torch.tensor(batch[k], dtype=torch.long)

        return batch

    def save_pretrained(self, save_directory):
        pass


def load_character_tokenizer(vocab: pathlib.Path) -> CharTokenizer:
    with open(vocab, 'r') as fp:
        vocab_dict = json.load(fp)
    return CharTokenizer(vocab_dict)

