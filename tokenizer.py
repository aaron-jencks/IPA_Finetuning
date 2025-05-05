import pathlib

from transformers import GPT2TokenizerFast


eod_token = "<|endoftext|>"


def load_tokenizer(vocab: pathlib.Path, merges: pathlib.Path) -> GPT2TokenizerFast:
    # ---- Load tokenizer ----
    tokenizer = GPT2TokenizerFast(
        vocab_file=str(vocab),
        merges_file=str(merges),
        add_prefix_space=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': [eod_token]})

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return tokenizer