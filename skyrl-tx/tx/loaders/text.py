import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from tx.loaders.common import LoaderIterator


def text(tokenizer: PreTrainedTokenizer, dataset: Dataset, batch_size: int) -> LoaderIterator:
    "Data loader for text data. It returns an iterator over (batch, metrics) elements."

    for data in dataset.iter(batch_size=batch_size):
        # We pad to multiples of 128 here for consistent shapes
        batch = tokenizer(data["text"], return_tensors="pt", padding=True, pad_to_multiple_of=128)
        yield {
            "text": batch["input_ids"][:, :-1],
            "attention_mask": batch["attention_mask"][:, :-1],
            "target": batch["input_ids"][:, 1:],
        }, {"shape": str(batch["input_ids"].shape), "tokens": str(batch["attention_mask"].sum().item())}
