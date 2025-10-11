import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from tx.loaders.common import LoaderIterator


def chat(tokenizer: PreTrainedTokenizer, dataset: Dataset, batch_size: int) -> LoaderIterator:
    "Data loader that applies the chat template. It returns an iterator over (batch, metrics) elements."

    for data in dataset.shuffle().iter(batch_size=batch_size):
        batch = tokenizer.apply_chat_template(data["messages"], tokenize=False)
        # We pad to multiples of 512 here for consistent shapes
        batch = tokenizer(batch, return_tensors="pt", padding=True, pad_to_multiple_of=512)
        yield {
            "text": batch["input_ids"][:, :-1],
            "attention_mask": batch["attention_mask"][:, :-1],
            "target": batch["input_ids"][:, 1:],
        }, {"shape": str(batch["input_ids"].shape), "tokens": str(batch["attention_mask"].sum().item())}
