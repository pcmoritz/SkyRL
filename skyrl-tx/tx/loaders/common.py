import importlib
from operator import attrgetter
from typing import Callable, Iterator

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

LoaderIterator = Iterator[tuple[dict[str, torch.Tensor], dict[str, str]]]


def get_loader(loader_name: str) -> Callable[[PreTrainedTokenizer, Dataset, int], LoaderIterator]:
    module_name, function_name = loader_name.split(".", 1)
    try:
        module = importlib.import_module(module_name)
        return attrgetter(function_name)(module)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load {function_name} from {module_name}: {e}")
