import torch
import torch.nn as nn
from typing import Callable


def Param(*shape: int, dtype: torch.dtype, kernel_init: Callable, device: torch.device | None = None):
    """Create a parameter with the given shape and initializer."""
    param = torch.empty(*shape, dtype=dtype, device=device)
    kernel_init(param)
    return nn.Parameter(param)


def prepare_routing(
    tokens: torch.Tensor, indices: torch.Tensor, num_groups: int, adapter_indices: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepare inputs for grouped_mm operations by sorting tokens by group.

    Args:
        tokens: Tensor of shape (num_tokens, ...) to be sorted by group
        indices: Tensor of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional tensor of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after grouped operations
        sorted_adapter_indices: Adapter indices sorted together with tokens (if provided)
    """
    sort_indices = torch.argsort(indices)
    sorted_tokens = tokens[sort_indices]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_indices]
    group_sizes = torch.bincount(indices, minlength=num_groups)
    unsort_indices = torch.argsort(sort_indices)
    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices
