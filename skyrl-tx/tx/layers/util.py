from dataclasses import dataclass
from flax import nnx
import jax
from jax import numpy as jnp


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


@jax.tree_util.register_dataclass
@dataclass
class LoRARoutingInfo:
    """Pre-computed routing information for LoRA layers.

    Computing this once and reusing across all LoRA layers avoids
    redundant argsort operations (1000+ per forward pass -> 1).
    """
    adapter_indices: jax.Array     # [B] original adapter indices per batch element
    sorted_indices: jax.Array      # [B*T] indices to sort tokens by adapter
    unsort_indices: jax.Array      # [B*T] indices to restore original order
    group_sizes: jax.Array         # [max_adapters] number of tokens per adapter
    adapter_indices_sorted: jax.Array  # [B*T] adapter indices in sorted order


def compute_lora_routing(
    adapter_indices: jax.Array,
    seq_len: int,
    max_lora_adapters: int,
) -> LoRARoutingInfo:
    """Compute routing info once for all LoRA layers in a forward pass.

    Args:
        adapter_indices: [B] adapter index per batch element
        seq_len: sequence length T
        max_lora_adapters: maximum number of adapters

    Returns:
        LoRARoutingInfo with pre-computed sorting indices
    """
    adapter_indices_expanded = jnp.repeat(adapter_indices, seq_len)
    sorted_indices = jnp.argsort(adapter_indices_expanded)
    unsort_indices = jnp.argsort(sorted_indices)
    group_sizes = jnp.bincount(adapter_indices_expanded, length=max_lora_adapters)
    adapter_indices_sorted = adapter_indices_expanded[sorted_indices]

    return LoRARoutingInfo(
        adapter_indices=adapter_indices,
        sorted_indices=sorted_indices,
        unsort_indices=unsort_indices,
        group_sizes=group_sizes,
        adapter_indices_sorted=adapter_indices_sorted,
    )


def prepare_routing(
    tokens: jax.Array, indices: jax.Array, num_groups: int, adapter_indices: jax.Array | None = None
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Prepare inputs for ragged_dot operations by sorting tokens by group.

    Args:
        tokens: Array of shape (num_tokens, ...) to be sorted by group
        indices: Array of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional array of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after ragged operations
    """
    sort_indices = jnp.argsort(indices)
    sorted_tokens = tokens[sort_indices]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_indices]
    group_sizes = jnp.bincount(indices, length=num_groups)
    unsort_indices = jnp.argsort(sort_indices)
    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices
