"""Shared attention utilities for transformer models."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, get_abstract_mesh


def _attention_output_sharding(q: jax.Array) -> NamedSharding:
    sharding = getattr(q, "sharding", None)
    if isinstance(sharding, NamedSharding):
        return sharding
    return NamedSharding(get_abstract_mesh(), PartitionSpec("fsdp", None, "tp", None))

def dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    is_causal: bool,
    head_dim: int,
) -> jax.Array:
    """Compute dot-product attention with automatic backend selection.

    Uses cuDNN on GPU for memory-efficient attention. Falls back to XLA for CPU/TPU.

    Args:
        q: Query tensor of shape [batch, q_len, num_heads, head_dim]
        k: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        v: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        attention_mask: Mask of shape [batch, kv_len] where 1 = valid, 0 = masked.
            Sequences must be right-padded (valid tokens first, then padding).
        is_causal: Whether to apply causal masking (for prefill/training)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [batch, q_len, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5
    output_sharding = _attention_output_sharding(q)

    if jax.default_backend() == "gpu":
        attention_impl = jax.sharding.auto_axes(
            lambda query, key, value, mask: jax.nn.dot_product_attention(
                query,
                key,
                value,
                scale=scale,
                mask=mask[:, None, None, :].astype(bool),
                is_causal=is_causal,
                implementation="xla",
            ),
            out_sharding=output_sharding,
        )
        return jax.sharding.reshard(attention_impl(q, k, v, attention_mask), output_sharding)

    # CPU/TPU fallback
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale, mask=attention_mask[:, None, None, :].astype(bool), is_causal=is_causal
    )
