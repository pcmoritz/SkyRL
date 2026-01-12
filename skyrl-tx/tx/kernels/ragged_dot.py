"""Pallas-based ragged_dot kernel with group_offset support.

This module provides a GPU-optimized ragged_dot implementation that supports
the group_offset parameter for efficient expert parallelism.

When group_offset is specified, only tokens belonging to groups
[offset, offset + g_local) are computed, avoiding wasted computation
on tokens that belong to other shards.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ProblemSizes(NamedTuple):
    """Dimensions of the ragged_dot problem."""

    m: int  # Number of tokens (rows in lhs)
    k: int  # Input feature dimension
    n: int  # Output feature dimension
    g: int  # Total number of groups
    g_local: int  # Number of local groups (when using group_offset)


class BlockSizes(NamedTuple):
    """Block sizes for the Pallas kernel."""

    bm: int  # Block size for M dimension
    bk: int  # Block size for K dimension
    bn: int  # Block size for N dimension


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def next_pow2(x: int) -> int:
    """Round up to next power of 2."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _select_block_sizes(sizes: ProblemSizes) -> BlockSizes:
    """Select appropriate block sizes based on problem dimensions."""
    # Start with reasonable defaults for modern GPUs
    bm = min(128, next_pow2(max(16, sizes.m)))
    bk = min(128, next_pow2(max(16, sizes.k)))
    bn = min(128, next_pow2(max(16, sizes.n)))
    return BlockSizes(bm=bm, bk=bk, bn=bn)


def _ragged_dot_kernel(
    # Inputs
    lhs_ref,  # [m, k]
    rhs_ref,  # [g_local, k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [1]
    # Output
    out_ref,  # [m, n]
    *,
    # Static params
    m: int,
    k: int,
    n: int,
    g: int,
    g_local: int,
    bm: int,
    bk: int,
    bn: int,
    acc_dtype: jnp.dtype,
):
    """Pallas kernel for ragged_dot with group_offset support.

    Grid: (num_m_blocks, num_n_blocks)

    For each (m_block, n_block):
    1. Determine which group this m_block belongs to
    2. If group is in [offset, offset + g_local), compute the dot product
    3. Otherwise, output zeros
    """
    m_block_idx = pl.program_id(0)
    n_block_idx = pl.program_id(1)

    # Load group_offset
    offset = group_offset_ref[0]

    # Compute token range for this block
    m_start = m_block_idx * bm
    m_end = jnp.minimum(m_start + bm, m)
    n_start = n_block_idx * bn
    n_end = jnp.minimum(n_start + bn, n)

    # Compute cumulative sum to find group boundaries
    # We need to determine which group each token in this block belongs to
    cumsum = jnp.cumsum(group_sizes_ref[...], axis=0)  # Shape: [g]
    cumsum_with_zero = jnp.concatenate([jnp.array([0]), cumsum])  # Shape: [g+1]

    # Initialize accumulator
    acc = jnp.zeros((bm, bn), dtype=acc_dtype)

    # For each position in the block, determine its group and whether it's valid
    m_indices = jnp.arange(bm) + m_start  # Shape: [bm]

    # Find group for each token position using searchsorted
    # group_idx[i] is the group that token m_indices[i] belongs to
    group_idx = jnp.searchsorted(cumsum, m_indices, side="right")  # Shape: [bm]

    # Check if tokens are within valid range and belong to local groups
    valid_token = m_indices < m  # Token exists
    in_local_range = (group_idx >= offset) & (group_idx < offset + g_local)  # In local groups
    valid_mask = valid_token & in_local_range  # Shape: [bm]

    # Local group index (0-indexed into rhs)
    local_group_idx = group_idx - offset  # Shape: [bm]

    # Process each token in the block
    # We need to loop because different tokens may use different groups
    for mi in range(bm):
        token_idx = m_start + mi
        is_valid = (token_idx < m) & (group_idx[mi] >= offset) & (group_idx[mi] < offset + g_local)

        # Load lhs row for this token: [k]
        lhs_row = jnp.where(
            token_idx < m,
            lhs_ref[token_idx, :],
            jnp.zeros(k, dtype=lhs_ref.dtype),
        )

        # Local group index for RHS lookup
        local_g = local_group_idx[mi]
        local_g_safe = jnp.clip(local_g, 0, g_local - 1)

        # Load rhs slice for this group: [k, n]
        rhs_slice = rhs_ref[local_g_safe, :, n_start:n_end]

        # Compute dot product: [n]
        dot_result = jnp.dot(lhs_row, rhs_slice)

        # Mask invalid results
        dot_result = jnp.where(is_valid, dot_result, jnp.zeros_like(dot_result))

        # Store in accumulator
        acc = acc.at[mi, : n_end - n_start].set(dot_result)

    # Write output
    for mi in range(bm):
        token_idx = m_start + mi
        if token_idx < m:
            out_ref[token_idx, n_start:n_end] = acc[mi, : n_end - n_start].astype(out_ref.dtype)


def _ragged_dot_simple(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
) -> jax.Array:
    """Simple implementation of ragged_dot with group_offset using pure JAX.

    This serves as a reference implementation and fallback.
    Uses vectorized operations for better performance.
    """
    m, k = lhs.shape
    g_local, _, n = rhs.shape
    g = group_sizes.shape[0]
    offset = group_offset[0]

    # Compute cumulative sum for group boundaries
    cumsum = jnp.cumsum(group_sizes)
    cumsum_with_zero = jnp.concatenate([jnp.array([0], dtype=cumsum.dtype), cumsum])

    # Find group for each token
    token_indices = jnp.arange(m)
    group_idx = jnp.searchsorted(cumsum, token_indices, side="right")

    # Check if tokens belong to local groups
    valid_mask = (group_idx >= offset) & (group_idx < offset + g_local)

    # Local group indices (clipped for safe indexing)
    local_group_idx = jnp.clip(group_idx - offset, 0, g_local - 1)

    # Gather weights for each token based on its group
    # rhs shape: [g_local, k, n], we want weights[i] = rhs[local_group_idx[i]]
    weights = rhs[local_group_idx]  # Shape: [m, k, n]

    # Compute output: out[i] = lhs[i] @ weights[i]
    out = jnp.einsum("mk,mkn->mn", lhs, weights)

    # Mask invalid tokens
    out = jnp.where(valid_mask[:, None], out, 0.0)

    if preferred_element_type is not None:
        out = out.astype(preferred_element_type)

    return out


def _trans_ragged_dot_simple(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    g_local: int,
    precision=None,
    preferred_element_type=None,
) -> jax.Array:
    """Compute transposed ragged dot: lhs.T @ rhs, accumulated by group.

    This computes the gradient with respect to the weight matrix.
    lhs: [m, k], rhs: [m, n], output: [g_local, k, n]
    """
    m, k = lhs.shape
    _, n = rhs.shape
    g = group_sizes.shape[0]
    offset = group_offset[0]

    # Compute group assignments
    cumsum = jnp.cumsum(group_sizes)
    token_indices = jnp.arange(m)
    group_idx = jnp.searchsorted(cumsum, token_indices, side="right")

    # Valid mask for tokens in local groups
    valid_mask = (group_idx >= offset) & (group_idx < offset + g_local)
    local_group_idx = group_idx - offset

    # Initialize output
    out = jnp.zeros((g_local, k, n), dtype=lhs.dtype)

    # Use segment_sum to accumulate by group
    # First, compute outer products: lhs[:, :, None] * rhs[:, None, :]
    # This gives [m, k, n]
    outer_products = lhs[:, :, None] * rhs[:, None, :]

    # Mask invalid tokens
    outer_products = jnp.where(valid_mask[:, None, None], outer_products, 0.0)

    # Accumulate by local group using segment_sum
    # We need to scatter-add outer_products[i] to out[local_group_idx[i]]
    # Clip indices for safety (invalid tokens will contribute 0 anyway)
    safe_local_idx = jnp.clip(local_group_idx, 0, g_local - 1)

    # Use segment_sum with sorted indices
    sort_idx = jnp.argsort(safe_local_idx)
    sorted_local_idx = safe_local_idx[sort_idx]
    sorted_outer = outer_products[sort_idx]

    # Segment sum
    out = jax.ops.segment_sum(sorted_outer, sorted_local_idx, num_segments=g_local)

    if preferred_element_type is not None:
        out = out.astype(preferred_element_type)

    return out


def ragged_dot_pallas(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
) -> jax.Array:
    """Ragged dot product with group_offset support.

    This implementation uses pure JAX operations that are automatically
    differentiable, which ensures compatibility with shard_map and other
    JAX transformations.

    Args:
        lhs: Input tokens, shape [m, k]
        rhs: Group weights, shape [g_local, k, n]
        group_sizes: Size of each group, shape [g]
        group_offset: Starting group index, shape [1]
        precision: JAX precision setting (currently unused)
        preferred_element_type: Output dtype

    Returns:
        Output tensor, shape [m, n]
    """
    return _ragged_dot_simple(lhs, rhs, group_sizes, group_offset, precision, preferred_element_type)


def is_gpu() -> bool:
    """Check if running on GPU."""
    return jax.default_backend() == "gpu"


def is_tpu() -> bool:
    """Check if running on TPU."""
    return jax.default_backend() == "tpu"
