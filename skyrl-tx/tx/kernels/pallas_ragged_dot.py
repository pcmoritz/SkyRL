# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pallas ragged_dot implementation with boundary group skipping."""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
from jax import lax
from jax import numpy as jnp
from jax import custom_vjp
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu


class _KernelConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    max_concurrent_steps: int
    grid_block_n: int


def _choose_kernel_config(k: int) -> _KernelConfig:
    """Pick a conservative kernel configuration that works on most shapes."""

    def _select_block_k(value: int) -> int:
        for candidate in (128, 64, 32, 16):
            if value % candidate == 0:
                return candidate
        raise ValueError(f"k={value} must be divisible by 16")

    block_k = _select_block_k(k)
    # Use a single block_n that is supported by the WGMMA kernel.
    block_n = 64
    return _KernelConfig(block_m=64, block_n=block_n, block_k=block_k, max_concurrent_steps=3, grid_block_n=1)




@custom_vjp
def ragged_dot_with_group_offset(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    return _ragged_dot_forward_impl(lhs, rhs, group_sizes, group_offset)


def ragged_dot_with_group_offset_fwd(lhs, rhs, group_sizes, group_offset):
    y = _ragged_dot_forward_impl(lhs, rhs, group_sizes, group_offset)
    return y, (lhs, rhs, group_sizes, group_offset)


def ragged_dot_with_group_offset_bwd(res, cotangent):
    lhs, rhs, group_sizes, group_offset = res
    grad_lhs, grad_rhs = _ragged_dot_backward(lhs, rhs, group_sizes, group_offset, cotangent)
    return grad_lhs, grad_rhs, None, None


ragged_dot_with_group_offset.defvjp(ragged_dot_with_group_offset_fwd, ragged_dot_with_group_offset_bwd)


def _ragged_dot_forward_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    """GPU ragged_dot implementation that skips boundary tokens.

    Args:
        lhs: Token matrix of shape (M, K) sorted by expert id.
        rhs: Expert weights of shape (G_local, K, N).
        group_sizes: Global group sizes (num_experts_total,).
        group_offset: Scalar array with the starting expert index for this shard.

    Returns:
        Output array of shape (M, N) containing zeros for non-local experts.
    """
    if os.environ.get("SKYRL_DEBUG_RAGGED_DOT_LAYOUTS"):
        def _spec(x):
            sharding = getattr(x, "sharding", None)
            spec = getattr(sharding, "spec", None)
            return spec if spec is not None else sharding

        jax.debug.print(
            "ragged_dot kernel specs lhs={} rhs={} groups={} offset={}",
            _spec(lhs),
            _spec(rhs),
            _spec(group_sizes),
            _spec(group_offset),
        )

    (m, k) = lhs.shape
    g_local, k_rhs, n = rhs.shape
    orig_n = n

    if k != k_rhs:
        raise ValueError(f"Incompatible shapes lhs={lhs.shape} rhs={rhs.shape}")
    if g_local == 0:
        raise ValueError("rhs must contain at least one expert")

    offset = jnp.asarray(group_offset, dtype=jnp.int32).reshape(())
    sizes = jnp.asarray(group_sizes, dtype=jnp.int32)
    if sizes.ndim != 1:
        raise ValueError("group_sizes must be rank 1")

    group_boundaries = jnp.cumulative_sum(sizes, include_initial=True)
    shard_start = lax.dynamic_index_in_dim(group_boundaries, offset, axis=0, keepdims=False)
    shard_end = lax.dynamic_index_in_dim(group_boundaries, offset + g_local, axis=0, keepdims=False)
    prefix = shard_start
    suffix = jnp.asarray(m, dtype=jnp.int32) - shard_end
    local_group_sizes = lax.dynamic_slice_in_dim(sizes, offset, g_local, axis=0)
    extended_group_sizes = jnp.concatenate([prefix[jnp.newaxis], local_group_sizes, suffix[jnp.newaxis]], axis=0)

    config = _choose_kernel_config(k)
    n_pad = (-n) % config.block_n
    if n_pad:
        rhs = jnp.pad(rhs, ((0, 0), (0, 0), (0, n_pad)))
        n = n + n_pad
    if m > 0:
        config = config._replace(block_m=min(config.block_m, m))
    if n > 0:
        config = config._replace(block_n=min(config.block_n, n))
    zero_slice = jnp.zeros((1, k, n), dtype=rhs.dtype)
    padded_rhs = jnp.concatenate([zero_slice, rhs, zero_slice], axis=0)

    result = _pallas_ragged_dot(
        lhs,
        padded_rhs,
        group_sizes=extended_group_sizes,
        block_m=config.block_m,
        block_n=config.block_n,
        block_k=config.block_k,
        max_concurrent_steps=config.max_concurrent_steps,
        grid_block_n=config.grid_block_n,
    )
    if n_pad:
        result = result[:, :orig_n]
    token_idx = jnp.arange(m, dtype=jnp.int32)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)
    return jnp.where(valid_mask[:, None], result, 0)


def _pallas_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    group_sizes: jax.Array,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
) -> jax.Array:
    """Pallas kernel based on jax.lax.ragged_dot that skips boundary groups."""
    if lhs.dtype != rhs.dtype:
        raise NotImplementedError(f"dtype mismatch: lhs={lhs.dtype} rhs={rhs.dtype}")
    m, k = lhs.shape
    g_ext, k_rhs, n = rhs.shape
    if g_ext != group_sizes.shape[0]:
        raise ValueError(
            f"Expected group_sizes to have length {g_ext} but got {group_sizes.shape[0]}"
        )
    if k != k_rhs:
        raise ValueError(f"lhs.shape[1]={k} must match rhs.shape[1]={k_rhs}")
    if k % block_k != 0:
        raise ValueError(f"k={k} must be a multiple of block_k={block_k}")

    return ragged_dot_mgpu.ragged_dot(
        lhs,
        rhs,
        group_sizes=group_sizes,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=max_concurrent_steps,
        grid_block_n=grid_block_n,
        load_group_sizes_to_register=False,
    )


def _ragged_dot_backward(lhs, rhs, group_sizes, group_offset, cotangent):
    """Backward pass that masks non-local tokens and accumulates group grads."""
    g_local = rhs.shape[0]
    m = lhs.shape[0]
    shard_start, shard_end, _, group_ids = _local_group_metadata(
        group_sizes, group_offset, g_local, m, return_ids=True
    )
    token_idx = jnp.arange(m, dtype=jnp.int32)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)

    rhs_t = jnp.swapaxes(rhs, -1, -2)
    grad_lhs = _fallback_ragged_dot(cotangent, rhs_t, group_sizes, group_offset)
    grad_lhs = jnp.where(valid_mask[:, None], grad_lhs, 0)

    safe_group_ids = jnp.clip(group_ids, 0, g_local - 1)
    lhs_masked = jnp.where(valid_mask[:, None], lhs, 0)
    cot_masked = jnp.where(valid_mask[:, None], cotangent, 0)
    updates = lhs_masked[:, :, None] * cot_masked[:, None, :]
    grad_rhs = jnp.zeros_like(rhs).at[safe_group_ids].add(updates)

    return grad_lhs, grad_rhs


def _fallback_ragged_dot(lhs, rhs, group_sizes, group_offset):
    offset = jnp.asarray(group_offset, dtype=jnp.int32).reshape(())
    m = lhs.shape[0]
    g_local = rhs.shape[0]

    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]

    token_idx = jnp.arange(m, dtype=jnp.int32)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)

    local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
    adjusted_group_sizes = local_group_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)

    result = lax.ragged_dot(lhs, rhs, adjusted_group_sizes)
    return jnp.where(valid_mask[:, None], result, 0)


def _local_group_metadata(group_sizes, group_offset, g_local, m, return_ids=False):
    sizes = jnp.asarray(group_sizes, dtype=jnp.int32)
    offset = jnp.asarray(group_offset, dtype=jnp.int32).reshape(())
    cumsum = jnp.cumulative_sum(sizes, include_initial=True)
    shard_start = lax.dynamic_index_in_dim(cumsum, offset, axis=0, keepdims=False)
    shard_end = lax.dynamic_index_in_dim(cumsum, offset + g_local, axis=0, keepdims=False)
    local_group_sizes = lax.dynamic_slice_in_dim(sizes, offset, g_local, axis=0)
    if return_ids:
        # Use m (total tokens) as the fixed size since local_len is traced
        # and jnp.arange requires a concrete stop value
        token_idx = jnp.arange(m, dtype=jnp.int32)
        bins = jnp.cumsum(local_group_sizes, dtype=jnp.int32)[:-1]
        # Compute group_ids for all m tokens, then slice to local range
        # For tokens before shard_start, subtract shard_start to get negative indices
        # which will be clamped when used
        all_group_ids = jnp.searchsorted(bins, token_idx - shard_start, side="right")
        # The group_ids are only valid for indices in [shard_start, shard_end)
        # We return all_group_ids which caller will use with proper slicing
        return shard_start, shard_end, local_group_sizes, all_group_ids
    return shard_start, shard_end, local_group_sizes, None
