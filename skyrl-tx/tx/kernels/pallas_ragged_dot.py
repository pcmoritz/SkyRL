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

import dataclasses
import functools
import math
import os
from typing import NamedTuple

import jax
from jax import lax
from jax import numpy as jnp
from jax import custom_vjp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax import ops

_DEFAULT_SMS = int(os.environ.get("SKYRL_RAGGED_DOT_NUM_SMS", "132"))


class _KernelConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    max_concurrent_steps: int
    grid_block_n: int


def _choose_kernel_config(k: int, n: int) -> _KernelConfig:
    """Pick a conservative kernel configuration that works on most shapes."""

    def _select_block_k(value: int) -> int:
        for candidate in (128, 64, 32, 16):
            if value % candidate == 0:
                return candidate
        raise ValueError(f"k={value} must be divisible by 16")

    block_k = _select_block_k(k)
    # Use a wide tile for large outputs, otherwise fall back to a smaller tile.
    block_n = 128 if n >= 128 else 64 if n >= 64 else 32
    return _KernelConfig(block_m=128, block_n=block_n, block_k=block_k, max_concurrent_steps=4, grid_block_n=1)


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        """Get the group info for the current block."""

        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]

        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, b in enumerate(group_boundaries):
            start = end
            end = start + b
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
            this_is_group = (tid_begin <= tid) & (tid < tid_end)
            block = lax.select(this_is_group, tid - tid_begin + start_block, block)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(
            group_id=group,
            block=block,
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=start_within_block,
            actual_size=actual_size,
        )


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
    sharding = y.sharding
    return y, (lhs, rhs, group_sizes, group_offset, sharding)


def ragged_dot_with_group_offset_bwd(res, cotangent):
    lhs, rhs, group_sizes, group_offset, sharding = res
    if sharding is not None:
        cotangent = sharding.replicate(cotangent)
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
    (m, k) = lhs.shape
    g_local, k_rhs, n = rhs.shape

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

    config = _choose_kernel_config(k, n)
    return _pallas_ragged_dot(
        lhs,
        rhs,
        group_sizes=extended_group_sizes,
        block_m=config.block_m,
        block_n=config.block_n,
        block_k=config.block_k,
        max_concurrent_steps=config.max_concurrent_steps,
        grid_block_n=config.grid_block_n,
    )


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
    """Pallas kernel based on jax.lax.ragged_dot."""
    if lhs.dtype != rhs.dtype:
        raise NotImplementedError(f"dtype mismatch: lhs={lhs.dtype} rhs={rhs.dtype}")
    m, k = lhs.shape
    g_local, k_rhs, n = rhs.shape
    g_ext = group_sizes.shape[0]

    if g_ext != g_local + 2:
        raise ValueError(
            f"Expected group_sizes to have length {g_local + 2} (with padding) but got {g_ext}"
        )
    if k != k_rhs:
        raise ValueError(f"lhs.shape[1]={k} must match rhs.shape[1]={k_rhs}")
    if k % block_k != 0:
        raise ValueError(f"k={k} must be a multiple of block_k={block_k}")

    def _ceil_div(x: int, y: int) -> int:
        return -(-x // y)

    grid_m = _ceil_div(m, block_m) + g_ext - 1
    grid_n = _ceil_div(n, block_n)
    grid_size = grid_m * grid_n
    # Keep at most _DEFAULT_SMS warpgroups but avoid launching more SMs than work tiles.
    num_sms = max(1, min(_DEFAULT_SMS, grid_size))

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem):
        rows_per_expert = [rows_per_expert_gmem[i] for i in range(len(rows_per_expert_gmem))]
        thread_idx = pl.program_id(0)
        num_threads = pl.num_programs(0)
        total_iters = _ceil_div(grid_size, num_threads)

        @pl.loop(0, total_iters)
        def _worker(loop_idx):
            tile = thread_idx + loop_idx * num_threads

            @pl.when(tile < grid_size)
            def _process_tile():
                mi, ni = plgpu.planar_snake(
                    tile,
                    (grid_m, grid_n),
                    1,
                    grid_block_n,
                )
                group_info = GroupInfo.create(rows_per_expert, block_m, mi)
                is_real_group = (group_info.group_id > 0) & (group_info.group_id < g_ext - 1)
                rhs_index = group_info.group_id - 1

                def acc_scope(acc_ref):
                    @pl.when(is_real_group & (group_info.actual_size > 0))
                    def _():
                        plgpu.emit_pipeline(
                            lambda _, lhs_smem, rhs_smem: plgpu.wgmma(acc_ref, lhs_smem, rhs_smem),
                            grid=(k // block_k,),
                            in_specs=[
                                plgpu.BlockSpec(
                                    (block_m, block_k),
                                    lambda kk: (group_info.block, kk),
                                    delay_release=1,
                                ),
                                plgpu.BlockSpec(
                                    (block_k, block_n),
                                    lambda kk: (kk, ni),
                                    delay_release=1,
                                ),
                            ],
                            max_concurrent_steps=max_concurrent_steps,
                        )(lhs_gmem, rhs_gmem.at[rhs_index])
                    return acc_ref[...]

                acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

                @functools.partial(
                    pl.run_scoped,
                    o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype),
                )
                def store_scope(o_smem):  # pylint: disable=unused-variable
                    o_smem[...] = acc.astype(o_smem.dtype)
                    plgpu.commit_smem()

                    smem_start = group_info.start_within_block
                    remaining_rows = min(block_m, m)
                    while remaining_rows > 0:
                        const_rows_len = 1 << int(math.log2(remaining_rows))
                        remaining_rows //= 2

                        @pl.when(group_info.actual_size & const_rows_len != 0)
                        def _():
                            o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
                            o_gref_slice = o_gmem.at[
                                pl.ds(group_info.block_start + smem_start, const_rows_len),
                                pl.ds(ni * block_n, block_n),
                            ]
                            plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                        smem_start += group_info.actual_size & const_rows_len
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

        # pl.loop executes immediately upon definition.

    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(group_sizes, lhs, rhs)


def _ragged_dot_backward(lhs, rhs, group_sizes, group_offset, cotangent):
    """Backward pass that reuses the masking implementation for gradients."""

    def fallback(lhs_arg, rhs_arg):
        return _fallback_ragged_dot(lhs_arg, rhs_arg, group_sizes, group_offset)

    _, pullback = jax.vjp(fallback, lhs, rhs)
    return pullback(cotangent)


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
