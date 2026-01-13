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
from functools import lru_cache
from typing import NamedTuple

import jax
from jax import lax
from jax import numpy as jnp
from jax import custom_vjp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


class _KernelConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    max_concurrent_steps: int
    grid_block_n: int


@dataclasses.dataclass(frozen=True)
class _GroupInfo:
    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
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


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None else None


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "t", "yes", "y")


def _bucket_sizes(m: int) -> tuple[int, ...]:
    raw = os.environ.get("SKYRL_RAGGED_DOT_BUCKETS")
    if raw:
        base = [int(value) for value in raw.split(",") if value.strip()]
    else:
        base = [256, 512, 1024, 2048, 4096, 8192, 16384]
    sizes = [size for size in base if size < m]
    sizes.append(m)
    return tuple(sorted(set(sizes)))


@lru_cache(maxsize=None)
def _is_h100() -> bool:
    try:
        return any("H100" in device.device_kind for device in jax.devices("gpu"))
    except Exception:
        return False


def _choose_kernel_config(m: int, k: int, n: int) -> _KernelConfig:
    """Pick a conservative kernel configuration that works on most shapes."""

    def _grid_block_n_for(value: int, block_n: int) -> int:
        padded_n = value + (-value) % block_n
        grid_n = max(1, (padded_n + block_n - 1) // block_n)
        if grid_n >= 16:
            return 8
        if grid_n >= 8:
            return 4
        if grid_n >= 4:
            return 2
        return 1

    def _select_block_k(value: int) -> int:
        for candidate in (64, 32, 16):
            if value % candidate == 0:
                return candidate
        raise ValueError(f"k={value} must be divisible by 16")

    def _select_block_n(value: int, prefer_large_tiles: bool) -> int:
        candidates = (192, 128, 64) if prefer_large_tiles else (128, 64)
        for candidate in candidates:
            if value >= candidate:
                pad = (-value) % candidate
                if pad == 0 or pad <= candidate // 4:
                    return candidate
        return 64

    block_k = _select_block_k(k)
    prefer_large_tiles = _is_h100()
    block_n = _select_block_n(n, prefer_large_tiles)
    if prefer_large_tiles and m >= 192:
        block_m = 192
    elif m >= 128:
        block_m = 128
    else:
        block_m = 64
    if block_k >= 64:
        max_concurrent_steps = 6 if prefer_large_tiles and (block_m >= 128 or block_n >= 128) else 4
    else:
        max_concurrent_steps = 3

    grid_block_n = _grid_block_n_for(n, block_n)

    env_block_m = _env_int("SKYRL_RAGGED_DOT_BLOCK_M")
    env_block_n = _env_int("SKYRL_RAGGED_DOT_BLOCK_N")
    env_block_k = _env_int("SKYRL_RAGGED_DOT_BLOCK_K")
    env_steps = _env_int("SKYRL_RAGGED_DOT_MAX_STEPS")
    env_grid_block_n = _env_int("SKYRL_RAGGED_DOT_GRID_BLOCK_N")

    if env_block_m is not None:
        block_m = env_block_m
    if env_block_n is not None:
        block_n = env_block_n
    if env_block_k is not None:
        block_k = env_block_k
    if env_steps is not None:
        max_concurrent_steps = env_steps
    if env_grid_block_n is not None:
        grid_block_n = env_grid_block_n

    return _KernelConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=max_concurrent_steps,
        grid_block_n=grid_block_n,
    )


def _estimate_smem_bytes(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    dtype_size: int,
) -> int:
    stage_bytes = (block_m * block_k + block_k * block_n) * dtype_size
    acc_bytes = block_m * block_n * dtype_size
    # Account for accumulator + output smem buffers.
    return stage_bytes * max_concurrent_steps + acc_bytes * 2


def _adjust_config_for_smem(
    m: int,
    n: int,
    dtype_size: int,
    config: _KernelConfig,
) -> _KernelConfig:
    max_smem_bytes = _env_int("SKYRL_RAGGED_DOT_MAX_SMEM_BYTES") or 232_448

    def _grid_block_n_for(value: int, block_n: int) -> int:
        padded_n = value + (-value) % block_n
        grid_n = max(1, (padded_n + block_n - 1) // block_n)
        if grid_n >= 16:
            return 8
        if grid_n >= 8:
            return 4
        if grid_n >= 4:
            return 2
        return 1

    def _block_n_candidates(value: int) -> list[int]:
        candidates: list[int] = []
        for candidate in (config.block_n, 192, 128, 64):
            if candidate in candidates:
                continue
            pad = (-value) % candidate
            if pad <= candidate // 4:
                candidates.append(candidate)
        return candidates

    def _block_m_candidates(value: int) -> list[int]:
        candidates: list[int] = []
        for candidate in (config.block_m, 192, 128, 64):
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _step_candidates(value: int) -> list[int]:
        candidates: list[int] = []
        for candidate in (value, 6, 5, 4, 2):
            if candidate <= value and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    for steps in _step_candidates(config.max_concurrent_steps):
        for block_m in _block_m_candidates(m):
            for block_n in _block_n_candidates(n):
                smem = _estimate_smem_bytes(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=config.block_k,
                    max_concurrent_steps=steps,
                    dtype_size=dtype_size,
                )
                if smem <= max_smem_bytes:
                    grid_block_n = _grid_block_n_for(n, block_n)
                    return config._replace(
                        block_m=block_m,
                        block_n=block_n,
                        max_concurrent_steps=steps,
                        grid_block_n=grid_block_n,
                    )

    return config




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
    local_len = shard_end - shard_start
    local_group_sizes = lax.dynamic_slice_in_dim(sizes, offset, g_local, axis=0)

    if m == 0:
        return jnp.zeros((0, orig_n), dtype=lhs.dtype)

    bucket_sizes = _bucket_sizes(m)
    bucket_sizes_arr = jnp.array(bucket_sizes, dtype=jnp.int32)
    idxs = jnp.arange(len(bucket_sizes), dtype=jnp.int32)
    mask = local_len <= bucket_sizes_arr
    bucket_idx = jnp.min(jnp.where(mask, idxs, idxs[-1]))

    def _run_bucket(bucket: int) -> jax.Array:
        window_len = int(bucket)
        window_start = jnp.minimum(
            shard_start, jnp.asarray(m - window_len, dtype=jnp.int32)
        )
        window_start = jnp.maximum(window_start, jnp.int32(0))
        window_end = window_start + window_len

        prefix = shard_start - window_start
        suffix = window_end - shard_end
        extended_group_sizes = jnp.concatenate(
            [prefix[jnp.newaxis], local_group_sizes, suffix[jnp.newaxis]], axis=0
        )

        dtype_size = jnp.dtype(lhs.dtype).itemsize
        config = _choose_kernel_config(window_len, k, n)
        config = _adjust_config_for_smem(window_len, n, dtype_size, config)
        n_pad = (-n) % config.block_n
        rhs_padded = rhs
        n_eff = n
        if n_pad:
            rhs_padded = jnp.pad(rhs_padded, ((0, 0), (0, 0), (0, n_pad)))
            n_eff = n + n_pad

        zero_slice = jnp.zeros((1, k, n_eff), dtype=rhs_padded.dtype)
        padded_rhs = jnp.concatenate([zero_slice, rhs_padded, zero_slice], axis=0)
        lhs_window = lax.dynamic_slice_in_dim(lhs, window_start, window_len, axis=0)

        result = _pallas_ragged_dot(
            lhs_window,
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

        output = jnp.zeros((m, orig_n), dtype=result.dtype)
        return lax.dynamic_update_slice(output, result, (window_start, 0))

    fns = [functools.partial(_run_bucket, bucket) for bucket in bucket_sizes]
    return lax.switch(bucket_idx, fns)


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

    env_load_group_sizes = _env_bool("SKYRL_RAGGED_DOT_LOAD_GROUP_SIZES")
    load_group_sizes = False if env_load_group_sizes is None else env_load_group_sizes

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem):
        grid_m = pl.cdiv(m, block_m) + g_ext - 1
        grid_n = pl.cdiv(n, block_n)
        grid = (grid_m * grid_n,)
        if load_group_sizes:
            rows_per_expert = [rows_per_expert_gmem[i] for i in range(len(rows_per_expert_gmem))]
        else:
            rows_per_expert = rows_per_expert_gmem

        last_group = jnp.int32(g_ext - 1)

        @plgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):  # pylint: disable=unused-variable
            mi, ni = plgpu.planar_snake(
                loop_info.index[0],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )
            group_info = _GroupInfo.create(rows_per_expert, block_m, mi)
            is_local_group = (group_info.group_id > 0) & (group_info.group_id < last_group)

            def acc_scope(acc_ref):
                acc_ref[...] = jnp.zeros((block_m, block_n), dtype=acc_ref.dtype)

                @pl.when(is_local_group)
                def _():
                    plgpu.emit_pipeline(
                        lambda _, lhs_smem, rhs_smem: plgpu.wgmma(
                            acc_ref,
                            lhs_smem,
                            rhs_smem,
                        ),
                        grid=(k // block_k,),
                        in_specs=[
                            plgpu.BlockSpec(
                                (block_m, block_k),
                                lambda k: (group_info.block, k),
                                delay_release=1,
                            ),
                            plgpu.BlockSpec(
                                (block_k, block_n),
                                lambda k: (k, ni),
                                delay_release=1,
                            ),
                        ],
                        max_concurrent_steps=max_concurrent_steps,
                    )(lhs_gmem, rhs_gmem.at[group_info.group_id])
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

    num_sms = 132
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(group_sizes, lhs, rhs)


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
