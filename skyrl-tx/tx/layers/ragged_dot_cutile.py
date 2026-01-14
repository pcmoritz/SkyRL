# Copyright 2025 SkyRL Authors
# Licensed under the Apache License, Version 2.0

"""Grouped matrix multiplication kernels for GPU written in cuTile.

This module provides cuTile (NVIDIA CUDA Tile) kernels for grouped matrix
multiplication with expert parallelism support via `group_offset`.

Requires: CUDA Toolkit 13.1+, NVIDIA Driver r580+, pip install cuda-tile
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

TILE_M = 128
TILE_N = 128
TILE_K = 128


def _jax_to_torch(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor via DLPack."""
    return torch.from_dlpack(arr)


def _compute_tile_schedule(
    group_sizes: np.ndarray,
    m: int,
    tm: int,
    start_group: int,
    num_local_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute tile schedule for grouped matmul."""
    group_offsets = np.concatenate([[0], np.cumsum(group_sizes)]).astype(np.int32)

    group_ids = []
    m_tile_ids = []

    for g in range(start_group, start_group + num_local_groups):
        if g >= len(group_sizes) or group_sizes[g] == 0:
            continue
        g_start = group_offsets[g]
        g_end = group_offsets[g + 1]
        for t in range(g_start // tm, (g_end + tm - 1) // tm):
            if t * tm < m:
                group_ids.append(g)
                m_tile_ids.append(t)

    if not group_ids:
        return group_offsets, np.array([], np.int32), np.array([], np.int32), 0

    return (
        group_offsets,
        np.array(group_ids, np.int32),
        np.array(m_tile_ids, np.int32),
        len(group_ids),
    )


@ct.kernel
def _gmm_kernel(
    lhs,  # [m, k]
    rhs,  # [num_local_groups, k, n]
    out,  # [m, n]
    group_offsets,  # [num_groups + 1]
    group_ids,  # [num_tiles]
    m_tile_ids,  # [num_tiles]
    start_group: ConstInt,
    num_tiles: ConstInt,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
):
    """Forward: out[group] = lhs[group] @ rhs[local_group]."""
    n_tile = ct.bid(0)
    tile_idx = ct.bid(1)

    if tile_idx >= num_tiles:
        return

    gid = ct.load(group_ids, index=(tile_idx,), shape=(1,))[0]
    m_tile = ct.load(m_tile_ids, index=(tile_idx,), shape=(1,))[0]
    g_start = ct.load(group_offsets, index=(gid,), shape=(1,))[0]
    g_end = ct.load(group_offsets, index=(gid + 1,), shape=(1,))[0]
    local_g = gid - start_group

    acc = ct.zeros((tm, tn), dtype=ct.f32)
    k_dim = ct.shape(lhs, 1)

    for ki in range(ct.cdiv(k_dim, tk)):
        a = ct.load(lhs, index=(m_tile * tm, ki * tk), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(rhs, index=(local_g, ki * tk, n_tile * tn), shape=(1, tk, tn), padding_mode=ct.PaddingMode.ZERO)[0]
        acc = ct.mma(a, b, acc)

    rows = ct.arange(tm, dtype=ct.i32) + m_tile * tm
    mask = (rows >= g_start) & (rows < g_end)
    mask2d = ct.broadcast_to(mask[:, None], (tm, tn))
    result = ct.where(mask2d, acc, ct.zeros_like(acc))

    ct.store(out, index=(m_tile * tm, n_tile * tn), tile=result.to(ct.dtype(out)))


@ct.kernel
def _tgmm_kernel(
    lhs,  # [k, m]
    rhs,  # [m, n]
    out,  # [num_local_groups, k, n]
    group_offsets,  # [num_groups + 1]
    group_ids,  # [num_tiles]
    m_tile_ids,  # [num_tiles]
    start_group: ConstInt,
    num_tiles: ConstInt,
    tm: ConstInt,
    tk: ConstInt,
    tn: ConstInt,
):
    """Backward: out[g] = lhs[:, group].T @ rhs[group, :]."""
    n_tile = ct.bid(0)
    k_tile = ct.bid(1)
    tile_idx = ct.bid(2)

    if tile_idx >= num_tiles:
        return

    gid = ct.load(group_ids, index=(tile_idx,), shape=(1,))[0]
    m_tile = ct.load(m_tile_ids, index=(tile_idx,), shape=(1,))[0]
    g_start = ct.load(group_offsets, index=(gid,), shape=(1,))[0]
    g_end = ct.load(group_offsets, index=(gid + 1,), shape=(1,))[0]
    local_g = gid - start_group

    m_start = m_tile * tm
    rows = ct.arange(tm, dtype=ct.i32) + m_start
    mask = (rows >= g_start) & (rows < g_end)

    a = ct.load(lhs, index=(k_tile * tk, m_start), shape=(tk, tm), padding_mode=ct.PaddingMode.ZERO)
    mask_a = ct.broadcast_to(mask[None, :], (tk, tm))
    a = ct.where(mask_a, a, ct.zeros_like(a))

    b = ct.load(rhs, index=(m_start, n_tile * tn), shape=(tm, tn), padding_mode=ct.PaddingMode.ZERO)
    mask_b = ct.broadcast_to(mask[:, None], (tm, tn))
    b = ct.where(mask_b, b, ct.zeros_like(b))

    contrib = ct.mma(a, b, ct.zeros((tk, tn), dtype=ct.f32))
    ct.atomic_add(out, index=(local_g, k_tile * tk, n_tile * tn), tile=contrib.to(ct.dtype(out)))


def _torch_dtype(jax_dtype):
    """Map JAX dtype to torch dtype."""
    dtype_map = {
        jnp.float32: torch.float32,
        jnp.float16: torch.float16,
        jnp.bfloat16: torch.bfloat16,
        jnp.int32: torch.int32,
    }
    return dtype_map.get(jax_dtype, torch.float32)


def gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
    """Grouped matmul: lhs[group] @ rhs[local_group] for each group."""
    tm, tk, tn = tiling or (TILE_M, TILE_K, TILE_N)
    m, k = lhs.shape
    num_local_groups, _, n = rhs.shape
    start_group = 0 if group_offset is None else int(group_offset.flatten()[0])

    group_sizes_np = np.asarray(group_sizes)
    group_offsets, group_ids, m_tile_ids, num_tiles = _compute_tile_schedule(
        group_sizes_np, m, tm, start_group, num_local_groups
    )

    if num_tiles == 0:
        return jnp.zeros((m, n), dtype=preferred_element_type)

    # Convert to torch tensors (cutile requires __cuda_array_interface__)
    lhs_t = _jax_to_torch(lhs)
    rhs_t = _jax_to_torch(rhs)
    out_t = torch.zeros((m, n), dtype=_torch_dtype(preferred_element_type), device="cuda")
    offsets_t = torch.from_numpy(group_offsets).cuda()
    gids_t = torch.from_numpy(group_ids).cuda()
    mids_t = torch.from_numpy(m_tile_ids).cuda()

    tiles_n = (n + tn - 1) // tn
    ct.launch(
        torch.cuda.current_stream(),
        (tiles_n, num_tiles, 1),
        _gmm_kernel,
        (lhs_t, rhs_t, out_t, offsets_t, gids_t, mids_t, start_group, num_tiles, tm, tn, tk),
    )
    torch.cuda.synchronize()

    # Convert back to JAX
    out = jax.dlpack.from_dlpack(out_t)

    # Zero rows outside local groups
    if num_local_groups < len(group_sizes_np):
        shard_start = group_offsets[start_group]
        shard_end = group_offsets[start_group + num_local_groups]
        valid = (jnp.arange(m) >= shard_start) & (jnp.arange(m) < shard_end)
        out = jnp.where(valid[:, None], out, 0)

    return out


def tgmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | None = None,
    group_offset: jax.Array | None = None,
    num_actual_groups: int | None = None,
) -> jax.Array:
    """Transposed grouped matmul for backward pass."""
    tm, tk, tn = tiling or (TILE_M, TILE_K, TILE_N)
    k, m = lhs.shape
    _, n = rhs.shape
    start_group = 0 if group_offset is None else int(group_offset.flatten()[0])
    group_sizes_np = np.asarray(group_sizes)
    num_actual_groups = num_actual_groups or len(group_sizes_np)

    group_offsets, group_ids, m_tile_ids, num_tiles = _compute_tile_schedule(
        group_sizes_np, m, tm, start_group, num_actual_groups
    )

    if num_tiles == 0:
        return jnp.zeros((num_actual_groups, k, n), dtype=preferred_element_type)

    # Convert to torch tensors
    lhs_t = _jax_to_torch(lhs)
    rhs_t = _jax_to_torch(rhs)
    out_t = torch.zeros((num_actual_groups, k, n), dtype=_torch_dtype(preferred_element_type), device="cuda")
    offsets_t = torch.from_numpy(group_offsets).cuda()
    gids_t = torch.from_numpy(group_ids).cuda()
    mids_t = torch.from_numpy(m_tile_ids).cuda()

    tiles_n = (n + tn - 1) // tn
    tiles_k = (k + tk - 1) // tk
    ct.launch(
        torch.cuda.current_stream(),
        (tiles_n, tiles_k, num_tiles),
        _tgmm_kernel,
        (lhs_t, rhs_t, out_t, offsets_t, gids_t, mids_t, start_group, num_tiles, tm, tk, tn),
    )
    torch.cuda.synchronize()

    return jax.dlpack.from_dlpack(out_t)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tiling: tuple[int, int, int] | None = None,
    preferred_element_type: jnp.dtype = jnp.float32,
    group_offset: jax.Array | None = None,
) -> jax.Array:
    """Ragged dot with cuTile acceleration and autodiff support."""
    return gmm(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset)


def _fwd(lhs, rhs, group_sizes, tiling, dtype, group_offset):
    out = gmm(lhs, rhs, group_sizes, dtype, tiling, group_offset)
    return out, (lhs, rhs, group_sizes, group_offset)


def _bwd(tiling, dtype, res, g):
    lhs, rhs, group_sizes, group_offset = res
    num_local = rhs.shape[0]
    grad_lhs = gmm(g, jnp.swapaxes(rhs, 1, 2), group_sizes, dtype, tiling, group_offset)
    grad_rhs = tgmm(lhs.T, g, group_sizes, dtype, tiling, group_offset, num_local)
    return grad_lhs, grad_rhs, None, None


ragged_dot.defvjp(_fwd, _bwd)
