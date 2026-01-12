"""Pallas ragged_dot kernel with group_offset support for expert parallelism.

Based on https://github.com/rdyro/gpu_ragged_dot but extended with group_offset
support to only compute on tokens belonging to local groups.

When group_offset is specified, rhs contains groups [offset, offset + g_local).
Only tokens in these groups are computed - tokens outside get zeros.
"""

from collections import namedtuple
from functools import partial
from typing import NamedTuple, Optional

import jax
from jax import numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

# Try to import triton compiler params (for GPU optimization)
try:
    from jax.experimental.pallas import triton as plgpu
    CompilerParams = getattr(plgpu, "CompilerParams", getattr(plgpu, "TritonCompilerParams", None))
    HAS_TRITON = CompilerParams is not None
except ImportError:
    HAS_TRITON = False
    plgpu = None
    CompilerParams = None


DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64

cdiv = lambda a, b: (a + b - 1) // b


class ProblemSizes(NamedTuple):
    m: int      # Total tokens
    k: int      # Input features
    n: int      # Output features
    g: int      # Total groups
    g_local: int  # Local groups (for group_offset)


class BlockSizes(NamedTuple):
    m: int
    k: int
    n: int


def _ragged_dot_kernel_with_offset(
    # inputs
    x_ref,              # [m, k]
    A_ref,              # [g_local, k, n]
    group_sizes_ref,    # [g]
    group_offsets_ref,  # [g] - cumsum of group_sizes
    offset_ref,         # [1] - group_offset value
    # outputs
    y_ref,              # [m, n]
    # static params
    size: ProblemSizes,
    block: BlockSizes,
    compute_dtype: Optional[jnp.dtype] = None,
    acc_dtype: jnp.dtype = jnp.float32,
):
    """Pallas kernel for ragged_dot with group_offset.

    Grid: (g_local, n_blocks)
    Each kernel instance processes one local group for a slice of output columns.
    """
    # Program IDs
    local_gi = pl.program_id(0)  # Local group index (0 to g_local-1)
    nj = pl.program_id(1)        # Output column block index

    # Global group index
    offset = offset_ref[0]
    global_gi = offset + local_gi

    # Get group size for this group
    group_sz = group_sizes_ref[global_gi]

    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    # Compute start token index for this group
    # start_ridx = sum(group_sizes[0:global_gi])
    start_ridx = jnp.where(global_gi == 0, 0, group_offsets_ref[global_gi - 1])

    # Output column range
    n_start = nj * block.n
    cols_mask = (n_start + jnp.arange(block.n)) < size.n

    @pl.when(group_sz > 0)
    def _compute():
        def outer_loop(m_block_idx, _):
            """Process one m-block of tokens within this group."""
            # Token indices for this block
            ridx = start_ridx + m_block_idx * block.m
            rows_mask = (m_block_idx * block.m + jnp.arange(block.m)) < group_sz

            def inner_loop(k_block_idx, acc):
                """Accumulate over k dimension."""
                k_start = k_block_idx * block.k
                k_mask = (k_start + jnp.arange(block.k)) < size.k

                # Load x block: [block_m, block_k]
                x_block = pl.load(
                    x_ref,
                    (pl.ds(ridx, block.m), pl.ds(k_start, block.k)),
                    mask=rows_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # Load A block for LOCAL group: [block_k, block_n]
                # A_ref is [g_local, k, n], we index by local_gi
                A_block = pl.load(
                    A_ref,
                    (local_gi, pl.ds(k_start, block.k), pl.ds(n_start, block.n)),
                    mask=k_mask[:, None] & cols_mask[None, :],
                    other=0.0,
                )

                # Matrix multiply and accumulate
                result = jax.lax.dot_general(
                    x_block.astype(compute_dtype),
                    A_block.astype(compute_dtype),
                    dimension_numbers=(((1,), (0,)), ((), ())),
                    preferred_element_type=acc_dtype,
                )
                return acc + result.astype(acc_dtype)

            # Accumulate over k blocks
            acc = jnp.zeros((block.m, block.n), dtype=acc_dtype)
            acc = jax.lax.fori_loop(0, cdiv(size.k, block.k), inner_loop, acc)

            # Store result
            pl.store(
                y_ref,
                (pl.ds(ridx, block.m), pl.ds(n_start, block.n)),
                acc.astype(y_ref.dtype),
                mask=rows_mask[:, None] & cols_mask[None, :],
            )
            return None

        # Loop over m-blocks in this group
        jax.lax.fori_loop(0, cdiv(group_sz, block.m), outer_loop, None)


def ragged_dot_pallas(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
    block_m: int = DEFAULT_BLOCK_M,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
) -> jax.Array:
    """Ragged dot product with group_offset support using Pallas.

    Only computes on tokens belonging to groups [offset, offset + g_local).
    Tokens outside this range are set to zero.

    Args:
        lhs: Input tokens [m, k]
        rhs: Group weights [g_local, k, n]
        group_sizes: Size of each group [g]
        group_offset: Starting group index [1]
        precision: JAX precision (unused, for API compatibility)
        preferred_element_type: Output dtype
        block_m, block_k, block_n: Tile sizes for the kernel

    Returns:
        Output [m, n] with zeros for non-local tokens
    """
    m, k = lhs.shape
    g_local, _, n = rhs.shape
    g = group_sizes.shape[0]
    offset = group_offset[0]

    out_dtype = lhs.dtype if preferred_element_type is None else preferred_element_type

    # Normalize block sizes
    block_m = min(block_m, m)
    block_k = min(block_k, k)
    block_n = min(block_n, n)

    # Make block sizes powers of 2, minimum 16
    def normalize_block(b, s):
        b = min(b, s)
        b = max(16, 1 << (b - 1).bit_length() if b > 0 else 16)
        return b

    block_m = normalize_block(block_m, m)
    block_k = normalize_block(block_k, k)
    block_n = normalize_block(block_n, n)

    size = ProblemSizes(m=m, k=k, n=n, g=g, g_local=g_local)
    block = BlockSizes(m=block_m, k=block_k, n=block_n)

    # Precompute group offsets (cumsum)
    group_offsets = jnp.cumsum(group_sizes)

    # Grid: (g_local groups, n_blocks)
    grid = (g_local, cdiv(n, block_n))

    # Input specs
    in_specs = [
        pl.BlockSpec((m, k), lambda gi, nj: (0, 0)),           # x: full array
        pl.BlockSpec((g_local, k, n), lambda gi, nj: (0, 0, 0)),  # A: full array
        pl.BlockSpec((g,), lambda gi, nj: (0,)),               # group_sizes
        pl.BlockSpec((g,), lambda gi, nj: (0,)),               # group_offsets
        pl.BlockSpec((1,), lambda gi, nj: (0,)),               # offset
    ]

    # Output spec
    out_shape = jax.ShapeDtypeStruct((m, n), dtype=out_dtype)
    out_specs = pl.BlockSpec((m, n), lambda gi, nj: (0, 0))

    # Build kernel
    kernel = partial(
        _ragged_dot_kernel_with_offset,
        size=size,
        block=block,
        compute_dtype=None,
        acc_dtype=jnp.float32,
    )

    # Compiler params for GPU (if available)
    compiler_params = {}
    if HAS_TRITON and CompilerParams is not None:
        compiler_params = {"compiler_params": CompilerParams(num_warps=4, num_stages=2)}

    # Initialize output to zeros (non-local tokens will remain zero)
    # Then run kernel to fill in local tokens
    y = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=not HAS_TRITON,  # Use interpret mode if no Triton
        **compiler_params,
    )(lhs, rhs, group_sizes, group_offsets, group_offset)

    return y


def is_gpu() -> bool:
    """Check if running on GPU."""
    return jax.default_backend() == "gpu"


def is_tpu() -> bool:
    """Check if running on TPU."""
    return jax.default_backend() == "tpu"
