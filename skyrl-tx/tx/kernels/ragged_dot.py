"""Pallas ragged_dot kernel with group_offset support for expert parallelism.

Based on https://github.com/rdyro/gpu_ragged_dot but extended with group_offset
support to only compute on tokens belonging to local groups.

When group_offset is specified, rhs contains groups [offset, offset + g_local).
Only tokens in these groups are computed - tokens outside get zeros.
"""

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


def _ragged_dot_grad_lhs_kernel(
    # inputs
    dy_ref,             # [m, n] - gradient w.r.t. output
    A_ref,              # [g_local, k, n] - weights
    group_sizes_ref,    # [g]
    group_offsets_ref,  # [g] - cumsum of group_sizes
    offset_ref,         # [1] - group_offset value
    # outputs
    dx_ref,             # [m, k] - gradient w.r.t. lhs
    # static params
    size: ProblemSizes,
    block: BlockSizes,
    compute_dtype: Optional[jnp.dtype] = None,
    acc_dtype: jnp.dtype = jnp.float32,
):
    """Pallas kernel for d_lhs = dy @ rhs^T (ragged).

    Grid: (g_local, k_blocks)
    Each kernel instance processes one local group for a slice of k columns.

    d_lhs[i, :] = dy[i, :] @ rhs[group(i), :, :].T
    """
    # Program IDs
    local_gi = pl.program_id(0)  # Local group index (0 to g_local-1)
    kj = pl.program_id(1)        # Output k-column block index

    # Global group index
    offset = offset_ref[0]
    global_gi = offset + local_gi

    # Get group size for this group
    group_sz = group_sizes_ref[global_gi]

    compute_dtype = compute_dtype if compute_dtype is not None else dy_ref.dtype

    # Compute start token index for this group
    start_ridx = jnp.where(global_gi == 0, 0, group_offsets_ref[global_gi - 1])

    # Output k-column range
    k_start = kj * block.k
    k_cols_mask = (k_start + jnp.arange(block.k)) < size.k

    @pl.when(group_sz > 0)
    def _compute():
        def outer_loop(m_block_idx, _):
            """Process one m-block of tokens within this group."""
            ridx = start_ridx + m_block_idx * block.m
            rows_mask = (m_block_idx * block.m + jnp.arange(block.m)) < group_sz

            def inner_loop(n_block_idx, acc):
                """Accumulate over n dimension."""
                n_start = n_block_idx * block.n
                n_mask = (n_start + jnp.arange(block.n)) < size.n

                # Load dy block: [block_m, block_n]
                dy_block = pl.load(
                    dy_ref,
                    (pl.ds(ridx, block.m), pl.ds(n_start, block.n)),
                    mask=rows_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

                # Load A block transposed: A is [g_local, k, n], we want [block_n, block_k]
                # A[local_gi, k_start:k_start+block_k, n_start:n_start+block_n].T
                A_block = pl.load(
                    A_ref,
                    (local_gi, pl.ds(k_start, block.k), pl.ds(n_start, block.n)),
                    mask=k_cols_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                # A_block is [block_k, block_n], need [block_n, block_k] for dy @ A^T
                A_block_T = A_block.T  # [block_n, block_k]

                # Matrix multiply: [block_m, block_n] @ [block_n, block_k] -> [block_m, block_k]
                result = jax.lax.dot_general(
                    dy_block.astype(compute_dtype),
                    A_block_T.astype(compute_dtype),
                    dimension_numbers=(((1,), (0,)), ((), ())),
                    preferred_element_type=acc_dtype,
                )
                return acc + result.astype(acc_dtype)

            # Accumulate over n blocks
            acc = jnp.zeros((block.m, block.k), dtype=acc_dtype)
            acc = jax.lax.fori_loop(0, cdiv(size.n, block.n), inner_loop, acc)

            # Store result
            pl.store(
                dx_ref,
                (pl.ds(ridx, block.m), pl.ds(k_start, block.k)),
                acc.astype(dx_ref.dtype),
                mask=rows_mask[:, None] & k_cols_mask[None, :],
            )
            return None

        jax.lax.fori_loop(0, cdiv(group_sz, block.m), outer_loop, None)


def _ragged_dot_grad_rhs_kernel(
    # inputs
    x_ref,              # [m, k] - lhs input
    dy_ref,             # [m, n] - gradient w.r.t. output
    group_sizes_ref,    # [g]
    group_offsets_ref,  # [g] - cumsum of group_sizes
    offset_ref,         # [1] - group_offset value
    # outputs
    dA_ref,             # [g_local, k, n] - gradient w.r.t. rhs
    # static params
    size: ProblemSizes,
    block: BlockSizes,
    compute_dtype: Optional[jnp.dtype] = None,
    acc_dtype: jnp.dtype = jnp.float32,
):
    """Pallas kernel for d_rhs[g] = lhs[tokens_in_g]^T @ dy[tokens_in_g].

    Grid: (g_local, k_blocks, n_blocks)
    Each kernel instance computes one block of d_rhs for one local group.

    d_rhs[g, :, :] = sum over tokens in group g of: lhs[i, :, None] * dy[i, None, :]
                   = lhs[tokens_in_g].T @ dy[tokens_in_g]
    """
    # Program IDs
    local_gi = pl.program_id(0)  # Local group index (0 to g_local-1)
    kj = pl.program_id(1)        # k block index
    nj = pl.program_id(2)        # n block index

    # Global group index
    offset = offset_ref[0]
    global_gi = offset + local_gi

    # Get group size for this group
    group_sz = group_sizes_ref[global_gi]

    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    # Compute start token index for this group
    start_ridx = jnp.where(global_gi == 0, 0, group_offsets_ref[global_gi - 1])

    # Output ranges
    k_start = kj * block.k
    n_start = nj * block.n
    k_mask = (k_start + jnp.arange(block.k)) < size.k
    n_mask = (n_start + jnp.arange(block.n)) < size.n

    @pl.when(group_sz > 0)
    def _compute():
        def loop_body(m_block_idx, acc):
            """Accumulate x^T @ dy over m-blocks in this group."""
            ridx = start_ridx + m_block_idx * block.m
            rows_mask = (m_block_idx * block.m + jnp.arange(block.m)) < group_sz

            # Load x block: [block_m, block_k]
            x_block = pl.load(
                x_ref,
                (pl.ds(ridx, block.m), pl.ds(k_start, block.k)),
                mask=rows_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Load dy block: [block_m, block_n]
            dy_block = pl.load(
                dy_ref,
                (pl.ds(ridx, block.m), pl.ds(n_start, block.n)),
                mask=rows_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # Compute x^T @ dy: [block_k, block_m] @ [block_m, block_n] -> [block_k, block_n]
            result = jax.lax.dot_general(
                x_block.T.astype(compute_dtype),
                dy_block.astype(compute_dtype),
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=acc_dtype,
            )
            return acc + result.astype(acc_dtype)

        # Accumulate over all m-blocks in this group
        acc = jnp.zeros((block.k, block.n), dtype=acc_dtype)
        acc = jax.lax.fori_loop(0, cdiv(group_sz, block.m), loop_body, acc)

        # Store result
        pl.store(
            dA_ref,
            (local_gi, pl.ds(k_start, block.k), pl.ds(n_start, block.n)),
            acc.astype(dA_ref.dtype),
            mask=k_mask[:, None] & n_mask[None, :],
        )

    @pl.when(group_sz == 0)
    def _zero():
        # Zero out this block for empty groups
        pl.store(
            dA_ref,
            (local_gi, pl.ds(k_start, block.k), pl.ds(n_start, block.n)),
            jnp.zeros((block.k, block.n), dtype=dA_ref.dtype),
            mask=k_mask[:, None] & n_mask[None, :],
        )


def _normalize_block(b, s):
    """Normalize block size to power of 2, minimum 16."""
    b = min(b, s)
    b = max(16, 1 << (b - 1).bit_length() if b > 0 else 16)
    return b


def _compute_valid_mask(group_sizes: jax.Array, group_offset: jax.Array, m: int, g_local: int):
    """Compute mask for tokens belonging to local groups."""
    offset = group_offset[0]
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]
    token_idx = jnp.arange(m)
    return (token_idx >= shard_start) & (token_idx < shard_end)


def _ragged_dot_pallas_impl(
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
    """Internal Pallas implementation (no autodiff)."""
    m, k = lhs.shape
    g_local, _, n = rhs.shape
    g = group_sizes.shape[0]

    out_dtype = lhs.dtype if preferred_element_type is None else preferred_element_type

    # Normalize block sizes
    block_m = _normalize_block(block_m, m)
    block_k = _normalize_block(block_k, k)
    block_n = _normalize_block(block_n, n)

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

    # Output spec - vma=set() indicates no variation across mesh axes (for shard_map compatibility)
    out_shape = jax.ShapeDtypeStruct((m, n), dtype=out_dtype, vma=set())
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

    y = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=not HAS_TRITON,
        **compiler_params,
    )(lhs, rhs, group_sizes, group_offsets, group_offset)

    # Pallas doesn't zero-initialize output, so tokens outside local groups
    # may contain garbage. Explicitly mask them to zero.
    valid_mask = _compute_valid_mask(group_sizes, group_offset, m, g_local)
    y = jnp.where(valid_mask[:, None], y, 0)

    return y


def _ragged_dot_grad_lhs_impl_no_mask(
    dy: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    out_dtype: jnp.dtype,
    block_m: int = DEFAULT_BLOCK_M,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
) -> jax.Array:
    """Compute d_lhs = dy @ rhs^T (ragged) using Pallas. No masking - caller handles it."""
    m, n = dy.shape
    g_local, k, _ = rhs.shape
    g = group_sizes.shape[0]

    # Normalize block sizes
    block_m = _normalize_block(block_m, m)
    block_k = _normalize_block(block_k, k)
    block_n = _normalize_block(block_n, n)

    size = ProblemSizes(m=m, k=k, n=n, g=g, g_local=g_local)
    block = BlockSizes(m=block_m, k=block_k, n=block_n)

    group_offsets = jnp.cumsum(group_sizes)

    # Grid: (g_local groups, k_blocks)
    grid = (g_local, cdiv(k, block_k))

    in_specs = [
        pl.BlockSpec((m, n), lambda gi, kj: (0, 0)),           # dy
        pl.BlockSpec((g_local, k, n), lambda gi, kj: (0, 0, 0)),  # A
        pl.BlockSpec((g,), lambda gi, kj: (0,)),               # group_sizes
        pl.BlockSpec((g,), lambda gi, kj: (0,)),               # group_offsets
        pl.BlockSpec((1,), lambda gi, kj: (0,)),               # offset
    ]

    out_shape = jax.ShapeDtypeStruct((m, k), dtype=out_dtype, vma=set())
    out_specs = pl.BlockSpec((m, k), lambda gi, kj: (0, 0))

    kernel = partial(
        _ragged_dot_grad_lhs_kernel,
        size=size,
        block=block,
        compute_dtype=None,
        acc_dtype=jnp.float32,
    )

    compiler_params = {}
    if HAS_TRITON and CompilerParams is not None:
        compiler_params = {"compiler_params": CompilerParams(num_warps=4, num_stages=2)}

    dx = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=not HAS_TRITON,
        **compiler_params,
    )(dy, rhs, group_sizes, group_offsets, group_offset)

    return dx


def _ragged_dot_grad_rhs_impl(
    lhs: jax.Array,
    dy: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    g_local: int,
    out_dtype: jnp.dtype,
    block_m: int = DEFAULT_BLOCK_M,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
) -> jax.Array:
    """Compute d_rhs[g] = lhs[tokens_in_g]^T @ dy[tokens_in_g] using Pallas."""
    m, k = lhs.shape
    _, n = dy.shape
    g = group_sizes.shape[0]

    # Normalize block sizes
    block_m = _normalize_block(block_m, m)
    block_k = _normalize_block(block_k, k)
    block_n = _normalize_block(block_n, n)

    size = ProblemSizes(m=m, k=k, n=n, g=g, g_local=g_local)
    block = BlockSizes(m=block_m, k=block_k, n=block_n)

    group_offsets = jnp.cumsum(group_sizes)

    # Grid: (g_local groups, k_blocks, n_blocks)
    grid = (g_local, cdiv(k, block_k), cdiv(n, block_n))

    in_specs = [
        pl.BlockSpec((m, k), lambda gi, kj, nj: (0, 0)),       # x
        pl.BlockSpec((m, n), lambda gi, kj, nj: (0, 0)),       # dy
        pl.BlockSpec((g,), lambda gi, kj, nj: (0,)),           # group_sizes
        pl.BlockSpec((g,), lambda gi, kj, nj: (0,)),           # group_offsets
        pl.BlockSpec((1,), lambda gi, kj, nj: (0,)),           # offset
    ]

    out_shape = jax.ShapeDtypeStruct((g_local, k, n), dtype=out_dtype, vma=set())
    out_specs = pl.BlockSpec((g_local, k, n), lambda gi, kj, nj: (0, 0, 0))

    kernel = partial(
        _ragged_dot_grad_rhs_kernel,
        size=size,
        block=block,
        compute_dtype=None,
        acc_dtype=jnp.float32,
    )

    compiler_params = {}
    if HAS_TRITON and CompilerParams is not None:
        compiler_params = {"compiler_params": CompilerParams(num_warps=4, num_stages=2)}

    dA = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=not HAS_TRITON,
        **compiler_params,
    )(lhs, dy, group_sizes, group_offsets, group_offset)

    return dA


@jax.custom_vjp
def ragged_dot_pallas(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
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

    Returns:
        Output [m, n] with zeros for non-local tokens
    """
    return _ragged_dot_pallas_impl(
        lhs, rhs, group_sizes, group_offset,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


def _ragged_dot_pallas_fwd(lhs, rhs, group_sizes, group_offset, precision, preferred_element_type):
    """Forward pass for custom_vjp."""
    m, k = lhs.shape
    g_local = rhs.shape[0]

    result = _ragged_dot_pallas_impl(
        lhs, rhs, group_sizes, group_offset,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # Compute valid mask here (in forward context) to preserve correct VMA
    # This mask will be reused in backward to ensure VMA consistency
    valid_mask = _compute_valid_mask(group_sizes, group_offset, m, g_local)

    # Save residuals for backward pass
    return result, (lhs, rhs, group_sizes, group_offset, valid_mask)


def _ragged_dot_pallas_bwd(residuals, g):
    """Backward pass using Pallas kernels.

    Uses the valid_mask computed in forward pass to ensure VMA consistency.
    The gradients are computed as:
    - d_lhs = g @ rhs^T (for tokens in local groups)
    - d_rhs[i] = lhs[tokens_in_group_i]^T @ g[tokens_in_group_i]
    """
    lhs, rhs, group_sizes, group_offset, valid_mask = residuals
    g_local = rhs.shape[0]
    m = lhs.shape[0]

    # Compute raw gradients with Pallas kernels (no masking inside)
    d_lhs_raw = _ragged_dot_grad_lhs_impl_no_mask(
        g, rhs, group_sizes, group_offset,
        out_dtype=lhs.dtype,
    )
    d_rhs_raw = _ragged_dot_grad_rhs_impl(
        lhs, g, group_sizes, group_offset,
        g_local=g_local,
        out_dtype=rhs.dtype,
    )

    # Apply mask using lhs as base to ensure correct VMA
    # jax.lax.select preserves the type of the first array when shapes match
    d_lhs = jax.lax.select(
        jnp.broadcast_to(valid_mask[:, None], lhs.shape),
        d_lhs_raw.astype(lhs.dtype),
        jnp.zeros_like(lhs),
    )

    # For d_rhs, use rhs as base for VMA
    d_rhs = jnp.zeros_like(rhs) + d_rhs_raw

    # Return gradients for all inputs (None for non-differentiable args)
    return (d_lhs, d_rhs, None, None, None, None)


ragged_dot_pallas.defvjp(_ragged_dot_pallas_fwd, _ragged_dot_pallas_bwd)


def is_gpu() -> bool:
    """Check if running on GPU."""
    return jax.default_backend() == "gpu"


def is_tpu() -> bool:
    """Check if running on TPU."""
    return jax.default_backend() == "tpu"
