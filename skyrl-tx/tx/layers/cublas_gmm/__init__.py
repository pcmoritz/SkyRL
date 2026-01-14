"""cuBLAS grouped GEMM via JAX FFI."""

from pathlib import Path
import ctypes
import jax
from jax import numpy as jnp

_registered = False

def _register():
    global _registered
    if _registered: return
    lib = ctypes.CDLL(str(Path(__file__).parent / "libgrouped_gemm.so"))
    jax.ffi.register_ffi_target("grouped_gemm_bf16", jax.ffi.pycapsule(lib.GroupedGemmBf16), platform="CUDA")
    jax.ffi.register_ffi_target("grouped_gemm_bf16_trans", jax.ffi.pycapsule(lib.GroupedGemmBf16Trans), platform="CUDA")
    jax.ffi.register_ffi_target("grouped_gemm_bf16_dw", jax.ffi.pycapsule(lib.GroupedGemmBf16Dw), platform="CUDA")
    _registered = True


def _fwd_impl(lhs, rhs, group_sizes, group_offset):
    """Forward: out = lhs @ rhs per group."""
    g_local = rhs.shape[0]
    m, n = lhs.shape[0], rhs.shape[2]
    # Output shapes: A_ptrs, B_ptrs, C_ptrs (scratch), and result
    out_shapes = (
        jax.ShapeDtypeStruct((g_local,), jnp.int64),      # A_ptrs
        jax.ShapeDtypeStruct((g_local,), jnp.int64),      # B_ptrs
        jax.ShapeDtypeStruct((g_local,), jnp.int64),      # C_ptrs
        jax.ShapeDtypeStruct((m, n), lhs.dtype),          # result
    )
    _, _, _, result = jax.ffi.ffi_call(
        "grouped_gemm_bf16", out_shapes,
    )(lhs, rhs, group_sizes.astype(jnp.int32), group_offset.astype(jnp.int32))
    return result


def _dlhs_impl(dout, rhs, group_sizes, group_offset):
    """d_lhs = dout @ rhs^T per group."""
    g_local = rhs.shape[0]
    m, k = dout.shape[0], rhs.shape[1]
    out_shapes = (
        jax.ShapeDtypeStruct((g_local,), jnp.int64),
        jax.ShapeDtypeStruct((g_local,), jnp.int64),
        jax.ShapeDtypeStruct((g_local,), jnp.int64),
        jax.ShapeDtypeStruct((m, k), dout.dtype),
    )
    _, _, _, result = jax.ffi.ffi_call(
        "grouped_gemm_bf16_trans", out_shapes,
    )(dout, rhs, group_sizes.astype(jnp.int32), group_offset.astype(jnp.int32))
    return result


def _drhs_impl(lhs, dout, group_sizes, group_offset, g_local):
    """d_rhs[i] = lhs[group_i]^T @ dout[group_i]."""
    k, n = lhs.shape[1], dout.shape[1]
    out_shapes = (
        jax.ShapeDtypeStruct((g_local,), jnp.int64),
        jax.ShapeDtypeStruct((g_local,), jnp.int64),
        jax.ShapeDtypeStruct((g_local,), jnp.int64),
        jax.ShapeDtypeStruct((g_local, k, n), lhs.dtype),
    )
    _, _, _, result = jax.ffi.ffi_call(
        "grouped_gemm_bf16_dw", out_shapes,
    )(lhs, dout, group_sizes.astype(jnp.int32), group_offset.astype(jnp.int32))
    return result


@jax.custom_vjp
def grouped_gemm_bf16(lhs, rhs, group_sizes, group_offset):
    """Grouped GEMM with cuBLAS forward and backward."""
    _register()
    return _fwd_impl(lhs, rhs, group_sizes, group_offset)


def _fwd(lhs, rhs, group_sizes, group_offset):
    _register()
    out = _fwd_impl(lhs, rhs, group_sizes, group_offset)
    return out, (lhs, rhs, group_sizes, group_offset)


def _bwd(res, g):
    lhs, rhs, group_sizes, group_offset = res
    # Use JAX for backward to preserve shard_map axis annotations
    from tx.layers.util import ragged_dot as jax_ragged_dot

    def fwd_jax(lhs, rhs):
        return jax_ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset, use_cublas=False)

    _, vjp_fn = jax.vjp(fwd_jax, lhs, rhs)
    d_lhs, d_rhs = vjp_fn(g)
    return d_lhs, d_rhs, None, None


grouped_gemm_bf16.defvjp(_fwd, _bwd)
