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
    _registered = True


def grouped_gemm_bf16(lhs, rhs, group_sizes, group_offset=0, trans_rhs=False):
    """Grouped GEMM: out[offsets[i]:offsets[i+1]] = lhs[...] @ rhs[i - offset]"""
    _register()

    m, k = lhs.shape
    g_local, n = rhs.shape[0], rhs.shape[1 if trans_rhs else 2]

    # Group boundaries and sizes
    offsets = jnp.cumsum(jnp.pad(group_sizes, (1, 0)))
    starts = offsets[group_offset:group_offset + g_local]
    group_ms = offsets[group_offset + 1:group_offset + g_local + 1] - starts

    # Byte offsets (bf16 = 2 bytes)
    lhs_off = (starts * k * 2).astype(jnp.int64)
    out_off = (starts * n * 2).astype(jnp.int64)
    rhs_off = (jnp.arange(g_local) * k * n * 2).astype(jnp.int64)

    # cuBLAS params (row-major -> col-major swap)
    Ms, Ns, Ks = jnp.full(g_local, n, jnp.int32), group_ms.astype(jnp.int32), jnp.full(g_local, k, jnp.int32)
    lda = jnp.full(g_local, k if trans_rhs else n, jnp.int32)
    ldb, ldc = jnp.full(g_local, k, jnp.int32), jnp.full(g_local, n, jnp.int32)

    # Pointer workspaces
    ptrs = jnp.zeros(g_local, jnp.int64)

    return jax.ffi.ffi_call(
        "grouped_gemm_bf16", jax.ShapeDtypeStruct((m, n), lhs.dtype),
        lhs, rhs, lhs_off, rhs_off, out_off, Ms, Ns, Ks, lda, ldb, ldc, ptrs, ptrs, ptrs,
        trans_rhs=trans_rhs,
    )
