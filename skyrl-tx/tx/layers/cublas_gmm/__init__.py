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


def grouped_gemm_bf16(lhs, rhs, group_sizes, group_offset, trans_rhs=False):
    """Grouped GEMM: out[offsets[i]:offsets[i+1]] = lhs[...] @ rhs[i - offset]"""
    _register()

    m, k = lhs.shape
    g_local, n = rhs.shape[0], rhs.shape[1 if trans_rhs else 2]

    # Pointer workspaces
    ptrs = jnp.zeros(g_local, jnp.int64)

    return jax.ffi.ffi_call(
        "grouped_gemm_bf16", jax.ShapeDtypeStruct((m, n), lhs.dtype),
        attributes=(("trans_rhs", trans_rhs),),
    )(lhs, rhs, group_sizes.astype(jnp.int32), group_offset.astype(jnp.int32), ptrs, ptrs, ptrs)
