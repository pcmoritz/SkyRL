import ctypes
import os

from jax import ffi

_TARGET_NAME = "cublas_gemm_grouped_batched_ex"
_DEFAULT_PLATFORM = "CUDA"
_REGISTERED = False
_ATTEMPTED = False


def _default_library_path() -> str | None:
    lib_path = os.path.join(os.path.dirname(__file__), "libcublas_grouped_gemm.so")
    return lib_path if os.path.exists(lib_path) else None


def register(path: str | None = None, *, platform: str | None = None) -> bool:
    """Register the cublas grouped GEMM FFI handler if a shared library is available."""
    global _REGISTERED, _ATTEMPTED
    if _ATTEMPTED:
        return _REGISTERED
    _ATTEMPTED = True

    lib_path = path or os.environ.get("TX_CUBLAS_GROUPED_GEMM_PATH") or _default_library_path()
    if not lib_path:
        return False

    platform = platform or os.environ.get("TX_CUBLAS_GROUPED_GEMM_PLATFORM", _DEFAULT_PLATFORM)
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return False

    try:
        fn = lib.cublas_gemm_grouped_batched_ex
    except AttributeError:
        return False

    ffi.register_ffi_target(_TARGET_NAME, ffi.pycapsule(fn), platform=platform, api_version=1)
    _REGISTERED = True
    return True


def target_name() -> str:
    return _TARGET_NAME
