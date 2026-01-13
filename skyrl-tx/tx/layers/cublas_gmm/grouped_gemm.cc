// Minimal cuBLAS grouped GEMM - all setup done in JAX
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static thread_local cublasHandle_t g_handle = nullptr;

ffi::Error GroupedGemmBf16Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> lhs,             // [m, k]
    ffi::Buffer<ffi::BF16> rhs,             // [g_local, k, n]
    ffi::Buffer<ffi::S64> lhs_offsets,      // [g_local] byte offsets into lhs
    ffi::Buffer<ffi::S64> rhs_offsets,      // [g_local] byte offsets into rhs
    ffi::Buffer<ffi::S64> out_offsets,      // [g_local] byte offsets into out
    ffi::Buffer<ffi::S32> Ms,               // [g_local]
    ffi::Buffer<ffi::S32> Ns,               // [g_local]
    ffi::Buffer<ffi::S32> Ks,               // [g_local]
    ffi::Buffer<ffi::S32> lda,              // [g_local]
    ffi::Buffer<ffi::S32> ldb,              // [g_local]
    ffi::Buffer<ffi::S32> ldc,              // [g_local]
    ffi::Buffer<ffi::S64> A_ptrs,           // [g_local] workspace for pointers
    ffi::Buffer<ffi::S64> B_ptrs,           // [g_local] workspace
    ffi::Buffer<ffi::S64> C_ptrs,           // [g_local] workspace
    bool trans_rhs,
    ffi::ResultBuffer<ffi::BF16> out        // [m, n]
) {
    if (!g_handle) cublasCreate(&g_handle);
    cublasSetStream(g_handle, stream);

    int g_local = static_cast<int>(lhs_offsets.dimensions()[0]);
    if (g_local == 0) return ffi::Error::Success();

    const char* lhs_base = reinterpret_cast<const char*>(lhs.typed_data());
    const char* rhs_base = reinterpret_cast<const char*>(rhs.typed_data());
    char* out_base = reinterpret_cast<char*>(out->typed_data());

    // Fill pointer arrays: ptr = base + offset
    const int64_t* lhs_off = lhs_offsets.typed_data();
    const int64_t* rhs_off = rhs_offsets.typed_data();
    const int64_t* out_off = out_offsets.typed_data();
    int64_t* A_ptr = const_cast<int64_t*>(A_ptrs.typed_data());
    int64_t* B_ptr = const_cast<int64_t*>(B_ptrs.typed_data());
    int64_t* C_ptr = const_cast<int64_t*>(C_ptrs.typed_data());

    for (int i = 0; i < g_local; i++) {
        A_ptr[i] = reinterpret_cast<int64_t>(rhs_base + rhs_off[i]);
        B_ptr[i] = reinterpret_cast<int64_t>(lhs_base + lhs_off[i]);
        C_ptr[i] = reinterpret_cast<int64_t>(out_base + out_off[i]);
    }

    // Create operation arrays (all same value)
    std::vector<cublasOperation_t> transa(g_local, trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb(g_local, CUBLAS_OP_N);
    // Each "group" has 1 matrix
    std::vector<int> group_size(g_local, 1);

    float alpha = 1.0f, beta = 0.0f;

    cublasGemmGroupedBatchedEx(
        g_handle,
        transa.data(), transb.data(),
        const_cast<int*>(Ms.typed_data()),
        const_cast<int*>(Ns.typed_data()),
        const_cast<int*>(Ks.typed_data()),
        &alpha,
        reinterpret_cast<const void**>(A_ptr), CUDA_R_16BF, const_cast<int*>(lda.typed_data()),
        reinterpret_cast<const void**>(B_ptr), CUDA_R_16BF, const_cast<int*>(ldb.typed_data()),
        &beta,
        reinterpret_cast<void**>(C_ptr), CUDA_R_16BF, const_cast<int*>(ldc.typed_data()),
        g_local,
        group_size.data(),
        CUBLAS_COMPUTE_32F
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GroupedGemmBf16,
    GroupedGemmBf16Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // lhs
        .Arg<ffi::Buffer<ffi::BF16>>()   // rhs
        .Arg<ffi::Buffer<ffi::S64>>()    // lhs_offsets
        .Arg<ffi::Buffer<ffi::S64>>()    // rhs_offsets
        .Arg<ffi::Buffer<ffi::S64>>()    // out_offsets
        .Arg<ffi::Buffer<ffi::S32>>()    // Ms
        .Arg<ffi::Buffer<ffi::S32>>()    // Ns
        .Arg<ffi::Buffer<ffi::S32>>()    // Ks
        .Arg<ffi::Buffer<ffi::S32>>()    // lda
        .Arg<ffi::Buffer<ffi::S32>>()    // ldb
        .Arg<ffi::Buffer<ffi::S32>>()    // ldc
        .Arg<ffi::Buffer<ffi::S64>>()    // A_ptrs workspace
        .Arg<ffi::Buffer<ffi::S64>>()    // B_ptrs workspace
        .Arg<ffi::Buffer<ffi::S64>>()    // C_ptrs workspace
        .Attr<bool>("trans_rhs")
        .Ret<ffi::Buffer<ffi::BF16>>()   // out
);
