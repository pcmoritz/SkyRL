// cuBLAS grouped GEMM - computes metadata from group_sizes and group_offset
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
    ffi::Buffer<ffi::S32> group_sizes,      // [num_groups]
    ffi::Buffer<ffi::S32> group_offset_buf, // [1]
    ffi::Buffer<ffi::S64> A_ptrs,           // [g_local] workspace
    ffi::Buffer<ffi::S64> B_ptrs,           // [g_local] workspace
    ffi::Buffer<ffi::S64> C_ptrs,           // [g_local] workspace
    bool trans_rhs,
    ffi::ResultBuffer<ffi::BF16> out        // [m, n]
) {
    if (!g_handle) cublasCreate(&g_handle);
    cublasSetStream(g_handle, stream);

    // Get dimensions from arrays
    int64_t m = lhs.dimensions()[0];
    int64_t k = lhs.dimensions()[1];
    int64_t g_local = rhs.dimensions()[0];
    int64_t n = rhs.dimensions()[2];  // rhs is [g_local, k, n]
    int num_groups = static_cast<int>(group_sizes.dimensions()[0]);

    if (g_local == 0) return ffi::Error::Success();

    // Read group_offset from device
    int32_t group_offset;
    cudaMemcpyAsync(&group_offset, group_offset_buf.typed_data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

    // Read group_sizes from device
    std::vector<int32_t> h_group_sizes(num_groups);
    cudaMemcpyAsync(h_group_sizes.data(), group_sizes.typed_data(), num_groups * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Compute group boundaries (prefix sum)
    std::vector<int64_t> offsets(num_groups + 1);
    offsets[0] = 0;
    for (int i = 0; i < num_groups; i++) {
        offsets[i + 1] = offsets[i] + h_group_sizes[i];
    }

    const char* lhs_base = reinterpret_cast<const char*>(lhs.typed_data());
    const char* rhs_base = reinterpret_cast<const char*>(rhs.typed_data());
    char* out_base = reinterpret_cast<char*>(out->typed_data());

    // Fill pointer arrays and size arrays
    int64_t* A_ptr = const_cast<int64_t*>(A_ptrs.typed_data());
    int64_t* B_ptr = const_cast<int64_t*>(B_ptrs.typed_data());
    int64_t* C_ptr = const_cast<int64_t*>(C_ptrs.typed_data());

    std::vector<int> Ms(g_local), Ns(g_local), Ks(g_local);
    std::vector<int> lda(g_local), ldb(g_local), ldc(g_local);
    std::vector<cublasOperation_t> transa(g_local, trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb(g_local, CUBLAS_OP_N);
    std::vector<int> group_size(g_local, 1);

    for (int i = 0; i < g_local; i++) {
        int g = group_offset + i;
        int64_t start = offsets[g];
        int group_m = static_cast<int>(offsets[g + 1] - start);

        A_ptr[i] = reinterpret_cast<int64_t>(rhs_base + i * k * n * 2);
        B_ptr[i] = reinterpret_cast<int64_t>(lhs_base + start * k * 2);
        C_ptr[i] = reinterpret_cast<int64_t>(out_base + start * n * 2);

        Ms[i] = static_cast<int>(n);
        Ns[i] = group_m;
        Ks[i] = static_cast<int>(k);
        lda[i] = trans_rhs ? static_cast<int>(k) : static_cast<int>(n);
        ldb[i] = static_cast<int>(k);
        ldc[i] = static_cast<int>(n);
    }

    float alpha = 1.0f, beta = 0.0f;

    cublasGemmGroupedBatchedEx(
        g_handle,
        transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(),
        &alpha,
        reinterpret_cast<const void**>(A_ptr), CUDA_R_16BF, lda.data(),
        reinterpret_cast<const void**>(B_ptr), CUDA_R_16BF, ldb.data(),
        &beta,
        reinterpret_cast<void**>(C_ptr), CUDA_R_16BF, ldc.data(),
        static_cast<int>(g_local),
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
        .Arg<ffi::Buffer<ffi::S32>>()    // group_sizes
        .Arg<ffi::Buffer<ffi::S32>>()    // group_offset
        .Arg<ffi::Buffer<ffi::S64>>()    // A_ptrs workspace
        .Arg<ffi::Buffer<ffi::S64>>()    // B_ptrs workspace
        .Arg<ffi::Buffer<ffi::S64>>()    // C_ptrs workspace
        .Attr<bool>("trans_rhs")
        .Ret<ffi::Buffer<ffi::BF16>>()   // out
);
