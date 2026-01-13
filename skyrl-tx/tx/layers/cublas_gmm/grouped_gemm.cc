// cuBLAS grouped GEMM with forward and backward passes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static thread_local cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (!g_handle) cublasCreate(&g_handle);
    return g_handle;
}

// Helper to read group boundaries
std::vector<int64_t> get_offsets(cudaStream_t stream, const int32_t* d_sizes, int num_groups) {
    std::vector<int32_t> h_sizes(num_groups);
    cudaMemcpyAsync(h_sizes.data(), d_sizes, num_groups * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int64_t> offsets(num_groups + 1);
    offsets[0] = 0;
    for (int i = 0; i < num_groups; i++) {
        offsets[i + 1] = offsets[i] + h_sizes[i];
    }
    return offsets;
}

// Forward: out = lhs @ rhs per group
// lhs: [m, k], rhs: [g_local, k, n] -> out: [m, n]
ffi::Error GroupedGemmBf16Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> rhs,
    ffi::Buffer<ffi::S32> group_sizes,
    ffi::Buffer<ffi::S32> group_offset_buf,
    ffi::ResultBuffer<ffi::BF16> out
) {
    cublasSetStream(get_handle(), stream);

    int64_t m = lhs.dimensions()[0], k = lhs.dimensions()[1];
    int64_t g_local = rhs.dimensions()[0], n = rhs.dimensions()[2];
    int num_groups = group_sizes.dimensions()[0];

    if (g_local == 0) return ffi::Error::Success();

    int32_t group_offset;
    cudaMemcpyAsync(&group_offset, group_offset_buf.typed_data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    auto offsets = get_offsets(stream, group_sizes.typed_data(), num_groups);

    const char* lhs_base = reinterpret_cast<const char*>(lhs.typed_data());
    const char* rhs_base = reinterpret_cast<const char*>(rhs.typed_data());
    char* out_base = reinterpret_cast<char*>(out->typed_data());

    // Allocate pointer arrays on CPU
    std::vector<const void*> A_ptrs(g_local), B_ptrs(g_local);
    std::vector<void*> C_ptrs(g_local);

    std::vector<int> Ms(g_local), Ns(g_local), Ks(g_local);
    std::vector<int> lda(g_local), ldb(g_local), ldc(g_local);
    std::vector<cublasOperation_t> transa(g_local, CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb(g_local, CUBLAS_OP_N);
    std::vector<int> group_size(g_local, 1);

    for (int i = 0; i < g_local; i++) {
        int64_t start = offsets[group_offset + i];
        int group_m = offsets[group_offset + i + 1] - start;

        // Row-major: C = A @ B becomes C^T = B^T @ A^T in col-major
        A_ptrs[i] = rhs_base + i * k * n * 2;
        B_ptrs[i] = lhs_base + start * k * 2;
        C_ptrs[i] = out_base + start * n * 2;

        Ms[i] = n; Ns[i] = group_m; Ks[i] = k;
        lda[i] = n; ldb[i] = k; ldc[i] = n;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmGroupedBatchedEx(get_handle(), transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(), &alpha,
        A_ptrs.data(), CUDA_R_16BF, lda.data(),
        B_ptrs.data(), CUDA_R_16BF, ldb.data(),
        &beta, C_ptrs.data(), CUDA_R_16BF, ldc.data(),
        g_local, group_size.data(), CUBLAS_COMPUTE_32F);

    return ffi::Error::Success();
}

// Backward for lhs: d_lhs = dout @ rhs^T per group
// dout: [m, n], rhs: [g_local, k, n] -> d_lhs: [m, k]
ffi::Error GroupedGemmBf16TransImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> dout,
    ffi::Buffer<ffi::BF16> rhs,
    ffi::Buffer<ffi::S32> group_sizes,
    ffi::Buffer<ffi::S32> group_offset_buf,
    ffi::ResultBuffer<ffi::BF16> d_lhs
) {
    cublasSetStream(get_handle(), stream);

    int64_t m = dout.dimensions()[0], n = dout.dimensions()[1];
    int64_t g_local = rhs.dimensions()[0], k = rhs.dimensions()[1];
    int num_groups = group_sizes.dimensions()[0];

    if (g_local == 0) return ffi::Error::Success();

    int32_t group_offset;
    cudaMemcpyAsync(&group_offset, group_offset_buf.typed_data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    auto offsets = get_offsets(stream, group_sizes.typed_data(), num_groups);

    const char* dout_base = reinterpret_cast<const char*>(dout.typed_data());
    const char* rhs_base = reinterpret_cast<const char*>(rhs.typed_data());
    char* dlhs_base = reinterpret_cast<char*>(d_lhs->typed_data());

    std::vector<const void*> A_ptrs(g_local), B_ptrs(g_local);
    std::vector<void*> C_ptrs(g_local);

    std::vector<int> Ms(g_local), Ns(g_local), Ks(g_local);
    std::vector<int> lda(g_local), ldb(g_local), ldc(g_local);
    std::vector<cublasOperation_t> transa(g_local, CUBLAS_OP_T);  // rhs transposed
    std::vector<cublasOperation_t> transb(g_local, CUBLAS_OP_N);
    std::vector<int> group_size(g_local, 1);

    for (int i = 0; i < g_local; i++) {
        int64_t start = offsets[group_offset + i];
        int group_m = offsets[group_offset + i + 1] - start;

        // d_lhs = dout @ rhs^T: [group_m, n] @ [n, k] -> [group_m, k]
        A_ptrs[i] = rhs_base + i * k * n * 2;
        B_ptrs[i] = dout_base + start * n * 2;
        C_ptrs[i] = dlhs_base + start * k * 2;

        Ms[i] = k; Ns[i] = group_m; Ks[i] = n;
        lda[i] = n;  // rhs is [k, n], transposed access
        ldb[i] = n;
        ldc[i] = k;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmGroupedBatchedEx(get_handle(), transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(), &alpha,
        A_ptrs.data(), CUDA_R_16BF, lda.data(),
        B_ptrs.data(), CUDA_R_16BF, ldb.data(),
        &beta, C_ptrs.data(), CUDA_R_16BF, ldc.data(),
        g_local, group_size.data(), CUBLAS_COMPUTE_32F);

    return ffi::Error::Success();
}

// Backward for rhs: d_rhs[i] = lhs[group_i]^T @ dout[group_i]
// lhs: [m, k], dout: [m, n] -> d_rhs: [g_local, k, n]
ffi::Error GroupedGemmBf16DwImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> dout,
    ffi::Buffer<ffi::S32> group_sizes,
    ffi::Buffer<ffi::S32> group_offset_buf,
    ffi::ResultBuffer<ffi::BF16> d_rhs
) {
    cublasSetStream(get_handle(), stream);

    int64_t m = lhs.dimensions()[0], k = lhs.dimensions()[1];
    int64_t n = dout.dimensions()[1];
    int64_t g_local = d_rhs->dimensions()[0];
    int num_groups = group_sizes.dimensions()[0];

    if (g_local == 0) return ffi::Error::Success();

    int32_t group_offset;
    cudaMemcpyAsync(&group_offset, group_offset_buf.typed_data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    auto offsets = get_offsets(stream, group_sizes.typed_data(), num_groups);

    const char* lhs_base = reinterpret_cast<const char*>(lhs.typed_data());
    const char* dout_base = reinterpret_cast<const char*>(dout.typed_data());
    char* drhs_base = reinterpret_cast<char*>(d_rhs->typed_data());

    std::vector<const void*> A_ptrs(g_local), B_ptrs(g_local);
    std::vector<void*> C_ptrs(g_local);

    std::vector<int> Ms(g_local), Ns(g_local), Ks(g_local);
    std::vector<int> lda(g_local), ldb(g_local), ldc(g_local);
    std::vector<cublasOperation_t> transa(g_local, CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb(g_local, CUBLAS_OP_T);  // lhs transposed
    std::vector<int> group_size(g_local, 1);

    for (int i = 0; i < g_local; i++) {
        int64_t start = offsets[group_offset + i];
        int group_m = offsets[group_offset + i + 1] - start;

        // d_rhs = lhs^T @ dout: [k, group_m] @ [group_m, n] -> [k, n]
        A_ptrs[i] = dout_base + start * n * 2;
        B_ptrs[i] = lhs_base + start * k * 2;
        C_ptrs[i] = drhs_base + i * k * n * 2;

        Ms[i] = n; Ns[i] = k; Ks[i] = group_m;
        lda[i] = n;
        ldb[i] = k;  // lhs is [group_m, k], transposed access
        ldc[i] = n;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmGroupedBatchedEx(get_handle(), transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(), &alpha,
        A_ptrs.data(), CUDA_R_16BF, lda.data(),
        B_ptrs.data(), CUDA_R_16BF, ldb.data(),
        &beta, C_ptrs.data(), CUDA_R_16BF, ldc.data(),
        g_local, group_size.data(), CUBLAS_COMPUTE_32F);

    return ffi::Error::Success();
}

#define BINDING \
    ffi::Ffi::Bind() \
        .Ctx<ffi::PlatformStream<cudaStream_t>>() \
        .Arg<ffi::Buffer<ffi::BF16>>() \
        .Arg<ffi::Buffer<ffi::BF16>>() \
        .Arg<ffi::Buffer<ffi::S32>>() \
        .Arg<ffi::Buffer<ffi::S32>>() \
        .Ret<ffi::Buffer<ffi::BF16>>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmBf16, GroupedGemmBf16Impl, BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmBf16Trans, GroupedGemmBf16TransImpl, BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmBf16Dw, GroupedGemmBf16DwImpl, BINDING);
