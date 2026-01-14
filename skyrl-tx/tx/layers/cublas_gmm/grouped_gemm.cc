// cuBLAS grouped GEMM with forward and backward passes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static thread_local cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (!g_handle) {
        cublasStatus_t status = cublasCreate(&g_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS create failed: %d\n", status);
        }
    }
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
// ptrs: [g_local * 2] int32 buffer to hold pointers (reinterpreted as int64)
ffi::Error GroupedGemmBf16Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> rhs,
    ffi::Buffer<ffi::S32> group_sizes,
    ffi::Buffer<ffi::S32> group_offset_buf,
    ffi::ResultBuffer<ffi::S32> A_ptrs_buf,  // mutable
    ffi::ResultBuffer<ffi::S32> B_ptrs_buf,
    ffi::ResultBuffer<ffi::S32> C_ptrs_buf,
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

    // Zero-initialize output buffer (for tokens outside local groups)
    cudaMemsetAsync(out_base, 0, m * n * 2, stream);

    // Mutable device pointer arrays
    int64_t* d_A_ptrs = reinterpret_cast<int64_t*>(A_ptrs_buf->typed_data());
    int64_t* d_B_ptrs = reinterpret_cast<int64_t*>(B_ptrs_buf->typed_data());
    int64_t* d_C_ptrs = reinterpret_cast<int64_t*>(C_ptrs_buf->typed_data());

    // Build pointer arrays on host, skipping zero-sized groups
    std::vector<int64_t> h_A_ptrs, h_B_ptrs, h_C_ptrs;
    std::vector<int> Ms, Ns, Ks;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<cublasOperation_t> transa, transb;

    for (int i = 0; i < g_local; i++) {
        int64_t start = offsets[group_offset + i];
        int group_m = offsets[group_offset + i + 1] - start;

        // Skip zero-sized groups
        if (group_m == 0) continue;

        // Row-major: C = A @ B becomes C^T = B^T @ A^T in col-major
        h_A_ptrs.push_back(reinterpret_cast<int64_t>(rhs_base + i * k * n * 2));
        h_B_ptrs.push_back(reinterpret_cast<int64_t>(lhs_base + start * k * 2));
        h_C_ptrs.push_back(reinterpret_cast<int64_t>(out_base + start * n * 2));

        Ms.push_back(n); Ns.push_back(group_m); Ks.push_back(k);
        lda_vec.push_back(n); ldb_vec.push_back(k); ldc_vec.push_back(n);
        transa.push_back(CUBLAS_OP_N);
        transb.push_back(CUBLAS_OP_N);
    }

    int num_active = h_A_ptrs.size();
    if (num_active == 0) {
        // Sync to ensure memset completes before returning
        cudaStreamSynchronize(stream);
        return ffi::Error::Success();
    }

    std::vector<int> group_size(num_active, 1);

    // Copy pointer arrays to device
    cudaMemcpyAsync(d_A_ptrs, h_A_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_ptrs, h_B_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_ptrs, h_C_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Sync to ensure pointer arrays are ready
    cudaStreamSynchronize(stream);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmGroupedBatchedEx(get_handle(), transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(), &alpha,
        reinterpret_cast<const void**>(d_A_ptrs), CUDA_R_16BF, lda_vec.data(),
        reinterpret_cast<const void**>(d_B_ptrs), CUDA_R_16BF, ldb_vec.data(),
        &beta, reinterpret_cast<void**>(d_C_ptrs), CUDA_R_16BF, ldc_vec.data(),
        num_active, group_size.data(), CUBLAS_COMPUTE_32F);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS grouped GEMM fwd failed: %d (m=%ld, k=%ld, n=%ld, num_active=%d)\n",
                status, m, k, n, num_active);
        return ffi::Error(ffi::ErrorCode::kInternal, "cuBLAS grouped GEMM failed");
    }

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
    ffi::ResultBuffer<ffi::S32> A_ptrs_buf,
    ffi::ResultBuffer<ffi::S32> B_ptrs_buf,
    ffi::ResultBuffer<ffi::S32> C_ptrs_buf,
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

    // Zero-initialize output buffer
    cudaMemsetAsync(dlhs_base, 0, m * k * 2, stream);

    int64_t* d_A_ptrs = reinterpret_cast<int64_t*>(A_ptrs_buf->typed_data());
    int64_t* d_B_ptrs = reinterpret_cast<int64_t*>(B_ptrs_buf->typed_data());
    int64_t* d_C_ptrs = reinterpret_cast<int64_t*>(C_ptrs_buf->typed_data());

    std::vector<int64_t> h_A_ptrs, h_B_ptrs, h_C_ptrs;
    std::vector<int> Ms, Ns, Ks;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<cublasOperation_t> transa, transb;

    for (int i = 0; i < g_local; i++) {
        int64_t start = offsets[group_offset + i];
        int group_m = offsets[group_offset + i + 1] - start;

        if (group_m == 0) continue;

        // d_lhs = dout @ rhs^T: [group_m, n] @ [n, k] -> [group_m, k]
        h_A_ptrs.push_back(reinterpret_cast<int64_t>(rhs_base + i * k * n * 2));
        h_B_ptrs.push_back(reinterpret_cast<int64_t>(dout_base + start * n * 2));
        h_C_ptrs.push_back(reinterpret_cast<int64_t>(dlhs_base + start * k * 2));

        Ms.push_back(k); Ns.push_back(group_m); Ks.push_back(n);
        lda_vec.push_back(n);  // rhs is [k, n], transposed access
        ldb_vec.push_back(n);
        ldc_vec.push_back(k);
        transa.push_back(CUBLAS_OP_T);
        transb.push_back(CUBLAS_OP_N);
    }

    int num_active = h_A_ptrs.size();
    if (num_active == 0) {
        cudaStreamSynchronize(stream);
        return ffi::Error::Success();
    }

    std::vector<int> group_size(num_active, 1);

    cudaMemcpyAsync(d_A_ptrs, h_A_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_ptrs, h_B_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_ptrs, h_C_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmGroupedBatchedEx(get_handle(), transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(), &alpha,
        reinterpret_cast<const void**>(d_A_ptrs), CUDA_R_16BF, lda_vec.data(),
        reinterpret_cast<const void**>(d_B_ptrs), CUDA_R_16BF, ldb_vec.data(),
        &beta, reinterpret_cast<void**>(d_C_ptrs), CUDA_R_16BF, ldc_vec.data(),
        num_active, group_size.data(), CUBLAS_COMPUTE_32F);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS grouped GEMM trans failed: %d\n", status);
        return ffi::Error(ffi::ErrorCode::kInternal, "cuBLAS grouped GEMM trans failed");
    }

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
    ffi::ResultBuffer<ffi::S32> A_ptrs_buf,
    ffi::ResultBuffer<ffi::S32> B_ptrs_buf,
    ffi::ResultBuffer<ffi::S32> C_ptrs_buf,
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

    // Zero-initialize output buffer
    cudaMemsetAsync(drhs_base, 0, g_local * k * n * 2, stream);

    int64_t* d_A_ptrs = reinterpret_cast<int64_t*>(A_ptrs_buf->typed_data());
    int64_t* d_B_ptrs = reinterpret_cast<int64_t*>(B_ptrs_buf->typed_data());
    int64_t* d_C_ptrs = reinterpret_cast<int64_t*>(C_ptrs_buf->typed_data());

    std::vector<int64_t> h_A_ptrs, h_B_ptrs, h_C_ptrs;
    std::vector<int> Ms, Ns, Ks;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<cublasOperation_t> transa, transb;

    for (int i = 0; i < g_local; i++) {
        int64_t start = offsets[group_offset + i];
        int group_m = offsets[group_offset + i + 1] - start;

        // For dw, we still need to write to d_rhs[i] even if group_m=0
        // But with K=0, the result should be zeros (which we already set via memset)
        if (group_m == 0) continue;

        // d_rhs = lhs^T @ dout: [k, group_m] @ [group_m, n] -> [k, n]
        h_A_ptrs.push_back(reinterpret_cast<int64_t>(dout_base + start * n * 2));
        h_B_ptrs.push_back(reinterpret_cast<int64_t>(lhs_base + start * k * 2));
        h_C_ptrs.push_back(reinterpret_cast<int64_t>(drhs_base + i * k * n * 2));

        Ms.push_back(n); Ns.push_back(k); Ks.push_back(group_m);
        lda_vec.push_back(n);
        ldb_vec.push_back(k);  // lhs is [group_m, k], transposed access
        ldc_vec.push_back(n);
        transa.push_back(CUBLAS_OP_N);
        transb.push_back(CUBLAS_OP_T);
    }

    int num_active = h_A_ptrs.size();
    if (num_active == 0) {
        cudaStreamSynchronize(stream);
        return ffi::Error::Success();
    }

    std::vector<int> group_size(num_active, 1);

    cudaMemcpyAsync(d_A_ptrs, h_A_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_ptrs, h_B_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_ptrs, h_C_ptrs.data(), num_active * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmGroupedBatchedEx(get_handle(), transa.data(), transb.data(),
        Ms.data(), Ns.data(), Ks.data(), &alpha,
        reinterpret_cast<const void**>(d_A_ptrs), CUDA_R_16BF, lda_vec.data(),
        reinterpret_cast<const void**>(d_B_ptrs), CUDA_R_16BF, ldb_vec.data(),
        &beta, reinterpret_cast<void**>(d_C_ptrs), CUDA_R_16BF, ldc_vec.data(),
        num_active, group_size.data(), CUBLAS_COMPUTE_32F);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS grouped GEMM dw failed: %d\n", status);
        return ffi::Error(ffi::ErrorCode::kInternal, "cuBLAS grouped GEMM dw failed");
    }

    return ffi::Error::Success();
}

#define BINDING \
    ffi::Ffi::Bind() \
        .Ctx<ffi::PlatformStream<cudaStream_t>>() \
        .Arg<ffi::Buffer<ffi::BF16>>() \
        .Arg<ffi::Buffer<ffi::BF16>>() \
        .Arg<ffi::Buffer<ffi::S32>>() \
        .Arg<ffi::Buffer<ffi::S32>>() \
        .Ret<ffi::Buffer<ffi::S32>>() \
        .Ret<ffi::Buffer<ffi::S32>>() \
        .Ret<ffi::Buffer<ffi::S32>>() \
        .Ret<ffi::Buffer<ffi::BF16>>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmBf16, GroupedGemmBf16Impl, BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmBf16Trans, GroupedGemmBf16TransImpl, BINDING);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmBf16Dw, GroupedGemmBf16DwImpl, BINDING);
