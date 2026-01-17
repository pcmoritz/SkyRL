#include "xla/ffi/api/ffi.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

struct DtypeInfo {
  cudaDataType data_type;
  cublasComputeType_t compute_type;
  size_t elem_size;
};

const char* CublasStatusToString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

xla::ffi::Error CudaError(cudaError_t status, const char* msg) {
  return xla::ffi::Error::Internal(std::string(msg) + ": " +
                                   cudaGetErrorString(status));
}

xla::ffi::Error CublasError(cublasStatus_t status, const char* msg) {
  return xla::ffi::Error::Internal(std::string(msg) + ": " +
                                   CublasStatusToString(status));
}

bool GetDtypeInfo(xla::ffi::DataType dtype, DtypeInfo* info) {
  switch (dtype) {
    case xla::ffi::DataType::F16:
      info->data_type = CUDA_R_16F;
      info->compute_type = CUBLAS_COMPUTE_32F;
      info->elem_size = 2;
      return true;
    case xla::ffi::DataType::BF16:
      info->data_type = CUDA_R_16BF;
      info->compute_type = CUBLAS_COMPUTE_32F;
      info->elem_size = 2;
      return true;
    case xla::ffi::DataType::F32:
      info->data_type = CUDA_R_32F;
      info->compute_type = CUBLAS_COMPUTE_32F;
      info->elem_size = 4;
      return true;
    case xla::ffi::DataType::F64:
      info->data_type = CUDA_R_64F;
      info->compute_type = CUBLAS_COMPUTE_64F;
      info->elem_size = 8;
      return true;
    default:
      return false;
  }
}

cublasHandle_t GetHandle() {
  thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      return nullptr;
    }
  }
  return handle;
}

xla::ffi::Error CublasGroupedGemmImpl(
    cudaStream_t stream, xla::ffi::AnyBuffer lhs, xla::ffi::AnyBuffer rhs,
    xla::ffi::BufferR1<xla::ffi::DataType::S32> group_sizes,
    xla::ffi::BufferR1<xla::ffi::DataType::S32> group_offset,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  auto lhs_dims = lhs.dimensions();
  auto rhs_dims = rhs.dimensions();
  auto out_dims = (*out).dimensions();

  if (lhs_dims.size() != 2 || rhs_dims.size() != 3 || out_dims.size() != 2) {
    return xla::ffi::Error::InvalidArgument("expected lhs(2d), rhs(3d), out(2d)");
  }

  if (group_sizes.dimensions().size() != 1 || group_offset.dimensions().size() != 1) {
    return xla::ffi::Error::InvalidArgument("expected group_sizes/group_offset rank 1");
  }

  if (group_offset.dimensions().front() != 1) {
    return xla::ffi::Error::InvalidArgument("group_offset must have shape (1,)");
  }

  if (lhs.element_type() != rhs.element_type() ||
      lhs.element_type() != (*out).element_type()) {
    return xla::ffi::Error::InvalidArgument("lhs/rhs/out must share dtype");
  }

  DtypeInfo dtype_info;
  if (!GetDtypeInfo(lhs.element_type(), &dtype_info)) {
    return xla::ffi::Error::InvalidArgument("unsupported dtype for cublas grouped gemm");
  }

  const int64_t m = lhs_dims[0];
  const int64_t k = lhs_dims[1];
  const int64_t g_local = rhs_dims[0];
  const int64_t rhs_k = rhs_dims[1];
  const int64_t n = rhs_dims[2];

  if (rhs_k != k) {
    return xla::ffi::Error::InvalidArgument("lhs and rhs k dimension mismatch");
  }
  if (out_dims[0] != m || out_dims[1] != n) {
    return xla::ffi::Error::InvalidArgument("out shape mismatch");
  }

  const int64_t num_groups = group_sizes.dimensions().front();
  if (g_local < 0 || g_local > num_groups) {
    return xla::ffi::Error::InvalidArgument("rhs group count mismatch");
  }
  if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    return xla::ffi::Error::InvalidArgument("matrix dimensions exceed int32");
  }

  std::vector<int32_t> h_group_sizes(num_groups);
  std::vector<int32_t> h_group_offset(1);

  cudaError_t cuda_status = cudaMemcpyAsync(
      h_group_sizes.data(), group_sizes.typed_data(),
      num_groups * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError(cuda_status, "failed to copy group_sizes to host");
  }
  cuda_status = cudaMemcpyAsync(h_group_offset.data(), group_offset.typed_data(),
                                sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError(cuda_status, "failed to copy group_offset to host");
  }
  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    return CudaError(cuda_status, "failed to sync stream for group metadata");
  }

  const int32_t offset = h_group_offset[0];
  if (offset < 0 || offset + g_local > num_groups) {
    return xla::ffi::Error::InvalidArgument("group_offset out of range");
  }

  std::vector<int64_t> offsets(num_groups + 1, 0);
  for (int64_t i = 0; i < num_groups; ++i) {
    offsets[i + 1] = offsets[i] + h_group_sizes[i];
  }
  if (offsets.back() != m) {
    return xla::ffi::Error::InvalidArgument("sum(group_sizes) must equal lhs rows");
  }

  size_t out_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * dtype_info.elem_size;
  cuda_status = cudaMemsetAsync((*out).untyped_data(), 0, out_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError(cuda_status, "failed to clear output");
  }

  std::vector<int> m_array;
  std::vector<int> n_array;
  std::vector<int> k_array;
  std::vector<int> lda_array;
  std::vector<int> ldb_array;
  std::vector<int> ldc_array;
  std::vector<cublasOperation_t> transa_array;
  std::vector<cublasOperation_t> transb_array;
  std::vector<const void*> a_array;
  std::vector<const void*> b_array;
  std::vector<void*> c_array;

  const char* lhs_base = reinterpret_cast<const char*>(lhs.untyped_data());
  const char* rhs_base = reinterpret_cast<const char*>(rhs.untyped_data());
  char* out_base = reinterpret_cast<char*>((*out).untyped_data());

  for (int64_t g = 0; g < g_local; ++g) {
    const int64_t global_group = offset + g;
    const int32_t group_m = h_group_sizes[global_group];
    if (group_m <= 0) {
      continue;
    }

    const int64_t row_start = offsets[global_group];
    const void* a_ptr = rhs_base + g * k * n * dtype_info.elem_size;
    const void* b_ptr = lhs_base + row_start * k * dtype_info.elem_size;
    void* c_ptr = out_base + row_start * n * dtype_info.elem_size;

    if (group_m > std::numeric_limits<int>::max()) {
      return xla::ffi::Error::InvalidArgument("group size exceeds int32");
    }
    m_array.push_back(static_cast<int>(n));
    n_array.push_back(static_cast<int>(group_m));
    k_array.push_back(static_cast<int>(k));
    lda_array.push_back(static_cast<int>(n));
    ldb_array.push_back(static_cast<int>(k));
    ldc_array.push_back(static_cast<int>(n));
    transa_array.push_back(CUBLAS_OP_N);
    transb_array.push_back(CUBLAS_OP_N);
    a_array.push_back(a_ptr);
    b_array.push_back(b_ptr);
    c_array.push_back(c_ptr);
  }

  const int group_count = static_cast<int>(a_array.size());
  if (group_count == 0) {
    return xla::ffi::Error::Success();
  }

  cublasHandle_t handle = GetHandle();
  if (handle == nullptr) {
    return xla::ffi::Error::Internal("failed to create cublas handle");
  }

  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "failed to set cublas stream");
  }
  status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "failed to set cublas pointer mode");
  }

  float alpha_f = 1.0f;
  float beta_f = 0.0f;
  double alpha_d = 1.0;
  double beta_d = 0.0;

  const void* alpha = (dtype_info.compute_type == CUBLAS_COMPUTE_64F)
                          ? static_cast<const void*>(&alpha_d)
                          : static_cast<const void*>(&alpha_f);
  const void* beta = (dtype_info.compute_type == CUBLAS_COMPUTE_64F)
                         ? static_cast<const void*>(&beta_d)
                         : static_cast<const void*>(&beta_f);

  status = cublasGemmGroupedBatchedEx(
      handle, transa_array.data(), transb_array.data(), m_array.data(),
      n_array.data(), k_array.data(), alpha, a_array.data(),
      dtype_info.data_type, lda_array.data(), b_array.data(),
      dtype_info.data_type, ldb_array.data(), beta, c_array.data(),
      dtype_info.data_type, ldc_array.data(), group_count,
      dtype_info.compute_type, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "cublasGemmGroupedBatchedEx failed");
  }

  return xla::ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cublas_gemm_grouped_batched_ex, CublasGroupedGemmImpl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::BufferR1<xla::ffi::DataType::S32>>()
        .Arg<xla::ffi::BufferR1<xla::ffi::DataType::S32>>()
        .Ret<xla::ffi::AnyBuffer>());
