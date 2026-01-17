#include "xla/ffi/api/ffi.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <cstring>
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

xla::ffi::Error CopyToHost(void* dst, const void* src, size_t bytes,
                           cudaStream_t stream) {
  if (bytes == 0) {
    return xla::ffi::Error::Success();
  }

  cudaPointerAttributes attrs{};
  cudaError_t attr_status = cudaPointerGetAttributes(&attrs, src);
  if (attr_status == cudaErrorInvalidValue) {
    std::memcpy(dst, src, bytes);
    return xla::ffi::Error::Success();
  }
  if (attr_status != cudaSuccess) {
    return CudaError(attr_status, "failed to query pointer attributes");
  }

#if CUDART_VERSION >= 10000
  if (attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged) {
#else
  if (attrs.memoryType == cudaMemoryTypeDevice) {
#endif
    cudaError_t copy_status =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    if (copy_status != cudaSuccess) {
      return CudaError(copy_status, "failed to copy device memory to host");
    }
    cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
      return CudaError(sync_status, "failed to sync stream for host copy");
    }
    return xla::ffi::Error::Success();
  }

  std::memcpy(dst, src, bytes);
  return xla::ffi::Error::Success();
}

bool IsDebugEnabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_DEBUG") != nullptr;
  return enabled;
}

bool IsVerboseEnabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_VERBOSE") != nullptr;
  return enabled;
}

bool IsDryRunEnabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_DRYRUN") != nullptr;
  return enabled;
}

bool IsTensorOpDisabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_NO_TENSOR_OP") != nullptr;
  return enabled;
}

bool IsSyncEnabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_SYNC") != nullptr;
  return enabled;
}

int GroupLogCount() {
  const char* env = std::getenv("TX_CUBLAS_GROUPED_GEMM_LOG_GROUPS");
  if (!env || env[0] == '\0') {
    return 0;
  }
  int count = std::atoi(env);
  return count < 0 ? 0 : count;
}

bool IsForceFp32Enabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_FORCE_FP32") != nullptr;
  return enabled;
}

bool IsUseGemmExEnabled() {
  static const bool enabled = std::getenv("TX_CUBLAS_GROUPED_GEMM_USE_GEMM_EX") != nullptr;
  return enabled;
}

xla::ffi::Error ValidateDevicePointer(const void* ptr, int device, bool allow_host,
                                      const char* name) {
  if (ptr == nullptr) {
    return xla::ffi::Error::InvalidArgument(std::string(name) + " pointer is null");
  }

  cudaPointerAttributes attrs{};
  cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
  if (status == cudaErrorInvalidValue) {
    if (allow_host) {
      return xla::ffi::Error::Success();
    }
    return xla::ffi::Error::InvalidArgument(
        std::string(name) + " pointer is not a device allocation");
  }
  if (status != cudaSuccess) {
    return CudaError(status, ("failed to query pointer attributes for " + std::string(name)).c_str());
  }

#if CUDART_VERSION >= 10000
  if (attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged) {
    if (attrs.device != device) {
      return xla::ffi::Error::InvalidArgument(
          std::string(name) + " pointer is on device " + std::to_string(attrs.device) +
          ", expected device " + std::to_string(device));
    }
    return xla::ffi::Error::Success();
  }
  if (attrs.type == cudaMemoryTypeHost) {
    if (allow_host) {
      return xla::ffi::Error::Success();
    }
    return xla::ffi::Error::InvalidArgument(
        std::string(name) + " pointer is host memory");
  }
#else
  if (attrs.memoryType == cudaMemoryTypeDevice) {
    if (attrs.device != device) {
      return xla::ffi::Error::InvalidArgument(
          std::string(name) + " pointer is on device " + std::to_string(attrs.device) +
          ", expected device " + std::to_string(device));
    }
    return xla::ffi::Error::Success();
  }
  if (attrs.memoryType == cudaMemoryTypeHost) {
    if (allow_host) {
      return xla::ffi::Error::Success();
    }
    return xla::ffi::Error::InvalidArgument(
        std::string(name) + " pointer is host memory");
  }
#endif
  return xla::ffi::Error::InvalidArgument(
      std::string(name) + " pointer has unsupported memory type");
}

const char* PointerTypeName(const cudaPointerAttributes& attrs) {
#if CUDART_VERSION >= 10000
  switch (attrs.type) {
    case cudaMemoryTypeDevice:
      return "device";
    case cudaMemoryTypeHost:
      return "host";
    case cudaMemoryTypeManaged:
      return "managed";
    default:
      return "unknown";
  }
#else
  switch (attrs.memoryType) {
    case cudaMemoryTypeDevice:
      return "device";
    case cudaMemoryTypeHost:
      return "host";
    default:
      return "unknown";
  }
#endif
}

bool CheckRange(size_t offset_bytes, size_t bytes_needed, size_t total_bytes) {
  if (offset_bytes > total_bytes) {
    return false;
  }
  return bytes_needed <= (total_bytes - offset_bytes);
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
#ifdef CUBLAS_COMPUTE_32F_FAST_16BF
      info->compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
#else
      info->compute_type = CUBLAS_COMPUTE_32F;
#endif
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

struct HandleSlot {
  std::mutex mu;
  cublasHandle_t handle = nullptr;
};

HandleSlot* GetHandleSlot(int device, xla::ffi::Error* err) {
  static std::once_flag init_once;
  static std::vector<std::unique_ptr<HandleSlot>> slots;
  std::call_once(init_once, []() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
      return;
    }
    slots.reserve(count);
    for (int i = 0; i < count; ++i) {
      slots.emplace_back(std::make_unique<HandleSlot>());
    }
  });

  if (device < 0 || device >= static_cast<int>(slots.size())) {
    *err = xla::ffi::Error::InvalidArgument("invalid device ordinal");
    return nullptr;
  }
  return slots[device].get();
}

bool EnsureHandleInitialized(int device, xla::ffi::Error* err) {
  HandleSlot* slot = GetHandleSlot(device, err);
  if (slot == nullptr) {
    return false;
  }
  std::unique_lock<std::mutex> lock(slot->mu);
  if (slot->handle != nullptr) {
    return true;
  }
  cublasStatus_t create_status = cublasCreate(&slot->handle);
  if (create_status != CUBLAS_STATUS_SUCCESS) {
    *err = CublasError(create_status, "failed to create cublas handle");
    return false;
  }
  return true;
}

xla::ffi::Error CublasGroupedGemmImpl(
    cudaStream_t stream, xla::ffi::AnyBuffer lhs, xla::ffi::AnyBuffer rhs,
    xla::ffi::BufferR1<xla::ffi::DataType::S32> group_sizes,
    xla::ffi::BufferR1<xla::ffi::DataType::S32> group_offset,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  int device = -1;
  cudaError_t cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) {
    return CudaError(cuda_status, "failed to get cuda device");
  }
  if (IsDebugEnabled()) {
    cudaError_t peek_status = cudaPeekAtLastError();
    if (peek_status != cudaSuccess) {
      return CudaError(peek_status, "previous CUDA error before cublas grouped gemm");
    }
  }

  int cc_major = 0;
  cuda_status = cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
  if (cuda_status != cudaSuccess) {
    return CudaError(cuda_status, "failed to query compute capability");
  }

  xla::ffi::Error ptr_err = ValidateDevicePointer(lhs.untyped_data(), device, false, "lhs");
  if (ptr_err.failure()) {
    return ptr_err;
  }
  ptr_err = ValidateDevicePointer(rhs.untyped_data(), device, false, "rhs");
  if (ptr_err.failure()) {
    return ptr_err;
  }
  ptr_err = ValidateDevicePointer((*out).untyped_data(), device, false, "out");
  if (ptr_err.failure()) {
    return ptr_err;
  }
  ptr_err = ValidateDevicePointer(group_sizes.untyped_data(), device, true, "group_sizes");
  if (ptr_err.failure()) {
    return ptr_err;
  }
  ptr_err = ValidateDevicePointer(group_offset.untyped_data(), device, true, "group_offset");
  if (ptr_err.failure()) {
    return ptr_err;
  }

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
  if ((lhs.element_type() == xla::ffi::DataType::F16 ||
       lhs.element_type() == xla::ffi::DataType::BF16) &&
      IsForceFp32Enabled()) {
    dtype_info.compute_type = CUBLAS_COMPUTE_32F;
  }
  if (lhs.element_type() == xla::ffi::DataType::BF16 && cc_major < 8) {
    return xla::ffi::Error::InvalidArgument("bf16 requires sm80+ for grouped gemm");
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

  xla::ffi::Error copy_err =
      CopyToHost(h_group_sizes.data(), group_sizes.typed_data(),
                 num_groups * sizeof(int32_t), stream);
  if (copy_err.failure()) {
    return copy_err;
  }
  copy_err = CopyToHost(h_group_offset.data(), group_offset.typed_data(),
                        sizeof(int32_t), stream);
  if (copy_err.failure()) {
    return copy_err;
  }

  const int32_t offset = h_group_offset[0];
  if (offset < 0 || offset + g_local > num_groups) {
    return xla::ffi::Error::InvalidArgument("group_offset out of range");
  }

  std::vector<int64_t> offsets(num_groups + 1, 0);
  int32_t min_group = std::numeric_limits<int32_t>::max();
  int32_t max_group = std::numeric_limits<int32_t>::min();
  for (int64_t i = 0; i < num_groups; ++i) {
    if (h_group_sizes[i] < 0) {
      return xla::ffi::Error::InvalidArgument("group_sizes must be non-negative");
    }
    min_group = std::min(min_group, h_group_sizes[i]);
    max_group = std::max(max_group, h_group_sizes[i]);
    offsets[i + 1] = offsets[i] + h_group_sizes[i];
  }
  if (offsets.back() != m) {
    return xla::ffi::Error::InvalidArgument("sum(group_sizes) must equal lhs rows");
  }
  if (IsDebugEnabled()) {
    for (int64_t g = 0; g < g_local; ++g) {
      const int64_t global_group = offset + g;
      const int64_t row_start = offsets[global_group];
      const int32_t group_m = h_group_sizes[global_group];
      if (row_start < 0 || row_start + group_m > m) {
        return xla::ffi::Error::InvalidArgument("group_sizes produce out-of-range rows");
      }
    }
  }
  if (IsVerboseEnabled()) {
    std::fprintf(stderr,
                 "cublas_grouped_gemm: device=%d dtype=%d m=%lld k=%lld n=%lld "
                 "num_groups=%lld g_local=%lld offset=%d min_group=%d max_group=%d\n",
                 device, static_cast<int>(lhs.element_type()),
                 static_cast<long long>(m), static_cast<long long>(k), static_cast<long long>(n),
                 static_cast<long long>(num_groups), static_cast<long long>(g_local),
                 offset, static_cast<int>(min_group), static_cast<int>(max_group));
    std::fprintf(stderr, "group_sizes[0..7]:");
    int64_t to_print = std::min<int64_t>(num_groups, 8);
    for (int64_t i = 0; i < to_print; ++i) {
      std::fprintf(stderr, " %d", h_group_sizes[i]);
    }
    std::fprintf(stderr, "\n");
    cudaPointerAttributes lhs_attrs{};
    cudaPointerAttributes rhs_attrs{};
    cudaPointerAttributes out_attrs{};
    cudaPointerAttributes group_sizes_attrs{};
    cudaPointerAttributes group_offset_attrs{};
    if (cudaPointerGetAttributes(&lhs_attrs, lhs.untyped_data()) == cudaSuccess &&
        cudaPointerGetAttributes(&rhs_attrs, rhs.untyped_data()) == cudaSuccess &&
        cudaPointerGetAttributes(&out_attrs, (*out).untyped_data()) == cudaSuccess) {
      std::fprintf(stderr,
                   "lhs_ptr=%p (%s dev=%d) rhs_ptr=%p (%s dev=%d) out_ptr=%p (%s dev=%d)\n",
                   lhs.untyped_data(), PointerTypeName(lhs_attrs), lhs_attrs.device,
                   rhs.untyped_data(), PointerTypeName(rhs_attrs), rhs_attrs.device,
                   (*out).untyped_data(), PointerTypeName(out_attrs), out_attrs.device);
    }
    if (cudaPointerGetAttributes(&group_sizes_attrs, group_sizes.untyped_data()) == cudaSuccess &&
        cudaPointerGetAttributes(&group_offset_attrs, group_offset.untyped_data()) == cudaSuccess) {
      std::fprintf(stderr, "group_sizes_ptr=%p (%s dev=%d) group_offset_ptr=%p (%s dev=%d)\n",
                   group_sizes.untyped_data(), PointerTypeName(group_sizes_attrs),
                   group_sizes_attrs.device, group_offset.untyped_data(),
                   PointerTypeName(group_offset_attrs), group_offset_attrs.device);
    }
    std::fflush(stderr);
  }

  size_t lhs_bytes = lhs.size_bytes();
  size_t rhs_bytes = rhs.size_bytes();
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
  std::vector<int> group_size_array;
  std::vector<const void*> a_array;
  std::vector<const void*> b_array;
  std::vector<void*> c_array;

  const char* lhs_base = reinterpret_cast<const char*>(lhs.untyped_data());
  const char* rhs_base = reinterpret_cast<const char*>(rhs.untyped_data());
  char* out_base = reinterpret_cast<char*>((*out).untyped_data());
  const int log_groups = GroupLogCount();
  int logged = 0;

  for (int64_t g = 0; g < g_local; ++g) {
    const int64_t global_group = offset + g;
    const int32_t group_m = h_group_sizes[global_group];
    if (group_m <= 0) {
      continue;
    }

    const int64_t row_start = offsets[global_group];
    const size_t a_offset_bytes =
        static_cast<size_t>(g) * static_cast<size_t>(k) * static_cast<size_t>(n) *
        dtype_info.elem_size;
    const size_t b_offset_bytes =
        static_cast<size_t>(row_start) * static_cast<size_t>(k) * dtype_info.elem_size;
    const size_t c_offset_bytes =
        static_cast<size_t>(row_start) * static_cast<size_t>(n) * dtype_info.elem_size;
    const size_t a_bytes =
        static_cast<size_t>(k) * static_cast<size_t>(n) * dtype_info.elem_size;
    const size_t b_bytes =
        static_cast<size_t>(group_m) * static_cast<size_t>(k) * dtype_info.elem_size;
    const size_t c_bytes =
        static_cast<size_t>(group_m) * static_cast<size_t>(n) * dtype_info.elem_size;
    if (IsDebugEnabled()) {
      if (!CheckRange(a_offset_bytes, a_bytes, rhs_bytes)) {
        return xla::ffi::Error::InvalidArgument("rhs pointer range out of bounds");
      }
      if (!CheckRange(b_offset_bytes, b_bytes, lhs_bytes)) {
        return xla::ffi::Error::InvalidArgument("lhs pointer range out of bounds");
      }
      if (!CheckRange(c_offset_bytes, c_bytes, out_bytes)) {
        return xla::ffi::Error::InvalidArgument("out pointer range out of bounds");
      }
    }
    if (log_groups > 0 && logged < log_groups) {
      std::fprintf(stderr,
                   "group[%lld] global=%lld group_m=%d row_start=%lld "
                   "a_off=%zu b_off=%zu c_off=%zu\n",
                   static_cast<long long>(g), static_cast<long long>(global_group),
                   static_cast<int>(group_m), static_cast<long long>(row_start),
                   a_offset_bytes, b_offset_bytes, c_offset_bytes);
      ++logged;
    }
    const void* a_ptr = rhs_base + a_offset_bytes;
    const void* b_ptr = lhs_base + b_offset_bytes;
    void* c_ptr = out_base + c_offset_bytes;

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
    group_size_array.push_back(1);
    a_array.push_back(a_ptr);
    b_array.push_back(b_ptr);
    c_array.push_back(c_ptr);
  }

  const int group_count = static_cast<int>(a_array.size());
  if (group_count == 0) {
    return xla::ffi::Error::Success();
  }
  if (IsVerboseEnabled()) {
    std::fprintf(stderr, "group_count=%d\n", group_count);
    if (!a_array.empty()) {
      std::fprintf(stderr,
                   "first_group: m=%d n=%d k=%d lda=%d ldb=%d ldc=%d a=%p b=%p c=%p\n",
                   m_array[0], n_array[0], k_array[0], lda_array[0], ldb_array[0], ldc_array[0],
                   a_array[0], b_array[0], c_array[0]);
    }
    std::fflush(stderr);
  }

  xla::ffi::Error err = xla::ffi::Error::Success();
  if (!EnsureHandleInitialized(device, &err)) {
    return err;
  }
  HandleSlot* slot = GetHandleSlot(device, &err);
  std::unique_lock<std::mutex> lock(slot->mu);
  cublasHandle_t handle = slot->handle;
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "failed to set cublas stream");
  }
  status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "failed to set cublas pointer mode");
  }
  if (dtype_info.data_type == CUDA_R_16F || dtype_info.data_type == CUDA_R_16BF) {
    status = cublasSetMathMode(handle,
                               IsTensorOpDisabled() ? CUBLAS_DEFAULT_MATH
                                                    : CUBLAS_TENSOR_OP_MATH);
  } else {
    status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "failed to set cublas math mode");
  }

  if (IsDryRunEnabled()) {
    return xla::ffi::Error::Success();
  }

  if (IsUseGemmExEnabled()) {
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
    for (int i = 0; i < group_count; ++i) {
      cublasStatus_t gemm_status = cublasGemmEx(
          handle, transa_array[i], transb_array[i], m_array[i], n_array[i], k_array[i],
          alpha, a_array[i], dtype_info.data_type, lda_array[i], b_array[i],
          dtype_info.data_type, ldb_array[i], beta, c_array[i], dtype_info.data_type,
          ldc_array[i], dtype_info.compute_type, CUBLAS_GEMM_DEFAULT);
      if (gemm_status != CUBLAS_STATUS_SUCCESS) {
        return CublasError(gemm_status, "cublasGemmEx failed");
      }
    }
    if (IsSyncEnabled()) {
      cuda_status = cudaStreamSynchronize(stream);
      if (cuda_status != cudaSuccess) {
        return CudaError(cuda_status, "failed to sync stream after cublas gemm ex");
      }
    }
    return xla::ffi::Error::Success();
  }

  std::vector<float> alpha_f(group_count, 1.0f);
  std::vector<float> beta_f(group_count, 0.0f);
  std::vector<double> alpha_d(group_count, 1.0);
  std::vector<double> beta_d(group_count, 0.0);

  const void* alpha = (dtype_info.compute_type == CUBLAS_COMPUTE_64F)
                          ? static_cast<const void*>(alpha_d.data())
                          : static_cast<const void*>(alpha_f.data());
  const void* beta = (dtype_info.compute_type == CUBLAS_COMPUTE_64F)
                         ? static_cast<const void*>(beta_d.data())
                         : static_cast<const void*>(beta_f.data());

  status = cublasGemmGroupedBatchedEx(
      handle, transa_array.data(), transb_array.data(), m_array.data(),
      n_array.data(), k_array.data(), alpha, a_array.data(),
      dtype_info.data_type, lda_array.data(), b_array.data(),
      dtype_info.data_type, ldb_array.data(), beta, c_array.data(),
      dtype_info.data_type, ldc_array.data(), group_count,
      group_size_array.data(), dtype_info.compute_type);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError(status, "cublasGemmGroupedBatchedEx failed");
  }
  if (IsSyncEnabled()) {
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
      return CudaError(cuda_status, "failed to sync stream after cublas grouped gemm");
    }
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

extern "C" bool cublas_gemm_grouped_batched_ex_init() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    return false;
  }
  int original_device = 0;
  if (cudaGetDevice(&original_device) != cudaSuccess) {
    return false;
  }
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      return false;
    }
    xla::ffi::Error err = xla::ffi::Error::Success();
    if (!EnsureHandleInitialized(device, &err)) {
      (void)cudaSetDevice(original_device);
      return false;
    }
  }
  if (cudaSetDevice(original_device) != cudaSuccess) {
    return false;
  }
  return true;
}
