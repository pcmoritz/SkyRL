# cuBLAS Grouped GEMM

Optimized ragged dot implementation using cuBLAS `cublasGemmGroupedBatchedEx` for GPU.

## Requirements

- CUDA 12.0+
- JAX with CUDA support

## Build

```bash
cd tx/layers/cublas_gmm
bash build.sh
```

## Usage

```python
from tx.layers.util import ragged_dot

# Use cuBLAS grouped GEMM (bf16 only)
result = ragged_dot(lhs, rhs, group_sizes, group_offset=offset, use_cublas=True)
```

Or directly:

```python
from tx.layers.cublas_gmm import grouped_gemm_bf16

result = grouped_gemm_bf16(lhs, rhs, group_sizes, group_offset=0)
```

## How it works

1. JAX computes all metadata (group offsets, sizes, leading dimensions)
2. C++ receives pre-computed arrays, fills pointer arrays, calls cuBLAS
3. No allocations in C++ - all memory managed by JAX
