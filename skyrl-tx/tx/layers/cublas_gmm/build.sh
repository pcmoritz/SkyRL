#!/bin/bash
# Build the cuBLAS grouped GEMM library
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Find XLA FFI headers (from jaxlib)
XLA_INCLUDE=$(python3 -c "import jaxlib; print(jaxlib.__path__[0])")/include

nvcc -shared -o libgrouped_gemm.so grouped_gemm.cc \
    -I"$XLA_INCLUDE" \
    -lcublas \
    -std=c++17 \
    -O3 \
    --compiler-options -fPIC

echo "Built libgrouped_gemm.so"
