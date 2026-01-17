#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_PATH="${1:-${SCRIPT_DIR}/libcublas_grouped_gemm.so}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME is not set." >&2
  exit 1
fi

JAX_INCLUDE="$(python - <<'PY'
import os
import jaxlib
print(os.path.join(os.path.dirname(jaxlib.__file__), "include"))
PY
)"

nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
  -I"${JAX_INCLUDE}" -I"${CUDA_HOME}/include" \
  "${SCRIPT_DIR}/cublas_grouped_gemm.cu" \
  -L"${CUDA_HOME}/lib64" -lcublas -lcudart \
  -o "${OUT_PATH}"

echo "Built ${OUT_PATH}"
