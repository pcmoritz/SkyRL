#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_PATH="${1:-${SCRIPT_DIR}/libcublas_grouped_gemm.so}"

JAX_INCLUDE="$(uv run --extra gpu python - <<'PY'
import os
import jaxlib
print(os.path.join(os.path.dirname(jaxlib.__file__), "include"))
PY
)"

nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
  -I"${JAX_INCLUDE}" \
  "${SCRIPT_DIR}/cublas_grouped_gemm.cu" \
  -lcublas -lcudart \
  -o "${OUT_PATH}"

echo "Built ${OUT_PATH}"
