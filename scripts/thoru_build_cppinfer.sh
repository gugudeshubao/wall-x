#!/usr/bin/env bash
set -euo pipefail

source /home/user/wy/wallx-venv/bin/activate

export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/home/user/wy/wallx-venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH="$(python3 - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

cd /home/user/wy/wall-x/cpp_infer
rm -rf build
mkdir -p build
cd build

cmake -DWALLX_CUDA_ARCH=110 -DCMAKE_CUDA_ARCHITECTURES=110 .. > /home/user/wy/thoru_cmake.log 2>&1
cmake --build . --target wallx_infer -j8 > /home/user/wy/thoru_build.log 2>&1
