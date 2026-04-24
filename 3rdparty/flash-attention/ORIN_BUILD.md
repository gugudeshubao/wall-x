# Flash Attention 2.8.3 (Patched for Orin)

This is Flash Attention **v2.8.3** with two patches for Jetson AGX Orin (SM 8.7):

## Patches applied

1. **`setup.py`**: Added SM 8.7 gencode (`-gencode arch=compute_87,code=sm_87`).
   Default `FLASH_ATTN_CUDA_ARCHS` now includes `87`.

2. **`flash_attn/ops/triton/rotary.py`**: Added pure PyTorch fallback for `apply_rotary`
   since triton does not support aarch64/Jetson. Auto-detects: uses triton on x86_64,
   falls back to PyTorch on aarch64.

## Build on Orin

```bash
cd /path/to/wall-x/3rdparty/flash-attention

# 1. Populate CUTLASS headers (required)
#    Option A: clone into csrc/cutlass
git clone --depth 1 https://github.com/NVIDIA/cutlass.git csrc/cutlass
#    Option B: symlink from wall-x's 3rdparty
# ln -s ../../cutlass csrc/cutlass

# 2. Build (Orin SM 8.7, ~50 min with MAX_JOBS=6)
source /data/wy/wall-x/venv/bin/activate
FLASH_ATTN_CUDA_ARCHS='87' \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=6 NVCC_THREADS=2 \
python setup.py build_ext --inplace

# 3. Install Python package (skip CUDA rebuild)
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install .

# 4. Copy the .so to venv site-packages
cp flash_attn_2_cuda.cpython-310-aarch64-linux-gnu.so \
   /data/wy/wall-x/venv/lib/python3.10/site-packages/
```

## Benchmark (Orin, wall-x VQA, 64 tokens)

| Backend | Mean Latency | Throughput | Peak GPU |
|---------|-------------|------------|----------|
| SDPA    | 9503 ms     | 6.7 tok/s  | 8.19 GB  |
| FA2     | 16289 ms    | 3.9 tok/s  | 15.97 GB |

> Note: SDPA outperforms FA2 on Orin for short VQA sequences (~420 input tokens).
> FA2 advantage is expected with longer sequences (>2K tokens).
