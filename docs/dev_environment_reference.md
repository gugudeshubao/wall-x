# wall-x 非敏感开发环境参考

> 本文档保存可进入仓库的开发/运行参考信息。机器登录信息、密码、IP、完整 SSH 命令等敏感内容请查看本机私有上下文。

## 目录结构

### 本地 Mac（开发机）

```text
/Users/sam/project/github/wall-x/          # 项目根目录
├── docs/                                   # 文章文档
├── scripts/                                # Python benchmark/test 脚本
├── cpp_infer/                              # C++ 推理引擎
└── csrc/                                   # CUDA 自定义算子
```

### Orin

| 路径 | 用途 |
|------|------|
| `/data/wy/wall-x` | 项目根目录 |
| `/data/wy/wall-x/venv` | Python venv |
| `/data/wy/models/wall-oss-flow` | 模型权重 |
| `/data/wy/wall-x/cpp_infer/build` | C++ 编译产物 |
| `/data/wy/wall-x/profiling` | nsys/ncu profiling 数据 |

### 5090

| 路径 | 用途 |
|------|------|
| `/home/ubuntu/project/github/wall-x` | 正式项目路径 |
| `/home/ubuntu/project/github/wall-x/wallx-venv` | Python venv |
| `/home/ubuntu/project/models/wall-oss-flow` | 模型权重 |
| `/home/ubuntu/project/profiling/wall-x-vqa` | profiling 集中存储 |

## 环境激活

**Orin**（激活 venv + 设置 `LD_LIBRARY_PATH`）：

```bash
source /data/wy/wall-x/venv/bin/activate
export LD_LIBRARY_PATH=/data/wy/wall-x/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

**5090**（激活 venv + 设置 `CUDA_HOME`）：

```bash
source /home/ubuntu/project/github/wall-x/wallx-venv/bin/activate
export CUDA_HOME=/usr/local/cuda
```

**Orin 锁频**（benchmark 前必须执行）：

```bash
sudo jetson_clocks    # 锁频 1300.5 MHz
```

## Python Benchmark 脚本

所有脚本位于 `scripts/` 目录。

### VQA Benchmark

| 脚本 | 用途 | 用法 |
|------|------|------|
| `bench_fa2_vs_sdpa.py` | FA2 vs SDPA VQA 推理对比 | `python scripts/bench_fa2_vs_sdpa.py --model_path $MODEL --attn both` |
| `bench_bnb_int8.py` | bitsandbytes INT8 量化 VQA | `python scripts/bench_bnb_int8.py --model_path $MODEL --mode all` |
| `test_vqa_bench.py` | 跨平台 VQA 批量测试（真实图片） | `python scripts/test_vqa_bench.py --model_path $MODEL --image_dir $IMG_DIR` |

### Flow Action Benchmark

| 脚本 | 用途 | 用法 |
|------|------|------|
| `bench_flow_action.py` | Python Flow Action 推理基线 | `python scripts/bench_flow_action.py --model_path $MODEL` |

> 条件：`seq=488`、`action_horizon=32`、ODE 5 steps、dummy inputs。数据来源：第三篇文章 Python 基线（912ms）。

### Attention Benchmark

| 脚本 | 用途 |
|------|------|
| `bench_fa_all.py` | 5 种 Attention 实现横评（MemEff SDPA / cuDNN SDPA / FA2 / Triton FA / Math） |
| `bench_trtllm_fmha_only.py` | TRT-LLM FMHA 独立 benchmark |
| `bench_trtllm_fmha_vs_fa2.py` | TRT-LLM FMHA vs FA2 对比 |

### GEMM / Triton

| 脚本 | 用途 |
|------|------|
| `bench_gemm_opt.py` | GEMM 优化评估（baseline / torch.compile / INT8） |
| `bench_triton_vs_pytorch_orin.py` | Triton vs PyTorch 算子性能对比 |
| `test_triton_orin.py` | Triton kernel 基础验证（SM 8.7） |
| `test_triton_rotary_orin.py` | flash-attn Triton rotary embedding 验证 |

## C++ 推理引擎

### 编译（Orin）

```bash
cd /data/wy/wall-x/cpp_infer/build
cmake .. \
  -DCMAKE_PREFIX_PATH=/home/dog/.local/lib/python3.10/site-packages/torch/share/cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
make -j4
```

### `wallx_infer` CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model, -m PATH` | 模型权重目录 | （必需） |
| `--kernels, -k PATH` | Triton cubin 目录 | （可选） |
| `--mode MODE` | 推理模式：`action` / `vqa` | `action` |
| `--dataset NAME` | normalizer 数据集 | `x2_normal` |
| `--timesteps N` | ODE 步数 | `5` |
| `--max_new_tokens N` | VQA 最大生成 token 数 | `64` |
| `--warmup N` | warmup 次数 | `2` |
| `--benchmark N` | benchmark N 次迭代 | - |

### 常用命令（Orin）

```bash
# Flow Action benchmark (10 次)
./wallx_infer --model /data/wy/models/wall-oss-flow --timesteps 5 --benchmark 10

# VQA benchmark (64 tokens, 10 次)
./wallx_infer --model /data/wy/models/wall-oss-flow --mode vqa --max_new_tokens 64 --benchmark 10

# VQA benchmark (20 tokens, 部署目标)
./wallx_infer --model /data/wy/models/wall-oss-flow --mode vqa --max_new_tokens 20 --benchmark 10
```

## Profiling 工具

### `nsys`（时间线分析）

```bash
nsys profile -o $OUTPUT --force-overwrite true -t cuda,nvtx \
  python scripts/profile_vqa.py --model_path $MODEL --image $IMAGE --max_new_tokens 16

nsys stats --report cuda_gpu_kern_sum $OUTPUT.nsys-rep   # 按 kernel 汇总
```

### `ncu`（单 kernel 分析，非常慢）

```bash
sudo ncu --set full --target-processes all --launch-skip 0 --launch-count 50 \
  -o $OUTPUT -f python scripts/profile_vqa.py --model_path $MODEL --image $IMAGE
```

### Profiling 数据统一存储路径（5090）

- `.nsys-rep` -> `/home/ubuntu/project/profiling/wall-x-vqa/nsys/`
- `.ncu-rep` -> `/home/ubuntu/project/profiling/wall-x-vqa/ncu/`

## 代码同步

实际目标机器地址、用户名和认证方式请查看私有上下文；这里仅保留命令模板。

```bash
# Orin: 同步 C++ 源码 + 重新编译
rsync -avz cpp_infer/src/ \
  <orin-user>@<orin-host>:/data/wy/wall-x/cpp_infer/src/

# Orin: 同步 Python 脚本
rsync -avz scripts/ \
  <orin-user>@<orin-host>:/data/wy/wall-x/scripts/

# 5090: 同步整个项目
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude 'build' --exclude 'dist' \
  /Users/sam/project/github/wall-x/ \
  <remote-user>@<remote-host>:/home/ubuntu/project/wall-x/
```

## GPU 监控

**Orin**（统一内存，`nvidia-smi` 不显示显存）：

```bash
tegrastats
python3 -c "import torch; print(torch.cuda.max_memory_allocated()/1e9, 'GB')"
```

**5090**：

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 性能基线数据（文章引用）

### Flow Action（Orin, bf16, batch=1, 488 tokens, 5 步 ODE）

| 配置 | 延迟 | 吞吐 | 加速比 |
|------|------|------|--------|
| Python SDPA | 912.4 ms | 1.10 infer/s | - |
| **C++ libtorch** | **553.9 ms** | **1.81 infer/s** | **1.65x** |
| 每步 ODE | 86.5 -> 26.2 ms | - | 3.3x |

### VQA（Orin, bf16, batch=1, 456 tokens, 64 tokens 生成）

| 配置 | 延迟 | 吞吐 | 加速比 |
|------|------|------|--------|
| Python SDPA | 8681 ms | 7.4 tok/s | - |
| Python FA2 | 6360 ms | 10.1 tok/s | - |
| **C++ libtorch** | **3469 ms** | **18.45 tok/s** | **1.83x vs FA2** |
| 每步 decode | 97.2 -> 49.0 ms | - | 2.0x |

### 部署目标

| 任务 | 目标频率 | 目标延迟 | 约束 |
|------|---------|---------|------|
| **Flow Action** | **2-3 Hz** | **333-500 ms** | - |
| **VQA** | **~1 Hz** | **< 1.2s** | `max_new_tokens <= 20` |

## 注意事项

- 5090 IP 可能变化；连不上时先查看私有上下文中的最新地址。
- 5090 无法直接访问外网（`huggingface.co` 不通），模型需通过 `hf-mirror` 或本地传输。
- Orin 上 `fake_inference.py` 只测部分 forward，不是真实 Flow Action 延迟。
- Orin benchmark 前必须执行 `sudo jetson_clocks` 锁频。
- Orin venv 激活后需设置 `LD_LIBRARY_PATH` 才能加载 torch 动态库。
- `max_new_tokens` 是脚本参数，不是模型限制。
- 文章 3/4 只编辑知乎主文档，不生成小红书和微信 HTML。
