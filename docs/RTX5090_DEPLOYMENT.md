# Wall-X RTX 5090 部署指南

## 硬件环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, SM 12.0) |
| VRAM | 32 GB GDDR7 |
| CPU | AMD Ryzen 9 9950X |
| RAM | 64 GB DDR5 |
| Storage | Samsung 9100 PRO 2TB NVMe |
| OS | Ubuntu 24.04 LTS |
| CUDA (system) | 13.1 (nvcc) |

## 软件环境

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.3 | 系统自带 |
| PyTorch | 2.11.0+cu130 | 通过 pip 安装，自动携带 CUDA 13.0 |
| Transformers | 4.57.6 | 需要 >=4.50 (AttentionInterface)，<5.0 (避免 API 破坏) |
| PEFT | 0.19.1 | 配合 transformers 4.57 |
| flash-attn | 2.8.3 | 源码编译，需绕过 CUDA 版本检查 |
| TorchVision | 0.26.0+cu130 | 必须与 PyTorch CUDA 版本一致 |
| diffusers | 0.37.1 | ActionHead 依赖 |
| wallx_csrc | 源码编译 | CUDA 扩展，SM 12.0 |

## 目录结构（远程机器）

```
/home/ubuntu/project/
├── wall-x/                          # 源码 (pip install -e .)
│   ├── wall_x/
│   ├── csrc/
│   ├── 3rdparty/cutlass/            # git clone --depth 1
│   ├── build_patch.py               # CUDA 版本检查绕过脚本
│   └── scripts/fake_inference.py
├── models/
│   └── wall-oss-flow/               # HuggingFace 模型权重 (~8GB)
│       ├── model.safetensors
│       ├── config.json              # 需要手动添加 pad_token_id
│       ├── tokenizer.json
│       └── ...
└── github/wall-x/wallx-venv/       # Python venv
```

## 部署步骤

### 1. 创建 Python 虚拟环境

```bash
python3 -m venv /home/ubuntu/project/github/wall-x/wallx-venv
source /home/ubuntu/project/github/wall-x/wallx-venv/bin/activate
```

### 2. 安装 PyTorch (cu128/cu130)

RTX 5090 需要 PyTorch >= 2.7。cu131 索引不存在，使用 cu128：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **注意**: flash-attn 安装时会自动将 PyTorch 升级到 cu130 版本（通过 nvidia 独立包分发），这是正常的。

### 3. 安装 Python 依赖

```bash
pip install transformers==4.57.6 accelerate peft scipy torchdiffeq qwen_vl_utils diffusers wheel
```

> ⚠️ **关键**: transformers 必须 >=4.50 且 <5.0。5.x 版本有大量 API 破坏性变更。

### 4. 编译 CUDA 扩展 (wallx_csrc)

系统 CUDA 13.1 与 PyTorch CUDA 12.8/13.0 不匹配，需要绕过版本检查：

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0"

# 方法1: 直接修补 torch 源码
python -c "
import torch.utils.cpp_extension as ext
import inspect
fpath = inspect.getfile(ext)
with open(fpath) as f:
    content = f.read()
old = 'def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:'
new = '''def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    import os
    if os.environ.get('SKIP_CUDA_MISMATCH_CHECK', ''):
        return'''
content = content.replace(old, new)
with open(fpath, 'w') as f:
    f.write(content)
"

# 编译 wallx_csrc
export SKIP_CUDA_MISMATCH_CHECK=1
pip install -e . --no-build-isolation
```

### 5. 编译 flash-attn

```bash
export MAX_JOBS=2          # 防止 OOM
export NVCC_THREADS=1
export SKIP_CUDA_MISMATCH_CHECK=1
export TORCH_CUDA_ARCH_LIST="12.0"
pip install flash-attn --no-build-isolation
```

> 编译约需 40-60 分钟。建议使用 `nohup` 后台运行。

### 6. 下载模型权重

机器无法直接访问 huggingface.co，使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='x-square-robot/wall-oss-flow', local_dir='/home/ubuntu/project/models/wall-oss-flow')
"
```

下载后需要给 config.json 添加 `pad_token_id`（transformers 4.57 要求）：

```bash
python -c "
import json
p = '/home/ubuntu/project/models/wall-oss-flow/config.json'
c = json.load(open(p))
c['pad_token_id'] = c.get('eos_token_id', 151645)
json.dump(c, open(p, 'w'), indent=2)
"
```

### 7. 运行推理测试

```bash
cd /home/ubuntu/project/wall-x
python scripts/fake_inference.py
```

预期输出：
```
✅ Fake inference test successful!
Output logits shape: torch.Size([1, 50, 153715])
✅ Output shape correct
✅ Output contains no NaN values
✅ Output contains no infinity values
```

## 已知问题与解决方案

### 1. SSH 连接不稳定 (Bad file descriptor)

**原因**: 本地 Mac 配置了 PAC 代理（alilang），间歇性拦截 TCP 连接。

**解决**: 
- 使用 SSH 密钥免密登录（替代 sshpass）
- 远程 sshd 添加 `ClientAliveInterval 30`、`TCPKeepAlive yes`
- 建议将目标 IP 加入代理排除列表

### 2. CUDA 版本不匹配

**原因**: 系统 nvcc 13.1，PyTorch 编译使用 CUDA 12.8/13.0。

**解决**: 修补 `torch.utils.cpp_extension._check_cuda_version`，设置 `SKIP_CUDA_MISMATCH_CHECK=1`。

### 3. flash-attn 编译 OOM

**原因**: 默认并行度过高，73 个编译单元 × 4 架构 × 多线程。

**解决**: `MAX_JOBS=2`、`NVCC_THREADS=1`。

### 4. HuggingFace 不可达

**原因**: 网络限制，huggingface.co IP 被拦截。

**解决**: 使用镜像 `HF_ENDPOINT=https://hf-mirror.com`。

### 5. transformers 版本兼容性

| 版本范围 | 问题 |
|----------|------|
| < 4.50 | 缺少 `AttentionInterface` |
| >= 5.0 | `ROPE_INIT_FUNCTIONS` 移除 `'default'`、`_tied_weights_keys` 格式变更、`is_flash_attn_greater_or_equal_2_10` 删除 |
| **4.50 ~ 4.57** | ✅ 推荐范围 |

## 推理性能参考

| 指标 | 值 |
|------|-----|
| 模型加载时间 | ~30s |
| Forward pass (seq_len=50) | ~120ms |
| GPU VRAM 峰值 | ~22 GB / 32 GB |
| Logits 范围 | [-17.6, 18.3] |
