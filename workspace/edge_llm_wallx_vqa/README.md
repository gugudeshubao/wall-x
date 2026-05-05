# Edge-LLM wall-x VQA Adapter

这条实验线专门回答一个更具体的问题：

> **`TensorRT-Edge-LLM` 能不能承接 `wall-x` 现有的 VQA case 集，而不只是跑通一个官方 Qwen-VL 样例。**

这里不碰 `Flow Action`，只看：

- `wall-x` 现成 VQA 图片
- `wall-x` 现成问题
- `wall-x` 现成 baseline 答案
- `TensorRT-Edge-LLM` 在相同 case 上的
  - 端到端时延
  - 输出文本
  - 与 baseline 的文本相似度

## 文件

- `extract_wallx_vqa_cases.py`
  - 从 `data/int8_vqa_results_orin.json` 提取批量 case
- `models.edge_llm.json`
  - 可用的 Edge-LLM engine 清单
- `run_edge_llm_wallx_vqa.py`
  - 批量跑 wall-x VQA case
  - 支持 `--compat-mode wallx_vqa`
- `compare_prompt_templates.py`
  - 静态比较 wall-x / Edge-LLM 的 tokenizer 和 chat template
- `edge_llm_vqa_backend.py`
  - 可直接调用的 Edge-LLM VQA backend 封装
- `smoke_test_backend.py`
  - 最小可运行 smoke test
- `vqa_wrapper_edge_llm.py`
  - 同接口 wrapper，尽量贴 `VQAWrapper.generate()`
- `run_wrapper_edge_llm.py`
  - wrapper 版 CLI
- `run_vqa_backend_switch.py`
  - 同一个 CLI 里切换 wall-x / Edge-LLM backend
- `compare_vqa_backends.py`
  - 同一张图同一问题下，直接对比 wall-x 和 Edge-LLM backend
- `wall_x.serving.VQAClient`
  - 主包内 websocket 客户端，直接连 VQA server
- `compare_vqa_backends.py`
  - 单个 case 上直接对比 wall-x baseline 和 Edge-LLM wrapper
- `backend_config.orin.qwen3.json`
  - Orin 当前可直接使用的 backend 配置
- `backend_config.orin.qwen25.json`
  - Orin 上 `Qwen2.5-VL-3B` 的 backend 配置
- `backend_config.thoru.qwen3.template.json`
  - Thor-U 配置模板，等 engine 到位后可直接改路径使用
- `RESULTS.md`
  - 当前已确认的结论

## 状态

- Orin：已跑出 16-case batch 结果
- Thor-U：已落下同一套适配代码和 case manifest，但还没有可直接复用的 Edge-LLM engine，因此当前还没出 Thor-U 的实跑数字

## 输入组织结论

- 默认 system prompt 会明显影响输出风格
- 去掉 system prompt、并把 `max_generate_length` 对齐到 wall-x reference token 数后，语义相似度会回升
- 但这层调整仍然不足以把 Edge-LLM 自动收敛成 wall-x baseline
- wall-x 有 `propri / action` 这套额外 token，Edge-LLM 的 VLM engine 里没有，这对 Flow 是硬边界

## 推荐命令

如果要跑当前最接近 wall-x VQA 的输入组织，直接用：

```bash
python3 run_edge_llm_wallx_vqa.py \
  --models-file models.edge_llm.json \
  --cases-file wallx_vqa_cases.bf16.json \
  --work-root /data/wy/wall-x/workspace/edge_llm_wallx_vqa/runs_best \
  --compat-mode wallx_vqa
```

最小 backend smoke test：

```bash
python3 smoke_test_backend.py \
  --build-dir /data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin \
  --engine-dir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen3-vl-2b \
  --visual-engine-dir /data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen3-vl-2b/visual \
  --plugin-path /data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so \
  --work-root /data/wy/wall-x/workspace/edge_llm_wallx_vqa/runtime_tmp \
  --image /data/wy/wall-x/test_images/fruits_on_table.png \
  --question "Describe what you see in this image." \
  --output-json /data/wy/wall-x/workspace/edge_llm_wallx_vqa/smoke.json
```

最小 backend 调用方式：

```python
from PIL import Image
from edge_llm_vqa_backend import EdgeLLMVQABackend, EdgeLLMVQABackendConfig

backend = EdgeLLMVQABackend(EdgeLLMVQABackendConfig(
    build_dir="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin",
    engine_dir="/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen3-vl-2b",
    multimodal_engine_dir="/data/wy/wall-x/workspace/edge_llm_exp/edge_engines/qwen3-vl-2b/visual",
    plugin_path="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so",
    work_root="/data/wy/wall-x/workspace/edge_llm_wallx_vqa/runtime_tmp",
    compat_mode="wallx_vqa",
))

answer = backend.generate(Image.open("/data/wy/wall-x/test_images/fruits_on_table.png"), "Describe what you see in this image.")
print(answer)
```

wrapper 版调用方式：

```bash
python3 run_wrapper_edge_llm.py \
  --backend-config backend_config.orin.qwen3.json \
  --image /data/wy/wall-x/test_images/fruits_on_table.png \
  --question "Describe what you see in this image."
```

backend switch 版调用方式：

```bash
python3 run_vqa_backend_switch.py \
  --backend edge \
  --edge-backend-config backend_config.orin.qwen3.json \
  --image /data/wy/wall-x/test_images/fruits_on_table.png \
  --question "Describe what you see in this image."
```

双后端对比方式：

```bash
python3 ../scripts/vqa_backend_compare.py \
  --edge-backend-config backend_config.orin.qwen3.json \
  --wallx-model-path /data/wy/models/wall-oss-flow \
  --image /data/wy/wall-x/test_images/fruits_on_table.png \
  --question "Describe what you see in this image."
```

推荐后续都从 `run_vqa_backend_switch.py` 这个入口继续扩，因为它最接近真正接 wall-x 外层调用链时需要的形态。

主工程 `scripts/` 入口也已经有了对应版本：

```bash
python3 scripts/vqa_backend_switch.py \
  --backend edge \
  --edge-backend-config /data/wy/wall-x/workspace/edge_llm_wallx_vqa/backend_config.orin.qwen3.json \
  --image /data/wy/wall-x/test_images/fruits_on_table.png \
  --question "Describe what you see in this image."
```

注意：`backend=wallx` 需要用 `/data/wy/wall-x/venv/bin/python` 跑主工程入口，因为 `wallx_csrc` 在这个 venv 里。

现在这个统一入口已经能直接做 6-case 的 wall-x vs Edge 双后端对照，不再只是单图 smoke。

最小 websocket server 启动方式：

```bash
/data/wy/wall-x/venv/bin/python -m wall_x.serving.launch_vqa_serving \
  --backend edge \
  --edge-backend-config /data/wy/wall-x/workspace/edge_llm_wallx_vqa/backend_config.orin.qwen3.json \
  --host 127.0.0.1 \
  --port 8792
```

主包内 websocket client：

```python
from wall_x.serving import VQAClient

client = VQAClient("ws://127.0.0.1:8792")
meta = client.connect_sync()
result = client.infer_sync("/data/wy/wall-x/test_images/fruits_on_table.png", "What objects are on the table?")
print(meta)
print(result)
client.close_sync()
```

单 case 对比方式：

```bash
python3 compare_vqa_backends.py \
  --backend-config backend_config.orin.qwen3.json \
  --cases-json wallx_vqa_cases.bf16.json \
  --image fruits_on_table.png \
  --question "Describe what you see in this image."
```

## 当前状态

- Orin 上已经完成：
  - batch benchmark
  - input 组织消融
  - backend smoke
  - wrapper smoke
- Thor-U 上已同步同一套代码和 case manifest
- Thor-U 还没有可直接复用的 Edge-LLM engine，因此还不能出同口径实跑数
