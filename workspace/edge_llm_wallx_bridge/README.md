# Edge-LLM + wall-x Bridge

这是一条单独的实验线，用来验证：

> **`TensorRT-Edge-LLM + plugin` 能不能直接承接 `wall-x` 的 VQA / VLM 主干。**

当前先做两件事：

1. `wall-x VQA` 的最小样例，直接走 `TensorRT-Edge-LLM` 的 `llm_inference`
2. 记录端到端 wall-clock，并和 `wall-x` / `cpp_infer` 的结果做对照

## 目录约定

- `bench_vqa_edge_llm.py`
  - 运行最小 VQA benchmark
- `bench_vqa_suite.py`
  - 按模型清单批量跑 benchmark，适合对比 `Qwen2.5-VL` / `Qwen3-VL`
- `compare_outputs.py`
  - 对比 Edge-LLM 输出和 wall-x baseline 输出
- `extract_wallx_reference.py`
  - 从已有 wall-x benchmark JSON 提取单题 reference，转成可直接对比的最小格式
- `models.edge_llm.json`
  - 当前可跑模型的路径清单，方便在 Orin / Thor-U 上切换
- `models.wallx_prompt.json`
  - 固定为 wall-x 常用问法 `Describe what you see in this image.` 的模型清单
- `RESULTS.md`
  - 当前已拿到的性能和输出对照结论

## 当前支持的输入

- 单图 VQA
- `Qwen2.5-VL-3B-Instruct`
- `Qwen3-VL-2B-Instruct`

## 推荐跑法

Orin / Thor-U 上如果已经准备好对应 engine，可以直接跑：

```bash
python3 bench_vqa_suite.py \
  --models-file models.wallx_prompt.json \
  --work-root /data/wy/wall-x/workspace/edge_llm_exp/bridge_runs_wallx_prompt \
  --runs 1 \
  --warmup 1
```

## 运行原则

- 只用已经在 Orin 上编译通过的 `TensorRT-Edge-LLM`
- 不修改 `cpp_infer/`
- 不污染 `trt_spike/`
- 所有中间产物放在本目录下

## 当前已经确认的事实

- `Qwen2.5-VL-3B-Instruct`
  - bridge benchmark：`mean ≈ 8194 ms`
- `Qwen3-VL-2B-Instruct`
  - bridge benchmark：`mean ≈ 5407 ms`
- 在 `fruits_on_table.png + "Describe what you see in this image."` 这个 wall-x 常用 case 上：
  - `Qwen2.5-VL-3B` 能输出和图相关的描述，但和 wall-x baseline 的文本相似度只有 `0.3083`
  - `Qwen3-VL-2B` 这条官方路线在该 case 上直接偏题，文本相似度只有 `0.3169`
