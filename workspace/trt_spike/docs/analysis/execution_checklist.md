# Execution Checklist

## Keep

- `cpp_infer` as the stable baseline
- hand TRT / TRT-LLM for Flow Action
- Edge-LLM custom bridge for Flow Action
- Edge-LLM router path for VQA

## Do next

### VQA

1. Keep `scripts/vqa_inference.py` supporting:
   - `backend=wallx`
   - `backend=edge`
   - `backend=edge_router`
2. Use Edge-LLM mainly as:
   - a backend candidate
   - a router candidate
   - a serving backend for object-enumeration prompts
3. Keep `cpp_infer` as the default fallback.

### Flow Action

1. Keep `workspace/edge_llm_wallx_flow/` as the custom bridge workspace.
2. Keep the bridge reproducible:
   - ONNX export
   - `action_build`
   - custom runner
3. Keep benchmarking on Orin only.
4. Do not assume stock `llm_inference` covers Flow.

## Do not mix

- Stock VLM runtime
- Edge custom bridge
- Self-owned runtime

These are different layers.

## Current best numbers

### VQA

- `cpp_infer`: `851.2 ms`
- hand TRT / TRT-LLM: `943.018 ms`
- Edge-LLM official VLM: `~5.1s - 6.8s` depending on model / prompt preset

### Flow Action

- `cpp_infer`: `290.7 ms`
- hand TRT / TRT-LLM: `176.837 ms`
- Edge-LLM custom bridge: `157.661 ms`

