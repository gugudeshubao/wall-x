#!/usr/bin/env python3
"""
Run a two-stage Flow TRT-LLM path with fused postfix engine.

Current structure:
  - host-side action step only for t0
  - TRT prefetch decoder engine
  - TRT fused postfix engine:
      action_step + postfix decoder + action_proj_back
  - host-side Euler loop only
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

import build_action_step_trtllm as action_step_mod
import build_flow_postfix_fused_trtllm as postfix_fused_mod
import build_flow_prefetch_trtllm as prefetch_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_session(engine_bytes):
    import tensorrt_llm

    return tensorrt_llm.runtime.Session.from_serialized_engine(engine_bytes)


def maybe_load_engine(path: Path):
    if path.exists():
        return path.read_bytes()
    return None


def load_engine_source(path_or_bytes):
    if isinstance(path_or_bytes, Path):
        return path_or_bytes.read_bytes()
    return path_or_bytes


def trim_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_flow_cfg(model_path: str):
    raw = json.loads((Path(model_path) / "config.json").read_text())
    return SimpleNamespace(
        action_hidden_size=raw.get("action_hidden_size", 2048),
        use_x_pred=raw.get("use_x_pred", False),
    )


def run_prefetch_session(session, refs, num_layers: int, precision="bfloat16", hidden_in=None):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]
    inputs = {
        "hidden_in": (refs["inputs_embeds_t0"] if hidden_in is None else hidden_in.cpu()).to("cuda").to(torch_dtype),
        "prefetch_cos_mrope": refs["prefetch_cos_mrope"].to("cuda").to(torch_dtype),
        "prefetch_sin_mrope": refs["prefetch_sin_mrope"].to("cuda").to(torch_dtype),
        "prefetch_causal_mask_4d": refs["prefetch_causal_mask_4d"].to("cuda").to(torch_dtype),
    }
    outputs = {
        "action_pred": torch.zeros_like(refs["action_pred_t0"], device="cuda", dtype=torch_dtype),
    }
    kv_heads = 2
    head_dim = 128
    seq_len = inputs["hidden_in"].shape[1]
    for i in range(num_layers):
        outputs[f"present_past_key_{i}"] = torch.zeros((1, kv_heads, seq_len, head_dim), dtype=torch_dtype, device="cuda")
        outputs[f"present_past_value_{i}"] = torch.zeros((1, kv_heads, seq_len, head_dim), dtype=torch_dtype, device="cuda")
    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    torch.cuda.synchronize()
    return outputs, ok, (time.perf_counter() - t0) * 1000


def run_postfix_fused_session(session, refs, noisy_action, dof_mask, time_embed, num_layers: int, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]
    inputs = {
        "noisy_action": noisy_action.to("cuda").to(torch_dtype),
        "dof_mask": dof_mask.to("cuda").to(torch_dtype),
        "time_embed": time_embed.to("cuda").to(torch_dtype),
        "postfix_cos_mrope": refs["postfix_cos_mrope"].to("cuda").to(torch_dtype),
        "postfix_sin_mrope": refs["postfix_sin_mrope"].to("cuda").to(torch_dtype),
        "postfix_attention_mask_additive_4d": refs["postfix_attention_mask_additive_4d"].to("cuda").to(torch_dtype),
    }
    for i in range(num_layers):
        inputs[f"prefix_past_key_{i}"] = refs[f"prefix_past_key_{i}"].to("cuda").to(torch_dtype)
        inputs[f"prefix_past_value_{i}"] = refs[f"prefix_past_value_{i}"].to("cuda").to(torch_dtype)
    outputs = {
        "action_pred": torch.zeros_like(refs["action_pred_t1"], device="cuda", dtype=torch_dtype),
    }
    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    torch.cuda.synchronize()
    return outputs, ok, (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser(description="Run two-stage Flow TRT-LLM path with fused postfix engine")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save-engines-dir", default="")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    refs = load_file(args.ref, device="cpu")
    cfg = load_flow_cfg(args.model_path)
    action_cfg, action_weights = action_step_mod.load_action_step_weights(args.model_path, args.checkpoint)

    save_dir = Path(args.save_engines_dir) if args.save_engines_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    prefetch_engine = None
    postfix_engine = None
    if save_dir:
        prefetch_engine = maybe_load_engine(save_dir / f"flow_prefetch_{args.num_layers}.engine")
        postfix_engine = maybe_load_engine(save_dir / f"flow_postfix_fused_{args.num_layers}.engine")

    if prefetch_engine is None:
        if args.run_only:
            raise FileNotFoundError("flow_prefetch engine not found")
        print("[BUILD] flow prefetch engine ...")
        prefetch_engine = prefetch_mod.build_engine(
            prefetch_mod.CheckpointReader(args.checkpoint),
            refs,
            num_layers=args.num_layers,
            precision=args.precision,
            output_kv=True,
            output_hidden=False,
            output_action_pred=True,
        )
        if save_dir:
            (save_dir / f"flow_prefetch_{args.num_layers}.engine").write_bytes(prefetch_engine)
            prefetch_engine = save_dir / f"flow_prefetch_{args.num_layers}.engine"
    else:
        print("[LOAD] flow prefetch engine from cache")
        if save_dir:
            prefetch_engine = save_dir / f"flow_prefetch_{args.num_layers}.engine"

    if postfix_engine is None:
        if args.run_only:
            raise FileNotFoundError("flow_postfix_fused engine not found")
        print("[BUILD] flow postfix fused engine ...")
        postfix_engine = postfix_fused_mod.build_engine(
            postfix_fused_mod.CheckpointReader(args.checkpoint),
            refs,
            precision=args.precision,
            num_layers=args.num_layers,
            output_hidden=False,
            output_action_pred=True,
        )
        if save_dir:
            (save_dir / f"flow_postfix_fused_{args.num_layers}.engine").write_bytes(postfix_engine)
            postfix_engine = save_dir / f"flow_postfix_fused_{args.num_layers}.engine"
    else:
        print("[LOAD] flow postfix fused engine from cache")
        if save_dir:
            postfix_engine = save_dir / f"flow_postfix_fused_{args.num_layers}.engine"

    timestep0 = refs["times"][0].reshape(1).float()
    time_embed0 = action_step_mod.sinusoidal_time_embed(timestep0, action_cfg.action_hidden_size).cpu()
    action_embed0 = action_step_mod.torch_reference(
        action_weights, refs["noise"].float(), refs["dof_mask"].float(), time_embed0
    ).to("cuda").to(refs["inputs_embeds_after_scatter"].dtype)
    hidden_in_prefetch = refs["inputs_embeds_after_scatter"].to("cuda").clone()
    action_mask = refs["flow_action_mask"].to("cuda").bool()
    hidden_in_prefetch[action_mask] = action_embed0.reshape(-1, hidden_in_prefetch.shape[-1])

    prefetch_session = make_session(load_engine_source(prefetch_engine))
    prefetch_outputs, ok, prefetch_ms = run_prefetch_session(
        prefetch_session, refs, args.num_layers, precision=args.precision, hidden_in=hidden_in_prefetch
    )
    if not ok:
        raise RuntimeError("prefetch session.run failed")
    del prefetch_session
    trim_memory()

    noise = refs["noise"].to("cuda").float()
    dof_mask = refs["dof_mask"].to("cuda").float()
    times = refs["times"].to("cuda").float()
    dt = times[1] - times[0]
    action_pred = prefetch_outputs["action_pred"].float()
    v_t = action_pred - noise if cfg.use_x_pred else action_pred
    noisy_action = refs["noisy_action_after_prefetch"].to("cuda").float()

    prefix_len = int(refs["prefix_length"][0].item())
    for i in range(args.num_layers):
        refs[f"prefix_past_key_{i}"] = prefetch_outputs[f"present_past_key_{i}"][:, :, :prefix_len, :].detach().cpu()
        refs[f"prefix_past_value_{i}"] = prefetch_outputs[f"present_past_value_{i}"][:, :, :prefix_len, :].detach().cpu()
    del prefetch_outputs
    trim_memory()

    postfix_session = make_session(load_engine_source(postfix_engine))
    postfix_times = []
    for step_idx in range(1, len(times) - 1):
        timestep = times[step_idx].reshape(1).float()
        time_embed = action_step_mod.sinusoidal_time_embed(timestep, action_cfg.action_hidden_size).cpu()
        outputs, ok, ms = run_postfix_fused_session(
            postfix_session, refs, noisy_action, dof_mask, time_embed, args.num_layers, precision=args.precision
        )
        if not ok:
            raise RuntimeError(f"postfix fused step {step_idx} failed")
        postfix_times.append(ms)
        action_pred = outputs["action_pred"].float()
        v_t = action_pred - noise if cfg.use_x_pred else action_pred
        noisy_action = noisy_action + dt * v_t

    del postfix_session
    trim_memory()

    ref_action = refs["predict_action"].float()
    pred_action = noisy_action.detach().cpu().float()
    diff = (pred_action - ref_action).abs()
    cos = torch.nn.functional.cosine_similarity(pred_action.flatten(), ref_action.flatten(), dim=0).item()

    print(f"[RESULT] prefetch_ms={prefetch_ms:.3f}")
    print(f"[RESULT] postfix_times_ms={postfix_times}")
    print(f"[RESULT] postfix_mean_ms={sum(postfix_times)/len(postfix_times):.3f}")
    print(f"[RESULT] total_ms={prefetch_ms + sum(postfix_times):.3f}")
    print(f"[RESULT] predict_action cosine={cos:.8f}")
    print(f"[RESULT] predict_action mean_abs={diff.mean().item():.8e}")
    print(f"[RESULT] predict_action max_abs={diff.max().item():.8e}")

    if save_dir:
        payload = {
            "prefetch_ms": prefetch_ms,
            "postfix_times_ms": postfix_times,
            "total_ms": prefetch_ms + sum(postfix_times),
            "predict_action_cosine": cos,
            "predict_action_mean_abs": diff.mean().item(),
            "predict_action_max_abs": diff.max().item(),
        }
        with open(save_dir / f"flow_two_stage_fused_{args.num_layers}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SAVE] {save_dir / f'flow_two_stage_fused_{args.num_layers}.json'}")


if __name__ == "__main__":
    main()
