#!/usr/bin/env python3
"""
Run the exported wall-x Flow Action engine and compare with the wall-x reference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def sinusoidal_time_embed(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = torch.exp(
        torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
        * (-math.log(10000.0) / (half_dim - 1))
    )
    emb = timestep[:, None].float() * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Run wall-x action.engine and compare against reference")
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--model-path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    refs = load_file(args.ref, device="cpu")
    engine_path = Path(args.engine_dir) / "action" / "action.engine"
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    engine = engine_path.read_bytes()
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

    noise = refs["noise"].clone().float().to("cuda")
    dof_mask = refs["dof_mask"].clone().float().to("cuda")
    times = refs["times"].clone().float().to("cuda")
    prefix_len = int(refs["prefix_length"][0].item())
    num_layers = len([k for k in refs.keys() if k.startswith("prefix_past_key_")])
    cfg = json.loads((Path(args.model_path) / "config.json").read_text())
    use_x_pred = bool(cfg.get("use_x_pred", False))

    inputs = {
        "noise_trajectory": noise,
        "time_steps_t0": times[0].reshape(1),
        "time_steps_t1": times[1].reshape(1),
        "seq_length": torch.tensor([prefix_len], device="cuda", dtype=torch.int64),
        "postfix_cos_mrope": refs["postfix_cos_mrope"].to("cuda").float(),
        "postfix_sin_mrope": refs["postfix_sin_mrope"].to("cuda").float(),
        "postfix_attention_mask_additive_4d": refs["postfix_attention_mask_additive_4d"].to("cuda").float(),
    }
    for i in range(num_layers):
        inputs[f"prefix_past_key_{i}"] = refs[f"prefix_past_key_{i}"].to("cuda").float()
        inputs[f"prefix_past_value_{i}"] = refs[f"prefix_past_value_{i}"].to("cuda").float()

    outputs = {
        "denoised_trajectory": torch.zeros_like(refs["noisy_action_after_prefetch"], device="cuda"),
        "action_pred": torch.zeros_like(refs["action_pred_t1"], device="cuda"),
    }
    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()

    def run_once_full():
        inputs["noise_trajectory"] = noise
        inputs["time_steps_t0"] = times[0].reshape(1)
        inputs["time_steps_t1"] = times[1].reshape(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        if not ok:
            raise RuntimeError("action engine run failed")
        first_ms = (time.perf_counter() - t0) * 1000
        first_pred = outputs["denoised_trajectory"].detach().cpu().float()
        first_action = outputs["action_pred"].detach().float()

        noisy_action = refs["noisy_action_after_prefetch"].clone().to("cuda").float()
        full_step_times = []
        total_t0 = time.perf_counter()
        for step_idx in range(1, len(times) - 1):
            inputs["noise_trajectory"] = noisy_action
            inputs["time_steps_t0"] = times[step_idx].reshape(1)
            inputs["time_steps_t1"] = times[step_idx + 1].reshape(1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
            torch.cuda.synchronize()
            if not ok:
                raise RuntimeError(f"action engine step {step_idx} failed")
            full_step_times.append((time.perf_counter() - t1) * 1000)
            action_pred = outputs["action_pred"].detach().float()
            v_t = action_pred - noise if use_x_pred else action_pred
            dt = (times[step_idx + 1] - times[step_idx]).view(1, 1, 1)
            noisy_action = noisy_action + dt * v_t
        total_ms = first_ms + (time.perf_counter() - total_t0) * 1000
        return first_ms, first_pred, first_action, noisy_action.cpu().float(), np.array(full_step_times), total_ms

    # Accuracy/reference run
    first_ms, pred, _first_action, pred_final, arr, total_ms = run_once_full()

    ref = refs["noisy_action_after_prefetch"].float()
    diff = (pred - ref).abs()
    cos = torch.nn.functional.cosine_similarity(pred.flatten(), ref.flatten(), dim=0).item()

    print(f"[RESULT] action_step_ms={first_ms:.3f}")
    print(f"[RESULT] denoised_vs_ref cosine={cos:.8f}")
    print(f"[RESULT] denoised_vs_ref mean_abs={diff.mean().item():.8e}")
    print(f"[RESULT] denoised_vs_ref max_abs={diff.max().item():.8e}")
    print(f"[RESULT] action_pred_shape={tuple(outputs['action_pred'].shape)}")

    ref_final = refs["predict_action"].float()
    final_diff = (pred_final - ref_final).abs()
    final_cos = torch.nn.functional.cosine_similarity(
        pred_final.flatten(), ref_final.flatten(), dim=0
    ).item()
    print(f"[RESULT] flow_final_cosine={final_cos:.8f}")
    print(f"[RESULT] flow_final_mean_abs={final_diff.mean().item():.8e}")
    print(f"[RESULT] flow_final_max_abs={final_diff.max().item():.8e}")
    print(f"[RESULT] flow_total_ms={total_ms:.3f}")
    if arr.size:
        print(f"[RESULT] flow_step_ms_mean={arr.mean():.3f}")
        print(f"[RESULT] flow_step_ms_std={arr.std():.3f}")

    # Benchmark repeated end-to-end loop with session reused.
    for _ in range(args.warmup):
        run_once_full()
    action_step_times = []
    total_times = []
    step_means = []
    for _ in range(args.iters):
        first_ms, _pred, _first_action, _pred_final, arr, total_ms = run_once_full()
        action_step_times.append(first_ms)
        total_times.append(total_ms)
        step_means.append(arr.mean() if arr.size else 0.0)
    action_step_times = np.array(action_step_times, dtype=np.float64)
    total_times = np.array(total_times, dtype=np.float64)
    step_means = np.array(step_means, dtype=np.float64)
    print(f"[BENCH] action_step_ms_mean={action_step_times.mean():.3f}")
    print(f"[BENCH] action_step_ms_std={action_step_times.std():.3f}")
    print(f"[BENCH] flow_step_ms_mean={step_means.mean():.3f}")
    print(f"[BENCH] flow_step_ms_std={step_means.std():.3f}")
    print(f"[BENCH] flow_total_ms_mean={total_times.mean():.3f}")
    print(f"[BENCH] flow_total_ms_std={total_times.std():.3f}")


if __name__ == "__main__":
    main()
