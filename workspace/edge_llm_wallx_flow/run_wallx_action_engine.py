#!/usr/bin/env python3
"""
Run the exported wall-x Flow Action engine and compare with the wall-x reference.

This file doubles as:
- a standalone benchmark CLI
- an importable runner used by the serving/policy adapter
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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


@dataclass
class WallXFlowActionRunnerConfig:
    engine_dir: str
    ref_path: str
    model_path: str = "/data/wy/models/wall-oss-flow"


class WallXFlowActionRunner:
    def __init__(self, config: WallXFlowActionRunnerConfig):
        self.config = config
        self.refs = load_file(config.ref_path, device="cpu")
        self.engine_path = Path(config.engine_dir) / "action" / "action.engine"
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)

        import tensorrt_llm
        from tensorrt_llm._utils import torch_dtype_to_trt

        self.tensorrt_llm = tensorrt_llm
        self.torch_dtype_to_trt = torch_dtype_to_trt

        engine = self.engine_path.read_bytes()
        self.session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

        self.noise = self.refs["noise"].clone().float().to("cuda")
        self.times = self.refs["times"].clone().float().to("cuda")
        self.prefix_len = int(self.refs["prefix_length"][0].item())
        self.num_layers = len(
            [k for k in self.refs.keys() if k.startswith("prefix_past_key_")]
        )
        cfg = json.loads((Path(config.model_path) / "config.json").read_text())
        self.use_x_pred = bool(cfg.get("use_x_pred", False))

        self.inputs = {
            "noise_trajectory": self.noise,
            "time_steps_t0": self.times[0].reshape(1),
            "time_steps_t1": self.times[1].reshape(1),
            "seq_length": torch.tensor(
                [self.prefix_len], device="cuda", dtype=torch.int64
            ),
            "postfix_cos_mrope": self.refs["postfix_cos_mrope"].to("cuda").float(),
            "postfix_sin_mrope": self.refs["postfix_sin_mrope"].to("cuda").float(),
            "postfix_attention_mask_additive_4d": self.refs[
                "postfix_attention_mask_additive_4d"
            ].to("cuda").float(),
        }
        for i in range(self.num_layers):
            self.inputs[f"prefix_past_key_{i}"] = self.refs[
                f"prefix_past_key_{i}"
            ].to("cuda").float()
            self.inputs[f"prefix_past_value_{i}"] = self.refs[
                f"prefix_past_value_{i}"
            ].to("cuda").float()

        self.outputs = {
            "denoised_trajectory": torch.zeros_like(
                self.refs["noisy_action_after_prefetch"], device="cuda"
            ),
            "action_pred": torch.zeros_like(self.refs["action_pred_t1"], device="cuda"),
        }
        infos = [
            self.tensorrt_llm.runtime.TensorInfo(
                k, self.torch_dtype_to_trt(v.dtype), v.shape
            )
            for k, v in self.inputs.items()
        ]
        self.session.infer_shapes(infos)
        self.stream = torch.cuda.current_stream()

    def run_once_full(self) -> Dict[str, Any]:
        self.inputs["noise_trajectory"] = self.noise
        self.inputs["time_steps_t0"] = self.times[0].reshape(1)
        self.inputs["time_steps_t1"] = self.times[1].reshape(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = self.session.run(
            inputs=self.inputs, outputs=self.outputs, stream=self.stream.cuda_stream
        )
        torch.cuda.synchronize()
        if not ok:
            raise RuntimeError("action engine run failed")
        first_ms = (time.perf_counter() - t0) * 1000
        first_pred = self.outputs["denoised_trajectory"].detach().cpu().float()
        first_action = self.outputs["action_pred"].detach().float()

        noisy_action = self.refs["noisy_action_after_prefetch"].clone().to(
            "cuda"
        ).float()
        full_step_times = []
        total_t0 = time.perf_counter()
        for step_idx in range(1, len(self.times) - 1):
            self.inputs["noise_trajectory"] = noisy_action
            self.inputs["time_steps_t0"] = self.times[step_idx].reshape(1)
            self.inputs["time_steps_t1"] = self.times[step_idx + 1].reshape(1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ok = self.session.run(
                inputs=self.inputs,
                outputs=self.outputs,
                stream=self.stream.cuda_stream,
            )
            torch.cuda.synchronize()
            if not ok:
                raise RuntimeError(f"action engine step {step_idx} failed")
            full_step_times.append((time.perf_counter() - t1) * 1000)
            action_pred = self.outputs["action_pred"].detach().float()
            v_t = action_pred - self.noise if self.use_x_pred else action_pred
            dt = (self.times[step_idx + 1] - self.times[step_idx]).view(1, 1, 1)
            noisy_action = noisy_action + dt * v_t
        total_ms = first_ms + (time.perf_counter() - total_t0) * 1000
        return {
            "action_step_ms": first_ms,
            "denoised_vs_ref": first_pred,
            "action_pred": first_action,
            "flow_final_action": noisy_action.cpu().float(),
            "flow_step_times_ms": np.array(full_step_times),
            "flow_total_ms": total_ms,
        }


def run_once_full(engine_dir: str, ref_path: str, model_path: str) -> Dict[str, Any]:
    runner = WallXFlowActionRunner(
        WallXFlowActionRunnerConfig(
            engine_dir=engine_dir, ref_path=ref_path, model_path=model_path
        )
    )
    return runner.run_once_full()


def main():
    parser = argparse.ArgumentParser(
        description="Run wall-x action.engine and compare against reference"
    )
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--model-path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    runner = WallXFlowActionRunner(
        WallXFlowActionRunnerConfig(
            engine_dir=args.engine_dir,
            ref_path=args.ref,
            model_path=args.model_path,
        )
    )

    # Accuracy/reference run
    result = runner.run_once_full()
    pred = result["denoised_vs_ref"]
    pred_final = result["flow_final_action"]
    arr = result["flow_step_times_ms"]
    total_ms = result["flow_total_ms"]

    ref = runner.refs["noisy_action_after_prefetch"].float()
    diff = (pred - ref).abs()
    cos = torch.nn.functional.cosine_similarity(pred.flatten(), ref.flatten(), dim=0).item()

    print(f"[RESULT] action_step_ms={result['action_step_ms']:.3f}")
    print(f"[RESULT] denoised_vs_ref cosine={cos:.8f}")
    print(f"[RESULT] denoised_vs_ref mean_abs={diff.mean().item():.8e}")
    print(f"[RESULT] denoised_vs_ref max_abs={diff.max().item():.8e}")
    print(f"[RESULT] action_pred_shape={tuple(result['action_pred'].shape)}")

    ref_final = runner.refs["predict_action"].float()
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
        runner.run_once_full()
    action_step_times = []
    total_times = []
    step_means = []
    for _ in range(args.iters):
        result = runner.run_once_full()
        action_step_times.append(result["action_step_ms"])
        total_times.append(result["flow_total_ms"])
        step_means.append(
            result["flow_step_times_ms"].mean() if result["flow_step_times_ms"].size else 0.0
        )
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
