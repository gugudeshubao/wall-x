#!/usr/bin/env python3
"""
Build and run a TRT-LLM engine for the weighted part of ActionProcessor.step().

Scope:
  - current wall-x config path with:
    - proj_with_mask = True
    - use_adarms = False
    - action_hidden_size = hidden_size = 2048
  - host provides:
    - noisy_action [B, H, action_dim]
    - dof_mask [B, H, action_dim]
    - time_embed [B, action_hidden_size]
  - engine computes:
    concat(noisy_action, dof_mask) -> w1
    concat(action_embed, repeated_time_embed) -> w2 -> silu -> w3
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


class CheckpointReader:
    def __init__(self, path: str):
        self.path = path
        self.f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self.f.keys())

    def get(self, key: str):
        return self.f.get_tensor(key)


def sinusoidal_time_embed(timestep: torch.Tensor, dim: int):
    half_dim = dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timestep.device, dtype=torch.float32) * -emb)
    emb = timestep[:, None].float() * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def load_action_step_weights(model_path: str, checkpoint: str):
    config_path = Path(model_path) / "config.json"
    raw = json.loads(config_path.read_text())
    cfg = SimpleNamespace(
        hidden_size=raw.get("hidden_size", 2048),
        action_hidden_size=raw.get("action_hidden_size", 2048),
        proj_with_mask=raw.get("proj_with_mask", True),
        use_adarms=raw.get("use_adarms", False),
    )
    ckpt = CheckpointReader(checkpoint)
    weights = {
        "w1": ckpt.get("action_preprocessor.w1.weight"),
        "w2": ckpt.get("action_preprocessor.w2.weight"),
        "w3": ckpt.get("action_preprocessor.w3.weight"),
    }
    return cfg, weights


def build_engine(cfg, weights, refs, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import Linear

    if not cfg.proj_with_mask or cfg.use_adarms:
        raise NotImplementedError("Current action_step TRT engine only supports proj_with_mask=True and use_adarms=False")

    action_embed_ref = refs["action_embed_t0"]
    B, horizon, hidden = action_embed_ref.shape
    action_dim = refs["noise"].shape[-1]
    action_hidden_size = cfg.action_hidden_size

    builder = Builder()
    builder_config = builder.create_builder_config(name="action_step", precision=precision)
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        noisy_action = Tensor(name="noisy_action", dtype=str_dtype_to_trt(precision), shape=[B, horizon, action_dim])
        dof_mask = Tensor(name="dof_mask", dtype=str_dtype_to_trt(precision), shape=[B, horizon, action_dim])
        time_embed = Tensor(name="time_embed", dtype=str_dtype_to_trt(precision), shape=[B, action_hidden_size])

        x = F.concat([noisy_action, dof_mask], dim=2)
        w1 = Linear(action_dim * 2, action_hidden_size, bias=False, dtype=precision, gather_output=True)
        w1.weight.value = weights["w1"]
        x1 = w1(x)

        t = F.view(time_embed, [B, 1, action_hidden_size])
        t = F.repeat_interleave(t, horizon, dim=1)
        x2 = F.concat([x1, t], dim=2)
        w2 = Linear(action_hidden_size * 2, action_hidden_size, bias=False, dtype=precision, gather_output=True)
        w2.weight.value = weights["w2"]
        h2 = w2(x2)
        h2 = F.silu(h2)
        w3 = Linear(action_hidden_size, hidden, bias=False, dtype=precision, gather_output=True)
        w3.weight.value = weights["w3"]
        out = w3(h2)
        out.mark_output("action_embed", str_dtype_to_trt(precision))

    return builder.build_engine(net, builder_config)


def torch_reference(weights, noisy_action, dof_mask, time_embed):
    w1 = weights["w1"].float()
    w2 = weights["w2"].float()
    w3 = weights["w3"].float()
    x = torch.cat([noisy_action, dof_mask], dim=-1)
    x1 = torch.matmul(x, w1.t())
    t = time_embed.unsqueeze(1).repeat(1, x1.shape[1], 1)
    x2 = torch.cat([x1, t], dim=-1)
    h2 = torch.matmul(x2, w2.t())
    h2 = torch.nn.functional.silu(h2)
    out = torch.matmul(h2, w3.t())
    return out


def run_engine(engine, noisy_action, dof_mask, time_embed, precision="bfloat16", warmup=3, iters=10):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]

    inputs = {
        "noisy_action": noisy_action.to("cuda").to(torch_dtype),
        "dof_mask": dof_mask.to("cuda").to(torch_dtype),
        "time_embed": time_embed.to("cuda").to(torch_dtype),
    }
    outputs = {
        "action_embed": torch.zeros(
            (inputs["noisy_action"].shape[0], inputs["noisy_action"].shape[1], time_embed.shape[-1]),
            dtype=torch_dtype,
            device="cuda",
        )
    }
    infos = [
        tensorrt_llm.runtime.TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape)
        for k, v in inputs.items()
    ]
    session.infer_shapes(infos)
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        if not ok:
            raise RuntimeError("warmup session.run failed")
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        if not ok:
            raise RuntimeError("bench session.run failed")
        times.append((time.perf_counter() - t0) * 1000)
    return outputs["action_embed"].detach().float().cpu(), np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Build and run ActionProcessor.step TRT-LLM block")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--which", default="t0", choices=["t0", "t1"])
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--save-engine", default="")
    args = parser.parse_args()

    refs = load_file(args.ref, device="cpu")
    cfg, weights = load_action_step_weights(args.model_path, args.checkpoint)

    engine = build_engine(cfg, weights, refs, precision=args.precision)
    if args.save_engine:
        with open(args.save_engine, "wb") as f:
            f.write(engine)
        print(f"[SAVE] {args.save_engine}")

    if args.which == "t0":
        noisy_action = refs["noise"].float()
        timestep = refs["times"][0].reshape(1).float()
        expected = refs["action_embed_t0"].float()
    else:
        noisy_action = refs["noisy_action_after_prefetch"].float()
        timestep = refs["times"][1].reshape(1).float()
        expected = refs["action_embed_t1"].float()
    dof_mask = refs["dof_mask"].float()
    time_embed = sinusoidal_time_embed(timestep, cfg.action_hidden_size).cpu()

    torch_out = torch_reference(weights, noisy_action, dof_mask, time_embed).float()
    trt_out, times = run_engine(engine, noisy_action, dof_mask, time_embed, precision=args.precision, warmup=args.warmup, iters=args.iters)

    diff_torch = (trt_out - torch_out).abs()
    cos_torch = torch.nn.functional.cosine_similarity(trt_out.flatten(), torch_out.flatten(), dim=0).item()
    diff_ref = (trt_out - expected).abs()
    cos_ref = torch.nn.functional.cosine_similarity(trt_out.flatten(), expected.flatten(), dim=0).item()

    print(f"[CMP][trt vs torch_ref] cosine={cos_torch:.8f} mean_abs={diff_torch.mean().item():.8e} max_abs={diff_torch.max().item():.8e}")
    print(f"[CMP][trt vs exported_ref] cosine={cos_ref:.8f} mean_abs={diff_ref.mean().item():.8e} max_abs={diff_ref.max().item():.8e}")
    print(f"[BENCH] mean={times.mean():.3f} ms std={times.std():.3f} ms min={times.min():.3f} ms max={times.max():.3f} ms")


if __name__ == "__main__":
    main()
