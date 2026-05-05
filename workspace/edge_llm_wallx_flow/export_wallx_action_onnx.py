#!/usr/bin/env python3
"""
Export a wall-x Flow Action single-step ONNX that can be built with
TensorRT-Edge-LLM's action_build.

This is intentionally minimal:
- one flow-matching step
- wall-x action preprocessor + transformer postfix stack
- output: denoised action trajectory and action velocity prediction

The ONNX schema keeps the Edge-LLM-friendly `seq_length` input so the action
builder can attach its optimization profile, but the actual wall-x flow logic
still runs through wall-x weights and wall-x token/cache layout.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def load_state_dict_from_dir(model_dir: str) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        sd.update(load_file(f, device="cpu"))
    if not sd and os.path.isfile(model_dir):
        sd.update(load_file(model_dir, device="cpu"))
    if not sd:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    return sd


def sinusoidal_time_embed(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = torch.exp(
        torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
        * (-math.log(10000.0) / (half_dim - 1))
    )
    emb = timestep[:, None].float() * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RmsNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weight", weight.clone().detach(), persistent=True)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight.clone().detach(), persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias.clone().detach(), persistent=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class WallXFlowActionStepExport(nn.Module):
    def __init__(self, model_dir: str, num_layers: int = 36):
        super().__init__()
        self.model_dir = model_dir
        self.state_dict = load_state_dict_from_dir(model_dir)
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        self.cfg = json.loads(cfg_path.read_text())
        self.num_layers = num_layers

        self.action_dim = int(
            self.state_dict["action_preprocessor.action_proj_back.weight"].shape[0]
        )
        self.action_hidden_size = int(
            self.state_dict["action_preprocessor.action_proj_back.weight"].shape[1]
        )
        self.hidden_size = int(self.cfg.get("hidden_size", 0))
        if self.hidden_size <= 0:
            raise ValueError("hidden_size missing from config.json")
        self.heads = int(self.cfg.get("num_attention_heads", 16))
        self.kv_heads = int(self.cfg.get("num_key_value_heads", 2))
        self.head_dim = int(self.cfg.get("head_dim", self.hidden_size // self.heads))
        self.kv_groups = self.heads // self.kv_heads
        self.rope_section = self.cfg["rope_scaling"]["mrope_section"]

        self.w1 = Linear(self.state_dict["action_preprocessor.w1.weight"].float())
        self.w2 = Linear(self.state_dict["action_preprocessor.w2.weight"].float())
        self.w3 = Linear(self.state_dict["action_preprocessor.w3.weight"].float())
        self.action_proj_back = Linear(
            self.state_dict["action_preprocessor.action_proj_back.weight"].float()
        )
        self.final_norm = RmsNorm(self.state_dict["model.norm.weight"].float())

        self.layer_ln1 = nn.ModuleList()
        self.layer_qkv = nn.ModuleList()
        self.layer_o = nn.ModuleList()
        self.layer_ln2 = nn.ModuleList()
        self.layer_gate = nn.ModuleList()
        self.layer_up = nn.ModuleList()
        self.layer_down = nn.ModuleList()
        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            self.layer_ln1.append(
                RmsNorm(self.state_dict[p + "input_layernorm.weight"].float())
            )
            q_w = self.state_dict[p + "self_attn.q_proj.weight"].float()
            k_w = self.state_dict[p + "self_attn.k_proj.weight"].float()
            v_w = self.state_dict[p + "self_attn.v_proj.weight"].float()
            q_b = self.state_dict[p + "self_attn.q_proj.bias"].float()
            k_b = self.state_dict[p + "self_attn.k_proj.bias"].float()
            v_b = self.state_dict[p + "self_attn.v_proj.bias"].float()
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
            self.layer_qkv.append(Linear(qkv_w, qkv_b))
            self.layer_o.append(
                Linear(self.state_dict[p + "self_attn.o_proj.weight"].float(), None)
            )
            self.layer_ln2.append(
                RmsNorm(self.state_dict[p + "post_attention_layernorm.weight"].float())
            )
            self.layer_gate.append(
                Linear(self.state_dict[p + "moe.experts.1.gate_proj.weight"].float(), None)
            )
            self.layer_up.append(
                Linear(self.state_dict[p + "moe.experts.1.up_proj.weight"].float(), None)
            )
            self.layer_down.append(
                Linear(self.state_dict[p + "moe.experts.1.down_proj.weight"].float(), None)
            )

    def forward(
        self,
        noise_trajectory: torch.Tensor,
        time_steps_t0: torch.Tensor,
        time_steps_t1: torch.Tensor,
        seq_length: torch.Tensor,
        postfix_cos_mrope: torch.Tensor,
        postfix_sin_mrope: torch.Tensor,
        postfix_attention_mask_additive_4d: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_layers = self.num_layers
        assert len(cache_tensors) == 2 * n_layers, (
            f"Expected 2*{n_layers} cache tensors, got {len(cache_tensors)}"
        )
        k_caches = list(cache_tensors[:n_layers])
        v_caches = list(cache_tensors[n_layers:])

        x = noise_trajectory.float()
        t0 = time_steps_t0.float()
        t1 = time_steps_t1.float()
        dof_mask = torch.ones_like(x)

        # action preprocessor step (mirrors wall-x ActionProcessor.step)
        hidden = torch.cat([x, dof_mask], dim=-1)
        hidden = self.w1(hidden)
        time_embed = sinusoidal_time_embed(t0, self.action_hidden_size)
        time_embed = time_embed.unsqueeze(1).repeat(1, hidden.shape[1], 1)
        hidden = torch.cat([hidden, time_embed], dim=-1)
        hidden = self.w2(hidden)
        hidden = torch.nn.functional.silu(hidden)
        hidden = self.w3(hidden)

        B, S, _ = hidden.shape
        cos = postfix_cos_mrope
        sin = postfix_sin_mrope
        mask = postfix_attention_mask_additive_4d

        for i in range(n_layers):
            residual = hidden
            h1 = self.layer_ln1[i](hidden)
            qkv_out = self.layer_qkv[i](h1)
            q, k, v = torch.split(
                qkv_out,
                [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim],
                dim=-1,
            )
            q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

            q_rot = (q * cos) + (rotate_half(q) * sin)
            k_rot = (k * cos) + (rotate_half(k) * sin)

            full_k = torch.cat([k_caches[i], k_rot], dim=2)
            full_v = torch.cat([v_caches[i], v], dim=2)
            k_rep = full_k.repeat_interleave(self.kv_groups, dim=1)
            v_rep = full_v.repeat_interleave(self.kv_groups, dim=1)

            scores = torch.matmul(q_rot, k_rep.transpose(-1, -2))
            scores = scores * (1.0 / math.sqrt(self.head_dim))
            scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            attn = torch.matmul(probs, v_rep)
            attn = attn.transpose(1, 2).contiguous().view(B, S, self.heads * self.head_dim)
            attn_out = self.layer_o[i](attn)
            x2 = residual + attn_out

            residual2 = x2
            h2 = self.layer_ln2[i](x2)
            hidden1 = torch.nn.functional.silu(self.layer_gate[i](h2)) * self.layer_up[i](h2)
            mlp_out = self.layer_down[i](hidden1)
            hidden = residual2 + mlp_out

        hidden = self.final_norm(hidden)
        action_hidden = hidden[:, :, : self.action_hidden_size]
        action_pred = self.action_proj_back(action_hidden)
        dt = (t1 - t0).view(B, 1, 1)
        denoised = x + dt * action_pred

        # Keep seq_length in the graph so action_build can profile it.
        # Use a runtime mask identity rather than a constant zero add so ONNX
        # export does not fold the input away.
        seq_mask = (
            torch.arange(S, device=denoised.device).view(1, S, 1)
            < seq_length.to(torch.long).view(B, 1, 1)
        ).to(denoised.dtype)
        denoised = denoised * seq_mask + denoised * (1.0 - seq_mask)
        return denoised, action_pred


def export_onnx(model: nn.Module, refs: dict, output_dir: str, num_layers: int, opset: int = 22) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model_cpu = model.cpu()

    B = refs["noise"].shape[0]
    prefix_len = int(refs["prefix_length"][0].item())
    action_horizon = int(refs["action_horizon"]) if "action_horizon" in refs else int(refs["action_pred_t0"].shape[1])
    action_dim = int(refs["noise"].shape[-1])
    hidden_size = int(refs["postfix_inputs_embeds_t1"].shape[-1])
    kv_heads = 2
    head_dim = 128

    noise_trajectory = refs["noise"].float()
    time_steps_t0 = refs["times"][0].reshape(1).float()
    time_steps_t1 = refs["times"][1].reshape(1).float()
    seq_length = torch.tensor([prefix_len], dtype=torch.int64)

    input_names = [
        "noise_trajectory",
        "time_steps_t0",
        "time_steps_t1",
        "seq_length",
        "postfix_cos_mrope",
        "postfix_sin_mrope",
        "postfix_attention_mask_additive_4d",
    ]
    inputs = [
        noise_trajectory,
        time_steps_t0,
        time_steps_t1,
        seq_length,
        refs["postfix_cos_mrope"].float(),
        refs["postfix_sin_mrope"].float(),
        refs["postfix_attention_mask_additive_4d"].float(),
    ]

    for i in range(num_layers):
        inputs.append(refs[f"prefix_past_key_{i}"].float())
        input_names.append(f"prefix_past_key_{i}")
    for i in range(num_layers):
        inputs.append(refs[f"prefix_past_value_{i}"].float())
        input_names.append(f"prefix_past_value_{i}")

    output_names = ["denoised_trajectory", "action_pred"]

    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            tuple(inputs),
            os.path.join(output_dir, "model.onnx"),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )

    cfg = {
        "model_type": "wallx_flow_action",
        "edgellm_version": "0.7.0",
        "hidden_size": hidden_size,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 16,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_theta": 1000000.0,
        "rope_scaling": {"type": "mrope", "rope_type": "mrope", "mrope_section": [16, 24, 24]},
        "action_hidden_size": int(model.action_hidden_size),
        "prefix_length": prefix_len,
        "postfix_length": action_horizon,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Export wall-x Flow Action step to ONNX for Edge-LLM action_build")
    parser.add_argument("--model-path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    args = parser.parse_args()

    refs = load_file(args.ref, device="cpu")
    model = WallXFlowActionStepExport(args.model_path, num_layers=args.num_layers)
    export_onnx(model, refs, args.output_dir, args.num_layers)
    print(f"[SAVE] ONNX + config written to {args.output_dir}")


if __name__ == "__main__":
    main()
