#!/usr/bin/env python3
"""
Export a wall-x Flow Action single-step ONNX using the same INT8-SQ QDQ
building blocks that Edge-LLM uses in `llm_loader`.

This is a custom experiment path:
- it does not use stock `tensorrt_edgellm export_action`
- it does not depend on ONNX Runtime post-quantization
- it emits standard Q/DQ + MatMul patterns through the `trt::int8_sq_*`
  custom ops translated by `llm_loader`'s ONNX exporter
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

# Make ONNX / protobuf imports deterministic on Orin.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn as nn
import onnxscript
from onnxscript import opset18 as _op18
from onnxscript import script
from safetensors.torch import load_file


def _ensure_llm_loader_import(edge_llm_root: str) -> None:
    root = Path(edge_llm_root)
    experimental_dir = root / "experimental"
    if str(experimental_dir) not in sys.path:
        sys.path.insert(0, str(experimental_dir))

def _load_llm_loader_int8_ops(edge_llm_root: str):
    ops_path = (
        Path(edge_llm_root)
        / "experimental"
        / "llm_loader"
        / "models"
        / "ops.py"
    )
    spec = importlib.util.spec_from_file_location("edge_llm_int8_ops", ops_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load llm_loader ops from {ops_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@script()
def _int8_sq_act_qdq_translation(
    hidden_states: onnxscript.FLOAT16,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    zero_i64 = _op18.Constant(value_int=0)
    zero_i8 = _op18.Cast(zero_i64, to=3)  # INT8
    q = _op18.QuantizeLinear(hidden_states, scale, zero_i8)
    dq = _op18.DequantizeLinear(q, scale, zero_i8)
    return _op18.Cast(dq, to=10)  # FLOAT16


@script()
def _int8_sq_weight_dq_translation(
    weight: onnxscript.INT8,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT16:
    zero_i64 = _op18.Constant(value_int=0)
    zero_i8 = _op18.Cast(zero_i64, to=3)  # INT8
    zero_vec = _op18.Expand(zero_i8, _op18.Shape(scale))
    dq = _op18.DequantizeLinear(weight, scale, zero_vec, axis=0)
    return _op18.Cast(dq, to=10)  # FLOAT16


def _build_local_translation_table() -> dict:
    return {
        torch.ops.trt.int8_sq_act_qdq.default: _int8_sq_act_qdq_translation,
        torch.ops.trt.int8_sq_weight_dq.default: _int8_sq_weight_dq_translation,
    }


def load_state_dict_from_dir(model_dir: str) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        sd.update(load_file(f, device="cpu"))
    if not sd and os.path.isfile(model_dir):
        sd.update(load_file(model_dir, device="cpu"))
    if not sd:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    return sd


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


def quantize_int8_per_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.float()
    max_abs = w.abs().amax(dim=1)
    scale = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_int8_per_tensor_scale(sample: torch.Tensor) -> torch.Tensor:
    s = torch.clamp(sample.float().abs().amax() / 127.0, min=1e-8)
    return torch.tensor([float(s.item())], dtype=torch.float32)


class Int8SQLinearBridge(nn.Module):
    def __init__(
        self,
        weight_fp: torch.Tensor,
        bias: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        pre_quant_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        qweight, w_scale = quantize_int8_per_channel(weight_fp)
        in_features = weight_fp.shape[1]
        self.register_buffer("weight", qweight, persistent=True)
        self.register_buffer("weight_scale", w_scale, persistent=True)
        self.register_buffer(
            "input_scale",
            input_scale.clone().detach()
            if input_scale is not None
            else torch.ones(1, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "pre_quant_scale",
            pre_quant_scale.clone().detach()
            if pre_quant_scale is not None
            else torch.ones(in_features, dtype=torch.float16),
            persistent=True,
        )
        if bias is not None:
            self.register_buffer("bias", bias.clone().detach().to(torch.float16), persistent=True)
        else:
            self.bias = None
        self._act_qdq = None
        self._weight_dq = None

    def bind_ops(self, int8_ops_module) -> None:
        self._act_qdq = int8_ops_module.int8_sq_act_qdq
        self._weight_dq = int8_ops_module.int8_sq_weight_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        x_smooth = x * self.pre_quant_scale
        if self._act_qdq is None or self._weight_dq is None:
            raise RuntimeError("INT8SQLinearBridge ops not bound")
        x_dq = self._act_qdq(x_smooth, self.input_scale)
        w_dq = self._weight_dq(self.weight, self.weight_scale)
        y = torch.matmul(x_dq, w_dq.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class WallXFlowActionStepExportINT8SQ(nn.Module):
    def __init__(self, model_dir: str, refs: Dict[str, torch.Tensor], edge_llm_root: str, num_layers: int = 36):
        super().__init__()
        _ensure_llm_loader_import(edge_llm_root)
        int8_ops = _load_llm_loader_int8_ops(edge_llm_root)

        self.ckpt = load_state_dict_from_dir(model_dir)
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        self.cfg = json.loads(cfg_path.read_text())
        self.num_layers = num_layers

        self.action_dim = int(self.ckpt["action_preprocessor.action_proj_back.weight"].shape[0])
        self.action_hidden_size = int(self.ckpt["action_preprocessor.action_proj_back.weight"].shape[1])
        self.hidden_size = int(self.cfg.get("hidden_size", 0))
        self.heads = int(self.cfg.get("num_attention_heads", 16))
        self.kv_heads = int(self.cfg.get("num_key_value_heads", 2))
        self.head_dim = int(self.cfg.get("head_dim", self.hidden_size // self.heads))
        self.kv_groups = self.heads // self.kv_heads

        example_noise = refs["noise"].float()
        example_hidden = refs["postfix_inputs_embeds_t1"].float()
        input_scale_small = quantize_int8_per_tensor_scale(example_noise)
        input_scale_hidden = quantize_int8_per_tensor_scale(example_hidden)

        self.w1 = Int8SQLinearBridge(self.ckpt["action_preprocessor.w1.weight"].float(), None, input_scale_small)
        self.w2 = Int8SQLinearBridge(self.ckpt["action_preprocessor.w2.weight"].float(), None)
        self.w3 = Int8SQLinearBridge(self.ckpt["action_preprocessor.w3.weight"].float(), None)
        self.action_proj_back = Int8SQLinearBridge(
            self.ckpt["action_preprocessor.action_proj_back.weight"].float(),
            None,
        )
        self.final_norm = RmsNorm(self.ckpt["model.norm.weight"].float())

        self.layer_ln1 = nn.ModuleList()
        self.layer_qkv = nn.ModuleList()
        self.layer_o = nn.ModuleList()
        self.layer_ln2 = nn.ModuleList()
        self.layer_gate = nn.ModuleList()
        self.layer_up = nn.ModuleList()
        self.layer_down = nn.ModuleList()

        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            self.layer_ln1.append(RmsNorm(self.ckpt[p + "input_layernorm.weight"].float()))

            q_w = self.ckpt[p + "self_attn.q_proj.weight"].float()
            k_w = self.ckpt[p + "self_attn.k_proj.weight"].float()
            v_w = self.ckpt[p + "self_attn.v_proj.weight"].float()
            q_b = self.ckpt[p + "self_attn.q_proj.bias"].float()
            k_b = self.ckpt[p + "self_attn.k_proj.bias"].float()
            v_b = self.ckpt[p + "self_attn.v_proj.bias"].float()
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
            self.layer_qkv.append(Int8SQLinearBridge(qkv_w, qkv_b, input_scale_hidden))

            self.layer_o.append(
                Int8SQLinearBridge(
                    self.ckpt[p + "self_attn.o_proj.weight"].float(),
                    None,
                )
            )
            self.layer_ln2.append(RmsNorm(self.ckpt[p + "post_attention_layernorm.weight"].float()))
            self.layer_gate.append(
                Int8SQLinearBridge(self.ckpt[p + "moe.experts.1.gate_proj.weight"].float(), None)
            )
            self.layer_up.append(
                Int8SQLinearBridge(self.ckpt[p + "moe.experts.1.up_proj.weight"].float(), None)
            )
            self.layer_down.append(
                Int8SQLinearBridge(self.ckpt[p + "moe.experts.1.down_proj.weight"].float(), None)
            )

        # Bind llm_loader INT8-SQ custom ops after all modules are created.
        for module in [self.w1, self.w2, self.w3, self.action_proj_back]:
            module.bind_ops(int8_ops)
        for module in self.layer_qkv:
            module.bind_ops(int8_ops)
        for module in self.layer_o:
            module.bind_ops(int8_ops)
        for module in self.layer_gate:
            module.bind_ops(int8_ops)
        for module in self.layer_up:
            module.bind_ops(int8_ops)
        for module in self.layer_down:
            module.bind_ops(int8_ops)

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
        assert len(cache_tensors) == 2 * n_layers
        k_caches = list(cache_tensors[:n_layers])
        v_caches = list(cache_tensors[n_layers:])

        x = noise_trajectory.float()
        t0 = time_steps_t0.float()
        t1 = time_steps_t1.float()
        dof_mask = torch.ones_like(x)

        hidden = torch.cat([x, dof_mask], dim=-1)
        hidden = self.w1(hidden)
        half_dim = self.action_hidden_size // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t0.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half_dim - 1))
        )
        emb = t0[:, None].float() * emb[None, :]
        time_embed = torch.cat((emb.sin(), emb.cos()), dim=-1).unsqueeze(1).repeat(1, hidden.shape[1], 1)
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

        seq_mask = (
            torch.arange(S, device=denoised.device).view(1, S, 1)
            < seq_length.to(torch.long).view(B, 1, 1)
        ).to(denoised.dtype)
        denoised = denoised * seq_mask + denoised * (1.0 - seq_mask)
        return denoised, action_pred


def export_onnx(
    model: nn.Module,
    refs: dict,
    output_dir: str,
    num_layers: int,
    edge_llm_root: str,
    opset: int = 18,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model_cpu = model.cpu()

    prefix_len = int(refs["prefix_length"][0].item())
    action_horizon = int(refs["action_horizon"]) if "action_horizon" in refs else int(refs["action_pred_t0"].shape[1])
    action_dim = int(refs["noise"].shape[-1])
    hidden_size = int(refs["postfix_inputs_embeds_t1"].shape[-1])
    kv_heads = 2
    head_dim = 128

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
        refs["noise"].float(),
        refs["times"][0].reshape(1).float(),
        refs["times"][1].reshape(1).float(),
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

    translation_table = _build_local_translation_table()

    with torch.no_grad():
        prog = torch.onnx.export(
            model_cpu,
            tuple(inputs),
            dynamo=True,
            input_names=input_names,
            output_names=["denoised_trajectory", "action_pred"],
            dynamic_shapes=None,
            opset_version=opset,
            custom_translation_table=translation_table,
            external_data=True,
            optimize=True,
        )
        prog.save(os.path.join(output_dir, "model.onnx"), external_data=True)

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
    (Path(output_dir) / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export wall-x Flow Action INT8-SQ style ONNX")
    parser.add_argument("--edge-llm-root", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
    parser.add_argument("--model-path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--opset", type=int, default=21)
    args = parser.parse_args()

    refs = load_file(args.ref, device="cpu")
    model = WallXFlowActionStepExportINT8SQ(
        model_dir=args.model_path,
        refs=refs,
        edge_llm_root=args.edge_llm_root,
        num_layers=args.num_layers,
    )
    export_onnx(
        model,
        refs,
        args.output_dir,
        args.num_layers,
        edge_llm_root=args.edge_llm_root,
        opset=args.opset,
    )
    print(f"[SAVE] INT8-SQ style ONNX + config written to {args.output_dir}")


if __name__ == "__main__":
    main()
