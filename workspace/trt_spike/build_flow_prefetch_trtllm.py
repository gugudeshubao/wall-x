#!/usr/bin/env python3
"""
Build and run a Flow Action prefetch-only TRT-LLM decoder.

Current scope:
  - dummy Flow reference exported by export_flow_dummy_reference.py
  - prefetch decoder only
  - shared attention path
  - shared norms
  - MLP MoE specialized to contiguous token blocks:
      expert0 for prefix tokens
      expert1 for postfix action tokens

This is intentionally narrower than the eventual full Flow TRT runtime.
It only targets the first prefetch pass inside generate_flow_action().
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as torch_f
from safetensors import safe_open
from safetensors.torch import load_file


def rotate_half_trt(x, F):
    head_dim = x.shape[-1]
    x1, x2 = F.split(x, [head_dim // 2, head_dim // 2], dim=3)
    neg_x2 = F.mul(x2, -1.0)
    return F.concat([neg_x2, x1], dim=3)


def rotate_half_torch(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class CheckpointReader:
    def __init__(self, path: str):
        self.path = path
        self.f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self.f.keys())

    def get(self, key: str):
        return self.f.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._keys


def build_engine(
    ckpt: CheckpointReader,
    refs,
    num_layers: int,
    precision="bfloat16",
    output_kv=True,
    output_hidden=True,
    output_action_pred=True,
):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import Linear, RmsNorm, RowLinear

    hidden_in = refs["inputs_embeds_t0"]
    B, S, H = hidden_in.shape
    prefix_len = int(refs["prefix_length"][0].item())
    postfix_len = S - prefix_len
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    q_dim = HEADS * HEAD_DIM
    k_dim = KV_HEADS * HEAD_DIM
    v_dim = KV_HEADS * HEAD_DIM
    action_proj_w = ckpt.get("action_preprocessor.action_proj_back.weight")
    action_dim, action_hidden_size = action_proj_w.shape

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=f"flow_prefetch_{num_layers}", precision=precision
    )
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        x = Tensor(name="hidden_in", dtype=str_dtype_to_trt(precision), shape=[B, S, H])
        cos = Tensor(
            name="prefetch_cos_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=list(refs["prefetch_cos_mrope"].shape),
        )
        sin = Tensor(
            name="prefetch_sin_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=list(refs["prefetch_sin_mrope"].shape),
        )
        mask = Tensor(
            name="prefetch_causal_mask_4d",
            dtype=str_dtype_to_trt(precision),
            shape=list(refs["prefetch_causal_mask_4d"].shape),
        )

        h = x
        present_keys = []
        present_vals = []
        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            ln1 = RmsNorm(H, eps=1e-5, dtype=precision)
            ln1.weight.value = ckpt.get(prefix + "input_layernorm.weight")

            q_w = ckpt.get(prefix + "self_attn.q_proj.weight")
            k_w = ckpt.get(prefix + "self_attn.k_proj.weight")
            v_w = ckpt.get(prefix + "self_attn.v_proj.weight")
            q_b = ckpt.get(prefix + "self_attn.q_proj.bias")
            k_b = ckpt.get(prefix + "self_attn.k_proj.bias")
            v_b = ckpt.get(prefix + "self_attn.v_proj.bias")
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()

            qkv = Linear(
                H,
                q_dim + k_dim + v_dim,
                bias=True,
                dtype=precision,
                gather_output=True,
                is_qkv=True,
            )
            qkv.weight.value = qkv_w
            qkv.bias.value = qkv_b

            o_proj = RowLinear(q_dim, H, bias=False, dtype=precision)
            o_proj.weight.value = ckpt.get(prefix + "self_attn.o_proj.weight")

            ln2 = RmsNorm(H, eps=1e-5, dtype=precision)
            ln2.weight.value = ckpt.get(prefix + "post_attention_layernorm.weight")

            gate0_w = ckpt.get(prefix + "moe.experts.0.gate_proj.weight")
            up0_w = ckpt.get(prefix + "moe.experts.0.up_proj.weight")
            down0_w = ckpt.get(prefix + "moe.experts.0.down_proj.weight")
            gate1_w = ckpt.get(prefix + "moe.experts.1.gate_proj.weight")
            up1_w = ckpt.get(prefix + "moe.experts.1.up_proj.weight")
            down1_w = ckpt.get(prefix + "moe.experts.1.down_proj.weight")
            inter0 = gate0_w.shape[0]
            inter1 = gate1_w.shape[0]

            gate0 = Linear(H, inter0, bias=False, dtype=precision, gather_output=True)
            gate0.weight.value = gate0_w
            up0 = Linear(H, inter0, bias=False, dtype=precision, gather_output=True)
            up0.weight.value = up0_w
            down0 = RowLinear(inter0, H, bias=False, dtype=precision)
            down0.weight.value = down0_w

            gate1 = Linear(H, inter1, bias=False, dtype=precision, gather_output=True)
            gate1.weight.value = gate1_w
            up1 = Linear(H, inter1, bias=False, dtype=precision, gather_output=True)
            up1.weight.value = up1_w
            down1 = RowLinear(inter1, H, bias=False, dtype=precision)
            down1.weight.value = down1_w

            residual = h
            h1 = ln1(h)

            qkv_out = qkv(h1)
            q, k, v = F.split(qkv_out, [q_dim, k_dim, v_dim], dim=2)
            q = F.permute(F.view(q, [B, S, HEADS, HEAD_DIM]), [0, 2, 1, 3])
            k = F.permute(F.view(k, [B, S, KV_HEADS, HEAD_DIM]), [0, 2, 1, 3])
            v = F.permute(F.view(v, [B, S, KV_HEADS, HEAD_DIM]), [0, 2, 1, 3])

            q_rot = F.add(F.mul(q, cos), F.mul(rotate_half_trt(q, F), sin))
            k_rot = F.add(F.mul(k, cos), F.mul(rotate_half_trt(k, F), sin))
            if output_kv:
                present_keys.append(k_rot)
                present_vals.append(v)

            k_rep = F.repeat_interleave(k_rot, KV_GROUPS, dim=1)
            v_rep = F.repeat_interleave(v, KV_GROUPS, dim=1)

            scores = F.matmul(q_rot, k_rep, transb=True)
            scores = F.mul(scores, 1.0 / np.sqrt(HEAD_DIM))
            scores = F.add(scores, mask)
            probs = F.softmax(scores, dim=-1)
            attn = F.matmul(probs, v_rep)
            attn = F.permute(attn, [0, 2, 1, 3])
            attn = F.view(attn, [B, S, q_dim])
            attn_out = o_proj(attn)
            x2 = F.add(residual, attn_out)

            residual2 = x2
            h2 = ln2(x2)

            h2_prefix = F.slice(h2, starts=[0, 0, 0], sizes=[B, prefix_len, H])
            h2_postfix = F.slice(h2, starts=[0, prefix_len, 0], sizes=[B, postfix_len, H])

            gate0_out = gate0(h2_prefix)
            up0_out = up0(h2_prefix)
            hidden0 = F.mul(F.silu(gate0_out), up0_out)
            mlp0 = down0(hidden0)

            gate1_out = gate1(h2_postfix)
            up1_out = up1(h2_postfix)
            hidden1 = F.mul(F.silu(gate1_out), up1_out)
            mlp1 = down1(hidden1)

            mlp_out = F.concat([mlp0, mlp1], dim=1)
            h = F.add(residual2, mlp_out)

        final_norm = RmsNorm(H, eps=1e-5, dtype=precision)
        final_norm.weight.value = ckpt.get("model.norm.weight")
        h = final_norm(h)
        action_h = F.slice(h, starts=[0, prefix_len, 0], sizes=[B, postfix_len, action_hidden_size])
        action_proj = Linear(
            action_hidden_size, action_dim, bias=False, dtype=precision, gather_output=True
        )
        action_proj.weight.value = action_proj_w
        action_pred = action_proj(action_h)
        if output_hidden:
            h.mark_output("hidden_out", str_dtype_to_trt(precision))
        if output_action_pred:
            action_pred.mark_output("action_pred", str_dtype_to_trt(precision))
        if output_kv:
            for i in range(num_layers):
                present_keys[i].mark_output(
                    f"present_past_key_{i}", str_dtype_to_trt(precision)
                )
                present_vals[i].mark_output(
                    f"present_past_value_{i}", str_dtype_to_trt(precision)
                )

    return builder.build_engine(net, builder_config)


def torch_reference(ckpt: CheckpointReader, refs, num_layers: int):
    hidden = refs["inputs_embeds_t0"].to("cuda").float()
    cos = refs["prefetch_cos_mrope"].to("cuda").float()
    sin = refs["prefetch_sin_mrope"].to("cuda").float()
    mask = refs["prefetch_causal_mask_4d"].to("cuda").float()
    prefix_len = int(refs["prefix_length"][0].item())
    B, S, H = hidden.shape
    postfix_len = S - prefix_len
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    action_proj_w = ckpt.get("action_preprocessor.action_proj_back.weight").to("cuda").float()

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        residual = hidden

        w = ckpt.get(prefix + "input_layernorm.weight").to("cuda").float()
        var = hidden.pow(2).mean(-1, keepdim=True)
        h1 = hidden * torch.rsqrt(var + 1e-5)
        h1 = h1 * w

        q_w = ckpt.get(prefix + "self_attn.q_proj.weight").to("cuda").float()
        k_w = ckpt.get(prefix + "self_attn.k_proj.weight").to("cuda").float()
        v_w = ckpt.get(prefix + "self_attn.v_proj.weight").to("cuda").float()
        q_b = ckpt.get(prefix + "self_attn.q_proj.bias").to("cuda").float()
        k_b = ckpt.get(prefix + "self_attn.k_proj.bias").to("cuda").float()
        v_b = ckpt.get(prefix + "self_attn.v_proj.bias").to("cuda").float()
        o_w = ckpt.get(prefix + "self_attn.o_proj.weight").to("cuda").float()

        q = torch.matmul(h1, q_w.t()) + q_b
        k = torch.matmul(h1, k_w.t()) + k_b
        v = torch.matmul(h1, v_w.t()) + v_b
        q = q.view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = (q * cos) + (rotate_half_torch(q) * sin)
        k = (k * cos) + (rotate_half_torch(k) * sin)
        k_rep = k.repeat_interleave(KV_GROUPS, dim=1)
        v_rep = v.repeat_interleave(KV_GROUPS, dim=1)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) / np.sqrt(HEAD_DIM)
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v_rep)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        attn_out = torch.matmul(attn, o_w.t())
        x2 = residual + attn_out

        residual2 = x2
        w2 = ckpt.get(prefix + "post_attention_layernorm.weight").to("cuda").float()
        var2 = x2.pow(2).mean(-1, keepdim=True)
        h2 = x2 * torch.rsqrt(var2 + 1e-5)
        h2 = h2 * w2

        prefix_h = h2[:, :prefix_len, :]
        postfix_h = h2[:, prefix_len:, :]

        gate0_w = ckpt.get(prefix + "moe.experts.0.gate_proj.weight").to("cuda").float()
        up0_w = ckpt.get(prefix + "moe.experts.0.up_proj.weight").to("cuda").float()
        down0_w = ckpt.get(prefix + "moe.experts.0.down_proj.weight").to("cuda").float()
        gate1_w = ckpt.get(prefix + "moe.experts.1.gate_proj.weight").to("cuda").float()
        up1_w = ckpt.get(prefix + "moe.experts.1.up_proj.weight").to("cuda").float()
        down1_w = ckpt.get(prefix + "moe.experts.1.down_proj.weight").to("cuda").float()

        hidden0 = torch_f.silu(torch.matmul(prefix_h, gate0_w.t())) * torch.matmul(prefix_h, up0_w.t())
        mlp0 = torch.matmul(hidden0, down0_w.t())
        hidden1 = torch_f.silu(torch.matmul(postfix_h, gate1_w.t())) * torch.matmul(postfix_h, up1_w.t())
        mlp1 = torch.matmul(hidden1, down1_w.t())
        hidden = residual2 + torch.cat([mlp0, mlp1], dim=1)

    w_final = ckpt.get("model.norm.weight").to("cuda").float()
    var = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(var + 1e-5)
    hidden = hidden * w_final
    action_hidden = hidden[:, prefix_len:, : action_proj_w.shape[1]]
    action_pred = torch.matmul(action_hidden, action_proj_w.t())
    return hidden.cpu(), action_pred.cpu()


def run_engine(engine, refs, num_layers: int, precision="bfloat16", warmup=3, iters=10):
    import tensorrt_llm
    from tensorrt_llm._utils import torch_dtype_to_trt

    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]

    inputs = {
        "hidden_in": refs["inputs_embeds_t0"].to("cuda").to(torch_dtype),
        "prefetch_cos_mrope": refs["prefetch_cos_mrope"].to("cuda").to(torch_dtype),
        "prefetch_sin_mrope": refs["prefetch_sin_mrope"].to("cuda").to(torch_dtype),
        "prefetch_causal_mask_4d": refs["prefetch_causal_mask_4d"].to("cuda").to(torch_dtype),
    }
    outputs = {
        "hidden_out": torch.zeros_like(inputs["hidden_in"])
    }
    postfix_len = refs["action_pred_t0"].shape[1]
    outputs["action_pred"] = torch.zeros(
        (1, postfix_len, refs["action_pred_t0"].shape[-1]), dtype=torch_dtype, device="cuda"
    )
    kv_heads = 2
    head_dim = 128
    seq_len = inputs["hidden_in"].shape[1]
    for i in range(num_layers):
        outputs[f"present_past_key_{i}"] = torch.zeros(
            (1, kv_heads, seq_len, head_dim), dtype=torch_dtype, device="cuda"
        )
        outputs[f"present_past_value_{i}"] = torch.zeros(
            (1, kv_heads, seq_len, head_dim), dtype=torch_dtype, device="cuda"
        )

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
    return outputs, np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Build and run Flow prefetch TRT-LLM block")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--save-engine", default="")
    args = parser.parse_args()

    refs = load_file(args.ref, device="cpu")
    ckpt = CheckpointReader(args.checkpoint)

    print(f"[BUILD] Flow prefetch engine ({args.num_layers} layers, {args.precision})")
    t0 = time.perf_counter()
    engine = build_engine(ckpt, refs, num_layers=args.num_layers, precision=args.precision, output_kv=True)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"[BUILD] done in {build_ms:.1f} ms")

    if args.save_engine:
        with open(args.save_engine, "wb") as f:
            f.write(engine)
        print(f"[SAVE] {args.save_engine}")

    print("[RUN] torch reference ...")
    exported_ref_hidden = refs["prefetch_hidden_states"].float()
    exported_ref_action = refs["action_pred_t0"].float()
    torch_hidden, torch_action = torch_reference(ckpt, refs, args.num_layers)
    torch_hidden = torch_hidden.float()
    torch_action = torch_action.float()
    if args.num_layers == 36:
        diff_torch_ref = (torch_hidden - exported_ref_hidden).abs()
        cos_torch_ref = torch.nn.functional.cosine_similarity(
            torch_hidden.flatten(), exported_ref_hidden.flatten(), dim=0
        ).item()
        print(
            f"[CMP][torch vs exported_ref] cosine={cos_torch_ref:.8f} "
            f"mean_abs={diff_torch_ref.mean().item():.8e} max_abs={diff_torch_ref.max().item():.8e}"
        )

    print("[RUN] TRT engine ...")
    outputs, times = run_engine(engine, refs, args.num_layers, precision=args.precision, warmup=args.warmup, iters=args.iters)
    trt_hidden = outputs["hidden_out"].detach().float().cpu()
    trt_action = outputs["action_pred"].detach().float().cpu()
    diff_trt_torch = (trt_hidden - torch_hidden).abs()
    cos_trt_torch = torch.nn.functional.cosine_similarity(
        trt_hidden.flatten(), torch_hidden.flatten(), dim=0
    ).item()
    print(
        f"[CMP][trt vs torch_ref] cosine={cos_trt_torch:.8f} "
        f"mean_abs={diff_trt_torch.mean().item():.8e} max_abs={diff_trt_torch.max().item():.8e}"
    )
    if args.num_layers == 36:
        diff_trt_ref = (trt_hidden - exported_ref_hidden).abs()
        cos_trt_ref = torch.nn.functional.cosine_similarity(
            trt_hidden.flatten(), exported_ref_hidden.flatten(), dim=0
        ).item()
        print(
            f"[CMP][trt vs exported_ref] cosine={cos_trt_ref:.8f} "
            f"mean_abs={diff_trt_ref.mean().item():.8e} max_abs={diff_trt_ref.max().item():.8e}"
        )
        diff_action_ref = (trt_action - exported_ref_action).abs()
        cos_action_ref = torch.nn.functional.cosine_similarity(
            trt_action.flatten(), exported_ref_action.flatten(), dim=0
        ).item()
        print(
            f"[CMP][action_pred trt vs exported_ref] cosine={cos_action_ref:.8f} "
            f"mean_abs={diff_action_ref.mean().item():.8e} max_abs={diff_action_ref.max().item():.8e}"
        )
    diff_action_torch = (trt_action - torch_action).abs()
    cos_action_torch = torch.nn.functional.cosine_similarity(
        trt_action.flatten(), torch_action.flatten(), dim=0
    ).item()
    print(
        f"[CMP][action_pred trt vs torch_ref] cosine={cos_action_torch:.8f} "
        f"mean_abs={diff_action_torch.mean().item():.8e} max_abs={diff_action_torch.max().item():.8e}"
    )

    prefix_len = int(refs["prefix_length"][0].item())
    k_ref = refs["prefix_past_key_0"].float()
    k_trt = outputs["present_past_key_0"][:, :, :prefix_len, :].detach().float().cpu()
    k_diff = (k_trt - k_ref).abs()
    k_cos = torch.nn.functional.cosine_similarity(k_trt.flatten(), k_ref.flatten(), dim=0).item()
    print(f"[CMP][kv0 prefix slice] cosine={k_cos:.8f} mean_abs={k_diff.mean().item():.8e} max_abs={k_diff.max().item():.8e}")

    print(
        f"[BENCH] mean={times.mean():.3f} ms std={times.std():.3f} ms min={times.min():.3f} ms max={times.max():.3f} ms"
    )


if __name__ == "__main__":
    main()
