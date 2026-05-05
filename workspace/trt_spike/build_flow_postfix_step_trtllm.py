#!/usr/bin/env python3
"""
Build and run a Flow Action postfix-step TRT-LLM decoder.

Current scope:
  - dummy Flow reference exported by export_flow_dummy_reference.py
  - the first postfix decoder step only
  - shared attention with prefix KV cache
  - expert1-only MLP on all postfix tokens

This targets the step_with_kvcache() body inside generate_flow_action().
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


def build_engine(ckpt: CheckpointReader, refs, num_layers: int, precision="bfloat16"):
    import tensorrt_llm
    from tensorrt_llm import Builder, Tensor, str_dtype_to_trt
    import tensorrt_llm.functional as F
    from tensorrt_llm.layers import Linear, RmsNorm, RowLinear

    hidden_in = refs["postfix_inputs_embeds_t1"]
    B, S, H = hidden_in.shape
    HEADS = 16
    KV_HEADS = 2
    HEAD_DIM = 128
    KV_GROUPS = HEADS // KV_HEADS
    q_dim = HEADS * HEAD_DIM
    k_dim = KV_HEADS * HEAD_DIM
    v_dim = KV_HEADS * HEAD_DIM
    past_len = refs["prefix_past_key_0"].shape[2]
    action_proj_w = ckpt.get("action_preprocessor.action_proj_back.weight")
    action_dim, action_hidden_size = action_proj_w.shape

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=f"flow_postfix_step_{num_layers}", precision=precision
    )
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        x = Tensor(name="hidden_in", dtype=str_dtype_to_trt(precision), shape=[B, S, H])
        cos = Tensor(
            name="postfix_cos_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=list(refs["postfix_cos_mrope"].shape),
        )
        sin = Tensor(
            name="postfix_sin_mrope",
            dtype=str_dtype_to_trt(precision),
            shape=list(refs["postfix_sin_mrope"].shape),
        )
        mask = Tensor(
            name="postfix_attention_mask_additive_4d",
            dtype=str_dtype_to_trt(precision),
            shape=list(refs["postfix_attention_mask_additive_4d"].shape),
        )

        past_keys = []
        past_vals = []
        for i in range(num_layers):
            pk = Tensor(
                name=f"prefix_past_key_{i}",
                dtype=str_dtype_to_trt(precision),
                shape=[B, KV_HEADS, past_len, HEAD_DIM],
            )
            pv = Tensor(
                name=f"prefix_past_value_{i}",
                dtype=str_dtype_to_trt(precision),
                shape=[B, KV_HEADS, past_len, HEAD_DIM],
            )
            past_keys.append(pk)
            past_vals.append(pv)

        h = x
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

            gate1_w = ckpt.get(prefix + "moe.experts.1.gate_proj.weight")
            up1_w = ckpt.get(prefix + "moe.experts.1.up_proj.weight")
            down1_w = ckpt.get(prefix + "moe.experts.1.down_proj.weight")
            inter1 = gate1_w.shape[0]

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

            full_k = F.concat([past_keys[i], k_rot], dim=2)
            full_v = F.concat([past_vals[i], v], dim=2)

            k_rep = F.repeat_interleave(full_k, KV_GROUPS, dim=1)
            v_rep = F.repeat_interleave(full_v, KV_GROUPS, dim=1)

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
            hidden1 = F.mul(F.silu(gate1(h2)), up1(h2))
            mlp_out = down1(hidden1)
            h = F.add(residual2, mlp_out)

        final_norm = RmsNorm(H, eps=1e-5, dtype=precision)
        final_norm.weight.value = ckpt.get("model.norm.weight")
        h = final_norm(h)
        action_h = F.slice(h, starts=[0, 0, 0], sizes=[B, S, action_hidden_size])
        action_proj = Linear(
            action_hidden_size, action_dim, bias=False, dtype=precision, gather_output=True
        )
        action_proj.weight.value = action_proj_w
        action_pred = action_proj(action_h)
        h.mark_output("hidden_out", str_dtype_to_trt(precision))
        action_pred.mark_output("action_pred", str_dtype_to_trt(precision))

    return builder.build_engine(net, builder_config)


def torch_reference(ckpt: CheckpointReader, refs, num_layers: int):
    hidden = refs["postfix_inputs_embeds_t1"].to("cuda").float()
    cos = refs["postfix_cos_mrope"].to("cuda").float()
    sin = refs["postfix_sin_mrope"].to("cuda").float()
    mask = refs["postfix_attention_mask_additive_4d"].to("cuda").float()
    B, S, H = hidden.shape
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

        past_k = refs[f"prefix_past_key_{i}"].to("cuda").float()
        past_v = refs[f"prefix_past_value_{i}"].to("cuda").float()
        full_k = torch.cat([past_k, k], dim=2)
        full_v = torch.cat([past_v, v], dim=2)

        k_rep = full_k.repeat_interleave(KV_GROUPS, dim=1)
        v_rep = full_v.repeat_interleave(KV_GROUPS, dim=1)
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

        gate1_w = ckpt.get(prefix + "moe.experts.1.gate_proj.weight").to("cuda").float()
        up1_w = ckpt.get(prefix + "moe.experts.1.up_proj.weight").to("cuda").float()
        down1_w = ckpt.get(prefix + "moe.experts.1.down_proj.weight").to("cuda").float()
        hidden1 = torch_f.silu(torch.matmul(h2, gate1_w.t())) * torch.matmul(h2, up1_w.t())
        mlp_out = torch.matmul(hidden1, down1_w.t())
        hidden = residual2 + mlp_out

    w_final = ckpt.get("model.norm.weight").to("cuda").float()
    var = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(var + 1e-5)
    hidden = hidden * w_final
    action_hidden = hidden[:, :, : action_proj_w.shape[1]]
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
        "hidden_in": refs["postfix_inputs_embeds_t1"].to("cuda").to(torch_dtype),
        "postfix_cos_mrope": refs["postfix_cos_mrope"].to("cuda").to(torch_dtype),
        "postfix_sin_mrope": refs["postfix_sin_mrope"].to("cuda").to(torch_dtype),
        "postfix_attention_mask_additive_4d": refs["postfix_attention_mask_additive_4d"].to("cuda").to(torch_dtype),
    }
    for i in range(num_layers):
        inputs[f"prefix_past_key_{i}"] = refs[f"prefix_past_key_{i}"].to("cuda").to(torch_dtype)
        inputs[f"prefix_past_value_{i}"] = refs[f"prefix_past_value_{i}"].to("cuda").to(torch_dtype)
    outputs = {
        "hidden_out": torch.zeros_like(inputs["hidden_in"])
    }
    outputs["action_pred"] = torch.zeros(
        (1, refs["action_pred_t1"].shape[1], refs["action_pred_t1"].shape[-1]),
        dtype=torch_dtype,
        device="cuda",
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
    parser = argparse.ArgumentParser(description="Build and run Flow postfix-step TRT-LLM block")
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

    print(f"[BUILD] Flow postfix-step engine ({args.num_layers} layers, {args.precision})")
    t0 = time.perf_counter()
    engine = build_engine(ckpt, refs, num_layers=args.num_layers, precision=args.precision)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"[BUILD] done in {build_ms:.1f} ms")

    if args.save_engine:
        with open(args.save_engine, "wb") as f:
            f.write(engine)
        print(f"[SAVE] {args.save_engine}")

    print("[RUN] torch reference ...")
    exported_ref_hidden = refs["postfix_hidden_states_t1"].float()
    exported_ref_action = refs["action_pred_t1"].float()
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

    print(
        f"[BENCH] mean={times.mean():.3f} ms std={times.std():.3f} ms min={times.min():.3f} ms max={times.max():.3f} ms"
    )


if __name__ == "__main__":
    main()
