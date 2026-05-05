#!/usr/bin/env python3
"""
Run TensorRT-Edge-LLM experimental/llm_loader tools in a hybrid Orin Python setup:

- CUDA torch from the user site-packages
- newer numpy / typing_extensions / onnxscript / modelopt from venv_edge
- llm_loader code from the repo checkout

This exists because the legacy venv used for Edge-LLM export currently has CPU-only
torch, while the system/user-site install has CUDA-enabled torch but older deps.
"""

from __future__ import annotations

import argparse
import builtins
import os
import inspect
import runpy
import sys
from pathlib import Path


USER_SITE = "/home/dog/.local/lib/python3.10/site-packages"
EDGE_VENV_SITE = "/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/lib/python3.10/site-packages"
EDGE_REPO_ROOT = "/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM"
EDGE_EXPERIMENTAL = "/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/experimental"


def prepare_hybrid_imports() -> None:
    tmpdir = "/data/wy/tmp"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    os.environ["TMP"] = tmpdir
    os.environ["TEMP"] = tmpdir
    # 1. Import CUDA torch from the user site path.
    if USER_SITE not in sys.path:
        sys.path.insert(0, USER_SITE)
    import torch  # noqa: F401

    # 2. Remove all user-site paths so later imports do not pull stale numpy / typing_extensions.
    sys.path = [p for p in sys.path if "/home/dog/.local" not in p]

    # 3. Drop potentially polluted modules.
    for key in list(sys.modules.keys()):
        if key == "numpy" or key.startswith("numpy."):
            del sys.modules[key]
        if key == "typing_extensions" or key.startswith("typing_extensions."):
            del sys.modules[key]

    # 4. Add the newer dep set and llm_loader repo path.
    sys.path.insert(0, EDGE_VENV_SITE)
    sys.path.insert(0, EDGE_REPO_ROOT)
    sys.path.insert(0, EDGE_EXPERIMENTAL)


def patch_llm_loader_awq_repack() -> None:
    """
    Monkey-patch llm_loader AWQ repacking for numpy 2.x on Orin.

    The stock code uses:
      packed_int16.view(np.int8).reshape(rows * 2, cols)
    which is failing in the current mixed environment.

    Replacing it with an explicit frombuffer() over the raw bytes is more robust
    and preserves the intended byte layout for export-time repacking.
    """
    import numpy as np
    import torch
    import llm_loader.checkpoint.repacking as repacking

    def patched_repack_awq_to_plugin(qweight: torch.Tensor, qzeros: torch.Tensor) -> torch.Tensor:
        in_features, out_div8 = qweight.shape
        out_features = out_div8 * 8
        group_size = in_features // qzeros.shape[0]

        qw = qweight.cpu().to(torch.int32)
        qz = qzeros.cpu().to(torch.int32)

        bit_to_ch = [0, 2, 4, 6, 1, 3, 5, 7]

        nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
        for k in range(8):
            nibbles[:, bit_to_ch[k]::8] = (qw >> (4 * k)) & 0xF

        zeros = torch.zeros(in_features // group_size, out_features, dtype=torch.int32)
        for k in range(8):
            zeros[:, bit_to_ch[k]::8] = (qz >> (4 * k)) & 0xF

        zeros_expanded = zeros.repeat_interleave(group_size, dim=0)
        nibbles = (nibbles - zeros_expanded + 8).clamp(0, 15)
        nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)

        packed_u16 = np.asarray(repacking._pack_intweights(nibbles_nk), dtype=np.uint16)
        packed_u16 = np.ascontiguousarray(packed_u16)
        rows, cols = packed_u16.shape
        packed_int8 = np.frombuffer(packed_u16.tobytes(), dtype=np.int8).reshape(rows * 2, cols)
        return torch.tensor(packed_int8, dtype=torch.int8).to(qweight.device)

    repacking.repack_awq_to_plugin = patched_repack_awq_to_plugin


def patch_torch_onnx_export_compat() -> None:
    """
    Make llm_loader's dynamo ONNX export tolerant to torch 2.5's older export API.

    We only strip unsupported kwargs here. This is enough for a fixed-shape export
    benchmark path and avoids patching the upstream Edge-LLM source tree.
    """
    import torch

    original_export = torch.onnx.export
    supported = set(inspect.signature(original_export).parameters.keys())

    def _dim_name(dim: object, default: str) -> str:
        for attr in ("__name__", "name"):
            value = getattr(dim, attr, None)
            if isinstance(value, str) and value:
                return value
        return default

    def _dynamic_shapes_to_axes(dynamic_shapes, input_names):
        if not dynamic_shapes or not input_names:
            return {}
        dynamic_axes = {}
        for name, shape_spec in zip(input_names, dynamic_shapes):
            if not isinstance(shape_spec, dict):
                continue
            axes = {}
            for axis, dim in shape_spec.items():
                if dim is None:
                    continue
                axes[int(axis)] = _dim_name(dim, f"{name}_dim{axis}")
            if axes:
                dynamic_axes[name] = axes
        return dynamic_axes

    class _LegacySaveShim:
        def __init__(self, model, args, kwargs):
            self.model = model
            self.args = args
            self.kwargs = kwargs

        def save(self, output_path, external_data=True):
            legacy_kwargs = dict(self.kwargs)
            dynamic_axes = _dynamic_shapes_to_axes(
                legacy_kwargs.get("dynamic_shapes"),
                legacy_kwargs.get("input_names"),
            )
            for key in [
                "dynamo",
                "dynamic_shapes",
                "custom_translation_table",
                "external_data",
                "optimize",
            ]:
                legacy_kwargs.pop(key, None)
            legacy_kwargs["f"] = str(output_path)
            legacy_kwargs["dynamo"] = False
            legacy_kwargs["dynamic_axes"] = dynamic_axes or legacy_kwargs.get("dynamic_axes")
            legacy_kwargs.pop("external_data", None)
            legacy_kwargs.pop("optimize", None)
            if "opset_version" in legacy_kwargs and legacy_kwargs["opset_version"] is not None:
                legacy_kwargs["opset_version"] = min(int(legacy_kwargs["opset_version"]), 20)
            return original_export(self.model, self.args, **legacy_kwargs)

    def wrapped_export(*args, **kwargs):
        unsupported = []
        for key in list(kwargs.keys()):
            if key not in supported:
                unsupported.append(key)
                kwargs.pop(key)
        if unsupported:
            builtins.print("PATCHED_TORCH_ONNX_EXPORT_DROP_KWARGS", unsupported, flush=True)
        if kwargs.get("dynamo", False):
            return _LegacySaveShim(args[0], args[1], kwargs)
        return original_export(*args, **kwargs)

    torch.onnx.export = wrapped_export


def patch_legacy_int4_groupwise_symbolic() -> None:
    """
    Register a legacy ONNX symbolic for trt::int4_groupwise_gemm.

    This is the missing piece for torch 2.5 legacy exporter after we drop the
    dynamo custom translation table. We emit the same TRT custom-op name that
    Edge-LLM's ONNX parser expects.
    """
    from torch.onnx import register_custom_op_symbolic, symbolic_helper
    from torch.onnx.symbolic_helper import _get_tensor_sizes
    from torch.onnx.symbolic_helper import _get_tensor_sizes

    @symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
    def symbolic_int4_groupwise_gemm(g, input, qweight, scales, gemm_n, gemm_k, group_size):
        out = g.op(
            "trt::Int4GroupwiseGemmPlugin",
            input,
            qweight,
            scales,
            gemm_n_i=gemm_n,
            gemm_k_i=gemm_k,
            group_size_i=group_size,
        )
        return out

    register_custom_op_symbolic("trt::int4_groupwise_gemm", symbolic_int4_groupwise_gemm, 20)
    builtins.print("PATCHED_LEGACY_SYMBOLIC", "trt::int4_groupwise_gemm", flush=True)


def patch_existing_legacy_symbolics() -> None:
    """
    Reuse the legacy symbolic registrations that already exist in the legacy
    tensorrt_edgellm package for attention_plugin / gather_nd / vit attention.
    """
    import torch
    from torch.onnx import register_custom_op_symbolic, symbolic_helper
    from torch.onnx.symbolic_helper import _get_tensor_sizes

    @symbolic_helper.parse_args(
        "v", "v", "v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "v", "v", "v"
    )
    def symbolic_attention_plugin(
        g,
        q,
        k,
        v,
        past_key_value,
        context_lengths,
        rope_rotary_cos_sin,
        kvcache_start_index,
        num_q_heads,
        num_kv_heads,
        enable_tree_attention,
        head_size,
        enable_fp8_kv_cache,
        sliding_window_size,
        attention_mask=None,
        position_ids=None,
        qkv_scales=None,
        _get_tensor_sizes=_get_tensor_sizes,
    ):
        q_sizes = _get_tensor_sizes(q)
        resolved_head_size = head_size
        if isinstance(q_sizes, (list, tuple)) and q_sizes and q_sizes[-1] is not None:
            resolved_head_size = int(q_sizes[-1])
        if resolved_head_size is None or int(resolved_head_size) <= 0:
            resolved_head_size = 128

        inputs = [
            q,
            k,
            v,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
        ]
        # For the current AWQ export path on Orin, forcing the non-tree attention
        # form is more robust than emitting empty optional inputs.
        _enable_tree_attention = 0
        attrs = dict(
            num_q_heads_i=num_q_heads,
            num_kv_heads_i=num_kv_heads,
            head_size_i=resolved_head_size,
            enable_tree_attention_i=_enable_tree_attention,
            enable_fp8_kv_cache_i=1 if enable_fp8_kv_cache else 0,
            sliding_window_size_i=sliding_window_size,
        )
        if qkv_scales is not None:
            attrs["qkv_scales_f"] = []
        attn_output, present_key_value = g.op(
            "trt::AttentionPlugin", *inputs, **attrs, outputs=2
        )
        attn_output.setType(q.type())
        present_key_value.setType(past_key_value.type())
        return attn_output, present_key_value

    def symbolic_gather_nd(g, value, indices, batch_dims=1):
        unsqueeze_axes = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        indices_expanded = g.op("Unsqueeze", indices, unsqueeze_axes)
        return g.op("GatherND", value, indices_expanded, batch_dims_i=batch_dims)

    register_custom_op_symbolic("trt::attention_plugin", symbolic_attention_plugin, 20)
    register_custom_op_symbolic("trt::gather_nd", symbolic_gather_nd, 20)
    builtins.print("PATCHED_LEGACY_SYMBOLIC", "attention_plugin/gather_nd", flush=True)


def patch_attention_forward_dynamic_reshape() -> None:
    """
    Avoid baking seq_len into the attention reshape path.

    The default Attention.forward() destructures hidden_states.shape into Python
    ints. Under the legacy exporter that turns seq_len into a constant, which in
    turn hard-codes a [1,1,hidden] reshape into the ONNX graph. For AWQ we need
    the reshape to stay symbolic so the graph can handle prefill-style lengths.
    """
    import llm_loader.models.default.modeling_default as modeling_default

    def patched_forward(
        self,
        hidden_states,
        past_key_value,
        rope_rotary_cos_sin,
        context_lengths,
        kvcache_start_index,
        attention_mask=None,
        attention_pos_id=None,
    ):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            ).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            ).reshape(batch_size, seq_len, self.num_kv_heads * self.head_dim)

        enable_tree = attention_mask is not None and attention_pos_id is not None
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": enable_tree,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
        }
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float", [1.0, 1.0, 1.0])

        attn_output, present_key_value = modeling_default.attention_plugin(
            query_states,
            key_states,
            value_states,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value

    modeling_default.Attention.forward = patched_forward
    builtins.print("PATCHED_LEGACY_FORWARD", "Attention.forward dynamic reshape", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run llm_loader tools in a hybrid Orin environment")
    parser.add_argument("tool", choices=["export_all_cli"], help="llm_loader module to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the target tool")
    ns = parser.parse_args()

    prepare_hybrid_imports()
    patch_llm_loader_awq_repack()
    patch_torch_onnx_export_compat()
    patch_legacy_int4_groupwise_symbolic()
    patch_existing_legacy_symbolics()

    module_map = {
        "export_all_cli": "llm_loader.export_all_cli",
    }
    mod = module_map[ns.tool]
    sys.argv = [mod, *ns.args]
    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
