#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
from onnx import numpy_helper


def set_dim_param(value_info: onnx.ValueInfoProto, axis: int, name: str) -> None:
    dim = value_info.type.tensor_type.shape.dim[axis]
    dim.ClearField("dim_value")
    dim.dim_param = name


def patch_model(model: onnx.ModelProto) -> None:
    for value in model.graph.input:
        name = value.name
        if name == "inputs_embeds":
            set_dim_param(value, 0, "batch")
            set_dim_param(value, 1, "seq_len")
        elif name.startswith("past_key_values_"):
            set_dim_param(value, 0, "batch")
            set_dim_param(value, 3, "past_len")
        elif name == "rope_rotary_cos_sin":
            set_dim_param(value, 0, "rope_batch")
            set_dim_param(value, 1, "max_pos")
        elif name == "context_lengths":
            set_dim_param(value, 0, "batch")
        elif name == "kvcache_start_index":
            set_dim_param(value, 0, "kv_batch")
        elif name == "last_token_ids":
            set_dim_param(value, 0, "batch")

    # The AWQ export currently bakes a [1, 1, hidden] reshape target into each
    # self-attention block. Switch those shape constants to [0, 0, hidden] so
    # ONNX Reshape copies the dynamic batch/seq dimensions from its input.
    for node in model.graph.node:
        if node.op_type != "Constant" or "/self_attn/Constant" not in node.name:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            arr = numpy_helper.to_array(attr.t)
            if list(arr.shape) == [3] and arr.tolist() == [1, 1, 2048]:
                arr = arr.copy()
                arr[0] = 0
                arr[1] = 0
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=attr.t.name))

    for value in model.graph.output:
        name = value.name
        if name == "logits":
            set_dim_param(value, 0, "batch")
            set_dim_param(value, 1, "seq_len")
        elif name.startswith("present_key_values_"):
            set_dim_param(value, 0, "batch")
            set_dim_param(value, 3, "present_len")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Edge-LLM AWQ ONNX static dims back to dynamic symbolic dims")
    parser.add_argument("--src-dir", required=True, help="Source llm export directory containing model.onnx and sidecars")
    parser.add_argument("--dst-dir", required=True, help="Destination llm export directory for patched artifacts")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in src_dir.iterdir():
        if src.name == "model.onnx":
            continue
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)

    model = onnx.load(str(src_dir / "model.onnx"), load_external_data=False)
    patch_model(model)
    onnx.save(model, str(dst_dir / "model.onnx"))
    print(f"patched -> {dst_dir / 'model.onnx'}")


if __name__ == "__main__":
    main()
