#!/usr/bin/env python3
"""
Post-training INT8 quantization for the custom wall-x Flow Action ONNX.

This does not use Edge-LLM's stock `export_action` path, which is currently
FP16-only. Instead, it quantizes our custom exported ONNX with ONNX Runtime's
static QDQ flow so we can probe whether `action_build` can consume an INT8-ish
graph produced from the existing bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator

# ONNX import on Orin can trip over the compiled protobuf extension path.
# Force the pure-Python protobuf implementation for this standalone tool.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from safetensors.torch import load_file


def build_feed_dict(refs: Dict[str, np.ndarray], num_layers: int) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {
        "noise_trajectory": refs["noise"].float().cpu().numpy(),
        "time_steps_t0": refs["times"][0].reshape(1).float().cpu().numpy(),
        "time_steps_t1": refs["times"][1].reshape(1).float().cpu().numpy(),
        "seq_length": np.array([int(refs["prefix_length"][0].item())], dtype=np.int64),
        "postfix_cos_mrope": refs["postfix_cos_mrope"].float().cpu().numpy(),
        "postfix_sin_mrope": refs["postfix_sin_mrope"].float().cpu().numpy(),
        "postfix_attention_mask_additive_4d": refs[
            "postfix_attention_mask_additive_4d"
        ].float().cpu().numpy(),
    }
    for i in range(num_layers):
        feed[f"prefix_past_key_{i}"] = refs[f"prefix_past_key_{i}"].float().cpu().numpy()
    for i in range(num_layers):
        feed[f"prefix_past_value_{i}"] = refs[f"prefix_past_value_{i}"].float().cpu().numpy()
    return feed


class SingleSampleReader(CalibrationDataReader):
    def __init__(self, sample: Dict[str, np.ndarray]):
        self._sample = sample
        self._iter: Iterator[Dict[str, np.ndarray]] | None = None

    def get_next(self) -> Dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter([self._sample])
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize the custom wall-x Flow Action ONNX to INT8 QDQ"
    )
    parser.add_argument("--onnx-dir", required=True, help="Directory containing model.onnx and config.json")
    parser.add_argument("--ref", required=True, help="flow_dummy reference safetensors")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write quantized model.onnx and copied config.json",
    )
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument(
        "--op-types-to-quantize",
        nargs="+",
        default=["MatMul"],
        help="ONNX op types to quantize (default: MatMul only)",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic"],
        default="dynamic",
        help="Quantization mode. Use dynamic first for the custom Flow bridge.",
    )
    parser.add_argument(
        "--activation-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Activation quant type",
    )
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Weight quant type",
    )
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel weight quantization")
    parser.add_argument(
        "--tmp-dir",
        default="/data/wy/tmp/flow_action_int8q",
        help="Temporary directory for ONNX Runtime calibration artifacts",
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    src_model = onnx_dir / "model.onnx"
    src_cfg = onnx_dir / "config.json"
    if not src_model.exists():
        raise FileNotFoundError(src_model)
    if not src_cfg.exists():
        raise FileNotFoundError(src_cfg)

    refs = load_file(args.ref, device="cpu")
    feed = build_feed_dict(refs, args.num_layers)
    reader = SingleSampleReader(feed)

    dst_model = output_dir / "model.onnx"
    shutil.copy2(src_cfg, output_dir / "config.json")

    activation_type = QuantType.QInt8 if args.activation_type == "qint8" else QuantType.QUInt8
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    if args.mode == "static":
        quantize_static(
            model_input=str(src_model),
            model_output=str(dst_model),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            per_channel=args.per_channel,
            op_types_to_quantize=args.op_types_to_quantize,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "AddQDQPairToWeight": True,
            },
        )
    else:
        quantize_dynamic(
            model_input=str(src_model),
            model_output=str(dst_model),
            weight_type=weight_type,
            per_channel=args.per_channel,
            op_types_to_quantize=args.op_types_to_quantize,
            extra_options={
                "WeightSymmetric": True,
            },
        )

    # Validate the written ONNX at least parses.
    onnx.load(str(dst_model))

    summary = {
        "source_onnx": str(src_model),
        "output_onnx": str(dst_model),
        "mode": args.mode,
        "op_types_to_quantize": args.op_types_to_quantize,
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "per_channel": args.per_channel,
    }
    (output_dir / "quant_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
