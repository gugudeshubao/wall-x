#!/usr/bin/env python3
"""
Build a TensorRT engine from the Phase A decoder ONNX with fixed shapes.

This is intentionally minimal and uses trtexec via subprocess for the first pass.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build Phase A decoder TensorRT engine")
    parser.add_argument("--onnx", required=True, help="Input ONNX path")
    parser.add_argument("--engine", required=True, help="Output engine path")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16"],
        help="TensorRT precision mode",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trtexec_bin = "/usr/src/tensorrt/bin/trtexec"
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
        "--verbose",
    ]

    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "bf16":
        cmd.append("--bf16")

    print("[BUILD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[SAVE] {engine_path}")


if __name__ == "__main__":
    main()
