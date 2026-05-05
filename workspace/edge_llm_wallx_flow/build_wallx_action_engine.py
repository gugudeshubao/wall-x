#!/usr/bin/env python3
"""
Run TensorRT-Edge-LLM action_build on the exported wall-x action ONNX.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build wall-x action.engine via Edge-LLM action_build")
    parser.add_argument("--edge-llm-root", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--min-seq-len", type=int, default=456)
    parser.add_argument("--max-seq-len", type=int, default=488)
    parser.add_argument("--opt-seq-len", type=int, default=472)
    args = parser.parse_args()

    action_build = Path(args.edge_llm_root) / "build_orin/examples/multimodal/action_build"
    cmd = [
        str(action_build),
        "--onnxDir",
        args.onnx_dir,
        "--engineDir",
        args.engine_dir,
        "--minSeqLen",
        str(args.min_seq_len),
        "--maxSeqLen",
        str(args.max_seq_len),
        "--optSeqLen",
        str(args.opt_seq_len),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

