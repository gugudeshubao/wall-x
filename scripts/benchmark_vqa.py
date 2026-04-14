#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vqa_inference import VQAWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("WALL_X_MODEL_PATH", "/path/to/model_path"),
        help="Path to the downloaded Wall-X model directory.",
    )
    parser.add_argument(
        "--image-path",
        default=str(
            Path(__file__).resolve().parents[1] / "assets" / "cot_example_frame.png"
        ),
    )
    parser.add_argument(
        "--question",
        default="To move the red block in the plate with same color, what should you do next? Think step by step.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_path == "/path/to/model_path" or not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. "
            "Pass --model-path or set WALL_X_MODEL_PATH."
        )

    image = Image.open(args.image_path).convert("RGB")
    wrapper = VQAWrapper(model_path=args.model_path, train_config=None)
    run_stats = []

    for idx in range(args.runs):
        _, stats = wrapper.generate(
            image,
            args.question,
            max_new_tokens=args.max_new_tokens,
            return_stats=True,
        )
        run_stats.append(stats)
        print(f"run {idx + 1}: {stats}", flush=True)

    summary = {
        "init": wrapper.timings,
        "runs": run_stats,
        "avg_generate_s": mean(stat["generate_s"] for stat in run_stats),
        "avg_total_s": mean(stat["total_s"] for stat in run_stats),
        "max_new_tokens": args.max_new_tokens,
        "runs_count": args.runs,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
