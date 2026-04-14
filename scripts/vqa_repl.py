#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path

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
        "--train-config-path",
        default=os.environ.get("WALL_X_TRAIN_CONFIG_PATH"),
        help="Optional training config path.",
    )
    parser.add_argument(
        "--image-path",
        default=str(
            Path(__file__).resolve().parents[1] / "assets" / "cot_example_frame.png"
        ),
        help="Default image used for VQA.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Default generation length for each question.",
    )
    return parser.parse_args()


def load_train_config(train_config_path: str | None):
    if train_config_path and os.path.exists(train_config_path):
        import yaml

        with open(train_config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    return None


def print_help():
    print("Commands:")
    print("  :help                Show this help")
    print("  :quit                Exit the REPL")
    print("  :image <path>        Switch the current image")
    print("  :tokens <int>        Update max_new_tokens")
    print("  :stats               Show model initialization timings")
    print("Any other input is treated as a question.")


def main():
    args = parse_args()

    if args.model_path == "/path/to/model_path" or not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. "
            "Pass --model-path or set WALL_X_MODEL_PATH."
        )

    current_image_path = args.image_path
    max_new_tokens = args.max_new_tokens

    print(f"Loading model once from: {args.model_path}", flush=True)
    wrapper = VQAWrapper(
        model_path=args.model_path,
        train_config=load_train_config(args.train_config_path),
    )
    print("Model loaded. Enter questions. Use :help for commands.", flush=True)
    print(f"Init timings: {wrapper.timings}", flush=True)

    while True:
        try:
            raw = input("wall-x> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not raw:
            continue
        if raw == ":quit":
            break
        if raw == ":help":
            print_help()
            continue
        if raw == ":stats":
            print(wrapper.timings, flush=True)
            continue
        if raw.startswith(":image "):
            candidate = raw[len(":image ") :].strip()
            if not os.path.exists(candidate):
                print(f"Image not found: {candidate}", flush=True)
                continue
            current_image_path = candidate
            print(f"Using image: {current_image_path}", flush=True)
            continue
        if raw.startswith(":tokens "):
            candidate = raw[len(":tokens ") :].strip()
            try:
                max_new_tokens = int(candidate)
                print(f"max_new_tokens={max_new_tokens}", flush=True)
            except ValueError:
                print(f"Invalid integer: {candidate}", flush=True)
            continue

        image = Image.open(current_image_path).convert("RGB")
        t0 = time.perf_counter()
        answer, stats = wrapper.generate(
            image, raw, max_new_tokens=max_new_tokens, return_stats=True
        )
        wall_s = time.perf_counter() - t0
        print(
            f"[timing] total={wall_s:.2f}s generate={stats['generate_s']:.2f}s "
            f"preprocess={stats['preprocess_s']:.2f}s decode={stats['decode_s']:.2f}s",
            flush=True,
        )
        print(answer, flush=True)


if __name__ == "__main__":
    main()
