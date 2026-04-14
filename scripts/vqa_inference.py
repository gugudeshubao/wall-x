import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


class VQAWrapper(object):
    def __init__(self, model_path: str, train_config: dict = None):
        self.timings = {}
        self.device = self._setup_device()
        t0 = time.perf_counter()
        if train_config is None:
            try:
                with open(os.path.join(model_path, "config.yml"), "r") as f:
                    train_config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                print(f"load train_config.yml fail: {e}")
                train_config = None
        self.timings["config_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.processor = self._load_processor(model_path, train_config)
        self.timings["processor_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.model = self._load_model(model_path, train_config)
        self.timings["model_s"] = time.perf_counter() - t0
        self.timings["init_total_s"] = sum(self.timings.values())

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_processor(
        self, model_path: str, train_config: dict | None
    ) -> AutoProcessor:
        processor_path = (
            train_config["processor_path"]
            if train_config is not None and "processor_path" in train_config
            else model_path
        )
        return AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    def _load_model(
        self, model_path: str, train_config: dict | None
    ) -> Qwen2_5_VLMoEForAction:
        model = Qwen2_5_VLMoEForAction.from_pretrained(
            model_path, train_config=train_config
        )
        if self.device == "cuda":
            model = model.to(self.device, dtype=torch.bfloat16)
        else:
            model.to(self.device)
        model.eval()
        return model

    def generate(self, image: Image.Image, text: str, return_stats: bool = False, **kwargs):
        t0 = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        preprocess_s = time.perf_counter() - t0

        generation_params = {
            "max_new_tokens": 1024,  # default value, can be overridden by kwargs
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            **kwargs,
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_params)
        generate_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        decode_s = time.perf_counter() - t0

        stats = {
            "preprocess_s": preprocess_s,
            "generate_s": generate_s,
            "decode_s": decode_s,
            "total_s": preprocess_s + generate_s + decode_s,
            "max_new_tokens": generation_params["max_new_tokens"],
            "device": self.device,
        }
        if return_stats:
            return response, stats
        return response


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
        help="Optional training config path. If omitted, the script falls back to model_path/config.yml.",
    )
    parser.add_argument(
        "--image-path",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "cot_example_frame.png"),
        help="Image used for VQA inference.",
    )
    parser.add_argument(
        "--question",
        default="To move the red block in the plate with same color, what should you do next? Think step by step.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--show-stats", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_path == "/path/to/model_path" or not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. "
            "Pass --model-path or set WALL_X_MODEL_PATH."
        )

    train_config = None
    if args.train_config_path and os.path.exists(args.train_config_path):
        with open(args.train_config_path, "r") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

    wrapper = VQAWrapper(model_path=args.model_path, train_config=train_config)

    try:
        img = Image.open(args.image_path).convert("RGB")
        answer, stats = wrapper.generate(
            img, args.question, max_new_tokens=args.max_new_tokens, return_stats=True
        )
        print("model answer:", answer)
        if args.show_stats:
            print("timings:", wrapper.timings)
            print("run stats:", stats)
    except Exception as e:
        print(f"model answer fail: {e}")
