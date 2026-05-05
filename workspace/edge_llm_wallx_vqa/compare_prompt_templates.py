#!/usr/bin/env python3
"""Compare wall-x tokenizer/template organization against Edge-LLM Qwen-VL engines."""

import argparse
import json
from pathlib import Path


WALLX_SPECIAL_TOKENS = [
    "<|propri|>",
    "<|action|>",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def wallx_summary(path: Path) -> dict:
    data = load_json(path)
    chat_template = data.get("chat_template", "")
    additional_tokens = data.get("additional_special_tokens", [])
    added_decoder = data.get("added_tokens_decoder", {})
    all_token_values = set(additional_tokens)
    all_token_values.update(v.get("content") for v in added_decoder.values() if isinstance(v, dict))
    action_token_count = sum(
        1 for token in all_token_values if isinstance(token, str) and token.startswith("<|action_token_")
    )
    return {
        "processor_class": data.get("processor_class"),
        "has_helpful_assistant_fallback": "You are a helpful assistant." in chat_template,
        "has_tools_branch": "{%- if tools %}" in chat_template,
        "has_image_pad_token": "<|image_pad|>" in all_token_values,
        "has_video_pad_token": "<|video_pad|>" in all_token_values,
        "has_vision_start_token": "<|vision_start|>" in all_token_values,
        "has_vision_end_token": "<|vision_end|>" in all_token_values,
        "has_proprio_token": "<|propri|>" in all_token_values,
        "has_action_token": "<|action|>" in all_token_values,
        "action_token_count": action_token_count,
        "chat_template_prefix": chat_template[:400],
    }


def edge_summary(processed_template_path: Path, tokenizer_config_path: Path) -> dict:
    processed = load_json(processed_template_path)
    tokenizer_cfg = load_json(tokenizer_config_path)
    extra_tokens = tokenizer_cfg.get("extra_special_tokens", {}) or {}
    token_values = set(extra_tokens.values()) if isinstance(extra_tokens, dict) else set()
    return {
        "model_path": processed.get("model_path"),
        "processor_class": tokenizer_cfg.get("processor_class"),
        "default_system_prompt": processed.get("default_system_prompt"),
        "generation_prompt": processed.get("generation_prompt"),
        "image_format": processed.get("content_types", {}).get("image", {}).get("format"),
        "video_format": processed.get("content_types", {}).get("video", {}).get("format"),
        "roles": processed.get("roles"),
        "has_proprio_token": "<|propri|>" in token_values,
        "has_action_token": "<|action|>" in token_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare wall-x and Edge-LLM prompt templates")
    parser.add_argument("--wallx-tokenizer-config", required=True)
    parser.add_argument("--qwen25-processed-template", required=True)
    parser.add_argument("--qwen25-tokenizer-config", required=True)
    parser.add_argument("--qwen3-processed-template", required=True)
    parser.add_argument("--qwen3-tokenizer-config", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = {
        "wallx": wallx_summary(Path(args.wallx_tokenizer_config)),
        "qwen25_edge_llm": edge_summary(
            Path(args.qwen25_processed_template), Path(args.qwen25_tokenizer_config)
        ),
        "qwen3_edge_llm": edge_summary(
            Path(args.qwen3_processed_template), Path(args.qwen3_tokenizer_config)
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
