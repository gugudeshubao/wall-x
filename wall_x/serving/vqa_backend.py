from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import re

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
EDGE_WALLX_VQA_DIR = REPO_ROOT / "workspace" / "edge_llm_wallx_vqa"
if str(EDGE_WALLX_VQA_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_WALLX_VQA_DIR))


def ensure_wallx_venv_site_packages() -> None:
    """Allow system Python to reuse the wall-x venv extension packages if needed."""
    import importlib.util

    if importlib.util.find_spec("wallx_csrc") is not None:
        return

    candidates = [
        Path("/data/wy/wall-x/venv/lib/python3.10/site-packages"),
        REPO_ROOT / "venv" / "lib" / "python3.10" / "site-packages",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break


class WallXVQABackend:
    def __init__(self, model_path: str, train_config: dict):
        self.device = self._setup_device()
        self.processor = self._load_processor(train_config["processor_path"])
        self.model = self._load_model(model_path, train_config)

    def _setup_device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_processor(self, model_path: str):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _load_model(self, model_path: str, train_config: dict):
        import torch

        ensure_wallx_venv_site_packages()
        from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
            Qwen2_5_VLMoEForAction,
        )

        model = Qwen2_5_VLMoEForAction.from_pretrained(
            model_path, train_config=train_config
        )
        if self.device == "cuda":
            model = model.to(self.device, dtype=torch.bfloat16)
        else:
            model.to(self.device)
        model.eval()
        return model

    def _prepare_inputs(self, image: Image.Image, text: str):
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
        return text_prompt, inputs

    def generate(self, image: Image.Image, text: str, **kwargs: Any) -> str:
        import torch

        _, inputs = self._prepare_inputs(image, text)
        generation_params = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            **kwargs,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_params)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def generate_with_metadata(self, image: Image.Image, text: str, **kwargs: Any) -> dict[str, Any]:
        import torch

        text_prompt, inputs = self._prepare_inputs(image, text)
        generation_params = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            **kwargs,
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_params)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {
            "backend": "wallx",
            "question": text,
            "formatted_complete_request": text_prompt,
            "latency_ms": latency_ms,
            "output_text": response,
        }


class EdgeRouterVQABackend:
    """Question-type router across the two measured Edge-LLM VQA backends."""

    def __init__(self, qwen25_backend_config: str, qwen3_backend_config: str):
        from vqa_wrapper_edge_llm import EdgeLLMVQAWrapper

        self.qwen25 = EdgeLLMVQAWrapper(backend_config_path=qwen25_backend_config)
        self.qwen3 = EdgeLLMVQAWrapper(backend_config_path=qwen3_backend_config)

    def _route(self, question: str) -> str:
        q = question.lower().strip()
        if any(
            key in q
            for key in [
                "predict the next action",
                "predict next action",
                "proprioception",
                "<|propri|>",
                "<|action|>",
                "robot action",
            ]
        ):
            return "qwen25"
        if any(
            key in q
            for key in [
                "what objects",
                "objects are on the table",
                "what is on the table",
                "how many",
                "list the objects",
            ]
        ):
            return "qwen3"
        if any(
            key in q
            for key in [
                "describe",
                "what do you see",
                "what is in the image",
                "what can you see",
            ]
        ):
            return "qwen25"
        return "qwen3"

    def generate(self, image: Image.Image, text: str, **kwargs: Any) -> str:
        return self.generate_with_metadata(image, text, **kwargs)["output_text"]

    def generate_with_metadata(self, image: Image.Image, text: str, **kwargs: Any) -> dict[str, Any]:
        route = self._route(text)
        backend = self.qwen3 if route == "qwen3" else self.qwen25
        result = backend.generate_with_metadata(image, text, **kwargs)
        result["backend"] = "edge_router"
        result["routed_backend"] = route
        return result


def build_minimal_wallx_train_config(model_path: str) -> dict:
    config_json = Path(model_path) / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Could not find {config_json}")
    model_cfg = json.loads(config_json.read_text())
    return {
        "processor_path": model_path,
        "data": {
            "use_state_string_representation": False,
            "action_horizon_flow": model_cfg.get("action_horizon_flow", 32),
        },
        "dof_config": model_cfg.get("dof_config", {}),
        "agent_pos_config": model_cfg.get("agent_pos_config", {}),
    }


def build_vqa_backend(
    *,
    backend: str,
    model_path: str | None = None,
    train_config_path: str | None = None,
    edge_backend_config: str | None = None,
    edge_backend_config_qwen25: str | None = None,
    edge_backend_config_qwen3: str | None = None,
):
    if backend == "edge":
        if not edge_backend_config:
            raise ValueError("edge backend requires edge_backend_config")
        from vqa_wrapper_edge_llm import EdgeLLMVQAWrapper

        return EdgeLLMVQAWrapper(backend_config_path=edge_backend_config)

    if backend == "edge_router":
        if not edge_backend_config_qwen25 or not edge_backend_config_qwen3:
            raise ValueError("edge_router backend requires both qwen25 and qwen3 backend configs")
        return EdgeRouterVQABackend(
            qwen25_backend_config=edge_backend_config_qwen25,
            qwen3_backend_config=edge_backend_config_qwen3,
        )

    if not model_path:
        raise ValueError("wallx backend requires model_path")

    train_config = None
    if train_config_path and os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg_yml = os.path.join(model_path, "config.yml")
        if os.path.exists(cfg_yml):
            with open(cfg_yml, "r") as f:
                train_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            train_config = build_minimal_wallx_train_config(model_path)
    return WallXVQABackend(model_path=model_path, train_config=train_config)
