#!/usr/bin/env python3
"""Rule-based router across the two Edge-LLM VQA models we already benchmarked."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from edge_llm_vqa_backend import EdgeLLMVQABackend, EdgeLLMVQABackendConfig


@dataclass
class EdgeRuleRouterConfig:
    qwen25_backend_config: str
    qwen3_backend_config: str
    default_route: str = "qwen3"


class EdgeRuleRouter:
    """A very small text-rule router between Qwen2.5-VL and Qwen3-VL Edge-LLM backends."""

    def __init__(self, config: EdgeRuleRouterConfig):
        self.config = config
        self.qwen25 = EdgeLLMVQABackend(EdgeLLMVQABackendConfig(**self._load_config(config.qwen25_backend_config)))
        self.qwen3 = EdgeLLMVQABackend(EdgeLLMVQABackendConfig(**self._load_config(config.qwen3_backend_config)))

    def _load_config(self, path: str) -> dict[str, Any]:
        import json

        return json.loads(Path(path).read_text())

    def _route(self, question: str) -> str:
        q = question.lower().strip()
        if any(key in q for key in ["what objects", "objects are on the table", "what is on the table", "how many", "list the objects"]):
            return "qwen3"
        if any(key in q for key in ["describe", "what do you see", "what is in the image", "what can you see"]):
            return "qwen25"
        return self.config.default_route

    def generate_with_metadata(self, image: Image.Image, question: str, **kwargs: Any) -> dict[str, Any]:
        route = self._route(question)
        backend = self.qwen3 if route == "qwen3" else self.qwen25
        result = backend.generate_with_metadata(image, question, **kwargs)
        result["routed_backend"] = route
        return result

    def generate(self, image: Image.Image, question: str, **kwargs: Any) -> str:
        return self.generate_with_metadata(image, question, **kwargs)["output_text"]
