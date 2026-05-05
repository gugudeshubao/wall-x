#!/usr/bin/env python3
"""Drop-in style VQA wrapper backed by TensorRT-Edge-LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from edge_llm_vqa_backend import EdgeLLMVQABackend, EdgeLLMVQABackendConfig


class EdgeLLMVQAWrapper:
    """Wrapper with a `generate(image, text, **kwargs)` API similar to wall-x VQAWrapper."""

    def __init__(
        self,
        backend_config_path: str | None = None,
        backend_config: dict[str, Any] | None = None,
    ):
        if backend_config is None:
            if backend_config_path is None:
                raise ValueError("Either backend_config_path or backend_config must be provided.")
            backend_config = json.loads(Path(backend_config_path).read_text())
        self.backend_config = backend_config
        self.backend = EdgeLLMVQABackend(
            EdgeLLMVQABackendConfig(**backend_config)
        )

    def generate(self, image: Image.Image, text: str, **kwargs: Any) -> str:
        return self.backend.generate(image, text, **kwargs)

    def generate_with_metadata(self, image: Image.Image, text: str, **kwargs: Any) -> dict[str, Any]:
        return self.backend.generate_with_metadata(image, text, **kwargs)
