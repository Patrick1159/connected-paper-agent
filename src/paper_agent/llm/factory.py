from __future__ import annotations

from ..config.settings import LLMConfig
from .base import LLMClient
from .openai_client import OpenAIClient


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    if cfg.protocol in ("openai", "openai_compatible"):
        return OpenAIClient(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model_id=cfg.model_id,
        )
    raise ValueError(f"Unsupported LLM protocol: {cfg.protocol!r}. Supported: openai")
