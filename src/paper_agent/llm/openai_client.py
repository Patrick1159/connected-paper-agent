from __future__ import annotations

import json
from typing import List

import httpx

from .base import LLMClient, Message


class OpenAIClient(LLMClient):
    """OpenAI-compatible HTTP client (works with OpenAI, DeepSeek, Qwen, local servers, etc.)"""

    def __init__(self, base_url: str, api_key: str, model_id: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout

    def chat(self, messages: List[Message], **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [m.to_dict() for m in messages],
        }
        payload.update(kwargs)

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            content=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
