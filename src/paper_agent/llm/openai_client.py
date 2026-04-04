from __future__ import annotations

import json
from threading import Lock
from typing import List

import httpx

from .base import LLMClient, Message, TokenUsage


class OpenAIClient(LLMClient):
    """OpenAI-compatible HTTP client (works with OpenAI, DeepSeek, Qwen, local servers, etc.)"""

    def __init__(self, base_url: str, api_key: str, model_id: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self._usage = TokenUsage()
        self._usage_lock = Lock()

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        normalized = text.strip()
        if not normalized:
            return 0
        return max(1, (len(normalized) + 3) // 4)

    def _record_usage(self, messages: List[Message], reply: str, usage: dict | None) -> None:
        prompt_tokens = 0
        completion_tokens = 0
        estimated = False

        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)

        if prompt_tokens <= 0 and completion_tokens <= 0:
            estimated = True
            prompt_tokens = sum(self._estimate_token_count(message.content) for message in messages)
            completion_tokens = self._estimate_token_count(reply)

        total_tokens = prompt_tokens + completion_tokens
        with self._usage_lock:
            self._usage.prompt_tokens += prompt_tokens
            self._usage.completion_tokens += completion_tokens
            self._usage.total_tokens += total_tokens
            self._usage.request_count += 1
            if estimated:
                self._usage.estimated_request_count += 1

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
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        self._record_usage(messages, reply, data.get("usage"))
        return reply

    def get_token_usage(self) -> TokenUsage:
        with self._usage_lock:
            return TokenUsage(
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens,
                total_tokens=self._usage.total_tokens,
                request_count=self._usage.request_count,
                estimated_request_count=self._usage.estimated_request_count,
            )
