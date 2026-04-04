from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    estimated_request_count: int = 0


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> str:
        """Send messages and return the assistant reply as a string."""

    def get_token_usage(self) -> TokenUsage:
        return TokenUsage()
