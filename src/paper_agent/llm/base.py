from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> str:
        """Send messages and return the assistant reply as a string."""
