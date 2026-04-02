from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LLMConfig:
    protocol: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_id: str = "gpt-4o"


@dataclass
class AgentConfig:
    max_rounds: int = 3
    top_k: int = 3


@dataclass
class PathsConfig:
    outputs_dir: str = "outputs"
    data_dir: str = "data"


@dataclass
class ArxivConfig:
    enable_rate_limit: bool = True
    min_interval_seconds: float = 3.0
    request_timeout_seconds: float = 120.0
    num_retries: int = 3
    page_size: int = 100


@dataclass
class Settings:
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        if config_path is None:
            # look for config.yaml in project root (two levels up from this file)
            candidate = Path(__file__).parent.parent.parent.parent / "config.yaml"
            config_path = str(candidate) if candidate.exists() else None

        raw: dict = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}

        agent_raw = raw.get("agent", {})
        llm_raw = raw.get("llm", {})
        paths_raw = raw.get("paths", {})
        arxiv_raw = raw.get("arxiv", {})

        # env var overrides api_key
        api_key = os.environ.get("PAPER_AGENT_API_KEY", llm_raw.get("api_key", ""))

        return cls(
            agent=AgentConfig(
                max_rounds=agent_raw.get("max_rounds", 3),
                top_k=agent_raw.get("top_k", 3),
            ),
            llm=LLMConfig(
                protocol=llm_raw.get("protocol", "openai"),
                base_url=llm_raw.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
                model_id=llm_raw.get("model_id", "gpt-4o"),
            ),
            paths=PathsConfig(
                outputs_dir=paths_raw.get("outputs_dir", "outputs"),
                data_dir=paths_raw.get("data_dir", "data"),
            ),
            arxiv=ArxivConfig(
                enable_rate_limit=arxiv_raw.get("enable_rate_limit", True),
                min_interval_seconds=arxiv_raw.get("min_interval_seconds", 3.0),
                request_timeout_seconds=arxiv_raw.get("request_timeout_seconds", 120.0),
                num_retries=arxiv_raw.get("num_retries", 3),
                page_size=arxiv_raw.get("page_size", 100),
            ),
        )
