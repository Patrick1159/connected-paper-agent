from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    # Input
    root_url: str = ""
    root_id: str = ""

    # Runtime
    frontier: List[str] = field(default_factory=list)   # arXiv IDs yet to be processed
    current_round: int = 0
    skipped_refs: List[str] = field(default_factory=list)  # titles/ids not on arXiv
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Output
    lineage_chain: List[str] = field(default_factory=list)
    lineage_rationale: str = ""
    report_path: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
