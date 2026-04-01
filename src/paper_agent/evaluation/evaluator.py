from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..llm.base import LLMClient, Message


_SYSTEM = """You are a research historian analyzing the intellectual lineage of a research area.
Task: choose the single best lineage chain ending at the root paper.

Priorities:
1. Clear method evolution
2. Same problem setting
3. Chronological progression from older to newer
4. A focused chain, not a broad set

Return valid JSON only:
{
  "chain": ["arxiv_id_oldest", ..., "arxiv_id_newest"],
    "rationale": "<2-4 sentences explaining the evolution>"
}

Rules:
- The last item in `chain` should be the root paper when possible.
- Use only IDs from the provided paper list.
- No markdown, no extra text, no extra keys.
- Keep the rationale concise and evidence-based."""


def select_best_lineage(
    llm: LLMClient,
    nodes: List[Dict[str, Any]],
    root_id: str,
) -> dict:
    """Return {chain: [arxiv_id, ...], rationale: str}."""
    analyzed = [n for n in nodes if n.get("status") == "analyzed"]
    if not analyzed:
        return {"chain": [root_id], "rationale": "Only the root paper was analyzed."}

    nodes_text = "\n\n".join(
        f"[{n['arxiv_id']}] ({n.get('year', '?')}) {n.get('title', '')}\n"
        f"Method: {n.get('method', '')}\n"
        f"Problem: {n.get('problem_solved', '')}"
        for n in analyzed
    )
    user_msg = (
        f"Root paper: {root_id}\n\n"
        "Analyzed papers:\n"
        f"{nodes_text}"
    )

    reply = llm.chat([Message("system", _SYSTEM), Message("user", user_msg)])
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[\w]*\n?", "", reply)
        reply = re.sub(r"\n?```$", "", reply)

    return json.loads(reply)
