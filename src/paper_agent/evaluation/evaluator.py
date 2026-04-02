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
- Every adjacent pair in `chain` must follow a real citation edge from the provided graph.
- Use only IDs from the provided paper list.
- No markdown, no extra text, no extra keys.
- Keep the rationale concise and evidence-based."""


def _heuristic_lineage(root_id: str, nodes: List[Dict[str, Any]], edges: List[tuple[str, str]]) -> dict:
    analyzed_ids = {n["arxiv_id"] for n in nodes if n.get("status") == "analyzed"}
    adjacency: dict[str, List[str]] = {}
    for citing, cited in edges:
        if citing in analyzed_ids and cited in analyzed_ids:
            adjacency.setdefault(citing, []).append(cited)

    def dfs(current: str) -> List[str]:
        best = [current]
        for parent in adjacency.get(current, []):
            candidate = [current, *dfs(parent)]
            if len(candidate) > len(best):
                best = candidate
        return best

    if root_id not in analyzed_ids:
        return {"chain": [root_id], "rationale": "Only the root paper was analyzed."}

    newest_to_oldest = dfs(root_id)
    chain = list(reversed(newest_to_oldest))
    if len(chain) == 1:
        rationale = "No analyzed ancestor connected by citation edges was available, so the lineage remains the root paper only."
    else:
        rationale = "This lineage follows the longest analyzed citation path ending at the root paper and respects the observed citation edges in the traced graph."
    return {"chain": chain, "rationale": rationale}


def _is_valid_chain(chain: List[str], analyzed_ids: set[str], edges: List[tuple[str, str]], root_id: str) -> bool:
    if not chain or any(aid not in analyzed_ids for aid in chain):
        return False
    if chain[-1] != root_id:
        return False
    edge_set = set(edges)
    return all((newer, older) in edge_set for older, newer in zip(chain, chain[1:]))


def select_best_lineage(
    llm: LLMClient,
    nodes: List[Dict[str, Any]],
    edges: List[tuple[str, str]],
    root_id: str,
) -> dict:
    """Return {chain: [arxiv_id, ...], rationale: str}."""
    analyzed = [n for n in nodes if n.get("status") == "analyzed"]
    if not analyzed:
        return {"chain": [root_id], "rationale": "Only the root paper was analyzed."}

    analyzed_ids = {n["arxiv_id"] for n in analyzed}
    analyzed_edges = [
        (citing, cited) for citing, cited in edges
        if citing in analyzed_ids and cited in analyzed_ids
    ]
    heuristic = _heuristic_lineage(root_id, analyzed, analyzed_edges)

    nodes_text = "\n\n".join(
        f"[{n['arxiv_id']}] ({n.get('year', '?')}) {n.get('title', '')}\n"
        f"Method: {n.get('method', '')}\n"
        f"Problem: {n.get('problem_solved', '')}"
        for n in analyzed
    )
    edges_text = "\n".join(
        f"{citing} -> {cited}"
        for citing, cited in analyzed_edges
    ) or "(no analyzed citation edges available)"
    user_msg = (
        f"Root paper: {root_id}\n\n"
        "Citation edges (citing -> cited):\n"
        f"{edges_text}\n\n"
        "Analyzed papers:\n"
        f"{nodes_text}"
    )

    reply = llm.chat([Message("system", _SYSTEM), Message("user", user_msg)])
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[\w]*\n?", "", reply)
        reply = re.sub(r"\n?```$", "", reply)

    result = json.loads(reply)
    chain = result.get("chain", [])
    if not _is_valid_chain(chain, analyzed_ids, analyzed_edges, root_id):
        return heuristic
    if len(chain) == 1 and len(heuristic.get("chain", [])) > 1:
        return heuristic
    return result
