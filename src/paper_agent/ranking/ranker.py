from __future__ import annotations

import json
import re
from typing import List

from ..ingestion.arxiv_fetcher import _canonicalize_arxiv_id
from ..llm.base import LLMClient, Message


_SYSTEM = """You are a research assistant helping trace the intellectual lineage of a paper.
Task: rank referenced papers by how important they are for tracing the paper's intellectual lineage.

Selection criteria, in order:
1. Direct method influence
2. Same core problem
3. Foundational prior work
4. Earlier papers when relevance is similar

Return ONLY a valid JSON array of up to top-{k} arXiv IDs in priority order.

Rules:
- Output IDs must come from the candidate list only.
- No explanation, no markdown, no extra text.
- Prefer precision over recall.
- Exclude weakly related papers.

Example:
["2301.00234", "2210.11416", "2005.14165"]"""


_EMPTY_REVIEW_PROMPT = """Your previous answer selected no papers.
Re-review the candidates and choose the most important prior work for lineage tracing.

Rules:
- Return a non-empty JSON array if any candidate is even moderately relevant.
- Return [] only if every candidate is clearly unrelated to the method or problem.
- Use candidate IDs only.
- No explanation."""


def _parse_ranked_ids(reply: str, candidates: List[dict], top_k: int) -> List[str]:
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[\w]*\n?", "", reply)
        reply = re.sub(r"\n?```$", "", reply)

    parsed = json.loads(reply)
    if not isinstance(parsed, list):
        raise ValueError("Ranking response must be a JSON array.")

    allowed_ids = {_canonicalize_arxiv_id(c["arxiv_id"]) for c in candidates}
    ranked_ids: List[str] = []
    seen: set[str] = set()
    for aid in parsed:
        if not isinstance(aid, str):
            continue
        normalized = _canonicalize_arxiv_id(aid)
        if normalized in allowed_ids and normalized not in seen:
            ranked_ids.append(normalized)
            seen.add(normalized)
        if len(ranked_ids) >= top_k:
            break
    return ranked_ids


def rank_candidates(
    llm: LLMClient,
    current_summary: str,
    candidates: List[dict],  # [{arxiv_id, title, abstract}, ...]
    top_k: int,
) -> List[str]:
    """Return up to top_k arXiv IDs ranked by relevance."""
    if not candidates:
        return []

    system_prompt = _SYSTEM.replace("{k}", str(top_k))

    candidates_text = "\n\n".join(
        f"ID: {c['arxiv_id']}\nTitle: {c['title']}\nAbstract: {c.get('abstract', '')[:280]}"
        for c in candidates
    )
    user_msg = (
        "Current paper:\n"
        f"{current_summary}\n\n"
        f"Top-K: {top_k}\n\n"
        "Candidate papers:\n"
        f"{candidates_text}"
    )

    reply = llm.chat([Message("system", system_prompt), Message("user", user_msg)])
    ids = _parse_ranked_ids(reply, candidates, top_k)
    if ids:
        return ids

    review_reply = llm.chat(
        [
            Message("system", system_prompt),
            Message("user", user_msg),
            Message("assistant", "[]"),
            Message("user", _EMPTY_REVIEW_PROMPT),
        ]
    )
    return _parse_ranked_ids(review_reply, candidates, top_k)
