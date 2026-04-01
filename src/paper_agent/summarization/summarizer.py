from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm.base import LLMClient, Message


@dataclass
class PaperSummary:
    core_idea: str
    method: str
    problem_solved: str
    field: str


_SYSTEM = """You are a research assistant that analyzes academic papers.
Task: extract only the essential research information from the paper.

Return valid JSON only with exactly these keys:
{
    "core_idea": "<1-2 sentences on the main contribution>",
    "method": "<short phrase or 1 sentence on the method>",
    "problem_solved": "<1 sentence on the target problem>",
    "field": "<short field label>"
}
Rules:
- Be precise, concise, and result-oriented.
- Use plain English.
- Do not add markdown, code fences, or extra keys.
- Do not mention missing information unless necessary.
- Prefer concrete method/problem statements over generic wording."""


def summarize(llm: LLMClient, title: str, abstract: str) -> PaperSummary:
    user_msg = (
        "Summarize this paper for literature tracing.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )
    reply = llm.chat([Message("system", _SYSTEM), Message("user", user_msg)])

    # Strip markdown code fences if present
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[\w]*\n?", "", reply)
        reply = re.sub(r"\n?```$", "", reply)

    data = json.loads(reply)
    return PaperSummary(
        core_idea=data.get("core_idea", ""),
        method=data.get("method", ""),
        problem_solved=data.get("problem_solved", ""),
        field=data.get("field", ""),
    )

