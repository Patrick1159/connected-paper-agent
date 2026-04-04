from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..llm.base import TokenUsage


def _format_duration(total_seconds: float) -> str:
    seconds = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def render_report(
    root_id: str,
    chain: List[str],
    rationale: str,
    nodes: List[Dict[str, Any]],
    skipped: List[str],
    elapsed_seconds: float = 0.0,
    token_usage: TokenUsage | None = None,
) -> str:
    node_map = {n["arxiv_id"]: n for n in nodes}
    lines: List[str] = []

    lines.append("# Research Literature Trace Report")
    lines.append(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append(f"\n**Root paper:** `{root_id}`")
    lines.append(f"**Total search time:** {_format_duration(elapsed_seconds)}")
    if root_id in node_map:
        root = node_map[root_id]
        lines.append(f"**Title:** {root.get('title', 'N/A')}")

    usage = token_usage or TokenUsage()
    lines.append("\n## Run Statistics\n")
    lines.append(f"- LLM requests: {usage.request_count}")
    lines.append(f"- Prompt tokens: {usage.prompt_tokens}")
    lines.append(f"- Completion tokens: {usage.completion_tokens}")
    lines.append(f"- Total tokens: {usage.total_tokens}")
    if usage.estimated_request_count:
        lines.append(
            f"- Estimated token requests: {usage.estimated_request_count}"
        )

    lines.append("\n---\n")
    lines.append("## Best Lineage Chain\n")
    lines.append(rationale)
    lines.append("")

    for i, aid in enumerate(chain):
        n = node_map.get(aid, {})
        title = n.get("title") or aid
        year = n.get("year", "?")
        lines.append(f"{i + 1}. **[{aid}]** ({year}) {title}")
    lines.append("")

    lines.append("---\n")
    lines.append("## Reading List & Paper Summaries\n")

    for aid in chain:
        n = node_map.get(aid, {})
        title = n.get("title") or aid
        year = n.get("year", "?")
        authors = ", ".join(n.get("authors") or [])
        lines.append(f"### [{aid}] {title} ({year})")
        if authors:
            lines.append(f"_Authors: {authors}_\n")
        if n.get("abstract"):
            lines.append(f"**Abstract:** {n['abstract']}\n")
        if n.get("problem_solved"):
            lines.append(f"**Problem solved:** {n['problem_solved']}\n")
        if n.get("method"):
            lines.append(f"**Method:** {n['method']}\n")
        if n.get("core_idea"):
            lines.append(f"**Core idea:** {n['core_idea']}\n")
        lines.append(f"https://arxiv.org/abs/{aid}\n")

    if skipped:
        lines.append("---\n")
        lines.append("## Papers Not Available on arXiv\n")
        for s in skipped:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


def save_report(content: str, outputs_dir: str, root_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = root_id.replace("/", "_")
    path = Path(outputs_dir) / f"report_{safe_id}_{ts}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)
