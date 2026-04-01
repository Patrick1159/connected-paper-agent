from __future__ import annotations

import re
from typing import List, Optional

import arxiv

from ..ingestion.arxiv_fetcher import _canonicalize_arxiv_id


def resolve_arxiv_id_by_title(title: str) -> Optional[str]:
    """Best-effort: search arXiv by title, return ID of top result."""
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(query=f'ti:"{title}"', max_results=1)))
    if not results:
        return None
    url = results[0].entry_id  # e.g. http://arxiv.org/abs/2305.10601v1
    m = re.search(r"/abs/([\d.]+)", url)
    return m.group(1) if m else None


def extract_candidate_ids(
    arxiv_ids_from_pdf: List[str],
    references_raw: List[str],
    llm_suggested_titles: Optional[List[str]] = None,
) -> List[str]:
    """
    Merge arXiv IDs found directly in PDF with any resolved from titles.
    Returns a deduplicated list of arXiv IDs.
    """
    seen: dict[str, None] = {}
    for aid in arxiv_ids_from_pdf:
        seen[_canonicalize_arxiv_id(aid)] = None

    if llm_suggested_titles:
        for title in llm_suggested_titles:
            aid = resolve_arxiv_id_by_title(title)
            if aid and aid not in seen:
                seen[_canonicalize_arxiv_id(aid)] = None

    return list(seen.keys())
