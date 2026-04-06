from __future__ import annotations

import json
import logging
import re
from threading import Lock
from typing import Iterable, List, Optional

from ..arxiv_client import extract_arxiv_id_from_entry, search_arxiv_by_title
from ..ingestion.arxiv_fetcher import _canonicalize_arxiv_id
from ..llm.base import LLMClient, Message


logger = logging.getLogger(__name__)


_title_resolution_cache: dict[str, Optional[str]] = {}
_title_resolution_cache_lock = Lock()


_QUOTED_TITLE_RE = re.compile(r'["“](.+?)["”]')
_REF_PREFIX_RE = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s*")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_DOI_RE = re.compile(r"doi\s*:?\s*\S+", re.IGNORECASE)
_ARXIV_TOKEN_RE = re.compile(r"(?:arXiv:|arxiv\.org/(?:abs|pdf)/)\S+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VENUE_HINT_RE = re.compile(
    r"\b(?:in|proc(?:eedings)?|proceedings of|conference|journal|workshop|symposium|"
    r"ieee|acm|springer|elsevier|cvpr|iccv|eccv|neurips|nips|icra|iros|rss|aaai|ijcai|"
    r"transactions|letters|robotics and automation|pattern analysis|computer vision)\b",
    re.IGNORECASE,
)
_PAGE_HINT_RE = re.compile(r"\b(?:pp?\.?|pages?|vol\.?|no\.?|isbn|issn)\b", re.IGNORECASE)
_AUTHOR_INITIAL_RE = re.compile(r"\b[A-Z]\.")

_TITLE_PARSE_SYSTEM = """You extract paper titles from raw bibliography references.

Return valid JSON only as an array of objects:
[
    {"reference_index": 0, "title": "<paper title or null>"}
]

Rules:
- `reference_index` must match the input.
- `title` must be only the paper title, without authors, venue, year, pages, DOI, or arXiv ID.
- If the title cannot be identified reliably, use null.
- Prefer precision over recall.
- No markdown, no extra text."""

_TITLE_PREFILTER_SYSTEM = """You rank paper titles by likely relevance to a source paper.

Return valid JSON only as an array of objects:
[
    {"title": "<exact title>", "reason": "<short reason>"}
]

Rules:
- Choose at most the requested number of titles.
- Use only titles from the provided candidate list.
- Keep each `title` exactly unchanged from the input list.
- Prefer titles that look most related to the source paper's problem or method.
- If uncertain, return fewer titles instead of guessing.
- No markdown, no extra text."""

_DEFAULT_TITLE_PARSE_BATCH_SIZE = 12
_DEFAULT_TITLE_QUERY_BATCH_SIZE = 4
def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
                text = re.sub(r"^```[\w]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
        return text.strip()


def _clean_reference_text(reference: str) -> str:
    text = _REF_PREFIX_RE.sub("", reference)
    text = _URL_RE.sub(" ", text)
    text = _DOI_RE.sub(" ", text)
    text = _ARXIV_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;,")


def _looks_like_title(candidate: str) -> bool:
    candidate = candidate.strip(" .;,")
    if not candidate:
        return False

    words = candidate.split()
    if len(words) < 4 or len(words) > 32:
        return False
    if not re.search(r"[A-Za-z]", candidate):
        return False
    if _VENUE_HINT_RE.search(candidate) or _PAGE_HINT_RE.search(candidate):
        return False
    if len(_AUTHOR_INITIAL_RE.findall(candidate)) >= 3:
        return False
    if candidate.count(",") >= 4:
        return False
    if re.search(r"\b(?:et al\.?|and)\b", candidate, re.IGNORECASE) and len(words) < 6:
        return False

    alpha_tokens = sum(1 for w in words if re.search(r"[A-Za-z]", w))
    if alpha_tokens < 4:
        return False
    return True


def _iter_reference_title_candidates(reference: str) -> Iterable[str]:
    text = _clean_reference_text(reference)
    if not text:
        return []

    candidates: List[str] = []

    for match in _QUOTED_TITLE_RE.findall(text):
        title = re.sub(r"\s+", " ", match).strip(" .;,")
        if title:
            candidates.append(title)

    before_year = _YEAR_RE.split(text, maxsplit=1)[0].strip(" .;,")
    work_text = before_year or text

    # Try sentence-like segments first.
    sentence_parts = [
        seg.strip(" .;,")
        for seg in re.split(r"(?<=[.!?])\s+(?=[A-Z])", work_text)
        if seg.strip(" .;,")
    ]
    candidates.extend(sentence_parts)

    # Then try comma-delimited windows to handle common bibliography formats.
    comma_parts = [seg.strip(" .;,") for seg in re.split(r",\s*", work_text) if seg.strip(" .;,")]
    for window_size in (1, 2, 3):
        for idx in range(len(comma_parts) - window_size + 1):
            piece = ", ".join(comma_parts[idx: idx + window_size]).strip(" .;,")
            if piece:
                candidates.append(piece)

    # Keep order, drop duplicates.
    seen: dict[str, None] = {}
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip(" .;,")
        if normalized and normalized not in seen:
            seen[normalized] = None
    return seen.keys()


def _score_title_candidate(candidate: str) -> int:
    score = 0
    words = candidate.split()
    if 5 <= len(words) <= 18:
        score += 3
    elif 4 <= len(words) <= 24:
        score += 1
    if re.search(r"[a-z]", candidate):
        score += 1
    if not _VENUE_HINT_RE.search(candidate):
        score += 2
    if not _PAGE_HINT_RE.search(candidate):
        score += 1
    if len(_AUTHOR_INITIAL_RE.findall(candidate)) == 0:
        score += 2
    elif len(_AUTHOR_INITIAL_RE.findall(candidate)) == 1:
        score += 1
    if candidate.count(",") <= 1:
        score += 1
    if _YEAR_RE.search(candidate):
        score -= 2
    if re.search(r"\b(?:et al\.?|and)\b", candidate, re.IGNORECASE):
        score -= 1
    return score


def _is_plausible_title(title: str) -> bool:
    title = re.sub(r"\s+", " ", title).strip(" .;,")
    if not title:
        return False
    if not _looks_like_title(title):
        return False
    lowered = title.lower()
    if lowered.startswith(("http://", "https://", "doi", "arxiv:")):
        return False
    return True


def _normalize_title_cache_key(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip(" .;,").casefold()


def extract_title_from_reference(reference: str) -> Optional[str]:
    """Best-effort extraction of a paper title from a raw reference block."""
    best_title: Optional[str] = None
    best_score = float("-inf")

    for candidate in _iter_reference_title_candidates(reference):
        if not _looks_like_title(candidate):
            continue
        score = _score_title_candidate(candidate)
        if score > best_score:
            best_title = candidate
            best_score = score

    return best_title


def _parse_titles_with_llm(
    llm: LLMClient,
    references_raw: List[str],
    batch_size: int = _DEFAULT_TITLE_PARSE_BATCH_SIZE,
) -> List[str]:
    titles: List[str] = []
    seen: dict[str, None] = {}

    for start in range(0, len(references_raw), batch_size):
        batch = references_raw[start:start + batch_size]
        payload = [
            {"reference_index": start + idx, "reference": _clean_reference_text(ref)}
            for idx, ref in enumerate(batch)
        ]
        user_msg = json.dumps(payload, ensure_ascii=False)

        try:
            reply = llm.chat(
                [Message("system", _TITLE_PARSE_SYSTEM), Message("user", user_msg)],
                temperature=0,
            )
            data = json.loads(_strip_code_fences(reply))
            if not isinstance(data, list):
                raise ValueError("Title parsing reply must be a JSON array.")

            for item in data:
                if not isinstance(item, dict):
                    continue
                title = item.get("title")
                if not isinstance(title, str):
                    continue
                title = re.sub(r"\s+", " ", title).strip(" .;,")
                if _is_plausible_title(title) and title not in seen:
                    seen[title] = None
                    titles.append(title)
        except Exception as exc:
            logger.warning("[refs] LLM title parsing failed for batch starting at %s: %s", start, exc)
            for ref in batch:
                fallback = extract_title_from_reference(ref)
                if fallback and fallback not in seen:
                    seen[fallback] = None
                    titles.append(fallback)

    return titles


def _shortlist_titles_with_llm(
    llm: LLMClient,
    source_title: str,
    candidate_titles: List[str],
    shortlist_size: Optional[int],
) -> List[str]:
    if shortlist_size is None or shortlist_size <= 0:
        return []
    if len(candidate_titles) <= shortlist_size:
        return list(candidate_titles)

    payload = {
        "source_title": source_title,
        "max_titles": shortlist_size,
        "candidate_titles": candidate_titles,
    }

    try:
        reply = llm.chat(
            [
                Message("system", _TITLE_PREFILTER_SYSTEM),
                Message("user", json.dumps(payload, ensure_ascii=False)),
            ],
            temperature=0,
        )
        data = json.loads(_strip_code_fences(reply))
        if not isinstance(data, list):
            raise ValueError("Title shortlist reply must be a JSON array.")

        candidate_lookup = {_normalize_title_cache_key(title): title for title in candidate_titles}
        shortlisted: List[str] = []
        seen: dict[str, None] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if not isinstance(title, str):
                continue
            normalized = _normalize_title_cache_key(title)
            exact_title = candidate_lookup.get(normalized)
            if exact_title and normalized not in seen:
                seen[normalized] = None
                shortlisted.append(exact_title)
            if len(shortlisted) >= shortlist_size:
                break

        if shortlisted:
            logger.info(
                "[refs] LLM shortlisted %s/%s parsed titles before arXiv resolution",
                len(shortlisted),
                len(candidate_titles),
            )
            return shortlisted
    except Exception as exc:
        logger.warning("[refs] LLM title shortlist failed: %s", exc)

    fallback_titles = list(candidate_titles[:shortlist_size])
    logger.info(
        "[refs] Falling back to first %s/%s parsed titles before arXiv resolution",
        len(fallback_titles),
        len(candidate_titles),
    )
    return fallback_titles


def resolve_arxiv_id_by_title(title: str) -> Optional[str]:
    """Best-effort: search arXiv by title, return ID of top result."""
    cache_key = _normalize_title_cache_key(title)
    with _title_resolution_cache_lock:
        if cache_key in _title_resolution_cache:
            return _title_resolution_cache[cache_key]

    try:
        results = search_arxiv_by_title(title, max_results=1)
        if not results:
            with _title_resolution_cache_lock:
                _title_resolution_cache[cache_key] = None
            return None
        arxiv_id = extract_arxiv_id_from_entry(results[0].entry_id)
        canonical_arxiv_id = _canonicalize_arxiv_id(arxiv_id) if arxiv_id else None
        with _title_resolution_cache_lock:
            _title_resolution_cache[cache_key] = canonical_arxiv_id
        return canonical_arxiv_id
    except Exception as exc:
        logger.warning("[refs] arXiv title search failed for %r: %s", title, exc)
        return None


def _resolve_titles_in_batches(
    titles: List[str],
    *,
    max_title_resolutions: Optional[int],
    title_query_batch_size: int,
) -> List[str]:
    if max_title_resolutions is not None:
        titles = titles[:max_title_resolutions]

    resolved_ids: List[str] = []

    for batch_start in range(0, len(titles), title_query_batch_size):
        batch = titles[batch_start:batch_start + title_query_batch_size]
        logger.info(
            "[refs] Resolving parsed title batch %s-%s/%s",
            batch_start + 1,
            min(batch_start + len(batch), len(titles)),
            len(titles),
        )
        for title in batch:
            aid = resolve_arxiv_id_by_title(title)
            if aid:
                resolved_ids.append(aid)

    return resolved_ids


def extract_candidate_ids(
    llm: LLMClient,
    source_title: str,
    arxiv_ids_from_pdf: List[str],
    references_raw: List[str],
    llm_suggested_titles: Optional[List[str]] = None,
    max_title_resolutions: Optional[int] = None,
    title_shortlist_size: Optional[int] = None,
    title_parse_batch_size: int = _DEFAULT_TITLE_PARSE_BATCH_SIZE,
    title_query_batch_size: int = _DEFAULT_TITLE_QUERY_BATCH_SIZE,
) -> List[str]:
    """
    Merge arXiv IDs found directly in PDF with any resolved from titles.
    Returns a deduplicated list of arXiv IDs.
    """
    seen: dict[str, None] = {}
    for aid in arxiv_ids_from_pdf:
        seen[_canonicalize_arxiv_id(aid)] = None

    titles_to_resolve = _parse_titles_with_llm(
        llm,
        references_raw,
        batch_size=title_parse_batch_size,
    )
    titles_to_resolve = _shortlist_titles_with_llm(
        llm,
        source_title,
        titles_to_resolve,
        title_shortlist_size,
    )
    resolved_ids = _resolve_titles_in_batches(
        titles_to_resolve,
        max_title_resolutions=max_title_resolutions,
        title_query_batch_size=title_query_batch_size,
    )

    for aid in resolved_ids:
        if aid:
            seen[_canonicalize_arxiv_id(aid)] = None

    if llm_suggested_titles:
        for title in llm_suggested_titles:
            aid = resolve_arxiv_id_by_title(title)
            if aid and _canonicalize_arxiv_id(aid) not in seen:
                seen[_canonicalize_arxiv_id(aid)] = None

    return list(seen.keys())
