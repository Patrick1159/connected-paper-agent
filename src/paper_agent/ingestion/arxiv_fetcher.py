from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from ..arxiv_client import download_pdf_file, fetch_arxiv_paper_by_id


_metadata_cache: dict[str, "PaperMeta"] = {}
_metadata_cache_lock = Lock()


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    pdf_url: str
    pdf_path: Optional[str] = None  # local path after download


def _normalize_arxiv_id(url_or_id: str) -> str:
    """Extract bare arXiv ID from URL or raw ID string."""
    # e.g. https://arxiv.org/abs/2305.10601 or https://arxiv.org/pdf/2305.10601
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.]+)", url_or_id, re.IGNORECASE)
    if m:
        return m.group(1).removesuffix(".pdf")
    # bare id like 2305.10601 or arxiv:2305.10601
    bare = re.sub(r"^arxiv:", "", url_or_id, flags=re.IGNORECASE).strip()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", bare):
        return bare
    raise ValueError(f"Cannot parse arXiv ID from: {url_or_id!r}")


def _canonicalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize an arXiv ID to its versionless form."""
    return re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)


def fetch_metadata(url_or_id: str) -> PaperMeta:
    arxiv_id = _canonicalize_arxiv_id(_normalize_arxiv_id(url_or_id))

    with _metadata_cache_lock:
        cached = _metadata_cache.get(arxiv_id)
    if cached is not None:
        return PaperMeta(
            arxiv_id=cached.arxiv_id,
            title=cached.title,
            authors=list(cached.authors),
            abstract=cached.abstract,
            year=cached.year,
            pdf_url=cached.pdf_url,
            pdf_path=cached.pdf_path,
        )

    result = fetch_arxiv_paper_by_id(arxiv_id)
    if not result:
        raise ValueError(f"arXiv paper not found: {arxiv_id}")
    meta = PaperMeta(
        arxiv_id=arxiv_id,
        title=result.title,
        authors=[a.name for a in result.authors],
        abstract=result.summary.replace("\n", " "),
        year=result.published.year,
        pdf_url=result.pdf_url,
    )
    with _metadata_cache_lock:
        _metadata_cache[arxiv_id] = PaperMeta(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            authors=list(meta.authors),
            abstract=meta.abstract,
            year=meta.year,
            pdf_url=meta.pdf_url,
            pdf_path=meta.pdf_path,
        )
    return meta


def download_pdf(meta: PaperMeta, dest_dir: str = "data") -> str:
    path = Path(dest_dir) / f"{meta.arxiv_id.replace('/', '_')}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        meta.pdf_path = str(path)
        return str(path)
    download_pdf_file(meta.pdf_url, path)
    meta.pdf_path = str(path)
    return str(path)
