from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional, TypeVar

import arxiv
import httpx

from .config.settings import ArxivConfig
from .rate_limit import GlobalRateLimiter


logger = logging.getLogger(__name__)

T = TypeVar("T")


_config: Optional[ArxivConfig] = None
_rate_limiter = GlobalRateLimiter(0.0)
_arxiv_client: Optional[arxiv.Client] = None
_http_client: Optional[httpx.Client] = None


def configure_arxiv_access(config: ArxivConfig) -> None:
    global _config, _rate_limiter, _arxiv_client, _http_client

    _config = config
    _rate_limiter = GlobalRateLimiter(config.min_interval_seconds if config.enable_rate_limit else 0.0)
    _arxiv_client = arxiv.Client(
        page_size=config.page_size,
        delay_seconds=0.0,
        num_retries=config.num_retries,
    )

    if _http_client is not None:
        _http_client.close()
    _http_client = httpx.Client(
        follow_redirects=True,
        timeout=config.request_timeout_seconds,
    )

    logger.info(
        "[arxiv] Configured shared access: enabled=%s interval=%.2fs timeout=%.1fs retries=%s",
        config.enable_rate_limit,
        config.min_interval_seconds,
        config.request_timeout_seconds,
        config.num_retries,
    )


def _ensure_configured() -> None:
    if _config is None:
        configure_arxiv_access(ArxivConfig())


def _get_arxiv_client() -> arxiv.Client:
    _ensure_configured()
    assert _arxiv_client is not None
    return _arxiv_client


def _get_http_client() -> httpx.Client:
    _ensure_configured()
    assert _http_client is not None
    return _http_client


def _run_rate_limited(label: str, operation: Callable[[], T]) -> T:
    _ensure_configured()
    _rate_limiter.acquire(label)
    return operation()


def fetch_arxiv_results(search: arxiv.Search) -> list[arxiv.Result]:
    client = _get_arxiv_client()
    return _run_rate_limited("arXiv API query", lambda: list(client.results(search)))


def fetch_arxiv_paper_by_id(arxiv_id: str) -> Optional[arxiv.Result]:
    results = fetch_arxiv_results(arxiv.Search(id_list=[arxiv_id]))
    return results[0] if results else None


def search_arxiv_by_title(title: str, max_results: int = 1) -> list[arxiv.Result]:
    return fetch_arxiv_results(arxiv.Search(query=f'ti:"{title}"', max_results=max_results))


def extract_arxiv_id_from_entry(entry_id: str) -> Optional[str]:
    match = re.search(r"/abs/([\d.]+)", entry_id)
    return match.group(1) if match else None


def download_pdf_file(pdf_url: str, destination: str | Path) -> str:
    client = _get_http_client()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _download() -> str:
        with client.stream("GET", pdf_url) as response:
            response.raise_for_status()
            with open(target, "wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return str(target)

    return _run_rate_limited("arXiv PDF download", _download)
