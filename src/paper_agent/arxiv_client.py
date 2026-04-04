from __future__ import annotations

import logging
import random
import re
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
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
_connection_lock = Lock()
_results_cache: dict[str, "_CacheEntry[list[arxiv.Result]]"] = {}


@dataclass
class _CacheEntry:
    value: T
    expires_at: float


def configure_arxiv_access(config: ArxivConfig) -> None:
    global _config, _rate_limiter, _arxiv_client, _http_client, _results_cache

    _config = config
    _rate_limiter = GlobalRateLimiter(config.min_interval_seconds if config.enable_rate_limit else 0.0)
    _results_cache = {}
    _arxiv_client = arxiv.Client(
        page_size=config.page_size,
        delay_seconds=0.0,
        num_retries=0,
    )

    if _http_client is not None:
        _http_client.close()
    _http_client = httpx.Client(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        timeout=config.request_timeout_seconds,
    )

    logger.info(
        "[arxiv] Configured shared access: enabled=%s interval=%.2fs single_connection=%s cache_ttl=%.0fs timeout=%.1fs retries=%s retry_429=%s retry_network=%s",
        config.enable_rate_limit,
        config.min_interval_seconds,
        config.single_connection_only,
        config.cache_ttl_seconds,
        config.request_timeout_seconds,
        config.num_retries,
        config.retry_on_429,
        config.retry_on_network_errors,
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
    assert _config is not None

    if _config.single_connection_only:
        with _connection_lock:
            _rate_limiter.acquire(label)
            return operation()

    _rate_limiter.acquire(label)
    return operation()


def _is_http_429_error(error: Exception) -> bool:
    if isinstance(error, arxiv.HTTPError):
        return error.status == 429
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429
    return False


def _is_retryable_http_error(error: Exception) -> bool:
    if isinstance(error, arxiv.HTTPError):
        return error.status in {408, 429} or 500 <= error.status < 600
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code in {408, 429} or 500 <= status_code < 600
    return False


def _is_retryable_network_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
            httpx.ProtocolError,
            socket.timeout,
            TimeoutError,
            ssl.SSLError,
            ConnectionError,
        ),
    ):
        return True

    message = str(error).lower()
    retryable_markers = (
        "ssleoferror",
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
        "max retries exceeded",
        "connection aborted",
        "connection reset",
        "remote end closed connection",
        "temporary failure",
        "temporarily unavailable",
        "read timed out",
        "timed out",
    )
    return any(marker in message for marker in retryable_markers)


def _classify_retryable_error(error: Exception) -> Optional[str]:
    assert _config is not None

    if _config.retry_on_429 and _is_http_429_error(error):
        return "HTTP 429"
    if _config.retry_on_network_errors and _is_retryable_network_error(error):
        return "network error"
    if _is_retryable_http_error(error):
        return "HTTP error"
    return None


def _compute_backoff_seconds(attempt_index: int) -> float:
    assert _config is not None
    min_seconds = max(
        0.0,
        _config.backoff_min_seconds,
        _config.min_interval_seconds if _config.enable_rate_limit else 0.0,
    )
    max_seconds = max(min_seconds, _config.backoff_max_seconds)
    multiplier = max(1.0, _config.backoff_multiplier)
    base = min_seconds if min_seconds > 0 else 1.0
    wait_seconds = min(max_seconds, base * (multiplier ** attempt_index))
    jitter = random.uniform(0.0, max(0.0, _config.backoff_jitter_seconds))
    return min(max_seconds, wait_seconds + jitter)


def _run_with_retries(label: str, operation: Callable[[], T]) -> T:
    _ensure_configured()
    assert _config is not None

    max_retries = max(0, _config.num_retries)
    attempt = 0

    while True:
        try:
            return _run_rate_limited(label, operation)
        except Exception as error:
            error_kind = _classify_retryable_error(error)
            if error_kind is None or attempt >= max_retries:
                raise

            wait_seconds = _compute_backoff_seconds(attempt)
            logger.warning(
                "[arxiv] %s during %s; backing off for %.2fs before retry %d/%d: %s",
                error_kind,
                label,
                wait_seconds,
                attempt + 1,
                max_retries,
                error,
            )
            time.sleep(wait_seconds)
            attempt += 1


def _search_cache_key(search: arxiv.Search) -> str:
    return "|".join(
        [
            f"query={search.query or ''}",
            f"id_list={','.join(search.id_list or [])}",
            f"max_results={search.max_results}",
            f"sort_by={search.sort_by}",
            f"sort_order={search.sort_order}",
        ]
    )


def _get_cached_results(cache_key: str) -> Optional[list[arxiv.Result]]:
    assert _config is not None
    if _config.cache_ttl_seconds <= 0:
        return None

    entry = _results_cache.get(cache_key)
    if entry is None:
        return None
    if entry.expires_at <= time.monotonic():
        _results_cache.pop(cache_key, None)
        return None
    return list(entry.value)


def _set_cached_results(cache_key: str, results: list[arxiv.Result]) -> None:
    assert _config is not None
    if _config.cache_ttl_seconds <= 0:
        return
    _results_cache[cache_key] = _CacheEntry(
        value=list(results),
        expires_at=time.monotonic() + _config.cache_ttl_seconds,
    )


def fetch_arxiv_results(search: arxiv.Search) -> list[arxiv.Result]:
    client = _get_arxiv_client()
    cache_key = _search_cache_key(search)
    cached_results = _get_cached_results(cache_key)
    if cached_results is not None:
        logger.debug("[arxiv] Cache hit for query: %s", cache_key)
        return cached_results

    results = _run_with_retries("arXiv API query", lambda: list(client.results(search)))
    _set_cached_results(cache_key, results)
    return list(results)


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

    return _run_with_retries("arXiv PDF download", _download)
