from __future__ import annotations

import logging
import time
from threading import Lock


logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """Simple process-wide blocking rate limiter."""

    def __init__(self, min_interval_seconds: float = 0.0):
        self._min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = Lock()
        self._next_allowed_at = 0.0

    @property
    def min_interval_seconds(self) -> float:
        return self._min_interval_seconds

    def acquire(self, label: str = "request") -> None:
        if self._min_interval_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed_at - now)
            scheduled_at = max(now, self._next_allowed_at) + self._min_interval_seconds
            self._next_allowed_at = scheduled_at

        if wait_seconds > 0:
            logger.debug("[rate-limit] Waiting %.2fs before %s", wait_seconds, label)
            time.sleep(wait_seconds)
