"""Retry / rate-limit / circuit-breaker (ROADMAP bolum 60).

API cagrilarinda: timeout, retry, exponential backoff, rate-limit handling,
circuit breaker. Ayni veriyi tekrar isteme (cache) disaridan verilir.

Kullanim:
    @with_retry(max_attempts=3, backoff=2.0)
    def fetch():
        return requests.get(url, timeout=10).json()
"""

from __future__ import annotations

import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Basit circuit breaker: ardi ardina hata olursa bir sure acigi keser."""

    def __init__(self, failure_threshold: int = 5, reset_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._open_until = 0.0

    def allow(self) -> bool:
        if time.time() > self._open_until:
            return True
        return False

    def success(self) -> None:
        self._failures = 0

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.time() + self.reset_seconds
            self._failures = 0
            logger.warning("Circuit breaker ACILDI (%ss)", self.reset_seconds)


def with_retry(max_attempts: int = 3, backoff: float = 2.0, base_delay: float = 1.0):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    delay = base_delay * (backoff ** attempt)
                    logger.warning("Deneme %d/%d basarisiz (%s), %.1fs bekleniyor",
                                   attempt + 1, max_attempts, exc, delay)
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last
        return wrapper
    return deco
