import logging
import time
from typing import Any

import requests

from configs import settings

logger = logging.getLogger(__name__)


class ApiFootballError(RuntimeError):
    pass


class ApiFootballRateLimitError(ApiFootballError):
    pass


class ApiFootballClient:
    """API-Football v3 HTTP istemcisi.

    Özellikler:
    - timeout, retry, exponential backoff
    - 429 / 5xx durumlarında backoff + Retry-After saygısı
    - basit circuit breaker (art arda çok fazla hata -> geçici durdur)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        backoff_factor: float | None = None,
    ) -> None:
        self.api_key = api_key or settings.API_FOOTBALL_KEY
        if not self.api_key:
            raise ApiFootballError(
                "API_FOOTBALL_KEY tanımlı değil. .env dosyasına ekleyin."
            )
        self.base_url = (base_url or settings.API_FOOTBALL_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.REQUEST_TIMEOUT
        self.max_retries = max_retries or settings.REQUEST_MAX_RETRIES
        self.backoff_factor = backoff_factor or settings.REQUEST_BACKOFF_FACTOR
        self._session = requests.Session()
        self._session.headers.update({"x-apisports-key": self.api_key})
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _sleep_with_jitter(self, seconds: float) -> None:
        import random

        time.sleep(seconds * (1 + random.random() * 0.3))

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}

        for attempt in range(self.max_retries + 1):
            if time.time() < self._circuit_open_until:
                raise ApiFootballRateLimitError(
                    "Circuit breaker açık; istekler geçici olarak durduruldu."
                )

            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                self._consecutive_failures += 1
                if attempt >= self.max_retries:
                    raise ApiFootballError(f"İstek başarısız: {exc}") from exc
                self._sleep_with_jitter(self.backoff_factor ** attempt)
                continue

            if resp.status_code == 200:
                self._consecutive_failures = 0
                return resp.json()

            retry_after = resp.headers.get("Retry-After")
            wait = self.backoff_factor ** attempt
            if retry_after and retry_after.isdigit():
                wait = max(wait, float(retry_after))

            if resp.status_code in (429, 500, 502, 503, 504):
                self._consecutive_failures += 1
                if self._consecutive_failures >= 5:
                    self._circuit_open_until = time.time() + 60
                    logger.warning("Circuit breaker açıldı (5 ardışık hata).")
                if attempt >= self.max_retries:
                    raise ApiFootballRateLimitError(
                        f"Rate limit / sunucu hatası ({resp.status_code})"
                    )
                logger.warning(
                    "HTTP %s %s deneme %s, %.1fs bekleniyor",
                    resp.status_code,
                    endpoint,
                    attempt + 1,
                    wait,
                )
                self._sleep_with_jitter(wait)
                continue

            if resp.status_code == 401:
                raise ApiFootballError("API anahtarı geçersiz (401).")

            raise ApiFootballError(
                f"API hatası: HTTP {resp.status_code} - {resp.text[:200]}"
            )

        raise ApiFootballError("Retry sayısı aşıldı.")

    def _check_response(self, data: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
        if data.get("errors"):
            raise ApiFootballError(f"{endpoint} hataları: {data['errors']}")
        return data.get("response", []) or []

    def get_list(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._check_response(self.get(endpoint, params), endpoint)