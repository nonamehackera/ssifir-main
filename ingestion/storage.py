import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from configs import settings

logger = logging.getLogger(__name__)


class RawStorage:
    """Bronze katmanı: kaynak API yanıtlarını olduğu gibi saklar (ROADMAP bölüm 58).

    Yapı: data/bronze/<source>/<date>/<endpoint>_<params>.json
    """

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or settings.DATA_DIR) / "bronze"

    def _path(self, source: str, endpoint: str, params: dict) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        suffix = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        filename = f"{endpoint.replace('/', '_')}_{suffix}.json"
        return self.base_dir / source / date_str / filename

    def find(self, source: str, endpoint: str, params: dict) -> Path | None:
        """Parametreye göre en son saklanmış dosyayı bulur (bronze)."""
        suffix = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        pattern = f"{endpoint.replace('/', '_')}_{suffix}.json"
        candidates = sorted(self.base_dir.glob(f"{source}/*/{pattern}"))
        return candidates[-1] if candidates else None

    def save(self, source: str, endpoint: str, params: dict, payload: object) -> Path:
        path = self._path(source, endpoint, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "endpoint": endpoint,
            "params": params,
            "response": payload,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
        return path

    def load(self, source: str, endpoint: str, params: dict) -> dict | None:
        path = self._path(source, endpoint, params)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)


def save_raw(source: str, endpoint: str, params: dict, payload: object) -> Path:
    return RawStorage().save(source, endpoint, params, payload)