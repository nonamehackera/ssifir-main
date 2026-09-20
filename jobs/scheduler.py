"""Hafif scheduler (ROADMAP 69-70, 86).

APScheduler/Celery gibi agir bagimlilik yerine, standart kutuphane ile
calisan basit bir job scheduler. Iki gorev:

  - weekly_retrain: haftalik tam retrain (ROADMAP 69)
  - drift_check:   periyodik drift kontrolu (ROADMAP 70)

Gercek sistemde buna ek olarak MLflow/Redis kullanilabilir; bu modul
bagimlilik siz calisir ve test edilebilir.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SimpleScheduler:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread = None

    def add_job(self, name: str, func, interval_seconds: int, args=None, kwargs=None):
        self._jobs[name] = {
            "func": func, "interval": interval_seconds,
            "args": args or (), "kwargs": kwargs or {}, "last_run": None,
        }
        logger.info("Job eklendi: %s (interval=%ss)", name, interval_seconds)

    def _loop(self):
        while not self._stop.is_set():
            now = time.time()
            for name, job in self._jobs.items():
                last = job["last_run"] or 0
                if now - last >= job["interval"]:
                    job["last_run"] = now
                    try:
                        job["func"](*job["args"], **job["kwargs"])
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Job %s basarisiz: %s", name, exc)
            time.sleep(1.0)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler basladi (%d job)", len(self._jobs))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler durdu")

    def run_job_now(self, name: str):
        """Test/manuel tetikleme."""
        job = self._jobs.get(name)
        if not job:
            raise KeyError(name)
        job["func"](*job["args"], **job["kwargs"])
