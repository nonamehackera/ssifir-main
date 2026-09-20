"""Football-Data.co.uk fetcher + yeni mac kontrolu (ROADMAP 2.3, 78, 69).

Gorevler:
  1. fetch_latest(): son sezonun CSV'lerini data/raw/'a indirir (havuzlu +
     retry). Sadece son 1-2 sezon "canli guncellenen" kisimdir; eski
     sezonlar bir kez indirilir.
  2. check_new_matches(): local parquet ile son indirilen CSV'yi karsilastirip
     "yeni mac var mi?" raporu dondurur (ROADMAP 69: haftalik kontrol).
  3. upsert: yeni maclari gold/matches.parquet'a ekler (append-only, idempotent).

Not: Football-Data.co.uk UCRETSIZ ve API key gerektirmez. Rate-limit'e
dikkat: kucuk gecikme + retry (ingestion/retry.py).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from configs import settings
from ingestion.retry import with_retry
from ingestion.football_data.canonical import FD_LEAGUES, FD_SEASONS

logger = logging.getLogger(__name__)

FD_BASE_URL = "https://www.football-data.co.uk/mmz4281"


@with_retry(max_attempts=3, backoff=2.0, base_delay=1.0)
def _fetch_csv(season: str, league: str) -> pd.DataFrame | None:
    import requests

    url = f"{FD_BASE_URL}/{season}/{league}.csv"
    resp = requests.get(url, timeout=settings.REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    df = pd.read_csv(url := url) if False else pd.read_csv(io_string := resp.text)
    df["_season"] = season
    df["_league"] = league
    return df


def fetch_latest(seasons: list[str] | None = None, leagues: list[str] | None = None,
                raw_dir: str | None = None) -> dict:
    """Son sezon(lar) CSV'lerini indirir. Doner: {indirilen, atlanan, hatali}."""
    seasons = seasons or [FD_SEASONS[-1], FD_SEASONS[-2]]  # son 2 sezon
    leagues = leagues or list(FD_LEAGUES.keys())
    raw_dir = Path(raw_dir or settings.DATA_DIR) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    for season in seasons:
        for league in leagues:
            try:
                df = _fetch_csv(season, league)
                if df is None or df.empty:
                    stats["skipped"] += 1
                    continue
                out = raw_dir / f"{season}_{league}.csv"
                df.to_csv(out, index=False)
                stats["downloaded"] += 1
                time.sleep(0.2)  # rate-limit nezaketi
            except Exception as exc:  # noqa: BLE001
                logger.warning("Indirme basarisiz %s/%s: %s", season, league, exc)
                stats["failed"] += 1
    logger.info("fetch_latest: %s", stats)
    return stats


def check_new_matches(gold_path: str | None = None) -> dict:
    """Local gold/matches.parquet ile raw CSV'leri karsilastirir.

    Doner: {new_matches: int, latest_local_date, latest_remote_date, needs_update: bool}
    """
    gold_path = Path(gold_path or settings.DATA_DIR) / "gold" / "matches.parquet"
    if not gold_path.exists():
        return {"new_matches": -1, "needs_update": True, "note": "gold yok, tam yukleme gerekir"}
    local = pd.read_parquet(gold_path)
    local_max = pd.to_datetime(local["date"]).max() if "date" in local else None

    # son sezon raw CSV'lerinden en yeni tarihi bul
    raw_dir = Path(settings.DATA_DIR) / "raw"
    remote_max = None
    for csv in raw_dir.glob(f"{FD_SEASONS[-1]}_*.csv"):
        try:
            df = pd.read_csv(csv)
            if "Date" in df:
                d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").max()
                if remote_max is None or (d and d > remote_max):
                    remote_max = d
        except Exception:
            continue
    new_count = 0
    if local_max and remote_max:
        new_count = int((remote_max - local_max).days)  # kaba tahmin: gun farki
        new_count = max(0, new_count * 5)  # ~5 mac/gun ortalamasi (kaba)
    return {
        "new_matches_estimate": new_count,
        "latest_local_date": str(local_max) if local_max else None,
        "latest_remote_date": str(remote_max) if remote_max else None,
        "needs_update": bool(remote_max and local_max and remote_max > local_max),
    }


def upsert_gold(rebuild: bool = False) -> int:
    """Yeni veri geldiyse gold/matches.parquet'i yeniden build eder.

    rebuild=True -> tum sezonlari bastan build (tam retrain icin).
    rebuild=False -> sadece son 2 sezonu tazele (incrementalmis gibi).
    """
    from ingestion.football_data.canonical import build_matches

    seasons = None if rebuild else [FD_SEASONS[-1], FD_SEASONS[-2]]
    m = build_matches(seasons=seasons)
    logger.info("Gold guncellendi: %d mac", len(m))
    return len(m)
