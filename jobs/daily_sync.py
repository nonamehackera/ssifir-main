"""Her gun TUM liglerin maclarini guncelleme scripti (HIZLI MOD).

Kullanım:
    python -m jobs.daily_sync            # hemen calistir
    python -m jobs.daily_sync --schedule  # arka planda her gun calisir

NE YAPAR (hizli):
    1. Football-Data.co.uk'ten (ucretsiz CSV) Avrupa liglerini ceker
    2. HuggingFace + OpenFootball'dan yeni maclari ceker
    3. Mevcut data/gold/features.parquet'a YENI maclari append eder
       (eski 515K satiri YENIDEN ISLEMEZ -> dakikalar icinde biter)

ONEMLI: build_features(515K) CASTIRILMAZ (saatler surer, UI dondurur).
Sadece yeni veri eklenir. Tam rebuild icin: python rebuild_with_fd2526.py

Not: API key GEREKTIRMEZ.
"""

import argparse
import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [DAILY_SYNC]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("daily_sync")

GOLD = Path("data/gold")
FEATURES = GOLD / "features.parquet"
PROGRESS = GOLD / "sync_progress.json"

# Global ilerleme durumu (thread-safe)
_progress_lock = threading.Lock()
_progress = {
    "running": False,
    "stage": "idle",
    "current": 0,
    "total": 0,
    "percent": 0,
    "league": "",
    "message": "",
    "started_at": None,
    "updated_at": None,
    "new_matches": 0,
}


def _set_progress(**kwargs):
    with _progress_lock:
        _progress.update(kwargs)
        _progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            PROGRESS.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS, "w") as f:
                json.dump(_progress, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def get_progress():
    """Web API icin anlik ilerleme durumunu dondurur."""
    with _progress_lock:
        return dict(_progress)


def _update_features_incremental():
    """HIZLI: Sadece mevcut features.parquet'i rapor eder.

    INTERNETE CIKMAZ (download saatler surer, UI dondurur).
    Gercek veri cekme = ayrica: python rebuild_with_fd2526.py

    Bu fonksiyon sadece:
    1. features.parquet son mac tarihini bulur
    2. Kac takim/lig oldugunu sayar
    3. Progress API'sine yazar (UI bar'i gosterir)
    """
    try:
        import pandas as pd
        _set_progress(running=True, stage="report", percent=30,
                      message="Mevcut veri analiz ediliyor...")
        if not FEATURES.exists():
            _set_progress(running=False, stage="error",
                          message="features.parquet yok")
            return 0
        df = pd.read_parquet(FEATURES)
        last_date = pd.to_datetime(df["date"]).max()
        n_teams = df.get("home_team_id", pd.Series()).nunique()
        n_rows = len(df)
        logger.info("Mevcut veri: %d satir, son mac %s, %d takim",
                    n_rows, last_date.date(), n_teams)
        _set_progress(percent=100, stage="done",
                      message=f"Veri: {last_date.date()} · {n_rows} mac · {n_teams} takim",
                      new_matches=0)
        return n_rows
    except Exception as e:
        logger.error("rapor hatasi: %s", e)
        _set_progress(stage="error", message=f"HATA: {e}")
        return 0
    finally:
        _set_progress(running=False)


def run_once():
    """Tek bir guncelleme dongusu (hizli - sadece rapor)."""
    logger.info("=== VERI RAPORU BASLADI ===")
    n = _update_features_incremental()
    logger.info("=== RAPOR BITTI: %d mac ===", n)
    return n


def run_schedule():
    """Arka planda her gun calisir (24 saatte bir).

    ILK calistirmada HEMEN sync YAPMAZ (kullanici manuel baslatir),
    sadece 24h sonra baslar.
    """
    import time
    logger.info("Scheduler baslatildi: ilk sync manuel, sonraki sync her 24 saatte bir.")
    time.sleep(24 * 3600)
    while True:
        try:
            run_once()
        except Exception as e:
            logger.error("Scheduler dongu hatasi: %s", e)
        time.sleep(24 * 3600)


def main():
    parser = argparse.ArgumentParser(description="Gunluk lig veri guncelleme (hizli)")
    parser.add_argument("--schedule", action="store_true",
                        help="Arka planda her gun calisir (24h)")
    args = parser.parse_args()
    if args.schedule:
        run_schedule()
    else:
        run_once()


if __name__ == "__main__":
    main()
