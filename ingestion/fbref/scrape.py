"""FBref toplu scraping - arka planda calistirmak icin.

Kullanim:
  .venv/bin/python -m ingestion.fbref.scrape 2>&1 | tee /tmp/opencode/fbref_scrape.log
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fbref_scrape")

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from ingestion.fbref.provider import build_fbref_matches

    logger.info("=== FBref toplu scraping basliyor ===")
    df = build_fbref_matches(resume=True, cache_path="data/gold/fbref_matches.parquet")
    logger.info("=== TAMAMLANDI: %d mac, %d lig, %d istatistikli ===",
                len(df), df["league"].nunique() if not df.empty else 0,
                df["home_shots"].notna().sum() if not df.empty and "home_shots" in df.columns else 0)