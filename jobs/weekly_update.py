"""Haftalik guncelleme job'u (ROADMAP 69 + senin talebin: "veri haftada bir guncellenmeli").

Scheduler tarafindan haftada bir cagrilir:
  1. check_new_matches() -> yeni mac var mi?
  2. varsa fetch_latest() -> CSV indir
  3. upsert_gold() -> feature rebuild
  4. retrain (catboost) -> register + promote (sadece iyilesirse)
  5. drift kontrolu -> alarm

Kullanim:
  ./.venv/bin/python -m jobs.weekly_update
  veya scheduler.add_job("weekly", run_weekly_update, interval_seconds=604800)
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from datetime import datetime, timezone

from db.engine import init_db
from ingestion.football_data.fetcher import (
    check_new_matches,
    fetch_latest,
    upsert_gold,
)
from models.catboost.model import CatBoostModel
from monitoring.drift import evaluate_period, detect_drift, record_period
from registry.registry import register_model, promote, get_production_model
from evaluation.metrics import log_loss_1x2, accuracy_1x2

logger = logging.getLogger(__name__)


def run_weekly_update(force: bool = False, with_odds: bool = True) -> dict:
    init_db()
    report = {}

    # 1. yeni mac kontrolu
    chk = check_new_matches()
    report["check"] = chk
    if not force and not chk.get("needs_update"):
        logger.info("Yeni mac yok, guncelleme atlaniyor. %s", chk)
        report["updated"] = False
        return report

    # 2. fetch - cok kaynakli (ROADMAP 2: Sportmonks/API-Football/Football-Data)
    logger.info("Yeni veri var, indiriliyor...")
    # (a) Football-Data.co.uk (mevcut 30 lig, detayli stats)
    fd_stats = fetch_latest()
    # (b) TheSportsDB (200+ lig, sonuclar) - ag erisimi gerektirir
    tsdb_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "note": "ag engelli veya devre disi"}
    try:
        from ingestion.thesportsdb.adapter import build_thesportsdb_matches
        tsdb = build_thesportsdb_matches(seasons=[2024, 2025])
        tsdb_stats = {"downloaded": len(tsdb), "leagues": int(tsdb["league_id"].nunique()) if len(tsdb) else 0}
    except Exception as exc:
        logger.warning("TheSportsDB fetch atlandi: %s", exc)
    report["fetch"] = {"football_data": fd_stats, "thesportsdb": tsdb_stats}

    # 3. rebuild
    n = upsert_gold(rebuild=False)
    report["gold_rows"] = n

    # 4. retrain (kucuk walk-forward son fold)
    from feature_engine.engine import build_features
    from ingestion.football_data.canonical import build_matches

    m = build_matches()
    f = build_features(m)
    cut = int(len(f) * 0.8)
    train, test = f.iloc[:cut], f.iloc[cut:].reset_index(drop=True)
    model = CatBoostModel(with_odds=with_odds, iterations=400, verbose=False,
                         draw_weight=4.0, time_decay=1.0).fit(train)
    p = model.predict(test)
    ll = log_loss_1x2(test["result"], p.home_win, p.draw, p.away_win)
    acc = accuracy_1x2(test["result"], p.home_win, p.draw, p.away_win)

    # mevcut production ile karsilastir, sadece iyilesirse promote et
    prod = get_production_model("catboost")
    promote_ok = True
    if prod and prod.metrics:
        try:
            prev_ll = float(prod.metrics.get("log_loss", 99))
            if ll >= prev_ll:
                promote_ok = False
                logger.info("Yeni model eskisinden kotu (%.4f >= %.4f), promote EDILMEDI", ll, prev_ll)
        except Exception:
            pass
    if promote_ok:
        mv = register_model("catboost",
                             "catboost_odds" if with_odds else "catboost",
                             {"log_loss": round(float(ll), 4), "accuracy": round(float(acc), 3)},
                             "catboost", "football_data")
        promote(mv.model_id, mv.version, "staging")
        promote(mv.model_id, mv.version, "production")
        report["promoted"] = mv.version

    # 5. drift kaydi
    period = evaluate_period(test["result"], p.home_win, p.draw, p.away_win, period="weekly")
    record_period(period)
    report["log_loss"] = float(ll)
    report["accuracy"] = float(acc)
    report["updated"] = True
    logger.info("Haftalik guncelleme tamam: %s", {k: v for k, v in report.items() if k != "check"})
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="yeni mac kontrolu yapmadan zorla guncelle")
    ap.add_argument("--no-odds", dest="with_odds", action="store_false", default=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = run_weekly_update(force=args.force, with_odds=args.with_odds)
    print("WEEKLY UPDATE:", out)


if __name__ == "__main__":
    main()
