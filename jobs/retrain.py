"""Retrain orkestrasyon (ROADMAP bolum 66).

Tek komutla tum pipeline:
  extract -> validate -> normalize -> feature -> time-split ->
  train -> calibrate -> evaluate -> register -> deploy

Kullanim:
  .venv/bin/python -m jobs.retrain --with-odds --register
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from datetime import datetime, timezone

from db.engine import init_db
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.catboost.model import CatBoostModel
from registry.registry import register_model, promote
from monitoring.drift import evaluate_period, record_period

logger = logging.getLogger(__name__)


def run(with_odds: bool = True, register: bool = False, quick: bool = False) -> dict:
    init_db()
    steps = []

    # 1. extract + normalize + feature
    logger.info("1/7 extract + feature")
    m = build_matches()
    f = build_features(m)
    steps.append(f"matches={len(m)} features={len(f)}")

    # 2. time split (walk-forward son fold)
    cut = int(len(f) * (0.85 if quick else 0.8))
    train, val, test = f.iloc[:cut], f.iloc[cut:int(cut + (len(f) - cut) // 2)], f.iloc[int(cut + (len(f) - cut) // 2):].reset_index(drop=True)

    # 3. train
    logger.info("3/7 train catboost%s", " (with odds)" if with_odds else "")
    iters = 150 if quick else 400
    model = CatBoostModel(with_odds=with_odds, iterations=iters, verbose=False, 
                         draw_weight=4.0, time_decay=1.0).fit(train, val)
    steps.append("trained")

    # 4. evaluate (out-of-sample)
    from evaluation.metrics import log_loss_1x2, accuracy_1x2
    p = model.predict(test)
    ll = log_loss_1x2(test["result"], p.home_win, p.draw, p.away_win)
    acc = accuracy_1x2(test["result"], p.home_win, p.draw, p.away_win)
    period = evaluate_period(test["result"], p.home_win, p.draw, p.away_win, period="latest")
    record_period(period)
    steps.append(f"ll={ll:.4f} acc={acc:.3f}")

    # 5. register + promote (ROADMAP 68)
    if register:
        logger.info("5/7 register + promote")
        mv = register_model(
            "catboost", ("catboost_odds" if with_odds else "catboost"),
            {"log_loss": round(float(ll), 4), "accuracy": round(float(acc), 3)},
            "catboost", "football_data",
        )
        promote(mv.model_id, mv.version, "staging")
        promote(mv.model_id, mv.version, "production")
        steps.append(f"registered {mv.model_id} v{mv.version}")

    logger.info("Retrain tamam: %s", steps)
    return {"steps": steps, "log_loss": float(ll), "accuracy": float(acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-odds", action="store_true", default=True)
    ap.add_argument("--no-odds", dest="with_odds", action="store_false")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = run(with_odds=args.with_odds, register=args.register, quick=args.quick)
    print("RESULT:", out)


if __name__ == "__main__":
    main()
