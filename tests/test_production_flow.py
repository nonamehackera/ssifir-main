"""Uretim akisi (lifecycle + collector + scheduler) dogrulama testi.

Calistir:
    ./.venv/bin/python -m tests.test_production_flow
"""

import sys
import numpy as np
sys.path.insert(0, ".")

from datetime import datetime, timedelta, timezone

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.catboost.model import CatBoostModel

from api.lifecycle import MatchLifecycle
from live.collector import MockLiveSource, LiveCollector
from jobs.scheduler import SimpleScheduler

from db.engine import init_db
from db.models import ModelPrediction
from sqlalchemy import select, func


def _make_predict_fn(model):
    def predict_fn(features_row, snapshot_label=None, prediction_time=None):
        # pipeline benzeri cikti
        from prediction.pipeline import PredictionPipeline
        pip = PredictionPipeline(model, model_version="catboost_odds_v1", dataset_version="fd")
        # features_row tek satir DataFrame olmali
        import pandas as pd
        if isinstance(features_row, pd.DataFrame):
            row = features_row.iloc[[0]]
        else:
            row = pd.DataFrame([features_row])
        out = pip.predict(row)
        # pre-match lambda'yi da dondur (lifecycle icin)
        return out
    return predict_fn


def test_full_match_flow():
    init_db()
    m = build_matches(); f = build_features(m)
    cut = int(len(f) * 0.8)
    train, test = f.iloc[:cut], f.iloc[cut:].reset_index(drop=True)
    model = CatBoostModel(with_odds=True, iterations=200, verbose=False).fit(train)
    predict_fn = _make_predict_fn(model)

    # ornek mac (test'ten)
    row = test.iloc[0]
    fixture_id = int(row["match_id"])
    kickoff = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)

    orch = MatchLifecycle(predict_fn, source_versions={"football_data": "2122-2425"},
                          model_version="catboost_odds_v1")
    # pre-match lambda (pipeline ciktisindan)
    pre_payload = predict_fn(test.iloc[[0]])
    orch.set_pre_lambda(fixture_id, pre_payload["goals"]["home_lambda"],
                        pre_payload["goals"]["away_lambda"])

    # 1. PRE-MATCH
    emitted = orch.run_pre_match(fixture_id, kickoff, test.iloc[[0]])
    assert len(emitted) == 9, f"pre-match snapshot sayisi={len(emitted)}"

    # 2. LIVE (mock source ile)
    scripted = [
        {"minute": 10, "home_goals": 0, "away_goals": 0, "xg_home": 0.2, "xg_away": 0.1,
         "live_odds_home": 1.9, "live_odds_draw": 3.4, "live_odds_away": 4.0},
        {"minute": 45, "home_goals": 1, "away_goals": 0, "xg_home": 0.8, "xg_away": 0.4,
         "live_odds_home": 1.5, "live_odds_draw": 3.8, "live_odds_away": 7.0},
        {"minute": 90, "home_goals": 2, "away_goals": 0, "xg_home": 1.6, "xg_away": 0.7,
         "live_odds_home": 1.2, "live_odds_draw": 6.0, "live_odds_away": 12.0},
    ]
    src = MockLiveSource(scripted)
    collector = LiveCollector(src, fixture_id, interval=0.0, max_minute=95)
    collector.set_pre_lambda(pre_payload["goals"]["home_lambda"],
                              pre_payload["goals"]["away_lambda"])
    live_preds = collector.run()
    assert len(live_preds) == 3, f"live snapshot={len(live_preds)}"
    # 90. dakika tahmini: ev 2-0 onde -> yuksek ev olasiligi
    assert live_preds[-1]["result"]["home"] > 0.5

    # 3. RESULT (mac 2-0 bitti)
    final = orch.finalize(fixture_id, 2, 0)
    assert final["evaluated"] >= 10, f"degerlendirilen={final['evaluated']}"

    # DB kontrol
    sess = __import__("db.engine", fromlist=["SessionLocal"]).SessionLocal()
    n = sess.execute(select(func.count(ModelPrediction.id))
                     .where(ModelPrediction.fixture_id == fixture_id)).scalar()
    sess.close()
    assert n >= 12, f"DB'de tahmin sayisi={n}"
    print(f"[flow] pre={len(emitted)} live={len(live_preds)} DB_tahmin={n} "
          f"finalize={final['evaluated']}")
    return True


def test_scheduler():
    calls = {"n": 0}
    sched = SimpleScheduler()
    sched.add_job("dummy", lambda: calls.__setitem__("n", calls["n"] + 1),
                  interval_seconds=1)
    sched.run_job_now("dummy")
    assert calls["n"] == 1
    sched.start()
    import time as _t
    _t.sleep(2.2)
    sched.stop()
    assert calls["n"] >= 3, f"scheduler cagrim={calls['n']}"
    print(f"[scheduler] manuel+periyodik cagrim={calls['n']}")
    return True


if __name__ == "__main__":
    for t in [test_full_match_flow, test_scheduler]:
        try:
            t()
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            raise
    print("\nPRODUCTION FLOW OK")
