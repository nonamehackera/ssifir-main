"""GAP modulleri dogrulama testleri (spec 90 madde karsiligi).

Calistir:
    .venv/bin/python -m tests.test_gap_modules
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from feature_engine.coverage import coverage_matrix, print_report


def test_coverage():
    m = build_matches()
    f = build_features(m)
    mat = coverage_matrix(f)
    assert len(mat) > 0
    # free veride xg coverage = 0 olmali
    xg = mat[mat["feature"].str.startswith("xg")]
    assert (xg["coverage"] == 0).all(), "xg free veride yok"
    # corners kismen var
    cor = mat[mat["feature"] == "home_corners"]
    assert cor["coverage"].min() > 0.9, "corners cogunlukla var"
    print(f"[coverage] {len(mat)} hucre; xg=0 (uzak kaynak gerekir), corners>0.9")
    return True


def test_corners_cards():
    from models.markets.corners_cards import CornersModel, CardsModel
    from evaluation.metrics import mae_goals

    m = build_matches(); f = build_features(m)
    cut = int(len(f) * 0.8)
    train, test = f.iloc[:cut], f.iloc[cut:].reset_index(drop=True)
    cm = CornersModel(family="negbin").fit(train)
    cp = cm.predict(test)
    # ortalama yaklasilik
    assert abs(cp.home_lambda.mean() - test["home_corners"].mean()) < 1.0
    km = CardsModel(family="negbin").fit(train)
    kp = km.predict(test)
    act = test["home_yellow"].fillna(0) + test["home_red"].fillna(0) + test["away_yellow"].fillna(0) + test["away_red"].fillna(0)
    # cards pred total vs actual
    assert abs((kp.home_lambda + kp.away_lambda).mean() - act.mean()) < 1.0
    print(f"[corners+cards] corners pred={cp.home_lambda.mean():.2f}/{cp.away_lambda.mean():.2f} "
          f"actual={test['home_corners'].mean():.2f}/{test['away_corners'].mean():.2f}")
    print(f"                 cards total pred={(kp.home_lambda+kp.away_lambda).mean():.2f} actual={act.mean():.2f}")
    return True


def test_persistence():
    from prediction.persistence import save_prediction, record_result, get_prediction
    from db.engine import init_db

    init_db()
    pid = "GAP_1_T-24H"
    payload = {"result": {"home": 0.5, "draw": 0.27, "away": 0.23},
               "goals": {"home_lambda": 1.6, "away_lambda": 1.0},
               "btts": {"yes": 0.55}, "over_under_2_5": {"over": 0.52}}
    save_prediction(pid, 123456, datetime.now(timezone.utc), payload,
                    model_version="catboost_odds_v1", feature_version="v1",
                    source_versions={"football_data": "2122-2425"})
    rec = record_result(pid, 2, 1)
    got = get_prediction(pid)
    assert abs(float(got.home_win) - 0.5) < 1e-6
    assert rec.actual_result == "H"
    print(f"[persistence] DB'ye yazildi, sonuc kaydedildi (brier={float(rec.brier_score):.3f})")
    return True


def test_cache():
    from cache.redis_client import get_cache
    c = get_cache()  # redis yoksa in-memory fallback
    c.set("k", {"a": 1}, ttl=10)
    assert c.get("k")["a"] == 1
    print("[cache] in-memory fallback calisiyor (redis opsiyonel)")
    return True


def test_retry():
    from ingestion.retry import with_retry, CircuitBreaker
    calls = {"n": 0}

    @with_retry(max_attempts=3, backoff=1.0, base_delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("gecici hata")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2
    cb = CircuitBreaker(failure_threshold=2, reset_seconds=1)
    assert cb.allow()
    cb.failure(); cb.failure()
    assert not cb.allow()  # acildi
    print("[retry] retry + circuit breaker calisiyor")
    return True


def test_snapshot_lifecycle():
    from prediction.snapshot_lifecycle import snapshot_schedule, prediction_id, stability

    ko = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)
    sched = snapshot_schedule(ko)
    assert len(sched) == 9
    assert prediction_id(123, "T-24H") == "123_T-24H"
    # stability ornegi
    reports = [
        {"payload": {"result": {"home": 0.5, "draw": 0.27, "away": 0.23}}},
        {"payload": {"result": {"home": 0.51, "draw": 0.26, "away": 0.23}}},
    ]
    s = stability(reports)
    assert s >= 0
    print(f"[snapshot] 9 nokta (T-48H..KO), prediction_id formati OK, stability={s:.4f}")
    return True


def test_league_benchmark():
    from models.league_benchmark import benchmark, recommend_league_models

    m = build_matches(); f = build_features(m)
    rep = benchmark(f, with_odds=True)
    rec = recommend_league_models(rep)
    print(f"[league-benchmark] {len(rep)} lig karsilastirildi, ayri model onerilen: {rec}")
    return True


def test_drift_retrain_hook():
    from monitoring.drift import evaluate_period, detect_drift, maybe_retrain

    # retrain_fn'in gercekten cagrilmasini test et (mock ile)
    called = {"n": 0}
    def fake_retrain():
        called["n"] += 1
        return {"log_loss": 0.97}

    prev = evaluate_period(["H", "D", "A"], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], period="p")
    # onceki vs ayni -> drift yok -> retrain YOK
    r0 = maybe_retrain(detect_drift(prev, prev), retrain_fn=fake_retrain)
    assert r0["action"] == "skip"
    # kotu donem -> drift var -> retrain VAR
    bad = evaluate_period(["H", "H", "H"], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05], period="b")
    r1 = maybe_retrain(detect_drift(prev, bad), retrain_fn=fake_retrain)
    assert r1["action"] == "retrain" and called["n"] == 1
    print(f"[drift] drift yok=skip, drift var=retrain tetiklendi (called={called['n']})")
    return True


if __name__ == "__main__":
    tests = [
        test_coverage, test_corners_cards, test_persistence, test_cache,
        test_retry, test_snapshot_lifecycle, test_league_benchmark,
        test_drift_retrain_hook,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            raise
    print("\nALL GAP MODULES OK")
