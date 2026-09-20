"""Regression test: model MUST beat baselines honestly.

Runs a quick walk-forward on a single recent season (fast subset) and
asserts:
  1. Model 1X2 accuracy > majority baseline ('H')
  2. Model 1X2 accuracy > bookie favorite baseline
  3. Model log loss < 1.0986 (random 1X2)
  4. BTTS AUC > 0.5 (better than coin flip)
  5. Over2.5 AUC > 0.5

If ANY check fails, the test FAILS (no green-washing).
"""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
from models.catboost.model import CatBoostModel
from models.lightgbm.model import LightGBMModel
from models.ensemble.calibrated_ensemble import CalibratedEnsemble
from evaluation.metrics import accuracy_1x2, log_loss_1x2, auc_binary


def _bookie_pick(row):
    probs = {"H": row.get("mkt_home_prob", np.nan),
             "D": row.get("mkt_draw_prob", np.nan),
             "A": row.get("mkt_away_prob", np.nan)}
    probs = {k: (v if pd.notna(v) else -1) for k, v in probs.items()}
    if max(probs.values()) <= 0:
        return "H"
    return max(probs, key=probs.get)


@pytest.fixture(scope="module")
def data():
    feats = pd.read_parquet("data/gold/features.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    return feats


def test_model_beats_baselines(data):
    # Single season: 2023/24 (fast enough for CI, recent enough for honesty)
    vs = pd.Timestamp("2023-07-01")
    ve = pd.Timestamp("2024-07-01")
    valid = data[(data["date"] >= vs) & (data["date"] < ve)].copy()
    train_all = data[data["date"] < vs].copy().sort_values("date").reset_index(drop=True)
    n_cal = max(200, int(len(train_all) * 0.15))
    cal = train_all.iloc[-n_cal:]
    train = train_all.iloc[:-n_cal]

    # Baselines
    base_h = (valid["result"] == "H").mean()
    valid["bk"] = valid.apply(_bookie_pick, axis=1)
    base_book = (valid["bk"] == valid["result"]).mean()

    # Model (iterations kept modest for test speed)
    cat = CatBoostModel(with_odds=True, iterations=200, draw_weight=1.2, verbose=False)
    lgbm = LightGBMModel(with_odds=True, n_estimators=200, draw_weight=1.2, verbose=-1)
    ens = CalibratedEnsemble([cat, lgbm])
    ens.fit(train, cal)
    p = ens.predict(valid)

    y = valid["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ph, pd_, pa = np.asarray(p.home_win), np.asarray(p.draw), np.asarray(p.away_win)
    preds = np.select([ph >= pd_, pa >= ph], [0, 2], default=1)
    model_acc = (preds == y).mean()
    ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
    btts_auc = auc_binary(valid["btts"], p.btts_yes)
    over25_auc = auc_binary(valid["over25"], p.over25)

    print(f"\n  base_H={base_h:.3f} base_book={base_book:.3f}")
    print(f"  model_acc={model_acc:.3f} logloss={ll:.4f}")
    print(f"  btts_auc={btts_auc:.3f} over25_auc={over25_auc:.3f}")

    # HARD ASSERTIONS — no green-washing
    assert model_acc > base_h, f"Model %{model_acc*100:.1f} <= baseline H %{base_h*100:.1f}"
    assert model_acc > base_book, f"Model %{model_acc*100:.1f} <= bookie %{base_book*100:.1f}"
    assert ll < 1.0986, f"Log loss {ll:.4f} >= random 1.0986"
    assert btts_auc > 0.5, f"BTTS AUC {btts_auc:.3f} <= 0.5"
    assert over25_auc > 0.5, f"Over2.5 AUC {over25_auc:.3f} <= 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
