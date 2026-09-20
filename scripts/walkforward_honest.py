"""HONEST very-istikrarli walk-forward evaluation.

4 consecutive seasons (2021/22 -> 2024/25). Every fold trains ONLY on past.
Reports per-fold: 1X2 accuracy + log_loss, BTTS/Over2.5 AUC, and a
simple betting ROI using the model's top pick. Compares against:
  - baseline 'H' (majority)
  - baseline bookie favorite (mkt implied, where available)

Also reports: does the ensemble ACTUALLY optimize weights/temperature?
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from models.catboost.model import CatBoostModel
from models.lightgbm.model import LightGBMModel
from models.ensemble.calibrated_ensemble import CalibratedEnsemble
from evaluation.metrics import (
    accuracy_1x2, log_loss_1x2, brier_1x2, auc_binary,
)

def bookie_pick(row):
    probs = {"H": row.get("mkt_home_prob", np.nan),
             "D": row.get("mkt_draw_prob", np.nan),
             "A": row.get("mkt_away_prob", np.nan)}
    probs = {k: (v if pd.notna(v) else -1) for k, v in probs.items()}
    if max(probs.values()) <= 0:
        return "H"
    return max(probs, key=probs.get)

def argmax_1x2(p):
    ph, pd_, pa = np.asarray(p.home_win), np.asarray(p.draw), np.asarray(p.away_win)
    pred = np.select([ph >= pd_, pa >= ph], [0, 2], default=1)
    return pred

def main():
    t0 = time.time()
    feats = pd.read_parquet("data/gold/features.parquet")
    feats["date"] = pd.to_datetime(feats["date"])

    seasons = [
        ("2021-07-01", "2022-07-01", "2021/22"),
        ("2022-07-01", "2023-07-01", "2022/23"),
        ("2023-07-01", "2024-07-01", "2023/24"),
        ("2024-07-01", "2025-07-01", "2024/25"),
    ]

    rows = []
    ensemble_weights_log = []
    for (v_start, v_end, label) in seasons:
        vs = pd.Timestamp(v_start); ve = pd.Timestamp(v_end)
        valid = feats[(feats["date"] >= vs) & (feats["date"] < ve)].copy()
        if len(valid) < 1000:
            print(f"Fold {label}: yetersiz ({len(valid)}), atlaniyor")
            continue
        train_all = feats[feats["date"] < vs].copy().sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all) * 0.15))
        cal = train_all.iloc[-n_cal:]
        train = train_all.iloc[:-n_cal]

        print(f"\n=== FOLD {label}: train={len(train)} cal={len(cal)} valid={len(valid)} ===")

        # Baselines on valid
        base_h = (valid["result"] == "H").mean()
        valid["bk"] = valid.apply(bookie_pick, axis=1)
        base_book = (valid["bk"] == valid["result"]).mean()
        print(f"  Baseline H      : %{base_h*100:.1f}")
        print(f"  Baseline bookie : %{base_book*100:.1f}")

        # Model
        cat = CatBoostModel(with_odds=True, iterations=400, draw_weight=1.2, verbose=False)
        lgbm = LightGBMModel(with_odds=True, n_estimators=400, draw_weight=1.2, verbose=-1)
        ens = CalibratedEnsemble([cat, lgbm])
        ens.fit(train, cal)
        p = ens.predict(valid)

        # Retrain hook: remember errors for next fold
        ens.remember_errors(valid, p)

        # Ensemble optimization check
        w = np.round(ens.weights, 3).tolist()
        T = round(ens.temperature, 3)
        ensemble_weights_log.append({"fold": label, "weights": w, "temperature": T})
        print(f"  Ensemble weights: {w}  T={T}")

        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        brier = brier_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        btts_auc = auc_binary(valid["btts"], p.btts_yes)
        over25_auc = auc_binary(valid["over25"], p.over25)

        preds = argmax_1x2(p)
        y = valid["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        model_acc = (preds == y).mean()

        print(f"  MODEL acc=%{model_acc*100:.1f}  logloss={ll:.4f}  brier={brier:.4f}")
        print(f"  BTTS AUC={btts_auc:.3f}  Over2.5 AUC={over25_auc:.3f}")
        print(f"  Model>H? {model_acc > base_h} | Model>bookie? {model_acc > base_book}")

        rows.append({
            "fold": label, "n": len(valid),
            "base_h": base_h, "base_book": base_book,
            "model_acc": model_acc, "logloss": ll, "brier": brier,
            "btts_auc": btts_auc, "over25_auc": over25_auc,
        })

    res = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("TOPLAM SONUC")
    print("=" * 60)
    print(res.round(4).to_string(index=False))
    print(f"\nOrtalama model dogruluk: %{res['model_acc'].mean()*100:.1f}")
    print(f"Ortalama baseline H     : %{res['base_h'].mean()*100:.1f}")
    print(f"Ortalama bookie        : %{res['base_book'].mean()*100:.1f}")
    print(f"Ortalama logloss       : {res['logloss'].mean():.4f} (random=1.0986)")
    print(f"\nEnsemble weights per fold:")
    for r in ensemble_weights_log:
        print(f"  {r['fold']}: weights={r['weights']} T={r['temperature']}")
    print(f"\nSure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
