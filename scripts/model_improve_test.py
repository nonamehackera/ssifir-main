"""MODEL IYILESTIRME TESTI (walk-forward, dürüst).

Karsilastirma:
  - Elo baseline (aptal saptal mi, degil mi?)
  - Eski CatBoost (with_odds=False, draw_weight=1.2 sabit)
  - ImprovedMatchModel (feature selection + draw_weight tune + voting)

Metrikler:
  - accuracy (dogru tahmin orani)
  - log_loss (kalibrasyon)
  - max_streak / min_streak (model tutarli mi? 'aptal saptal' mi?)

Veri: hazir features.parquet (82K, 2021+).
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from models.elo.model import EloModel
from models.catboost.model import CatBoostModel
from models.improved.model import ImprovedMatchModel
from evaluation.metrics import accuracy_1x2, log_loss_1x2

def max_streak(y_true, y_pred):
    """En uzun dogru tahmin serisi (tutarli mi?)."""
    streak = run = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            run += 1; streak = max(streak, run)
        else:
            run = 0
    return streak

def min_streak(y_true, y_pred):
    """En uzun YANLIS tahmin serisi (surekli hata var mi?)."""
    streak = run = 0
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            run += 1; streak = max(streak, run)
        else:
            run = 0
    return streak

def argmax_1x2(p):
    ph, pd_, pa = np.asarray(p.home_win), np.asarray(p.draw), np.asarray(p.away_win)
    return np.select([ph >= pd_, pa >= ph, pa >= pd_], [0, 2, 1], default=1)

def run():
    t0 = time.time()
    feats = pd.read_parquet("data/gold/features.parquet")
    feats["date"] = pd.to_datetime(feats["date"])

    seasons = [
        ("2021-07-01","2022-07-01","2021/22"),
        ("2022-07-01","2023-07-01","2022/23"),
        ("2023-07-01","2024-07-01","2023/24"),
        ("2024-07-01","2025-07-01","2024/25"),
    ]

    rows = []
    for (vs, ve, label) in seasons:
        v_start, v_end = pd.Timestamp(vs), pd.Timestamp(ve)
        valid = feats[(feats["date"]>=v_start)&(feats["date"]<v_end)].copy()
        if len(valid) < 1000:
            continue
        train_all = feats[feats["date"]<v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all)*0.15))
        cal = train_all.iloc[-n_cal:]; train = train_all.iloc[:-n_cal:]
        print(f"\n=== FOLD {label}: train={len(train)} cal={len(cal)} valid={len(valid)} ===")

        # y_true
        y = valid["result"].map({"H":0,"D":1,"A":2}).to_numpy()

        results = {}

        # 1) Elo baseline
        elo = EloModel().fit(train, cal)
        p = elo.predict(valid)
        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        pred = argmax_1x2(p)
        results["elo"] = (acc, ll, max_streak(y, pred), min_streak(y, pred))

        # 2) Eski CatBoost (sabit draw_weight)
        cb = CatBoostModel(with_odds=False, iterations=800, draw_weight=1.2, verbose=False).fit(train, cal)
        p = cb.predict(valid)
        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        pred = argmax_1x2(p)
        results["catboost_old"] = (acc, ll, max_streak(y, pred), min_streak(y, pred))

        # 3) ImprovedMatchModel
        imp = ImprovedMatchModel(iterations=1200, n_models=2, verbose=False).fit(train, cal)
        p = imp.predict(valid)
        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        pred = argmax_1x2(p)
        results["improved"] = (acc, ll, max_streak(y, pred), min_streak(y, pred))

        for name, (acc, ll, mx, mn) in results.items():
            print(f"  [{name:12s}] acc={acc*100:.1f}% ll={ll:.4f} max_streak={mx} min_streak={mn}")

        for name, (acc, ll, mx, mn) in results.items():
            rows.append({"fold": label, "model": name, "acc": acc, "logloss": ll,
                         "max_streak": mx, "min_streak": mn})

    tab = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("MODEL IYILESTIRME SONUCI (ortalama)")
    print("="*70)
    summ = tab.groupby("model").agg(acc=("acc","mean"), logloss=("logloss","mean"),
                                    max_streak=("max_streak","mean"), min_streak=("min_streak","mean"))
    summ["acc"] = (summ["acc"]*100).round(1)
    summ["logloss"] = summ["logloss"].round(4)
    summ["max_streak"] = summ["max_streak"].round(0).astype(int)
    summ["min_streak"] = summ["min_streak"].round(0).astype(int)
    print(summ.to_string())
    print(f"\nSure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run()
