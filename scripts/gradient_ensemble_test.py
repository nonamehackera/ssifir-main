"""GRADIENT ENSEMBLE ACCURACY TESTI (hafif versiyon).

Karsilastirma:
  - ImprovedMatchModel (Balanced'siz, draw_weight tune)
  - GradientEnsemble (sadece LightGBM, n_models=1) -> bellek dostu

Hedef: %60+ accuracy.
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from models.improved.model import ImprovedMatchModel
from models.improved.gradient_ensemble import GradientEnsemble
from evaluation.metrics import accuracy_1x2, log_loss_1x2

R2I = {"H": 0, "D": 1, "A": 2}


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
    for (vs, ve, label) in seasons:
        v_start = pd.Timestamp(vs)
        valid = feats[(feats["date"] >= v_start) & (feats["date"] < pd.Timestamp(ve))].copy()
        if len(valid) < 1000:
            continue
        train_all = feats[feats["date"] < v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all) * 0.15))
        cal = train_all.iloc[-n_cal:]
        train = train_all.iloc[:-n_cal]
        print(f"\n=== FOLD {label}: train={len(train)} cal={len(cal)} valid={len(valid)} ===")

        # ImprovedMatchModel (Balanced'siz)
        cb = ImprovedMatchModel(iterations=1000, n_models=2, time_decay=1.0, verbose=False).fit(train, cal)
        p = cb.predict(valid)
        acc_cb = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll_cb = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        print(f"  [improved_catboost] acc={acc_cb*100:.1f}% ll={ll_cb:.4f}")

        # GradientEnsemble (LightGBM only, n_models=1)
        ge = GradientEnsemble(iterations=1000, n_models=1, time_decay=1.0, verbose=False).fit(train, cal)
        p = ge.predict(valid)
        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        print(f"  [gradient_ensemble] acc={acc*100:.1f}% ll={ll:.4f}")
        rows.append({"fold": label, "cb_acc": acc_cb, "ge_acc": acc, "ge_ll": ll})

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SONUC")
    print("=" * 60)
    print(f"  ImprovedMatchModel ort acc: {tab['cb_acc'].mean()*100:.1f}%")
    print(f"  GradientEnsemble   ort acc: {tab['ge_acc'].mean()*100:.1f}%")
    print(f"  Artis: {(tab['ge_acc'].mean()-tab['cb_acc'].mean())*100:+.1f} puan")
    print(f"\nSure: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
