"""SAF MODEL (+EV arayisi) - bookmaker kopyalamadan.

ImprovedMatchModel zaten with_odds=False (saf sinyal).
Bu script: saf modelin ACIKLIS ORANINA KARSI edge'ini olcer.
Eger VB roi > 0 ise -> model bookmaker'i beat ediyor = GERCEK +EV.

NOT: HF/FD tek oran verdigi icin kapanis yok -> CLV gercek olamaz,
sadece 'acilis edge' olceriz. Bu teorik +EV'dir.
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from models.improved.model import ImprovedMatchModel
from evaluation.metrics import accuracy_1x2, log_loss_1x2
from betting.strategy import evaluate_betting

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
        ob = valid[["avg_home_odds", "avg_draw_odds", "avg_away_odds"]].notna().all(axis=1)
        valid_odds = valid[ob]
        if len(valid_odds) < 500:
            continue
        train_all = feats[feats["date"] < v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all) * 0.15))
        cal = train_all.iloc[-n_cal:]
        train = train_all.iloc[:-n_cal]

        model = ImprovedMatchModel(iterations=1200, n_models=2, verbose=False).fit(train, cal)
        p = model.predict(valid_odds)

        acc = accuracy_1x2(valid_odds["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid_odds["result"], p.home_win, p.draw, p.away_win)

        # edge esiklerini tarayalim: 0.03, 0.05, 0.10
        for thr in [0.03, 0.05, 0.10]:
            vb = evaluate_betting(valid_odds, p, edge_threshold=thr)
            rows.append({
                "fold": label, "thr": thr, "acc": acc, "logloss": ll,
                "n_bets": vb["value_bet"]["n_bets"], "vb_roi": vb["value_bet"]["roi"],
                "vb_yield": vb["value_bet"]["yield"],
                "ab_roi": vb["always_best"]["roi"], "ab_hit": vb["always_best"]["hit_rate"],
            })

        print(f"FOLD {label}: acc={acc*100:.1f}% ll={ll:.4f} | odds'lu={len(valid_odds)}")

    tab = pd.DataFrame(rows)
    tab.to_csv("data/gold/ev_search_results.csv", index=False)
    print("\n" + "=" * 70)
    print("SAF MODEL +EV ARAMA (edge esigi taramasi)")
    print("=" * 70)
    for thr in [0.03, 0.05, 0.10]:
        sub = tab[tab["thr"] == thr]
        print(f"\nedge>={thr}:")
        print(f"  ort VB roi = {sub['vb_roi'].mean()*100:+.1f}%  (bets={sub['n_bets'].sum()})")
        print(f"  ort yield  = {sub['vb_yield'].mean()*100:+.2f}%")
        print(f"  ort acc    = {sub['acc'].mean()*100:.1f}%")
        print(f"  AB roi     = {sub['ab_roi'].mean()*100:+.1f}%  (basit favori bahsi)")
    print(f"\nSure: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
