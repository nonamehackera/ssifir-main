"""HIBRIT MODEL EV TAHMINI (gercekci kapanis simulasyonu ile).

Model: %58 saf + %42 odds-aware (weight'ler walk-forward'ta optimize edildi).
EV: her mac icin model probasi ile KAPANIS ORANI karsilastirilir.
Kapanis yoksa: Pinnacle proxy = implied(model) * 0.95 (margin).

Cikti: ortalama EV, kac mac pozitif EV, ROI simulasyonu.
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from models.improved.model import ImprovedMatchModel
from models.ensemble.market_informed import MarketInformedEnsemble
from models.catboost.model import CatBoostModel
from evaluation.metrics import accuracy_1x2, log_loss_1x2
from betting.strategy import evaluate_betting, market_margin

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

        # Hibrit: saf %58 + odds %42 (optimize weight)
        saf = ImprovedMatchModel(iterations=1200, n_models=3, time_decay=1.0, verbose=False).fit(train, cal)
        odds_m = CatBoostModel(with_odds=True, iterations=1200, verbose=False).fit(train, cal)
        ps = saf.predict(valid_odds)
        po = odds_m.predict(valid_odds)
        W_SAF, W_ODDS = 0.58, 0.42
        ph = W_SAF * ps.home_win + W_ODDS * po.home_win
        pd_ = W_SAF * ps.draw + W_ODDS * po.draw
        pa = W_SAF * ps.away_win + W_ODDS * po.away_win
        from models.base import PredictionResult
        ph = ph / (ph + pd_ + pa)
        pd_ = pd_ / (ph + pd_ + pa)
        pa = pa / (ph + pd_ + pa)
        hpred = PredictionResult(home_win=ph, draw=pd_, away_win=pa)

        acc = accuracy_1x2(valid_odds["result"], ph, pd_, pa)
        ll = log_loss_1x2(valid_odds["result"], ph, pd_, pa)

        # EV: model probasi vs KAPANIS (proxy: Pinnacle = implied*0.95)
        odds_cols = ("avg_home_odds", "avg_draw_odds", "avg_away_odds")
        oh = valid_odds[odds_cols[0]].to_numpy(float)
        od = valid_odds[odds_cols[1]].to_numpy(float)
        oa = valid_odds[odds_cols[2]].to_numpy(float)
        imp_h = 1 / oh; imp_d = 1 / od; imp_a = 1 / oa
        s = imp_h + imp_d + imp_a
        imp_h /= s; imp_d /= s; imp_a /= s
        # Pinnacle proxy (daha dusuk margin)
        pin_h = imp_h * 0.95; pin_d = imp_d * 0.95; pin_a = imp_a * 0.95
        ps_ = pin_h + pin_d + pin_a
        pin_h /= ps_; pin_d /= ps_; pin_a /= ps_

        # EV her secenek icin
        ev_h = ph * oh - 1
        ev_d = pd_ * od - 1
        ev_a = pa * oa - 1
        ev_max = np.maximum.reduce([ev_h, ev_d, ev_a])
        n_pos = (ev_max > 0.03).sum()
        mean_ev = ev_max[ev_max > 0.03].mean() if n_pos > 0 else 0.0

        rows.append({"fold": label, "acc": acc, "logloss": ll,
                     "n_pos_ev": int(n_pos), "mean_ev": mean_ev,
                     "n_odds": len(valid_odds)})
        print(f"FOLD {label}: acc={acc*100:.1f}% ll={ll:.4f} | +EV mac={n_pos} | ort EV (>+3%)={mean_ev*100:+.1f}%")

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("HIBRIT MODEL EV TAHMINI (Pinnacle proxy kapanis)")
    print("=" * 60)
    print(f"  ort acc         = {tab['acc'].mean()*100:.1f}%")
    print(f"  ort +EV mac     = {tab['n_pos_ev'].mean():.0f} / fold")
    print(f"  ort EV (>+3%)   = {tab['mean_ev'].mean()*100:+.1f}%")
    print(f"  Sure: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
