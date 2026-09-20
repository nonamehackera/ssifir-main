"""Kapsamli model degerlendirme - FBref dahil tum veriyle.

Calistirma:
  .venv/bin/python scripts/evaluate_full.py
"""
import logging
import sys
import time

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from evaluation.metrics import brier_1x2, log_loss_1x2


def load_data():
    feats = pd.read_parquet("data/gold/features.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    matches = pd.read_parquet("data/gold/matches.parquet")
    feats = feats.merge(
        matches[["match_id", "home_team", "away_team", "league_name"]],
        on="match_id", how="left")
    return feats


def main():
    feats = load_data()
    print("=" * 72)
    print("VERI DURUMU")
    print("=" * 72)
    for prefix, label in [("", "FD (istatistikli)"), ("XG_", "xgabora (38 lig)"),
                          ("FB_", "FBref (istatistikli)"), ("HF_", "HF (istatistiksiz)"),
                          ("OF_", "OpenFootball")]:
        mask = feats["league"].str.startswith(prefix) if prefix else ~feats["league"].str.startswith(("FB_", "HF_", "OF_", "TS_", "XG_"))
        sub = feats[mask]
        if len(sub) == 0:
            continue
        n_stats = sub["home_shots"].notna().sum()
        print(f"  {label:22s}: {len(sub):>7d} mac | istatistikli: {n_stats:>6d}")

    stats = feats["home_shots"].notna()
    print(f"\n  TOPLAM istatistikli mac: {stats.sum()} / {len(feats)}")

    # Egitim: TUM maclar (istatistikli + istatistiksiz).
    # Istatistiksiz maclar da kullanilir: Elo, form, h2h, oran feature'lari
    # mevcut; shot/corner NaN olunca CatBoost/LightGBM bunu dogal isler.
    train_all = feats[feats["date"] < "2024-01-01"].copy()
    n_cal = max(200, int(len(train_all) * 0.15))
    cal = train_all.iloc[-n_cal:]
    train = train_all.iloc[:-n_cal]

    # Test: 2024 ilk yari - TUM liglerden (istatistikli veya degil)
    test_all = feats[(feats["date"] >= "2024-01-01") & (feats["date"] < "2024-07-01")]
    test = test_all.sample(n=min(500, len(test_all)), random_state=42).sort_values("date")
    print(f"\nEgitim: {len(train)} | Kalibrasyon: {len(cal)} | Test: {len(test)} "
          f"(testte istatistikli: {test['home_shots'].notna().sum()})")

    from models.catboost.model import CatBoostModel
    from models.ensemble.calibrated_ensemble import CalibratedEnsemble
    from models.lightgbm.model import LightGBMModel

    t0 = time.time()
    cat = CatBoostModel(with_odds=True, iterations=300, draw_weight=1.2)
    lgbm = LightGBMModel(with_odds=True, verbose=-1, draw_weight=1.2)
    ens = CalibratedEnsemble([cat, lgbm])
    ens.fit(train, cal)
    p = ens.predict(test)

    ll = log_loss_1x2(test["result"], p.home_win, p.draw, p.away_win)
    brier = brier_1x2(test["result"], p.home_win, p.draw, p.away_win)
    preds = np.where(p.home_win > p.draw,
                     np.where(p.home_win > p.away_win, "H", "A"),
                     np.where(p.draw > p.away_win, "D", "A"))
    correct = (preds == test["result"].values).sum()
    probs = np.maximum(p.home_win, np.maximum(p.draw, p.away_win))

    print(f"\n{'='*72}")
    print(f"SONUC (istatistikli veri, {time.time()-t0:.0f}s)")
    print(f"{'='*72}")
    print(f"Log Loss:  {ll:.4f}  (baseline: 1.0986)")
    print(f"Brier:     {brier:.4f}")
    print(f"Dogruluk:  {correct}/{len(test)} = %{correct/len(test)*100:.1f}")

    pred_h, pred_d, pred_a = (preds == "H").sum(), (preds == "D").sum(), (preds == "A").sum()
    real_h, real_d, real_a = (test["result"] == "H").sum(), (test["result"] == "D").sum(), (test["result"] == "A").sum()
    print(f"\nTahmin:    H={pred_h:3d}(%{pred_h/len(test)*100:.1f}) D={pred_d:3d}(%{pred_d/len(test)*100:.1f}) A={pred_a:3d}(%{pred_a/len(test)*100:.1f})")
    print(f"Gercek:    H={real_h:3d}(%{real_h/len(test)*100:.1f}) D={real_d:3d}(%{real_d/len(test)*100:.1f}) A={real_a:3d}(%{real_a/len(test)*100:.1f})")

    print(f"\n--- Guven Esikleri ---")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        mask = probs > t
        if mask.sum() > 0:
            acc = (preds[mask] == test["result"].values[mask]).mean()
            print(f"  >%{t*100:.0f}: {mask.sum():3d} mac, dogruluk=%{acc*100:.1f}")

    real_d_mask = test["result"] == "D"
    pred_d_mask = preds == "D"
    hit = int((real_d_mask & pred_d_mask).sum())
    print(f"\n--- Draw ---")
    print(f"Gercek D: {real_d_mask.sum()}, Tahmin D: {pred_d_mask.sum()}, Dogru: {hit}, Kacirilan: {int((real_d_mask & ~pred_d_mask).sum())}")

    print(f"\n--- Lig Bazli ---")
    test2 = test.copy()
    test2["pred"] = preds
    test2["correct"] = preds == test["result"].values
    test2["prob"] = probs
    for lg, grp in test2.groupby("league"):
        if len(grp) < 10:
            continue
        lg_name = grp["league_name"].iloc[0] if "league_name" in grp else lg
        print(f"  {lg_name[:28]:28s}: {len(grp):3d} mac, dogruluk=%{grp['correct'].mean()*100:.1f}")

    print(f"\n--- Yanlis Tahminler (güven yuksek, ilk 15) ---")
    wrong = test2[~test2["correct"]].sort_values("prob", ascending=False)
    for _, row in wrong.head(15).iterrows():
        lg = row.get("league_name", row["league"])[:22]
        print(f"  {lg:22s} {str(row['date'].date())} {str(row['home_team'])[:16]:16s} vs {str(row['away_team'])[:16]:16s} | G={row['result']} T={row['pred']} P={row['prob']:.3f}")

    print(f"\n--- Dogru Tahminler (güven yuksek, ilk 10) ---")
    good = test2[test2["correct"]].sort_values("prob", ascending=False)
    for _, row in good.head(10).iterrows():
        lg = row.get("league_name", row["league"])[:22]
        print(f"  {lg:22s} {str(row['date'].date())} {str(row['home_team'])[:16]:16s} vs {str(row['away_team'])[:16]:16s} | G={row['result']} T={row['pred']} P={row['prob']:.3f}")


if __name__ == "__main__":
    main()