"""TAHMIN YAP + SONUCLARI KONTROL ET (ImprovedMatchModel).

Her fold'da:
  1. Gecmis veriyle modeli egit
  2. Gelecek sezon icin TAHMIN uret (en yuksek olasilikli sonuc)
  3. Gercek sonucla (result) karsilastir -> dogru/yanlis
  4. Accuracy + confusion + en uzun dogru/yanlis seri

Ayrica: bahis EDGE'i var mi diye odds ile kesisimde value-bet kontrolu.
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
I2R = {0: "H", 1: "D", 2: "A"}


def predict_and_check():
    t0 = time.time()
    feats = pd.read_parquet("data/gold/features.parquet")
    feats["date"] = pd.to_datetime(feats["date"])

    seasons = [
        ("2021-07-01", "2022-07-01", "2021/22"),
        ("2022-07-01", "2023-07-01", "2022/23"),
        ("2023-07-01", "2024-07-01", "2023/24"),
        ("2024-07-01", "2025-07-01", "2024/25"),
    ]

    all_true, all_pred = [], []
    odds_rows = []

    for (vs, ve, label) in seasons:
        v_start, v_end = pd.Timestamp(vs), pd.Timestamp(ve)
        valid = feats[(feats["date"] >= v_start) & (feats["date"] < v_end)].copy()
        if len(valid) < 1000:
            continue
        train_all = feats[feats["date"] < v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all) * 0.15))
        cal = train_all.iloc[-n_cal:]
        train = train_all.iloc[:-n_cal]

        # --- TAHMIN YAP ---
        model = ImprovedMatchModel(iterations=1200, n_models=2, verbose=False).fit(train, cal)
        p = model.predict(valid)

        # en yuksek olasilikli sonuc = tahmin
        pred_idx = np.argmax(np.vstack([p.home_win, p.draw, p.away_win]), axis=0)
        true_idx = valid["result"].map(R2I).to_numpy()

        acc = (pred_idx == true_idx).mean()
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        print(f"FOLD {label}: TAHMIN acc={acc*100:.1f}%  logloss={ll:.4f}  (n={len(valid)})")

        all_true.extend(true_idx.tolist())
        all_pred.extend(pred_idx.tolist())

        # --- ODDS var olanlarda bahis EDGE'i kontrolu ---
        ob = valid[["avg_home_odds", "avg_draw_odds", "avg_away_odds"]].notna().all(axis=1)
        if ob.sum() > 200:
            vb = evaluate_betting(valid[ob], p[ob] if False else
                                  type(p)(home_win=p.home_win[ob], draw=p.draw[ob], away_win=p.away_win[ob]),
                                  edge_threshold=0.03)
            odds_rows.append({
                "fold": label, "n_odds": int(ob.sum()),
                "vb_roi": vb["value_bet"]["roi"], "vb_bets": vb["value_bet"]["n_bets"],
                "clv": vb["clv"]["mean_clv"],
            })
            print(f"   -> odds'lu mac: {ob.sum()} | VB roi={vb['value_bet']['roi']*100:+.1f}% | CLV={vb['clv']['mean_clv']*100:+.2f}%")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall = (all_true == all_pred).mean()
    print("\n" + "=" * 60)
    print(f"GENEL TAHMIN DOGRULUGU: {overall*100:.1f}%  (toplam {len(all_true)} mac)")
    print("=" * 60)

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, pr in zip(all_true, all_pred):
        cm[t, pr] += 1
    print("\nConfusion (satir=gercek, kolon=tahmin):")
    print("        H     D     A")
    for i, r in enumerate("HDA"):
        print(f"  {r}:  {cm[i,0]:4d}  {cm[i,1]:4d}  {cm[i,2]:4d}")

    # En uzun dogru / yanlis seri
    run, mstr, mruns, yrun, ystr = 0, 0, 0, 0, 0
    for t, pr in zip(all_true, all_pred):
        if t == pr:
            run += 1; mstr = max(mstr, run); yrun = 0
        else:
            yrun += 1; ystr = max(ystr, yrun); run = 0
    print(f"\nEn uzun DOGRU seri: {mstr}")
    print(f"En uzun YANLIS seri: {ystr}")

    if odds_rows:
        odf = pd.DataFrame(odds_rows)
        print("\nOdds'lu maclar (bahis edge kontrolu):")
        print(odf.to_string(index=False))

    print(f"\nSure: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    predict_and_check()
