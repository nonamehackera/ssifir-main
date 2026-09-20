"""Hizli dogruluk diagnostici: baseline vs model.
- Baseline 1: her zaman 'H' (en sik sonuc)
- Baseline 2: bahis favorisi (dusuk odds = tahmini kazanan)
- Model: LightGBM (hizli) - sadece 'date' onceki veriyle egit, sonrakiyle test
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from models.lightgbm.model import LightGBMModel

t0 = time.time()
feats = pd.read_parquet("data/gold/features.parquet")
feats["date"] = pd.to_datetime(feats["date"])
# Sanity: target dagilimi
print("=== HEDEF DAGILIMI (tum veri) ===")
vc = feats["result"].value_counts(normalize=True)
print(vc.round(4).to_dict())
print(f"Majority (H) baseline dogruluk: {vc.max()*100:.1f}%")

# Bahis favorisi baseline: en dusuk avg odds -> o takim kazanir
def bookie_pick(row):
    odds = {"H": row["mkt_home_prob"], "D": row["mkt_draw_prob"], "A": row["mkt_away_prob"]}
    odds = {k: (v if pd.notna(v) else -1) for k, v in odds.items()}
    if max(odds.values()) <= 0:
        return "H"
    # en yuksek implied prob = favori
    best = max(odds, key=odds.get)
    return best

# Test: 2024-2025 sezonu (gercek out-of-sample)
test = feats[(feats["date"] >= "2024-07-01") & (feats["date"] < "2025-07-01")].copy()
train = feats[feats["date"] < "2024-07-01"].copy()

print(f"\n=== TEST SET: {len(test)} mac (2024/25 sezonu) ===")
# Baseline H
acc_h = (test["result"] == "H").mean()
# Baseline bookie
test["bookie"] = test.apply(bookie_pick, axis=1)
acc_book = (test["bookie"] == test["result"]).mean()
print(f"Baseline 'H'                 : %{acc_h*100:.1f}")
print(f"Baseline bahis favorisi     : %{acc_book*100:.1f}")

# Model (hizli: iterations dusuk, veri ornegi)
n_cal = max(200, int(len(train) * 0.15))
cal = train.iloc[-n_cal:]
tr = train.iloc[:-n_cal]
print(f"\nEgitim: {len(tr)} | Kalibrasyon: {len(cal)}")
lgb = LightGBMModel(with_odds=True, verbose=-1, draw_weight=1.2, n_estimators=200)
lgb.fit(tr, cal)
p = lgb.predict(test)

# LightGBM model returns PredictionResult
preds = np.where(p.home_win > p.draw,
                 np.where(p.home_win > p.away_win, "H", "A"),
                 np.where(p.draw > p.away_win, "D", "A"))
acc_model = (preds == test["result"].values).mean()
print(f"Model (LightGBM)            : %{acc_model*100:.1f}")
print(f"\nSure: {time.time()-t0:.0f}s")
print(f"\nSONUC: Model > Baseline H mi? {acc_model > acc_h} | Model > Bahis favorisi mi? {acc_model > acc_book}")
