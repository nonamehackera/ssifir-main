"""LightGBM hizli test - sadece buyuk ligler."""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import log_loss_1x2, accuracy_1x2
from models.lightgbm.model import LightGBMModel

print("Veri yukleniyor...")
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features

matches = build_matches()
feats = build_features(matches)

# Sadece buyuk ligleri al (hizlandirmak icin)
top_leagues = ["E0", "SP1", "D1", "I1", "F1", "N1", "P1", "T1", "B1"]
feats = feats[feats["league"].isin(top_leagues)].copy()
print(f"Toplam: {len(feats)} mac ({len(top_leagues)} buyuk lig)")

feats_sorted = feats.sort_values("date").reset_index(drop=True)
val_start = pd.Timestamp("2024-04-01")
val_end = pd.Timestamp("2024-10-01")
valid = feats_sorted[(feats_sorted["date"] >= val_start) & (feats_sorted["date"] < val_end)]
train_all = feats_sorted[feats_sorted["date"] < val_start]
n_cal = max(50, int(len(train_all) * 0.15))
cal = train_all.iloc[-n_cal:]
train = train_all.iloc[:-n_cal]
print(f"Train: {len(train)}, Cal: {len(cal)}, Valid: {len(valid)}")

naive_ll = 1.0986

# LightGBM (no odds)
print("\n--- LightGBM (no odds) ---")
t0 = time.time()
lgbm = LightGBMModel(with_odds=False, n_estimators=200, learning_rate=0.05, num_leaves=63, verbose=-1)
lgbm.fit(train, cal)
pred = lgbm.predict(valid)
ll = log_loss_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
acc = accuracy_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
print(f"  LL={ll:.4f} Acc={acc:.3f} Sure={time.time()-t0:.1f}s")

# LightGBM (with odds)
print("\n--- LightGBM (with odds) ---")
t0 = time.time()
lgbm_odds = LightGBMModel(with_odds=True, n_estimators=200, learning_rate=0.05, num_leaves=63, verbose=-1)
lgbm_odds.fit(train, cal)
pred = lgbm_odds.predict(valid)
ll_odds = log_loss_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
acc_odds = accuracy_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
print(f"  LL={ll_odds:.4f} Acc={acc_odds:.3f} Sure={time.time()-t0:.1f}s")

print(f"\nBaseline: {naive_ll:.4f}")
print(f"LightGBM no odds:  {'✓' if ll < naive_ll else '✗'} {ll:.4f}")
print(f"LightGBM w/ odds:  {'✓' if ll_odds < naive_ll else '✗'} {ll_odds:.4f}")
