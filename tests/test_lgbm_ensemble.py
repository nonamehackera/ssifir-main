"""LightGBM + Ensemble testi."""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import log_loss_1x2, brier_1x2, accuracy_1x2
from models.catboost.model import CatBoostModel
from models.ensemble.ensemble import EnsembleModel
from models.elo.model import EloModel
from models.lightgbm.model import LightGBMModel
from models.poisson.model import DixonColesModel

print("Veri yukleniyor...")
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features

matches = build_matches()
feats = build_features(matches)
print(f"Features: {len(feats)} satir, {feats.shape[1]} sutun")

# Split
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
results = {}

def run_model(name, model_fn):
    t0 = time.time()
    model = model_fn()
    model.fit(train, cal)
    pred = model.predict(valid)
    ll = log_loss_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    acc = accuracy_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    elapsed = time.time() - t0
    print(f"  {name:20s} LL={ll:.4f} Acc={acc:.3f} Sure={elapsed:.1f}s")
    results[name] = ll
    return model

# LightGBM (no odds)
print("\n--- LightGBM (no odds) ---")
try:
    run_model("lightgbm", lambda: LightGBMModel(with_odds=False, iterations=300, learning_rate=0.05))
except Exception as e:
    print(f"  HATA: {e}")

# LightGBM (with odds)
print("\n--- LightGBM (with odds) ---")
try:
    run_model("lightgbm_odds", lambda: LightGBMModel(with_odds=True, iterations=300, learning_rate=0.05))
except Exception as e:
    print(f"  HATA: {e}")

# Ensemble
print("\n--- Ensemble (Elo + DC + CatBoost + LightGBM) ---")
try:
    t0 = time.time()
    ens = EnsembleModel([EloModel(), DixonColesModel(), CatBoostModel(with_odds=False), LightGBMModel(with_odds=False)])
    ens.fit(train, cal)
    pred = ens.predict(valid)
    ll = log_loss_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    acc = accuracy_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    elapsed = time.time() - t0
    print(f"  ensemble          LL={ll:.4f} Acc={acc:.3f} Sure={elapsed:.1f}s")
    print(f"  Agirliklar: {np.round(ens.weights, 3)}")
    print(f"  Temperature: {ens.temperature:.3f}")
    results["ensemble"] = ll
except Exception as e:
    print(f"  HATA: {e}")

print()
print("=" * 60)
print("TUM SONUCLAR (log_loss kucukten buyuge)")
print("=" * 60)
for name, ll in sorted(results.items(), key=lambda x: x[1]):
    status = "✓" if ll < naive_ll else "✗"
    print(f"  {status} {name:20s} {ll:.4f}")
print(f"\n  Baseline: {naive_ll:.4f}")
