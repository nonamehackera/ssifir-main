"""TUM MODeller - parquet'ten oku, hizli karsilastirma."""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import log_loss_1x2, brier_1x2, accuracy_1x2, log_loss_binary, auc_binary
from models.catboost.model import CatBoostModel, feature_columns
from models.ensemble.ensemble import EnsembleModel
from models.elo.model import EloModel
from models.lightgbm.model import LightGBMModel
from models.poisson.model import DixonColesModel

print("=" * 70)
print("  TUM MODELLER - KARSILASTIRMA TESTI")
print("=" * 70)

# Feature'lari parquet'ten oku (cok hizli)
print("\n1. Feature parquet'ten okunuyor...")
t0 = time.time()
feats = pd.read_parquet("data/gold/features.parquet")
feats["date"] = pd.to_datetime(feats["date"])
print(f"   Sure: {time.time()-t0:.1f}s | {len(feats)} satir, {feats.shape[1]} sutun")

# Feature dogrulama
cat_cols = feature_columns(with_odds=False)
cat_cols_odds = feature_columns(with_odds=True)
missing = [c for c in cat_cols if c not in feats.columns]
print(f"   Feature: {len(cat_cols)} (no odds), {len(cat_cols_odds)} (odds)")
print(f"   Eksik: {missing if missing else 'YOK'}")
for f in ["home_w_corners", "away_w_corners"]:
    cnt = feats[f].notna().sum() if f in feats.columns else 0
    print(f"   {f}: {cnt}/{len(feats)} dolu")

# Split
print("\n2. Veri bolunuyor...")
feats_sorted = feats.sort_values("date").reset_index(drop=True)
val_start = pd.Timestamp("2024-04-01")
val_end = pd.Timestamp("2024-10-01")
valid = feats_sorted[(feats_sorted["date"] >= val_start) & (feats_sorted["date"] < val_end)]
train_all = feats_sorted[feats_sorted["date"] < val_start]
n_cal = max(50, int(len(train_all) * 0.15))
cal = train_all.iloc[-n_cal:]
train = train_all.iloc[:-n_cal]
print(f"   Train: {len(train)}, Cal: {len(cal)}, Valid: {len(valid)}")

naive_ll = 1.0986
results = {}

def run_and_eval(name, model):
    t0 = time.time()
    model.fit(train, cal)
    fit_time = time.time() - t0
    t1 = time.time()
    pred = model.predict(valid)
    pred_time = time.time() - t1
    ll = log_loss_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    br = brier_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    acc = accuracy_1x2(valid["result"], pred.home_win, pred.draw, pred.away_win)
    btts_ll = log_loss_binary(valid["btts"].to_numpy(), pred.btts_yes) if pred.btts_yes is not None else None
    over_auc = auc_binary(valid["over25"].to_numpy(), pred.over25) if pred.over25 is not None else None
    status = "✓" if ll < naive_ll else "✗"
    results[name] = {"log_loss": ll, "brier": br, "accuracy": acc, "btts_ll": btts_ll, "over_auc": over_auc}
    print(f"  {status} {name:20s} LL={ll:.4f} Br={br:.4f} Acc={acc:.3f} fit={fit_time:.1f}s pred={pred_time:.1f}s")
    return model

# 1. Elo
print("\n3. Modeller egitiliyor ve degerlendiriliyor:")
print("   " + "-" * 65)

run_and_eval("elo", EloModel())

# 2. Dixon-Coles
print("   [Dixon-Coles ~90s surebilir...]")
run_and_eval("dixon_coles", DixonColesModel())

# 3. CatBoost (no odds)
run_and_eval("catboost", CatBoostModel(with_odds=False))

# 4. CatBoost (with odds)
run_and_eval("catboost_odds", CatBoostModel(with_odds=True))

# 5. LightGBM (no odds)
run_and_eval("lightgbm", LightGBMModel(with_odds=False, verbose=-1))

# 6. LightGBM (with odds)
run_and_eval("lightgbm_odds", LightGBMModel(with_odds=True, verbose=-1))

# 7. Ensemble
print("\n   [Ensemble: 4 model egitimi + agirlik optimizasyonu]")
run_and_eval("ensemble", EnsembleModel([EloModel(), DixonColesModel(), CatBoostModel(with_odds=False), LightGBMModel(with_odds=False, verbose=-1)]))

# OZET
print("\n" + "=" * 70)
print("  SONUCLAR (log_loss kucukten buyuge)")
print("=" * 70)
print(f"  {'Model':20s} {'LogLoss':>8s} {'Brier':>8s} {'Acc':>6s} {'BTTS_LL':>8s} {'O25_AUC':>8s} {'Durum':>6s}")
print("  " + "-" * 65)
for name, m in sorted(results.items(), key=lambda x: x[1]["log_loss"]):
    status = "✓" if m["log_loss"] < naive_ll else "✗"
    btts = f"{m['btts_ll']:.4f}" if m["btts_ll"] else "N/A"
    oauc = f"{m['over_auc']:.3f}" if m["over_auc"] else "N/A"
    print(f"  {name:20s} {m['log_loss']:8.4f} {m['brier']:8.4f} {m['accuracy']:6.3f} {btts:>8s} {oauc:>8s} {status:>6s}")

print(f"\n  Naive baseline: {naive_ll:.4f}")
print(f"  Feature sayisi: {len(cat_cols)} (no odds), {len(cat_cols_odds)} (odds)")
print("=" * 70)
