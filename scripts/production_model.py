"""PRODUCTION MODEL v6 - HEPSI BIR ARADA
1. XGBoost (CatBoost yerine)
2. Dixon-Coles Poisson
3. Walk-forward validation
4. Real-time tahmin hazir

Tum modelleri train et, ensemble yap, degerlendir.
"""
import sys, warnings, time, gc, os
warnings.filterwarnings("ignore")
os.environ["LIGHTGBM_WARNINGS"] = "0"
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import json

from models.dixon_coles import DixonColesModel
from models.walk_forward import walk_forward_backtest

t0 = time.time()
print("=" * 80)
print("  PRODUCTION MODEL v6 - HEPSI BIR ARADA")
print("=" * 80)

# ===========================
# 1. DATA LOADING & PREP
# ===========================
print("\n[1/5] Veri yukleniyor...", flush=True)

df = pd.read_parquet("data/gold/features_enhanced_v5.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    df[c] = df[c].fillna(0).astype(int)
df["total_goals"] = df["home_goals"] + df["away_goals"]
df["result"] = df["result"].fillna("D")
df = df.dropna(subset=["result", "elo_diff"])

# Draw features
df["form_draw_pct"] = (df["home_draws_last5"].fillna(0) + df["away_draws_last5"].fillna(0)) / 10.0
df["elo_close_match"] = (df["elo_diff"].abs() < 100).astype(float)
df["both_weak_attack"] = ((df["home_gf_5"].fillna(0) < 1.0) & (df["away_gf_5"].fillna(0) < 1.0)).astype(float)
if "mkt_home_prob" in df.columns:
    df["mkt_total_prob"] = df["mkt_home_prob"].fillna(0) + df["mkt_draw_prob"].fillna(0) + df["mkt_away_prob"].fillna(0)
    df["mkt_overround"] = df["mkt_total_prob"] - 1.0

# Feature selection
LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading","ht_home_goals","ht_away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners","home_yellow","away_yellow","home_red","away_red","shots_diff","sot_diff","home_goals","away_goals","result","result_H","result_D","result_A","btts","over25","over15","over35","total_goals","match_id","season","date","referee"}
DEAD = {"home_xgot","away_xgot","home_big_chances","away_big_chances","home_xg_flash","away_xg_flash","home_fouls","away_fouls","home_possession","away_possession","home_xgot_5","away_xgot_5","home_big_chances_5","away_big_chances_5","home_xg_5","away_xg_5","home_fouls_5","away_fouls_5","home_possession_5","away_possession_5"}
CLOSING = {"mkt_close_home_prob","mkt_close_draw_prob","mkt_close_away_prob","mkt_close_over25_prob","avg_close_home_odds","avg_close_draw_odds","avg_close_away_odds","avg_close_over25_odds"}
exclude = LEAK | DEAD | CLOSING
CAT = ["league", "home_team_id", "away_team_id"]
cats = [c for c in CAT if c in df.columns]
nums = [c for c in df.columns if c not in exclude and c not in cats and df[c].dtype in ["float64","int64","float32","int32"]]
cols = cats + nums
for c in cols:
    if c not in CAT:
        df[c] = df[c].fillna(0)

# Encoding
df_lgb = df.copy()
for c in CAT:
    if c in df_lgb.columns:
        df_lgb[c] = df_lgb[c].astype(str).astype("category").cat.codes

print(f"  Veri: {len(df)} satir, {len(cols)} feature")
print(f"  Tarih: {df['date'].min().strftime('%Y-%m-%d')} -> {df['date'].max().strftime('%Y-%m-%d')}")

# Split
tr = df_lgb[(df_lgb["date"] >= "2020-01-01") & (df_lgb["date"] < "2025-03-01")]
va = df_lgb[(df_lgb["date"] >= "2025-03-01") & (df_lgb["date"] < "2025-09-01")]
te = df_lgb[df_lgb["date"] >= "2025-09-01"]
print(f"  Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")

# ===========================
# 2. XGBOOST ENSEMBLE
# ===========================
print(f"\n[2/5] XGBoost egitimi...", flush=True)

Xtr = tr[cols].astype(np.float32).values
ytr = tr["result"].map({"H":0,"D":1,"A":2}).values
Xva = va[cols].astype(np.float32).values
yva = va["result"].map({"H":0,"D":1,"A":2}).values
Xte = te[cols].astype(np.float32).values
yte = te["result"].map({"H":0,"D":1,"A":2}).values

# LGB x3
lgb_models, lgb_ws = [], []
for s in [42, 123, 789]:
    t1 = time.time()
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, num_leaves=50,
        learning_rate=0.03, n_estimators=500, max_depth=6,
        min_child_samples=80, subsample=0.75, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=s,
        verbose=-1, n_jobs=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    va_acc = accuracy_score(yva, m.predict(Xva))
    lgb_models.append(m); lgb_ws.append(va_acc)
    print(f"  LGB s={s}: {time.time()-t1:.0f}s val_acc={va_acc:.4f}")

# XGBoost x3
xgb_models, xgb_ws = [], []
for s in [42, 123, 789]:
    t1 = time.time()
    m = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, max_depth=6,
        learning_rate=0.03, n_estimators=500, subsample=0.75,
        colsample_bytree=0.65, reg_alpha=0.5, reg_lambda=5.0,
        random_state=s, eval_metric="mlogloss", use_label_encoder=False,
        n_jobs=-1, verbosity=0,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    va_acc = accuracy_score(yva, m.predict(Xva))
    xgb_models.append(m); xgb_ws.append(va_acc)
    print(f"  XGB s={s}: {time.time()-t1:.0f}s val_acc={va_acc:.4f}")

# ===========================
# 3. ENSEMBLE + CALIBRATION
# ===========================
print(f"\n[3/5] Ensemble + Kalibrasyon...", flush=True)

all_models = lgb_models + xgb_models
all_ws = np.array(lgb_ws + xgb_ws); all_ws = all_ws / all_ws.sum()
print(f"  Agirliklar: LGB={[f'{w:.3f}' for w in all_ws[:3]]} XGB={[f'{w:.3f}' for w in all_ws[3:]]}")

# Val
all_val = [m.predict_proba(Xva) for m in all_models]
raw_val = np.average(all_val, axis=0, weights=all_ws)

# Calibration
cal = []
for c in range(3):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val[:, c], (yva == c).astype(float))
    cal.append(ir)

# Test
all_te = [m.predict_proba(Xte) for m in all_models]
raw_te = np.average(all_te, axis=0, weights=all_ws)
cp = np.column_stack([cal[c].predict(raw_te[:, c]) for c in range(3)])
s = cp.sum(axis=1, keepdims=True); s = np.where(s > 0, s, 1); cp /= s

preds = np.argmax(cp, axis=1)
conf = cp.max(axis=1)

# ===========================
# 4. METRIKLER
# ===========================
print(f"\n[4/5] Metrikler...", flush=True)

acc = accuracy_score(yte, preds)
draw_r = (preds[yte == 1] == 1).mean() if (yte == 1).sum() else 0
brier = np.mean([brier_score_loss((yte == c).astype(float), cp[:, c]) for c in range(3)])
ll = log_loss(yte, np.clip(cp, 1e-7, 1 - 1e-7))

pred_dist = pd.Series(preds).map({0:"H",1:"D",2:"A"}).value_counts(normalize=True)
real_dist = pd.Series(yte).map({0:"H",1:"D",2:"A"}).value_counts(normalize=True)

print(f"\n  === ML ENSEMBLE (LGB + XGB) ===")
print(f"  Features: {len(cols)}")
print(f"  Accuracy: {acc*100:.1f}%")
print(f"  Brier: {brier:.4f}")
print(f"  Log Loss: {ll:.4f}")
print(f"  Draw Recall: {draw_r*100:.1f}%")
for k in ["H","D","A"]:
    print(f"  {k}: pred={pred_dist.get(k,0)*100:.1f}% real={real_dist.get(k,0)*100:.1f}%")
for thr_name, thr in [("HC50", 0.50), ("HC60", 0.60), ("HC65", 0.65), ("HC70", 0.70)]:
    mask = conf >= thr
    if mask.sum() > 0:
        a = accuracy_score(yte[mask], preds[mask])
        print(f"  {thr_name}: {a*100:.1f}% ({mask.sum()} mac)")

# Value Bet
has_odds = te["avg_home_odds"].notna() & te["avg_draw_odds"].notna() & te["avg_away_odds"].notna()
ot = te[has_odds]
print(f"\n  Odds matches: {len(ot)}/{len(te)}")
if len(ot) > 0:
    Xto = ot[cols].astype(np.float32).values
    yto = ot["result"].map({"H":0,"D":1,"A":2}).values
    all_op = [m.predict_proba(Xto) for m in all_models]
    raw_o = np.average(all_op, axis=0, weights=all_ws)
    cp_o = np.column_stack([cal[c].predict(raw_o[:, c]) for c in range(3)])
    s = cp_o.sum(axis=1, keepdims=True); s = np.where(s > 0, s, 1); cp_o /= s

    ho = ot["avg_home_odds"].values.astype(float)
    do_ = ot["avg_draw_odds"].values.astype(float)
    ao = ot["avg_away_odds"].values.astype(float)
    mh = 1.0 / ho; md = 1.0 / do_; ma = 1.0 / ao

    bets = profit = staked = wins = 0
    bt = {"H":0, "D":0, "A":0}
    for i in range(len(ot)):
        for out, ov, mp, mdl, ri in [("H",ho[i],mh[i],cp_o[i,0],0),("D",do_[i],md[i],cp_o[i,1],1),("A",ao[i],ma[i],cp_o[i,2],2)]:
            edge = mdl - mp
            if edge > 0.03 and ov > 1.01:
                kelly = edge / (ov - 1) * 0.20
                kelly = max(0, min(kelly, 0.05))
                if kelly > 0.005:
                    bets += 1; st = kelly * 100; staked += st; bt[out] += 1
                    if yto[i] == ri: profit += st * (ov - 1); wins += 1
                    else: profit -= st
    roi = profit / staked * 100 if staked > 0 else 0
    wr = wins / bets * 100 if bets > 0 else 0
    print(f"  Value Bets: {bets}")
    print(f"  Bet dist: H={bt['H']} D={bt['D']} A={bt['A']}")
    print(f"  Win: {wins}/{bets} ({wr:.1f}%)")
    print(f"  Profit: {profit:+.1f}")
    print(f"  ROI: {roi:+.2f}%")
    print(f"  100 birim bankroll: {100+profit:.1f}")

# Feature importance
fi_avg = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
fi_s = sorted(zip(cols, fi_avg), key=lambda x: -x[1])[:10]
print(f"\n  Top Features:")
for name, imp in fi_s:
    print(f"    {name}: {int(imp)}")

# ===========================
# 5. DIXON-COLES
# ===========================
print(f"\n[5/5] Dixon-Coles Poisson...", flush=True)

# Dixon-Coles icin ayri veri
df_raw = pd.read_parquet("data/gold/features_enhanced_v5.parquet")
df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
df_raw = df_raw.sort_values("date").reset_index(drop=True)
df_raw = df_raw.dropna(subset=["home_goals", "away_goals", "home_team_id", "away_team_id"])
df_raw["home_goals"] = df_raw["home_goals"].astype(int)
df_raw["away_goals"] = df_raw["away_goals"].astype(int)

dc_train = df_raw[(df_raw["date"] >= "2023-01-01") & (df_raw["date"] < "2025-09-01")]
dc_test = df_raw[df_raw["date"] >= "2025-09-01"]
dc_test = dc_test.head(5000)

dc_model = DixonColesModel()
dc_model.fit(dc_train, max_teams=80)

dc_probs = dc_model.predict_batch(dc_test)
dc_preds = np.argmax(dc_probs, axis=1)
dc_yte = dc_test["result"].map({"H":0,"D":1,"A":2}).values
dc_acc = (dc_preds == dc_yte).mean()
dc_conf = dc_probs.max(axis=1)
dc_draw_r = (dc_preds[dc_yte == 1] == 1).mean() if (dc_yte == 1).sum() else 0

print(f"\n  === DIXON-COLES ===")
print(f"  Accuracy: {dc_acc*100:.1f}%")
print(f"  Draw Recall: {dc_draw_r*100:.1f}%")
for thr_name, thr in [("HC50", 0.50), ("HC60", 0.60), ("HC65", 0.65)]:
    mask = dc_conf >= thr
    if mask.sum() > 0:
        a = (dc_preds[mask] == dc_yte[mask]).mean()
        print(f"  {thr_name}: {a*100:.1f}% ({mask.sum()} mac)")

# Dixon-Coles Value Bet
dc_has_odds = dc_test["avg_home_odds"].notna() & dc_test["avg_draw_odds"].notna() & dc_test["avg_away_odds"].notna()
dc_ot = dc_test[dc_has_odds]
if len(dc_ot) > 0:
    dc_yto = dc_ot["result"].map({"H":0,"D":1,"A":2}).values
    ho = dc_ot["avg_home_odds"].values.astype(float)
    do_ = dc_ot["avg_draw_odds"].values.astype(float)
    ao = dc_ot["avg_away_odds"].values.astype(float)
    
    dc_bets = dc_profit = dc_staked = dc_wins = 0
    for i in range(len(dc_ot)):
        mp = [dc_probs[dc_has_odds.values][i][0], dc_probs[dc_has_odds.values][i][1], dc_probs[dc_has_odds.values][i][2]]
        for out, ov, mdl, ri in [( "H", ho[i], mp[0], 0), ("D", do_[i], mp[1], 1), ("A", ao[i], mp[2], 2)]:
            market_p = 1.0 / ov
            edge = mdl - market_p
            if edge > 0.03 and ov > 1.01:
                kelly = edge / (ov - 1) * 0.20
                kelly = max(0, min(kelly, 0.05))
                if kelly > 0.005:
                    dc_bets += 1; st = kelly * 100; dc_staked += st
                    if dc_yto[i] == ri: dc_profit += st * (ov - 1); dc_wins += 1
                    else: dc_profit -= st
    dc_roi = dc_profit / dc_staked * 100 if dc_staked > 0 else 0
    print(f"  Value Bets: {dc_bets}")
    print(f"  ROI: {dc_roi:+.2f}%")

# ===========================
# WALK-FORWARD
# ===========================
print(f"\n{'='*80}")
print(f"  WALK-FORWARD BACKTEST")
print(f"{'='*80}")

wf_results, wf_avg = walk_forward_backtest(df_lgb, cols, CAT, n_splits=3, train_years=3, val_months=6, test_months=6)

# ===========================
# FINAL SUMMARY
# ===========================
print(f"\n{'='*80}")
print(f"  FINAL KARSILASTIRMA")
print(f"{'='*80}")
print(f"  {'Model':<30} {'Accuracy':>10} {'Draw R':>10} {'ROI':>10}")
print(f"  {'-'*60}")
print(f"  {'ML Ensemble (LGB+XGB)':<30} {acc*100:>9.1f}% {draw_r*100:>9.1f}% {roi:>+9.2f}%")
print(f"  {'Dixon-Coles Poisson':<30} {dc_acc*100:>9.1f}% {dc_draw_r*100:>9.1f}% {dc_roi:>+9.2f}%")
if wf_avg:
    print(f"  {'Walk-Forward (LGB)':<30} {wf_avg.get('accuracy',0)*100:>9.1f}% {wf_avg.get('draw_recall',0)*100:>9.1f}% {wf_avg.get('roi',0):>+9.2f}%")

# Feature importance
fi_avg = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
fi_s = sorted(zip(cols, fi_avg), key=lambda x: -x[1])[:10]

# Save
result = {
    "model": "v6_final",
    "ml_ensemble": {"accuracy": acc, "brier": brier, "log_loss": ll, "draw_recall": draw_r, "roi": roi, "n_bets": bets},
    "dixon_coles": {"accuracy": dc_acc, "draw_recall": dc_draw_r, "roi": dc_roi},
    "walk_forward": wf_avg,
    "features": len(cols),
    "top_features": [(n, int(v)) for n, v in fi_s],
}
with open("tahminler/production_v6_final.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# Save models for real-time
os.makedirs("models/trained", exist_ok=True)
for i, m in enumerate(lgb_models):
    m.booster_.save_model(f"models/trained/lgb_{i}.txt")
for i, m in enumerate(xgb_models):
    m.save_model(f"models/trained/xgb_{i}.json")
with open("models/trained/feature_cols.json", "w") as f:
    json.dump(cols, f)

print(f"\n  Modeller kaydedildi: models/trained/")
print(f"  Sonuclar: tahminler/production_v6_final.json")
print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
