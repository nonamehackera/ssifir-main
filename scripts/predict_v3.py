"""Test setini 2025/26 sezonuna kaydirarak GERCEK veri ile test et."""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import time

t0 = time.time()

feats = pd.read_parquet("data/gold/features.parquet")
feats = feats[feats["date"] >= "2015-01-01"].copy().sort_values("date").reset_index(drop=True)
matches = pd.read_parquet("data/gold/matches.parquet")
matches = matches[matches["date"] >= "2015-01-01"].copy().sort_values("date").reset_index(drop=True)

# Orijinal veri kalitesini isaretle
stat_cols = ["home_shots", "away_shots", "home_sot", "away_sot",
             "home_corners", "away_corners", "home_yellow", "away_yellow"]
for col in stat_cols:
    if col in matches.columns:
        feats[f"real_{col}"] = matches[col].notna().values
feats["has_real_corners"] = feats["real_home_corners"] & feats["real_away_corners"]

# TEST SETINI DEGISTIR: son 1000 degil, 2025/26 sezonundaki son 1000 mac
# FD 2025/26: Agustos 2025 - Haziran 2026 arasi
test_start = pd.Timestamp("2025-08-01")
test_end = pd.Timestamp("2026-06-01")

test_mask = (feats["date"] >= test_start) & (feats["date"] < test_end)
test_indices = feats[test_mask].index

if len(test_indices) < 1000:
    print(f"Uyari: 2025/26 sezonunda sadece {len(test_indices)} mac var")
    test_idx = test_indices
else:
    test_idx = test_indices[-1000:]

train_mask = feats.index.isin(test_idx) == False
train = feats[train_mask].copy()
test = feats.loc[test_idx].copy()

print("="*70)
print("  2025/26 SEZONU TEST SETI")
print("="*70)
print(f"  Train: {len(train)} mac (son tarih: {train['date'].max()})")
print(f"  Test: {len(test)} mac ({test['date'].min()} -> {test['date'].max()})")

# Test seti kaynak dagilimi
print("\n  Test seti kaynak dagilimi:")
for prefix in ["FD", "HF_", "XG_", "SB_", "US_", "OF_"]:
    if prefix == "FD":
        n = test[~test["league"].str.startswith(("HF_","OF_","TS_","FB_","XG_","SB_","US_"), na=False)]
    else:
        n = test[test["league"].str.startswith(prefix, na=False)]
    if len(n) > 0:
        print(f"    {prefix:>4}: {len(n):>4} mac")

# Test seti veri kalitesi
real_corners = test["has_real_corners"].sum()
real_odds = test["avg_home_odds"].notna().sum()
real_xg = test["home_xg"].notna().sum()
print(f"\n  Veri kalitesi:")
print(f"    GERCEK corner: {real_corners}/{len(test)} ({real_corners/len(test)*100:.1f}%)")
print(f"    GERCEK odds: {real_odds}/{len(test)} ({real_odds/len(test)*100:.1f}%)")
print(f"    GERCEK xG: {real_xg}/{len(test)} ({real_xg/len(test)*100:.1f}%)")

# Y结果lari
y_result = test["result"].values
y_btts = test["btts"].values.astype(int)
y_over25 = test["over25"].values.astype(int)

FEATURES_1X2 = [
    "elo_diff", "home_elo", "away_elo",
    "attack_elo_diff", "defence_elo_diff",
    "home_attack_elo", "home_defence_elo", "away_attack_elo", "away_defence_elo",
    "home_gf_5", "home_ga_5", "home_pts_5",
    "away_gf_5", "away_ga_5", "away_pts_5",
    "home_gf_3", "home_ga_3", "away_gf_3", "away_ga_3",
    "home_hgf_5", "home_hga_5", "home_hpts_5",
    "away_agf_5", "away_aga_5", "away_apts_5",
    "home_w_gf", "home_w_ga", "away_w_gf", "away_w_ga",
    "home_w_shots", "home_w_sot", "away_w_shots", "away_w_sot",
    "xg_diff_real", "xg_diff_abs", "total_xg_real",
    "home_score_prob", "away_score_prob", "btts_xprob", "draw_xprob", "over25_xprob",
    "home_rest_days", "away_rest_days",
    "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "mkt_home_prob", "mkt_draw_prob", "mkt_away_prob", "mkt_over25_prob",
]
FEATURES_BTTS = FEATURES_1X2[:]
FEATURES_GOALS = list(dict.fromkeys([
    f for f in FEATURES_1X2 if f not in ("mkt_home_prob", "mkt_draw_prob", "mkt_away_prob")
]))

def avail(cols, df):
    return [c for c in cols if c in df.columns]

f1x2 = avail(FEATURES_1X2, train)
fbtts = avail(FEATURES_BTTS, train)
fgoals = avail(FEATURES_GOALS, train)

def prep(df, cols):
    X = df[cols].copy()
    for c in ["season"]:
        if c in X.columns: X[c] = X[c].astype("category")
    return X

def clean(df, targets):
    return df.dropna(subset=targets).copy()

train_c = clean(train, ["result","btts","over25","home_goals","away_goals"])

print(f"\n  Features 1X2: {len(f1x2)}, Goals: {len(fgoals)}")

# 1X2
print("\n  1X2 modeli...")
X_tr = prep(train_c, f1x2)
y_enc = np.array([{"H":0,"D":1,"A":2}[r] for r in train_c["result"]])

p1_seeds = []
for s in [42, 123, 456]:
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        num_leaves=40, learning_rate=0.015, n_estimators=1200,
        max_depth=5, min_child_samples=60,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=8.0,
        max_bin=63, random_seed=s, verbose=-1,
    )
    split = int(len(X_tr) * 0.8)
    X_fit, X_cal = X_tr.iloc[:split], X_tr.iloc[split:]
    y_fit, y_cal = y_enc[:split], y_enc[split:]
    m.fit(X_fit, y_fit)
    raw_cal = m.predict_proba(X_cal)
    calibrators = []
    for cls in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal[:, cls], (y_cal == cls).astype(float))
        calibrators.append(ir)
    raw_test = m.predict_proba(prep(test, f1x2))
    cal_test = np.column_stack([calibrators[c].predict(raw_test[:, c]) for c in range(3)])
    cal_sum = cal_test.sum(axis=1, keepdims=True)
    cal_sum = np.where(cal_sum == 0, 1, cal_sum)
    cal_test /= cal_sum
    p1_seeds.append(cal_test)

p1x2 = np.mean(p1_seeds, axis=0)
pw = np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
     np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
conf1x2 = np.max(p1x2, axis=1)

# BTTS
print("  BTTS modeli...")
X_tr_bt = prep(train_c, fbtts)
pbt_seeds = []
for s in [42, 123, 456]:
    m = lgb.LGBMClassifier(objective="binary", num_leaves=36, learning_rate=0.015, n_estimators=800,
        max_depth=5, min_child_samples=60, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1)
    split = int(len(X_tr_bt) * 0.8)
    X_fit, X_cal = X_tr_bt.iloc[:split], X_tr_bt.iloc[split:]
    m.fit(X_fit, train_c["btts"].values[:split])
    raw_cal = m.predict_proba(X_cal)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, train_c["btts"].values[split:].astype(float))
    raw_test = m.predict_proba(prep(test, fbtts))[:, 1]
    pbt_seeds.append(ir.predict(raw_test))

pbt_p = np.mean(pbt_seeds, axis=0)
pbtth = (pbt_p > 0.5).astype(int)
confbtts = np.maximum(pbt_p, 1-pbt_p)

# GOL
print("  Gol modeli...")
X_tr_gl = prep(train_c, fgoals)
po2_seeds = []
for s in [42, 123, 456]:
    m = lgb.LGBMClassifier(objective="binary", num_leaves=36, learning_rate=0.015, n_estimators=800,
        max_depth=5, min_child_samples=60, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1)
    split = int(len(X_tr_gl) * 0.8)
    X_fit, X_cal = X_tr_gl.iloc[:split], X_tr_gl.iloc[split:]
    m.fit(X_fit, train_c["over25"].values[:split])
    raw_cal = m.predict_proba(X_cal)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, train_c["over25"].values[split:].astype(float))
    raw_test = m.predict_proba(prep(test, fgoals))[:, 1]
    po2_seeds.append(ir.predict(raw_test))

po2_p = np.mean(po2_seeds, axis=0)
phl, pal = [], []
for s in [42,123,456]:
    mh = lgb.LGBMRegressor(objective="poisson", num_leaves=36, learning_rate=0.015,
        n_estimators=600, max_depth=5, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1)
    mh.fit(X_tr_gl, train_c["home_goals"].values.astype(float))
    phl.append(np.maximum(mh.predict(prep(test, fgoals)),0.1))
    ma = lgb.LGBMRegressor(objective="poisson", num_leaves=36, learning_rate=0.015,
        n_estimators=600, max_depth=5, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1)
    ma.fit(X_tr_gl, train_c["away_goals"].values.astype(float))
    pal.append(np.maximum(ma.predict(prep(test, fgoals)),0.1))
hl, al = np.mean(phl,axis=0), np.mean(pal,axis=0)
total_lam = hl + al
po2_pois = 1.0 - np.exp(-total_lam) * (1.0 + total_lam + total_lam**2/2.0)
po2f = 0.5*po2_p + 0.5*po2_pois
po2ens = (po2f > 0.5).astype(int)
confou25 = np.maximum(po2f, 1-po2f)

# Over 1.5 & 3.5 (Poisson-based)
po15_pois = 1.0 - np.exp(-total_lam) * (1.0 + total_lam)
po15f = po15_pois
po15ens = (po15f > 0.5).astype(int)
confou15 = np.maximum(po15f, 1-po15f)

po35_pois = 1.0 - np.exp(-total_lam) * (1.0 + total_lam + total_lam**2/2.0 + total_lam**3/6.0)
po35f = po35_pois
po35ens = (po35f > 0.5).astype(int)
confou35 = np.maximum(po35f, 1-po35f)

# KORNER
print("  Korner modeli...")
real_mask = test["has_real_corners"].values
test_real = test[real_mask].copy()
real_mask_train = train["has_real_corners"].values
ct_clean = train[real_mask_train].dropna(subset=["home_corners","away_corners"]).copy()

KORNER_FEATS = ["home_elo", "away_elo", "elo_diff", "home_gf_5", "home_ga_5", "away_gf_5", "away_ga_5",
                "home_w_gf", "home_w_ga", "away_w_gf", "away_w_ga", "lg_corner_avg",
                "home_rest_days", "away_rest_days", "home_attack_elo", "away_defence_elo"]
fkorner = [f for f in KORNER_FEATS if f in ct_clean.columns]

pc75 = pc85 = pc95 = confc75 = confc85 = confc95 = act75 = act85 = act95 = None
if len(test_real) >= 30:
    X_kr = prep(ct_clean, fkorner)
    y_tc_train = pd.to_numeric(ct_clean["home_corners"],errors="coerce").fillna(0).values + \
                 pd.to_numeric(ct_clean["away_corners"],errors="coerce").fillna(0).values
    m75 = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=3.0, random_seed=41, verbose=-1)
    m75.fit(X_kr, (y_tc_train >= 8).astype(int))
    m85 = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=3.0, random_seed=42, verbose=-1)
    m85.fit(X_kr, (y_tc_train >= 9).astype(int))
    m95 = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=3.0, random_seed=43, verbose=-1)
    m95.fit(X_kr, (y_tc_train >= 10).astype(int))

    X_kt = prep(test_real, fkorner)
    raw75 = m75.predict_proba(X_kt)[:,1]
    raw85 = m85.predict_proba(X_kt)[:,1]
    raw95 = m95.predict_proba(X_kt)[:,1]

    split_k = int(len(X_kr) * 0.8)
    ir75 = IsotonicRegression(out_of_bounds="clip")
    ir75.fit(m75.predict_proba(X_kr.iloc[split_k:])[:,1], (y_tc_train[split_k:] >= 8).astype(int))
    ir85 = IsotonicRegression(out_of_bounds="clip")
    ir85.fit(m85.predict_proba(X_kr.iloc[split_k:])[:,1], (y_tc_train[split_k:] >= 9).astype(int))
    ir95 = IsotonicRegression(out_of_bounds="clip")
    ir95.fit(m95.predict_proba(X_kr.iloc[split_k:])[:,1], (y_tc_train[split_k:] >= 10).astype(int))

    pc75 = (ir75.predict(raw75) > 0.5).astype(int)
    pc85 = (ir85.predict(raw85) > 0.5).astype(int)
    pc95 = (ir95.predict(raw95) > 0.5).astype(int)
    confc75 = np.maximum(ir75.predict(raw75), 1-ir75.predict(raw75))
    confc85 = np.maximum(ir85.predict(raw85), 1-ir85.predict(raw85))
    confc95 = np.maximum(ir95.predict(raw95), 1-ir95.predict(raw95))

    y_tc_real = pd.to_numeric(test_real["home_corners"],errors="coerce").fillna(0).values + \
                pd.to_numeric(test_real["away_corners"],errors="coerce").fillna(0).values
    act75 = (y_tc_real >= 8).astype(int)
    act85 = (y_tc_real >= 9).astype(int)
    act95 = (y_tc_real >= 10).astype(int)

# RAPOR
def acc(p,a): return f"{(p==a).sum()}/{len(a)} = {(p==a).sum()/len(a)*100:.1f}%"
def section(title): print(f"\n{'='*70}\n  {title}\n{'='*70}")

section(f"2025/26 SEZONU - {len(test)} MAC")
print(f"  Tarih: {test['date'].min()} -> {test['date'].max()}")

for name, preds, conf, actual in [
    ("1X2", pw, conf1x2, y_result),
    ("BTTS", pbtth, confbtts, y_btts),
    ("GOL UST 1.5", po15ens, confou15, test["over15"].values.astype(int)),
    ("GOL UST 2.5", po2ens, confou25, y_over25),
    ("GOL UST 3.5", po35ens, confou35, test["over35"].values.astype(int)),
]:
    section(name)
    print(f"  Toplam: {acc(preds, actual)}")
    print(f"\n  {'Guven':>10} {'Adet':>6} {'Dogru':>6} {'Yanlis':>6} {'Acc':>8}")
    print("  " + "-"*45)
    for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.01)]:
        m=(conf>=lo)&(conf<hi); n=m.sum()
        if n>0:
            c=(preds[m]==actual[m]).sum()
            print(f"  {lo*100:>4.0f}-{hi*100:<4.0f}%  {n:>5}  {c:>5}  {n-c:>5}  {c/n*100:>7.1f}%")

if pc75 is not None and len(test_real) >= 30:
    section("KORNER (SADECE GERCEK VERI)")
    print(f"  Gercek corner: {len(test_real)} mac")
    for name, pc, conf, act in [("UST 7.5", pc75, confc75, act75), ("UST 8.5", pc85, confc85, act85), ("UST 9.5", pc95, confc95, act95)]:
        section(f"KORNER {name}")
        print(f"  Toplam: {acc(pc, act)}")
        print(f"\n  {'Guven':>10} {'Adet':>6} {'Dogru':>6} {'Yanlis':>6} {'Acc':>8}")
        print("  " + "-"*45)
        for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.01)]:
            m=(conf>=lo)&(conf<hi); n=m.sum()
            if n>0:
                c=(pc[m]==act[m]).sum()
                print(f"  {lo*100:>4.0f}-{hi*100:<4.0f}%  {n:>5}  {c:>5}  {n-c:>5}  {c/n*100:>7.1f}%")

section("70%+ GUVEN OZETI")
for name, preds, conf, actual in [
    ("1X2", pw, conf1x2, y_result),
    ("BTTS", pbtth, confbtts, y_btts),
    ("GOL UST 1.5", po15ens, confou15, test["over15"].values.astype(int)),
    ("GOL UST 2.5", po2ens, confou25, y_over25),
    ("GOL UST 3.5", po35ens, confou35, test["over35"].values.astype(int)),
]:
    m = conf >= 0.70
    n = m.sum()
    if n > 0:
        c = (preds[m] == actual[m]).sum()
        print(f"  {name:<20} {n:>5} oneri / {len(preds):>5} toplam  ->  {c}/{n} = {c/n*100:.1f}%")

if pc75 is not None and len(test_real) >= 30:
    for name, pc, conf, act in [("KORNER UST 7.5", pc75, confc75, act75), ("KORNER UST 8.5", pc85, confc85, act85), ("KORNER UST 9.5", pc95, confc95, act95)]:
        m = conf >= 0.70
        n = m.sum()
        if n > 0:
            c = (pc[m] == act[m]).sum()
            print(f"  {name:<20} {n:>5} oneri / {len(pc):>5} toplam  ->  {c}/{n} = {c/n*100:.1f}%")
        else:
            print(f"  {name:<20} 0 oneri")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
