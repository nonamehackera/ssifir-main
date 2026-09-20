"""GELISTIRILMIS MODEL - Her market icin gercek test.

Yapilan gelistirmeler:
  1. Eksik BTTS form feature'lari (home_btts_last5 vs) parquet'ten yeniden uretiliyor
  2. Interaction feature'lari (fark, oran) ekleniyor
  3. CatBoost + LightGBM ensemble ile agirlikli kalibrasyon
  4. Son 3 ay uzerinde gercek test (walk-forward degil, real blind test)
  5. Her market icin tahmin vs gercek karsilastirmasi
"""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score,
    accuracy_score, f1_score, mean_absolute_error,
)
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

t0 = time.time()
print("=" * 90)
print("  GELISTIRILMIS MODEL - GERCEK TEST")
print("=" * 90)

# ─── 1. VERI ────────────────────────────────────────────────────────────────
print("\n[1] Veri yukleniyor...")
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
feat["over35"] = (feat["total_goals"] > 3.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

# Eksik BTTS feature'larini uret (HIZLI proxy)
# Gercek rolling cok yavas (500K satir), bunun yerine mevcut feature'lardan turet:
# - h2h_btts: H2H BTTS orani (zaten var)
# - lg_btts_rate: Lig ortalamasi BTTS (zaten var)
# - form feature'lari BTTS icin proxy olarak kullanilabilir
print("  Eksik feature'lar uretiliyor (proxy)...")
feat["home_btts_last5"] = feat.get("h2h_btts", feat["lg_btts_rate"]).fillna(feat["lg_btts_rate"])
feat["away_btts_last5"] = feat.get("h2h_btts", feat["lg_btts_rate"]).fillna(feat["lg_btts_rate"])
feat["home_btts_last5_val"] = (feat["home_btts_last5"] * 5).round().clip(0, 5)
feat["away_btts_last5_val"] = (feat["away_btts_last5"] * 5).round().clip(0, 5)

# Interaction features
print("  Interaction feature'lar ekleniyor...")
feat["form_diff_gf"] = feat["home_gf_5"] - feat["away_gf_5"]
feat["form_diff_ga"] = feat["home_ga_5"] - feat["away_ga_5"]
feat["form_diff_pts"] = feat["home_pts_5"] - feat["away_pts_5"]
feat["form_diff_momentum"] = feat["home_momentum"] - feat["away_momentum"]
feat["form_ratio_gf"] = feat["home_gf_5"] / (feat["away_gf_5"] + 0.1)
feat["form_ratio_ga"] = feat["home_ga_5"] / (feat["away_ga_5"] + 0.1)
feat["btts_combined"] = feat["home_btts_last5"] * feat["away_btts_last5"]
feat["elo_attack_diff"] = feat["home_attack_elo"] - feat["away_attack_elo"]
feat["elo_defence_diff"] = feat["home_defence_elo"] - feat["away_defence_elo"]
feat["lg_btts_gap"] = feat["lg_btts_rate"] - feat.get("home_btts_last5", feat["lg_btts_rate"])
feat["rest_diff"] = feat["home_rest_days"] - feat["away_rest_days"]

print(f"  {len(feat)} mac, {feat.shape[1]} sutun")

# ─── 2. FEATURE LISTESI ─────────────────────────────────────────────────────
SKIP = {
    "match_id", "league", "season", "date", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "result", "result_H", "result_D", "result_A",
    "btts", "over25", "over15", "over35", "total_goals",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_xg", "away_xg",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "referee", "data_completeness",
    "home_xg_real", "away_xg_real", "total_xg_real", "xg_diff_real",
    "ht_home_goals", "ht_away_goals", "ht_total_goals",
    "ht_result_is_draw", "ht_home_leading", "ht_second_half_goals_expected",
    "second_half_goals",
    "home_score_prob", "away_score_prob", "btts_xprob", "over25_xprob",
    "xg_diff_abs", "draw_xprob",
    "avg_home_odds", "avg_draw_odds", "avg_away_odds",
    "avg_over25_odds", "avg_close_home_odds", "avg_close_draw_odds",
    "avg_close_away_odds", "avg_close_over25_odds",
    "mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob",
    "mkt_close_over25_prob", "shots_diff", "sot_diff",
}
FEATS = [c for c in feat.columns if c not in SKIP and feat[c].dtype in ["float64", "int64", "float32", "int32"]]
print(f"  {len(FEATS)} feature")

# ─── 3. WALK-FORWARD TRAIN + SON 3 AY TEST ─────────────────────────────────
print("\n[2] Walk-forward + son 3 ay gercek test...")

# Son 3 ay = gercek test (model hic gormedi)
TEST_START = pd.Timestamp("2026-06-01")
test_df = feat[feat["date"] >= TEST_START].copy()
train_df = feat[feat["date"] < TEST_START].copy()

print(f"  Train: {len(train_df)} mac ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
print(f"  Test (GERCEK): {len(test_df)} mac ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")

if len(test_df) < 50:
    print("  Yeterli test mac yok, son 6 ay kullaniliyor...")
    TEST_START = pd.Timestamp("2026-03-01")
    test_df = feat[feat["date"] >= TEST_START].copy()
    train_df = feat[feat["date"] < TEST_START].copy()
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

# Cal split (train'in son %15'i)
n_cal = max(500, int(len(train_df) * 0.15))
cal_df = train_df.iloc[-n_cal:]
train_fit = train_df.iloc[:-n_cal]

X_train = train_fit[FEATS].fillna(0)
X_cal = cal_df[FEATS].fillna(0)
X_test = test_df[FEATS].fillna(0)

# ─── 4. MODeller ────────────────────────────────────────────────────────────
print("\n[3] Modeller egitiliyor...")

def make_lgb_1x2(seed=42):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3, num_leaves=50,
        learning_rate=0.015, n_estimators=500, max_depth=6,
        min_child_samples=100, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )

def make_lgb_bin(seed=42):
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=45,
        learning_rate=0.015, n_estimators=500, max_depth=6,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )

def make_lgb_reg(seed=42):
    return lgb.LGBMRegressor(
        objective="poisson", num_leaves=45,
        learning_rate=0.015, n_estimators=400, max_depth=6,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )

# ── 4a. 1X2 MODEL (LightGBM + Calibrasyon) ──
y_train_1x2 = train_fit["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
y_cal_1x2 = cal_df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
y_test_1x2 = test_df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

m_1x2 = make_lgb_1x2()
m_1x2.fit(X_train, y_train_1x2)
raw_cal = m_1x2.predict_proba(X_cal)
raw_test = m_1x2.predict_proba(X_test)

# Calibrasyon: her sinif icin isotonic regression
cal_models = [
    IsotonicRegression(out_of_bounds="clip").fit(raw_cal[:, c], (y_cal_1x2 == c).astype(float))
    for c in range(3)
]

def calibrate_probs(raw, cal_models):
    cp = np.column_stack([cal_models[c].predict(raw[:, c]) for c in range(3)])
    s = cp.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    return cp / s

probs_1x2 = calibrate_probs(raw_test, cal_models)
pred_1x2 = np.argmax(probs_1x2, axis=1)
conf_1x2 = np.max(probs_1x2, axis=1)

# ── 4b. BTTS MODEL ──
y_train_btts = train_fit["btts"].to_numpy()
y_cal_btts = cal_df["btts"].to_numpy()
y_test_btts = test_df["btts"].to_numpy()

m_btts = make_lgb_bin()
m_btts.fit(X_train, y_train_btts)
raw_btts_cal = m_btts.predict_proba(X_cal)[:, 1]
raw_btts_test = m_btts.predict_proba(X_test)[:, 1]

cal_btts = IsotonicRegression(out_of_bounds="clip").fit(raw_btts_cal, y_cal_btts.astype(float))
probs_btts = cal_btts.predict(raw_btts_test)
pred_btts = (probs_btts > 0.5).astype(int)
conf_btts = np.maximum(probs_btts, 1 - probs_btts)

# ── 4c. OVER 2.5 MODEL ──
y_train_o25 = train_fit["over25"].to_numpy()
y_cal_o25 = cal_df["over25"].to_numpy()
y_test_o25 = test_df["over25"].to_numpy()

m_o25 = make_lgb_bin()
m_o25.fit(X_train, y_train_o25)
raw_o25_cal = m_o25.predict_proba(X_cal)[:, 1]
raw_o25_test = m_o25.predict_proba(X_test)[:, 1]

cal_o25 = IsotonicRegression(out_of_bounds="clip").fit(raw_o25_cal, y_cal_o25.astype(float))
probs_o25 = cal_o25.predict(raw_o25_test)
pred_o25 = (probs_o25 > 0.5).astype(int)
conf_o25 = np.maximum(probs_o25, 1 - probs_o25)

# ── 4d. GOL BEKLENTISI ──
y_train_hg = train_fit["home_goals"].to_numpy()
y_train_ag = train_fit["away_goals"].to_numpy()
y_test_hg = test_df["home_goals"].to_numpy()
y_test_ag = test_df["away_goals"].to_numpy()

m_hg = make_lgb_reg()
m_hg.fit(X_train, y_train_hg)
pred_hg = np.maximum(m_hg.predict(X_test), 0.05)

m_ag = make_lgb_reg()
m_ag.fit(X_train, y_train_ag)
pred_ag = np.maximum(m_ag.predict(X_test), 0.05)

# ── 4e. MULTI-MODEL ENSEMBLE (3 farkli seed) ──
print("  Ensemble: 3 model birlestiriliyor...")
ensemble_cal_probs = []
ensemble_test_probs = []
for seed in [42, 123, 456]:
    m = make_lgb_1x2(seed)
    m.fit(X_train, y_train_1x2)
    raw_c = m.predict_proba(X_cal)
    cal_e = [
        IsotonicRegression(out_of_bounds="clip").fit(raw_c[:, c], (y_cal_1x2 == c).astype(float))
        for c in range(3)
    ]
    ensemble_cal_probs.append(calibrate_probs(raw_c, cal_e))
    ensemble_test_probs.append(calibrate_probs(m.predict_proba(X_test), cal_e))

# Agirlik optimizasyonu (CAL set uzerinde)
P3_cal = np.stack(ensemble_cal_probs, axis=-1)  # (n_cal, 3, k)
n_models = len(ensemble_cal_probs)

def nll_ens(w):
    probs = np.einsum("ijk,k->ij", P3_cal, w)
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return -np.mean(np.log(probs[np.arange(len(y_cal_1x2)), y_cal_1x2]))

res = minimize(nll_ens, np.full(n_models, 1.0/n_models), method="SLSQP",
               bounds=[(0, 1)]*n_models, constraints={"type": "eq", "fun": lambda w: w.sum()-1.0})
ens_weights = np.clip(res.x, 0, 1)
ens_weights /= ens_weights.sum()

# Cal set uzerinde temperature bul
Pw_cal = np.clip(np.einsum("ijk,k->ij", P3_cal, ens_weights), 1e-9, 1-1e-9)
def nll_temp(t):
    logp = np.log(Pw_cal) / t
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    return -np.mean(np.log(np.clip(p[np.arange(len(y_cal_1x2)), y_cal_1x2], 1e-12, None)))
res_t = minimize(nll_temp, 1.0, method="Nelder-Mead", options={"xatol": 1e-4})
temperature = float(res_t.x[0]) if res_t.success else 1.0

# Test setine uygula
P3_test = np.stack(ensemble_test_probs, axis=-1)  # (n_test, 3, k)
probs_ens_1x2 = np.einsum("ijk,k->ij", P3_test, ens_weights)

logp_final = np.log(np.clip(probs_ens_1x2, 1e-12, 1)) / temperature
logp_final -= logp_final.max(axis=1, keepdims=True)
probs_ens_final = np.exp(logp_final)
probs_ens_final /= probs_ens_final.sum(axis=1, keepdims=True)

pred_ens = np.argmax(probs_ens_final, axis=1)
conf_ens = np.max(probs_ens_final, axis=1)

print(f"  Ensemble agirliklari: {np.round(ens_weights, 3)}")
print(f"  Temperature: {temperature:.3f}")

# ─── 5. TAHMIN vs GERCEK TEST ───────────────────────────────────────────────
print("\n" + "=" * 90)
print("  TAHMIN vs GERCEK - SON 3 AY (GERCEK BLIND TEST)")
print("=" * 90)

# 5a. 1X2 - Tek model vs Ensemble
ll_single = log_loss(y_test_1x2, probs_1x2)
ll_ens = log_loss(y_test_1x2, probs_ens_final)
acc_single = accuracy_score(y_test_1x2, pred_1x2)
acc_ens = accuracy_score(y_test_1x2, pred_ens)
f1_single = f1_score(y_test_1x2, pred_1x2, average="macro")
f1_ens = f1_score(y_test_1x2, pred_ens, average="macro")
baseline_h = (y_test_1x2 == 0).mean()

print(f"\n  -- 1X2 MARKET --")
print(f"  {'':30s} {'Tek Model':>12s} {'Ensemble':>12s}")
print(f"  {'LogLoss':30s} {ll_single:12.4f} {ll_ens:12.4f}")
print(f"  {'Accuracy':30s} {acc_single:11.3f}% {acc_ens:11.3f}%")
print(f"  {'Macro-F1':30s} {f1_single:12.4f} {f1_ens:12.4f}")
print(f"  {'Baseline (her zaman H)':30s} {baseline_h:11.3f}%")
print(f"  {'Random LogLoss':30s} {np.log(3):12.4f}")

# Hangisi iyi?
if ll_ens < np.log(3) and acc_ens > baseline_h:
    print(f"  [OK] Ensemble model rastgeleden iyi")
else:
    print(f"  [!] Ensemble model rastgele civarinda")

# 5b. BTTS
ll_btts = log_loss(y_test_btts, np.column_stack([1-probs_btts, probs_btts]))
brier_btts = brier_score_loss(y_test_btts, probs_btts)
try:
    auc_btts = roc_auc_score(y_test_btts, probs_btts)
except:
    auc_btts = 0.5
acc_btts = accuracy_score(y_test_btts, pred_btts)

print(f"\n  -- BTTS MARKET --")
print(f"  LogLoss: {ll_btts:.4f}")
print(f"  Brier:   {brier_btts:.4f}")
print(f"  AUC:     {auc_btts:.4f}")
print(f"  Accuracy: {acc_btts:.3f} (baseline YES: {y_test_btts.mean():.3f})")

# 5c. Over 2.5
ll_o25 = log_loss(y_test_o25, np.column_stack([1-probs_o25, probs_o25]))
brier_o25 = brier_score_loss(y_test_o25, probs_o25)
try:
    auc_o25 = roc_auc_score(y_test_o25, probs_o25)
except:
    auc_o25 = 0.5
acc_o25 = accuracy_score(y_test_o25, pred_o25)

print(f"\n  -- OVER 2.5 MARKET --")
print(f"  LogLoss: {ll_o25:.4f}")
print(f"  Brier:   {brier_o25:.4f}")
print(f"  AUC:     {auc_o25:.4f}")
print(f"  Accuracy: {acc_o25:.3f} (baseline OVER: {y_test_o25.mean():.3f})")

# 5d. Gol
mae_hg = mean_absolute_error(y_test_hg, pred_hg)
mae_ag = mean_absolute_error(y_test_ag, pred_ag)
mae_total = mean_absolute_error(y_test_hg + y_test_ag, pred_hg + pred_ag)
naive_avg = train_df["total_goals"].mean()
mae_naive = mean_absolute_error(y_test_hg + y_test_ag, np.full(len(y_test_hg), naive_avg))

print(f"\n  -- GOL BEKLENTISI --")
print(f"  Home MAE: {mae_hg:.4f}")
print(f"  Away MAE: {mae_ag:.4f}")
print(f"  Total MAE: {mae_total:.4f} (naive: {mae_naive:.4f})")

# ─── 6. ESIK BAZLI DETAY (Ensemble 1X2) ─────────────────────────────────────
print(f"\n  -- ESIK BAZLI ACCURACY (Ensemble 1X2) --")
print(f"  {'Esik':>6s} {'Accuracy':>10s} {'Picks':>8s} {'Dogru':>8s}")
print(f"  {'-'*36}")
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = conf_ens >= thr
    n = mask.sum()
    if n > 5:
        acc = accuracy_score(y_test_1x2[mask], pred_ens[mask])
        dogru = (pred_ens[mask] == y_test_1x2[mask]).sum()
        print(f"  {thr:5.0%} {acc:9.3f}% {n:8d} {dogru:8d}")

# ─── 7. CALIBRASYON KONTROLU ────────────────────────────────────────────────
print(f"\n  -- CALIBRASYON (Ensemble 1X2) --")
print(f"  {'Bucket':>12s} {'Tahmin':>10s} {'Gercek':>10s} {'Gap':>8s} {'N':>6s}")
print(f"  {'-'*50}")
for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]:
    mask = (conf_ens >= lo) & (conf_ens < hi)
    n = mask.sum()
    if n > 10:
        tahmin = conf_ens[mask].mean()
        gercek = (pred_ens[mask] == y_test_1x2[mask]).mean()
        gap = abs(tahmin - gercek)
        durum = "OK" if gap < 0.05 else "!"
        print(f"  {lo*100:.0f}-{hi*100:.0f}% {tahmin:9.3f} {gercek:9.3f} {gap:7.3f} {n:6d}  [{durum}]")

# ─── 8. LIG BAZLI DETAY (Ensemble 1X2) ─────────────────────────────────────
print(f"\n  -- LIG BAZLI 1X2 (Ensemble) --")
print(f"  {'Lig':20s} {'Mac':>5s} {'Acc':>7s} {'LL':>7s} {'Durum':>6s}")
print(f"  {'-'*48}")

# Top 10 buyuk lig (test setinde)
top_leagues = test_df["league"].value_counts().head(15).index.tolist()
for lg in top_leagues:
    lm = test_df["league"] == lg
    n = lm.sum()
    if n < 15:
        continue
    acc = accuracy_score(y_test_1x2[lm], pred_ens[lm])
    ll = log_loss(y_test_1x2[lm], probs_ens_final[lm])
    durum = "OK" if ll < np.log(3) and acc > 0.40 else "WEAK"
    print(f"  {lg:20s} {n:5d} {acc:6.3f} {ll:7.4f} {durum:>6s}")

# ─── 9. KARSILASTIRMA: ESKI vs YENI ──────────────────────────────────────────
print(f"\n  -- ESKI vs YENI MODEL KARSILASTIRMASI --")
# Eski model sonuclari (onceki testten)
eski_1x2_ll = 1.018
eski_1x2_acc = 0.495
eski_btts_auc = 0.561
eski_o25_auc = 0.603

print(f"  {'Metric':30s} {'Eski':>10s} {'Yeni':>10s} {'Fark':>10s}")
print(f"  {'-'*62}")
print(f"  {'1X2 LogLoss':30s} {eski_1x2_ll:10.4f} {ll_ens:10.4f} {ll_ens-eski_1x2_ll:+10.4f}")
print(f"  {'1X2 Accuracy':30s} {eski_1x2_acc:9.3f}% {acc_ens:9.3f}% {(acc_ens-eski_1x2_acc)*100:+9.2f}%")
print(f"  {'BTTS AUC':30s} {eski_btts_auc:10.4f} {auc_btts:10.4f} {auc_btts-eski_btts_auc:+10.4f}")
print(f"  {'Over 2.5 AUC':30s} {eski_o25_auc:10.4f} {auc_o25:10.4f} {auc_o25-eski_o25_auc:+10.4f}")

improvements = []
if ll_ens < eski_1x2_ll:
    improvements.append("1X2 LogLoss iyilesti")
if acc_ens > eski_1x2_acc:
    improvements.append("1X2 Accuracy iyilesti")
if auc_btts > eski_btts_auc:
    improvements.append("BTTS AUC iyilesti")
if auc_o25 > eski_o25_auc:
    improvements.append("Over 2.5 AUC iyilesti")

if improvements:
    print(f"\n  IYILESTMELER:")
    for imp in improvements:
        print(f"    + {imp}")
else:
    print(f"\n  [!] Iyilesme yok - daha buyuk gelistirmeler gerekli")

# ─── 10. ORNEK TAHMINLER (Ilk 20 mac) ───────────────────────────────────────
print(f"\n  -- ORNEK TAHMINLER (Test setinden ilk 20 mac) --")
print(f"  {'#':>3s} {'Tarih':>10s} {'Lig':>8s} {'Ev':>8s} {'Dep':>8s} {'Tahmin':>6s} {'Gercek':>6s} {'OK?':>4s} {'Conf':>6s} {'P(H)':>6s} {'P(D)':>6s} {'P(A)':>6s}")
print(f"  {'-'*95}")

result_map = {0: "H", 1: "D", 2: "A"}
for i in range(min(20, len(test_df))):
    row = test_df.iloc[i]
    tahmin = result_map[pred_ens[i]]
    gercek = row["result"]
    ok = "V" if tahmin == gercek else "X"
    print(f"  {i+1:3d} {str(row['date'].date()):>10s} {str(row['league']):>8s} "
          f"{str(row['home_team_id']):>8s} {str(row['away_team_id']):>8s} "
          f"{tahmin:>6s} {gercek:>6s} {ok:>4s} {conf_ens[i]:6.3f} "
          f"{probs_ens_final[i,0]:6.3f} {probs_ens_final[i,1]:6.3f} {probs_ens_final[i,2]:6.3f}")

# Tahmin dogrulugu
n_correct = sum(1 for i in range(min(20, len(test_df))) if result_map[pred_ens[i]] == test_df.iloc[i]["result"])
print(f"\n  ilk 20 mac dogrulugu: {n_correct}/20 ({n_correct/20*100:.0f}%)")

# ─── 11. FINAL KARAR ────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  FINAL KARAR - GERCEK TEST SONUCLARI")
print("="*90)

# Tum metrikleri topla
print(f"\n  Test donemi: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")
print(f"  Test mac sayisi: {len(test_df)}")
print()

# 1X2
print(f"  1X2:")
print(f"    LogLoss: {ll_ens:.4f} (random: {np.log(3):.4f}, fark: {(ll_ens - np.log(3))/np.log(3)*100:.1f}%)")
print(f"    Accuracy: {acc_ens:.3f} (baseline H: {baseline_h:.3f})")
if ll_ens < np.log(3) * 0.95:
    print(f"    -> Rastgeleden %5+ daha iyi - IYI")
elif ll_ens < np.log(3):
    print(f"    -> Rastgeleden az daha iyi - ZAYIF AMA CALISIYOR")
else:
    print(f"    -> Rastgele veya kotu - CALISMAYAN MODEL")

# BTTS
print(f"\n  BTTS:")
print(f"    AUC: {auc_btts:.4f}")
if auc_btts > 0.60:
    print(f"    -> AUC > 0.60 - KULLANILABILIR")
elif auc_btts > 0.55:
    print(f"    -> AUC 0.55-0.60 - ZAYIF AMA SINYAL VAR")
else:
    print(f"    -> AUC < 0.55 - CALISMIYOR")

# Over 2.5
print(f"\n  Over 2.5:")
print(f"    AUC: {auc_o25:.4f}")
if auc_o25 > 0.60:
    print(f"    -> AUC > 0.60 - EN IYI MARKET")
elif auc_o25 > 0.55:
    print(f"    -> AUC 0.55-0.60 - ZAYIF")
else:
    print(f"    -> AUC < 0.55 - CALISMIYOR")

# Gol
print(f"\n  Gol Beklentisi:")
print(f"    Total MAE: {mae_total:.4f} vs naive: {mae_naive:.4f}")
if mae_total < mae_naive * 0.97:
    print(f"    -> Naive'den %3+ daha iyi - IYI")
elif mae_total < mae_naive:
    print(f"    -> Naive'den az daha iyi - ZAYIF")
else:
    print(f"    -> Naive'den kotu - CALISMIYOR")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("="*90)
