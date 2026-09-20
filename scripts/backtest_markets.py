"""BACKTEST MARKETS v2: Hizli LightGBM ile son 2000 mac backtest."""
import sys, time, json, warnings, os
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson

t0 = time.time()

# 1. DATA
print("=" * 72)
print("ADIM 1: Veri yukleniyor...")
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
feat = feat.dropna(subset=["result", "elo_diff"])

hc = pd.to_numeric(feat["home_corners"], errors="coerce").fillna(0)
ac = pd.to_numeric(feat["away_corners"], errors="coerce").fillna(0)
feat["total_corners"] = hc + ac
feat["corner_over85"] = (feat["total_corners"] > 8.5).astype(int)
feat["corner_reliable"] = feat["total_corners"] >= 5

N_TEST = 2000
test = feat.iloc[-N_TEST:].reset_index(drop=True)
train = feat.iloc[:-N_TEST].reset_index(drop=True)

print(f"Train: {len(train)} mac | Test: {len(test)} mac")
print(f"Test: {test['date'].min().date()} -> {test['date'].max().date()}")
rd = test['result'].value_counts(normalize=True)
print(f"Sonuc: H={rd.get('H',0):.1%} D={rd.get('D',0):.1%} A={rd.get('A',0):.1%}")
print(f"BTTS={test['btts'].mean():.1%} O1.5={test['over15'].mean():.1%} O2.5={test['over25'].mean():.1%}")
cr = test['corner_reliable'].sum()
print(f"Corner reliable: {cr}/{len(test)} ({test['corner_reliable'].mean():.1%})")
if cr > 0:
    ct = test[test['corner_reliable']]
    print(f"  ort korner: {ct['total_corners'].mean():.1f} | O8.5: {ct['corner_over85'].mean():.1%}")

# 2. FEATURES (numeric only for speed)
LEAK = {
    "second_half_goals", "ht_total_goals", "ht_result_is_draw", "ht_home_leading",
    "home_xg", "away_xg", "total_xg_real", "xg_diff_real", "home_score_prob",
    "away_score_prob", "btts_xprob", "over25_xprob", "draw_xprob", "xg_diff_abs",
}
NUM_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_gf_5", "home_ga_5", "home_pts_5", "home_shots_5", "home_sot_5",
    "home_corners_5", "home_w_gf", "home_w_ga", "home_w_shots", "home_w_sot",
    "home_w_corners", "home_hgf_5", "home_hga_5", "home_hpts_5",
    "home_gf_3", "home_ga_3", "home_pts_3",
    "home_gf_8", "home_ga_8", "home_gf_20", "home_ga_20",
    "home_gf_std", "home_ga_std", "home_pts_std",
    "away_gf_5", "away_ga_5", "away_pts_5", "away_shots_5", "away_sot_5",
    "away_corners_5", "away_w_gf", "away_w_ga", "away_w_shots", "away_w_sot",
    "away_w_corners", "away_agf_5", "away_aga_5", "away_apts_5",
    "away_gf_3", "away_ga_3", "away_pts_3",
    "away_gf_8", "away_ga_8", "away_gf_20", "away_ga_20",
    "away_gf_std", "away_ga_std", "away_pts_std",
    "home_attack_elo", "home_defence_elo", "away_attack_elo", "away_defence_elo",
    "attack_elo_diff", "defence_elo_diff",
    "home_rest_days", "away_rest_days",
    "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts",
    "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
    "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg",
    "data_completeness",
    "mkt_home_prob", "mkt_draw_prob", "mkt_away_prob", "mkt_over25_prob",
    "home_momentum", "away_momentum",
    "home_wins_last5", "home_draws_last5", "away_wins_last5", "away_draws_last5",
    "home_form_std", "away_form_std", "home_gdiff5", "away_gdiff5",
    "home_gf_5_real", "home_ga_5_real", "away_gf_5_real", "away_ga_5_real",
]
SAFE = [c for c in NUM_FEATURES if c not in LEAK and c in train.columns]
print(f"\nFeature sayisi: {len(SAFE)}")

# Label encode categoricals -> numeric for LGBM
for c in ["league", "home_team_id", "away_team_id", "season"]:
    if c in train.columns:
        combined = pd.concat([train[c], test[c]], axis=0).astype("category").cat.codes
        train[c + "_enc"] = combined[:len(train)].values
        test[c + "_enc"] = combined[len(train):].values
        SAFE.append(c + "_enc")

X_tr = train[SAFE].fillna(0).values
X_te = test[SAFE].fillna(0).values
feat_names = SAFE

LGB_COMMON = dict(
    num_leaves=40, learning_rate=0.03, n_estimators=600,
    max_depth=5, min_child_samples=80, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0, verbose=-1,
    random_state=42,
)
sp = int(len(X_tr) * 0.85)

# 3. MODELS
print("\n" + "=" * 72)
print("ADIM 2: Modeller egitiliyor...")

# 3a) 1X2
print("  [1/5] 1X2...")
y1x2 = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
m1x2 = lgb.LGBMClassifier(objective="multiclass", num_class=3, **LGB_COMMON)
m1x2.fit(X_tr[:sp], y1x2[:sp], eval_set=[(X_tr[sp:], y1x2[sp:])])
p1x2_raw = m1x2.predict_proba(X_te)
# Simple calibration via validation set
from sklearn.isotonic import IsotonicRegression
p1x2_val = m1x2.predict_proba(X_tr[sp:])
cal = [IsotonicRegression(out_of_bounds="clip").fit(p1x2_val[:, c], (y1x2[sp:] == c).astype(float)) for c in range(3)]
ph_raw = cal[0].predict(p1x2_raw[:, 0])
pd_raw = cal[1].predict(p1x2_raw[:, 1])
pa_raw = cal[2].predict(p1x2_raw[:, 2])
s = np.column_stack([ph_raw, pd_raw, pa_raw]).sum(axis=1)
s = np.where(s == 0, 1, s)
ph, pd_, pa = ph_raw / s, pd_raw / s, pa_raw / s
print("    OK")

# 3b) BTTS
print("  [2/5] BTTS...")
m_btts = lgb.LGBMClassifier(objective="binary", **LGB_COMMON)
m_btts.fit(X_tr[:sp], train["btts"].values[:sp], eval_set=[(X_tr[sp:], train["btts"].values[sp:])])
raw = m_btts.predict_proba(X_tr[sp:])[:, 1]
cal_btts = IsotonicRegression(out_of_bounds="clip").fit(raw, train["btts"].values[sp:].astype(float))
btts_prob = cal_btts.predict(m_btts.predict_proba(X_te)[:, 1])
print("    OK")

# 3c) O/U 2.5
print("  [3/5] O/U 2.5...")
m_ov25 = lgb.LGBMClassifier(objective="binary", **LGB_COMMON)
m_ov25.fit(X_tr[:sp], train["over25"].values[:sp], eval_set=[(X_tr[sp:], train["over25"].values[sp:])])
raw = m_ov25.predict_proba(X_tr[sp:])[:, 1]
cal_ov25 = IsotonicRegression(out_of_bounds="clip").fit(raw, train["over25"].values[sp:].astype(float))
ov25_prob = cal_ov25.predict(m_ov25.predict_proba(X_te)[:, 1])
print("    OK")

# 3d) O/U 1.5
print("  [4/5] O/U 1.5...")
m_ov15 = lgb.LGBMClassifier(objective="binary", **LGB_COMMON)
m_ov15.fit(X_tr[:sp], train["over15"].values[:sp], eval_set=[(X_tr[sp:], train["over15"].values[sp:])])
raw = m_ov15.predict_proba(X_tr[sp:])[:, 1]
cal_ov15 = IsotonicRegression(out_of_bounds="clip").fit(raw, train["over15"].values[sp:].astype(float))
ov15_prob = cal_ov15.predict(m_ov15.predict_proba(X_te)[:, 1])
print("    OK")

# 3e) Corner total (Poisson regression -> O/U 8.5)
print("  [5/5] Corner O/U 8.5...")
corner_mask_tr = train["corner_reliable"].to_numpy().astype(bool)
corner_mask_te = test["corner_reliable"].to_numpy().astype(bool)
if corner_mask_tr.sum() > 500:
    y_cor_total = train.loc[corner_mask_tr, "total_corners"].values.astype(float)
    m_cor = lgb.LGBMRegressor(
        objective="poisson", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=40,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0,
        verbose=-1, random_state=42,
    )
    m_cor.fit(X_tr[corner_mask_tr], y_cor_total)
    cor_lambda = np.maximum(m_cor.predict(X_te[corner_mask_te]), 0.5)
    cor_prob_full = np.zeros(len(test))
    cor_prob_full[corner_mask_te] = 1 - poisson.cdf(8, cor_lambda)
    print(f"    OK (train: {corner_mask_tr.sum()}, test: {corner_mask_te.sum()})")
else:
    cor_prob_full = np.full(len(test), 0.5)
    print("    Atlandi - yeterli veri yok")

all_probs = np.column_stack([ph, pd_, pa])
pick = np.argmax(all_probs, axis=1)
conf = np.max(all_probs, axis=1)

# 4. METRICS
print(f"\n{'=' * 72}")
print("SONUCLAR (son 2000 mac, out-of-sample)")
print("=" * 72)

y_result = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

def logloss_binary(y, p):
    return float(-np.mean(y * np.log(np.clip(p, 1e-9, None)) + (1-y) * np.log(np.clip(1-p, 1e-9, None))))

# 1X2 metrics
ll_1x2 = float(-np.mean(np.log(np.clip(all_probs[np.arange(len(y_result)), y_result], 1e-9, None))))
brier_1x2 = float(np.mean(np.sum((all_probs - np.eye(3)[y_result]) ** 2, axis=1)))
acc_1x2 = float((pick == y_result).mean())
freq_h, freq_d, freq_a = (y_result == 0).mean(), (y_result == 1).mean(), (y_result == 2).mean()
ll_naive = -(freq_h * np.log(freq_h) + freq_d * np.log(freq_d) + freq_a * np.log(freq_a))

print(f"\n--- 1X2 ---")
print(f"  Logloss:  {ll_1x2:.4f}  (naive: {ll_naive:.4f})  calisma: {ll_1x2 - ll_naive:+.4f}")
print(f"  Brier:    {brier_1x2:.4f}")
print(f"  Accuracy: {acc_1x2*100:.1f}%  (naive H: {freq_h*100:.1f}%)")

lbl = {0: "Ev sahibi (H)", 1: "Beraberlik (D)", 2: "Deplasman (A)"}
print(f"\n  Tahmin turune gore:")
for c in [0, 1, 2]:
    msk = pick == c
    if msk.sum():
        print(f"    {lbl[c]:20s}  n={msk.sum():>4}  isabet={100*(pick[msk]==y_result[msk]).mean():5.1f}%  ort.conf={conf[msk].mean():.3f}")

print(f"\n  Kalibrasyon:")
print(f"  {'Guven':>10} {'n':>6} {'Gercek':>8} {'Tahmin avg':>10}")
for lo in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    hi = lo + 0.05
    msk = (conf >= lo) & (conf < hi)
    if msk.sum() > 10:
        real_acc = (pick[msk] == y_result[msk]).mean()
        print(f"  {lo*100:.0f}-{hi*100:.0f}%  {msk.sum():>6} {real_acc*100:>7.1f}%  {conf[msk].mean()*100:>9.1f}%")

# Binary markets
markets = [
    ("BTTS", btts_prob, test["btts"].values, test["btts"].mean()),
    ("Over 1.5 Gol", ov15_prob, test["over15"].values, test["over15"].mean()),
    ("Over 2.5 Gol", ov25_prob, test["over25"].values, test["over25"].mean()),
]
if corner_mask_te.sum() > 50:
    markets.append(("Corner O/U 8.5", cor_prob_full[corner_mask_te],
                      test.loc[corner_mask_te, "corner_over85"].values,
                      test.loc[corner_mask_te, "corner_over85"].mean()))

for name, probs, y_true, base_rate in markets:
    ll = logloss_binary(y_true, probs)
    bri = float(np.mean((probs - y_true) ** 2))
    conf_m = np.maximum(probs, 1 - probs).mean()
    maj_acc = max(base_rate, 1 - base_rate) * 100

    print(f"\n--- {name} ---")
    print(f"  Logloss: {ll:.4f}  |  Brier: {bri:.4f}  |  Ort guven: {conf_m:.3f}")
    print(f"  {'Esik':>8} {'Pick':>6} {'Dogruluk':>10} {'Toplama%':>10}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        msk = probs >= thr
        n = msk.sum()
        if n > 0:
            correct = (y_true[msk] == 1).mean()
            print(f"  {thr*100:>6.0f}% {n:>6} {correct*100:>9.1f}% {n/len(probs)*100:>9.1f}%")

    # Summary line
    acc50 = ((probs > 0.5).astype(int) == y_true).mean() * 100
    print(f"  => Accuracy@50%: {acc50:.1f}%  |  Baseline(majority): {maj_acc:.1f}%")

# 5. BAHIS
print(f"\n{'=' * 72}")
print("BAHIS DEGERLENDIRMESI (1X2, FD ort. oranlar)")
print("=" * 72)

oh = pd.to_numeric(test["avg_home_odds"], errors="coerce").to_numpy(float)
od = pd.to_numeric(test["avg_draw_odds"], errors="coerce").to_numpy(float)
oa = pd.to_numeric(test["avg_away_odds"], errors="coerce").to_numpy(float)
ok_odds = np.column_stack([oh, od, oa])
valid_m = np.isfinite(ok_odds).all(axis=1) & (ok_odds > 1.01).all(axis=1)

if valid_m.sum() > 0:
    edge = np.max(all_probs * ok_odds, axis=1) - 1
    prof_flat = np.where(y_result[valid_m] == pick[valid_m],
                         ok_odds[valid_m, pick[valid_m]] - 1, -1.0)
    print(f"  Flat stake (tum mac): n={valid_m.sum()}  ROI={prof_flat.mean()*100:+.1f}%  kar={prof_flat.sum():+.1f}")

    for et in [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]:
        vb = valid_m & (edge > et)
        if vb.sum() >= 5:
            p = np.where(y_result[vb] == pick[vb], ok_odds[vb, pick[vb]] - 1, -1.0)
            print(f"  Edge>{et*100:.0f}%: n={vb.sum():>4}  ROI={p.mean()*100:+.1f}%  kar={p.sum():+.1f}  isabet={(y_result[vb]==pick[vb]).mean()*100:.1f}%")

# 6. FEATURE IMPORTANCE (top 10)
print(f"\n{'=' * 72}")
print("FEATURE IMPORTANCE (1X2 model, top 15)")
print("=" * 72)
imp = m1x2.feature_importances_
order = np.argsort(-imp)[:15]
for i in order:
    print(f"  {feat_names[i]:30s}  {imp[i]:>6}")

print(f"\nTOPLAM SURE: {time.time()-t0:.0f}s")
