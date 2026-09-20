"""v12: Gol ve BTTS modellerini guclendir."""
import sys, os, warnings, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "gold")

print("=" * 70)
print("  v12: GOL + BTTS MODELLERINI GUCLENDIR")
print("=" * 70)

feat = pd.read_parquet(os.path.join(DATA, "features.parquet"))
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["is_draw"] = (feat["result"] == "D").astype(int)
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
feat["over35"] = (feat["total_goals"] > 3.5).astype(int)
feat["under25"] = 1 - feat["over25"]
feat = feat.dropna(subset=["result", "elo_diff"])

# Extra goal-specific features
feat["home_attack_power"] = feat.get("home_gf_5", 0).fillna(0) * feat.get("home_attack_elo", 1500).fillna(1500) / 1500
feat["away_attack_power"] = feat.get("away_gf_5", 0).fillna(0) * feat.get("away_attack_elo", 1500).fillna(1500) / 1500
feat["home_defence_power"] = feat.get("home_ga_5", 0).fillna(0) * feat.get("home_defence_elo", 1500).fillna(1500) / 1500
feat["away_defence_power"] = feat.get("away_ga_5", 0).fillna(0) * feat.get("away_defence_elo", 1500).fillna(1500) / 1500
feat["total_attack"] = feat["home_attack_power"] + feat["away_attack_power"]
feat["total_defence"] = feat["home_defence_power"] + feat["away_defence_power"]
feat["attack_defence_ratio"] = feat["total_attack"] / (feat["total_defence"] + 0.1)
feat["home_scoring_rate"] = feat.get("home_gf_5", 0).fillna(0) / (feat.get("home_ga_5", 1).fillna(1) + 0.1)
feat["away_scoring_rate"] = feat.get("away_gf_5", 0).fillna(0) / (feat.get("away_ga_5", 1).fillna(1) + 0.1)
feat["scoring_rate_diff"] = feat["home_scoring_rate"] - feat["away_scoring_rate"]
feat["both_score_potential"] = ((feat["home_gf_5"].fillna(0) > 0.8) & (feat["away_gf_5"].fillna(0) > 0.8)).astype(int)
feat["high_scoring_match"] = ((feat["home_gf_5"].fillna(0) + feat["away_gf_5"].fillna(0)) > 2.5).astype(int)
feat["low_scoring_match"] = ((feat["home_gf_5"].fillna(0) + feat["away_gf_5"].fillna(0)) < 1.5).astype(int)
feat["home_clean_sheets"] = (feat.get("home_ga_5", 1).fillna(1) < 0.5).astype(int)
feat["away_clean_sheets"] = (feat.get("away_ga_5", 1).fillna(1) < 0.5).astype(int)

# Rolling league stats
feat["lg_real_draw_rate"] = np.nan
feat["lg_real_home_rate"] = np.nan
feat["lg_real_goal_avg"] = np.nan
for league in feat["league"].unique():
    mask = feat["league"] == league
    lg = feat[mask].sort_values("date").copy()
    if len(lg) < 50: continue
    feat.loc[mask, "lg_real_draw_rate"] = lg["is_draw"].rolling(200, min_periods=50).mean().values
    feat.loc[mask, "lg_real_home_rate"] = (lg["result"] == "H").rolling(200, min_periods=50).mean().values
    feat.loc[mask, "lg_real_goal_avg"] = lg["total_goals"].rolling(200, min_periods=50).mean().values
feat = feat.dropna(subset=["lg_real_draw_rate"])

feat["elo_diff_abs"] = feat["elo_diff"].abs()
feat["form_diff"] = feat.get("home_pts_5", 0).fillna(0) - feat.get("away_pts_5", 0).fillna(0)
feat["form_diff_abs"] = feat["form_diff"].abs()
feat["gf_diff"] = feat.get("home_gf_5", 0).fillna(0) - feat.get("away_gf_5", 0).fillna(0)
feat["gf_diff_abs"] = feat["gf_diff"].abs()
feat["ga_diff"] = feat.get("home_ga_5", 0).fillna(0) - feat.get("away_ga_5", 0).fillna(0)
feat["ga_diff_abs"] = feat["ga_diff"].abs()
feat["teams_even"] = (feat["elo_diff_abs"] < 30).astype(int)
feat["teams_very_even"] = (feat["elo_diff_abs"] < 15).astype(int)
feat["form_similar"] = (feat["form_diff_abs"] < 1.0).astype(int)
feat["attack_similar"] = (feat["gf_diff_abs"] < 0.5).astype(int)
feat["h2h_draw_high"] = (feat.get("h2h_draw", 0).fillna(0) > 0.3).astype(int)
feat["lg_draw_signal"] = (feat["lg_real_draw_rate"] > 0.28).astype(int)

# GOAL-SPECIFIC FEATURES
GOAL_FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_attack_elo", "home_defence_elo", "away_attack_elo", "away_defence_elo",
    "attack_elo_diff", "defence_elo_diff",
    "home_gf_5", "home_ga_5", "home_pts_5",
    "away_gf_5", "away_ga_5", "away_pts_5",
    "home_gf_3", "away_gf_3",
    "home_hgf_5", "home_hga_5", "home_hpts_5",
    "away_agf_5", "away_aga_5", "away_apts_5",
    "home_w_gf", "home_w_ga", "away_w_gf", "away_w_ga",
    "home_w_shots", "home_w_sot", "away_w_shots", "away_w_sot",
    "home_rest_days", "away_rest_days",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "home_momentum", "away_momentum",
    "home_wins_last5", "away_wins_last5",
    "home_gdiff5", "away_gdiff5",
    "data_completeness",
    # GOAL-SPECIFIC
    "home_attack_power", "away_attack_power",
    "home_defence_power", "away_defence_power",
    "total_attack", "total_defence", "attack_defence_ratio",
    "home_scoring_rate", "away_scoring_rate", "scoring_rate_diff",
    "both_score_potential", "high_scoring_match", "low_scoring_match",
    "home_clean_sheets", "away_clean_sheets",
    "lg_real_draw_rate", "lg_real_home_rate", "lg_real_goal_avg",
    "form_diff", "gf_diff", "ga_diff", "gf_diff_abs", "ga_diff_abs",
]

FG = [c for c in GOAL_FEATURES if c in feat.columns]
print("  Goal features: %d" % len(FG))

# Load existing 1X2 model data
FINAL_CUTOFF = pd.Timestamp("2024-09-01")
final_tr = feat[feat["date"] >= FINAL_CUTOFF].copy().dropna(subset=["result"])
for c in FG:
    final_tr[c] = final_tr[c].fillna(0)

# Load existing web_model to keep 1X2 models
with open(os.path.join(ROOT, "web_model.pkl"), "rb") as f:
    wm = pickle.load(f)

# Also load original features for 1X2 model
F_ORIG = wm["features"]

X_f_orig = final_tr[[c for c in F_ORIG if c in final_tr.columns]].fillna(0)
X_f_goal = final_tr[FG].fillna(0)
t_f = (final_tr["date"].max() - final_tr["date"]).dt.days.clip(lower=0)
d_f = np.exp(-1.0 * t_f / 365.0).values
sp = int(len(X_f_goal) * 0.85)

print("  Train: %d, Cal: %d" % (sp, len(final_tr) - sp))

# ===== BTTS MODEL — 5 models ensemble =====
print("\n  BTTS model (5-ensemble)...")
y_btts = final_tr["btts"].values
btts_models = []
for seed in [42, 99, 123, 456, 789]:
    bm = lgb.LGBMClassifier(objective="binary", num_leaves=40, learning_rate=0.02,
        n_estimators=500, max_depth=6, min_child_samples=60,
        subsample=0.75, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
        verbose=-1, random_state=seed)
    bm.fit(X_f_goal.iloc[:sp], y_btts[:sp], sample_weight=d_f[:sp])
    # Calibrate
    raw = bm.predict_proba(X_f_goal.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw, y_btts[sp:].astype(float))
    btts_models.append((bm, ir))

# BTTS test
btts_cal = np.mean([ir.predict(bm.predict_proba(X_f_goal.iloc[sp:])[:, 1]) for bm, ir in btts_models], axis=0)
btts_actual = y_btts[sp:]
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    mask = btts_cal >= thr
    n = mask.sum()
    if n > 0:
        corr = btts_actual[mask].mean() * 100
        print("    BTTS > %.0f%%: %4d picks, %.1f%% correct" % (thr*100, n, corr))

# ===== OVER 2.5 MODEL — 5 models ensemble =====
print("\n  Over2.5 model (5-ensemble)...")
y_over25 = final_tr["over25"].values
over25_models = []
for seed in [42, 99, 123, 456, 789]:
    om = lgb.LGBMClassifier(objective="binary", num_leaves=40, learning_rate=0.02,
        n_estimators=500, max_depth=6, min_child_samples=60,
        subsample=0.75, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
        verbose=-1, random_state=seed)
    om.fit(X_f_goal.iloc[:sp], y_over25[:sp], sample_weight=d_f[:sp])
    raw = om.predict_proba(X_f_goal.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw, y_over25[sp:].astype(float))
    over25_models.append((om, ir))

o25_cal = np.mean([ir.predict(om.predict_proba(X_f_goal.iloc[sp:])[:, 1]) for om, ir in over25_models], axis=0)
o25_actual = y_over25[sp:]
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    mask = o25_cal >= thr
    n = mask.sum()
    if n > 0:
        corr = o25_actual[mask].mean() * 100
        print("    O25 > %.0f%%: %4d picks, %.1f%% correct" % (thr*100, n, corr))

# ===== OVER 1.5 MODEL =====
print("\n  Over1.5 model (5-ensemble)...")
y_over15 = final_tr["over15"].values
over15_models = []
for seed in [42, 99, 123, 456, 789]:
    om = lgb.LGBMClassifier(objective="binary", num_leaves=40, learning_rate=0.02,
        n_estimators=500, max_depth=6, min_child_samples=60,
        subsample=0.75, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
        verbose=-1, random_state=seed)
    om.fit(X_f_goal.iloc[:sp], y_over15[:sp], sample_weight=d_f[:sp])
    raw = om.predict_proba(X_f_goal.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw, y_over15[sp:].astype(float))
    over15_models.append((om, ir))

o15_cal = np.mean([ir.predict(om.predict_proba(X_f_goal.iloc[sp:])[:, 1]) for om, ir in over15_models], axis=0)
o15_actual = y_over15[sp:]
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    mask = o15_cal >= thr
    n = mask.sum()
    if n > 0:
        corr = o15_actual[mask].mean() * 100
        print("    O15 > %.0f%%: %4d picks, %.1f%% correct" % (thr*100, n, corr))

# ===== OVER 3.5 MODEL =====
print("\n  Over3.5 model (5-ensemble)...")
y_over35 = final_tr["over35"].values
over35_models = []
for seed in [42, 99, 123, 456, 789]:
    om = lgb.LGBMClassifier(objective="binary", num_leaves=40, learning_rate=0.02,
        n_estimators=500, max_depth=6, min_child_samples=60,
        subsample=0.75, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
        verbose=-1, random_state=seed)
    om.fit(X_f_goal.iloc[:sp], y_over35[:sp], sample_weight=d_f[:sp])
    raw = om.predict_proba(X_f_goal.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw, y_over35[sp:].astype(float))
    over35_models.append((om, ir))

o35_cal = np.mean([ir.predict(om.predict_proba(X_f_goal.iloc[sp:])[:, 1]) for om, ir in over35_models], axis=0)
o35_actual = y_over35[sp:]
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    mask = o35_cal >= thr
    n = mask.sum()
    if n > 0:
        corr = o35_actual[mask].mean() * 100
        print("    O35 > %.0f%%: %4d picks, %.1f%% correct" % (thr*100, n, corr))

# ===== SAVE =====
print("\n  Saving...")
wm["m_btts"] = btts_models
wm["m_over25"] = over25_models
wm["m_over15"] = over15_models
wm["m_over35"] = over35_models
wm["version"] = "v12_goals_enhanced"
wm["goal_features"] = FG

with open(os.path.join(ROOT, "web_model.pkl"), "wb") as f:
    pickle.dump(wm, f)

print("  Saved: web_model.pkl (v12)")
print("  Sure: %ds" % (time.time() - t0))
