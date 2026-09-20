"""v11 production: League-aware draw override ile final model."""
import sys, os, time, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  v11 PRODUCTION: LEAGUE-AWARE 1X2")
print("=" * 70)

# 1. DATA
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

hc = pd.to_numeric(feat.get("home_corners"), errors="coerce").fillna(0)
ac = pd.to_numeric(feat.get("away_corners"), errors="coerce").fillna(0)
feat["total_corners"] = hc + ac
feat["corner_over85"] = (feat["total_corners"] > 8.5).astype(int)
feat["corner_reliable"] = feat["total_corners"] >= 5

# ROLLING LEAGUE STATS
print("  Rolling league stats...")
feat["lg_real_draw_rate"] = np.nan
feat["lg_real_home_rate"] = np.nan
feat["lg_real_goal_avg"] = np.nan
for league in feat["league"].unique():
    mask = feat["league"] == league
    lg = feat[mask].sort_values("date").copy()
    if len(lg) < 50: continue
    feat.loc[mask, "lg_real_draw_rate"] = lg["is_draw"].rolling(200, min_periods=50).mean().values
    feat.loc[mask, "lg_real_home_rate"] = (lg["result"]=="H").rolling(200, min_periods=50).mean().values
    feat.loc[mask, "lg_real_goal_avg"] = lg["total_goals"].rolling(200, min_periods=50).mean().values
feat = feat.dropna(subset=["lg_real_draw_rate"])

# DRAW FEATURES
feat["elo_diff_abs"] = feat["elo_diff"].abs()
feat["form_diff"] = feat.get("home_pts_5",0).fillna(0) - feat.get("away_pts_5",0).fillna(0)
feat["form_diff_abs"] = feat["form_diff"].abs()
feat["gf_diff"] = feat.get("home_gf_5",0).fillna(0) - feat.get("away_gf_5",0).fillna(0)
feat["gf_diff_abs"] = feat["gf_diff"].abs()
feat["ga_diff"] = feat.get("home_ga_5",0).fillna(0) - feat.get("away_ga_5",0).fillna(0)
feat["ga_diff_abs"] = feat["ga_diff"].abs()
feat["teams_even"] = (feat["elo_diff_abs"] < 30).astype(int)
feat["teams_very_even"] = (feat["elo_diff_abs"] < 15).astype(int)
feat["form_similar"] = (feat["form_diff_abs"] < 1.0).astype(int)
feat["attack_similar"] = (feat["gf_diff_abs"] < 0.5).astype(int)
feat["h2h_draw_high"] = (feat.get("h2h_draw",0).fillna(0) > 0.3).astype(int)
feat["lg_draw_signal"] = (feat["lg_real_draw_rate"] > 0.28).astype(int)

# FEATURES
F = [
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
    "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts",
    "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "home_momentum", "away_momentum",
    "home_wins_last5", "away_wins_last5",
    "home_gdiff5", "away_gdiff5",
    "home_pts_std", "away_pts_std", "data_completeness",
    "form_diff", "form_diff_abs", "gf_diff", "gf_diff_abs", "ga_diff", "ga_diff_abs",
    "teams_even", "teams_very_even", "form_similar", "attack_similar",
    "h2h_draw_high", "lg_draw_signal",
    "lg_real_draw_rate", "lg_real_home_rate", "lg_real_goal_avg",
]
F = [c for c in F if c in feat.columns]

# 2. FINAL TRAIN
FINAL_CUTOFF = pd.Timestamp("2024-09-01")
final_tr = feat[feat["date"] >= FINAL_CUTOFF].copy().dropna(subset=["result"])
for c in F:
    final_tr[c] = final_tr[c].fillna(0)
X_f = final_tr[F]
t_f = (final_tr["date"].max() - final_tr["date"]).dt.days.clip(lower=0)
d_f = np.exp(-1.0 * t_f / 365.0).values
sp = int(len(X_f) * 0.85)

# 1X2 model
print("  1X2 model egitimi...")
y_1x2 = final_tr["result"].map({"H":0,"D":1,"A":2}).values
main_models = []
cal_models = []
for seed in [42, 99]:
    m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
        learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=seed)
    m.fit(X_f.iloc[:sp], y_1x2[:sp], sample_weight=d_f[:sp])
    main_models.append(m)
    # Calibration for each model separately
    cal = []
    raw_cal = m.predict_proba(X_f.iloc[sp:])
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal[:, c], (y_1x2[sp:] == c).astype(float))
        cal.append(ir)
    cal_models.append(cal)

# Draw model
print("  Draw model egitimi...")
y_draw = final_tr["is_draw"].values
draw_models = []
for seed in [42, 123, 456]:
    dm = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=80,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=seed)
    dm.fit(X_f.iloc[:sp], y_draw[:sp], sample_weight=d_f[:sp])
    raw_cal = dm.predict_proba(X_f.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, y_draw[sp:].astype(float))
    draw_models.append((dm, ir))

# BTTS model
print("  BTTS model egitimi...")
y_btts = final_tr["btts"].values
btts_models = []
for seed in [42, 99]:
    bm = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.03,
        n_estimators=350, max_depth=4, min_child_samples=120,
        subsample=0.65, colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=15.0,
        verbose=-1, random_state=seed)
    bm.fit(X_f.iloc[:sp], y_btts[:sp], sample_weight=d_f[:sp])
    btts_models.append(bm)

# Over2.5 model
print("  Over2.5 model egitimi...")
y_over25 = final_tr["over25"].values
over_models = []
for seed in [42, 99]:
    om = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.03,
        n_estimators=350, max_depth=4, min_child_samples=120,
        subsample=0.65, colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=15.0,
        verbose=-1, random_state=seed)
    om.fit(X_f.iloc[:sp], y_over25[:sp], sample_weight=d_f[:sp])
    over_models.append(om)

# Goals regressors (home + away separately)
from scipy.stats import poisson
print("  Goals regressors...")
y_home_goals = final_tr["home_goals"].values.astype(float)
y_away_goals = final_tr["away_goals"].values.astype(float)
home_goal_models = []
away_goal_models = []
for seed in [42, 99]:
    gm_h = lgb.LGBMRegressor(objective="regression", num_leaves=30, learning_rate=0.03,
        n_estimators=350, max_depth=4, min_child_samples=120,
        subsample=0.65, colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=15.0,
        verbose=-1, random_state=seed)
    gm_h.fit(X_f.iloc[:sp], y_home_goals[:sp], sample_weight=d_f[:sp])
    home_goal_models.append(gm_h)
    
    gm_a = lgb.LGBMRegressor(objective="regression", num_leaves=30, learning_rate=0.03,
        n_estimators=350, max_depth=4, min_child_samples=120,
        subsample=0.65, colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=15.0,
        verbose=-1, random_state=seed+1)
    gm_a.fit(X_f.iloc[:sp], y_away_goals[:sp], sample_weight=d_f[:sp])
    away_goal_models.append(gm_a)

# 3. SAVE
model_data = {
    "m_1x2": (main_models[0], main_models[1], cal_models[0], cal_models[1]),
    "m_draw": draw_models,
    "m_btts": tuple(btts_models),
    "m_over25": tuple(over_models),
    "m_goals": home_goal_models + away_goal_models,
    "m_home_goals": home_goal_models[0] if home_goal_models else None,
    "m_away_goals": away_goal_models[0] if away_goal_models else None,
    "features": F,
    "version": "v11_league_aware",
    "train_date": str(pd.Timestamp.now().date()),
    "train_n": len(final_tr),
    # V5 draw override thresholds
    "draw_thresholds": {
        "high_draw_rate": 0.28,   # lg_real_draw_rate > 0.28 -> high draw lig
        "low_draw_rate": 0.22,    # lg_real_draw_rate < 0.22 -> low draw lig
        "threshold_high_draw": 0.32,  # high draw ligde esik
        "threshold_mid_draw": 0.38,   # mid draw ligde esik
        "threshold_low_draw": 0.45,   # low draw ligde esik
        "close_match_gap": 0.15,      # 1 ve 2 olasiliklari arasindaki max fark
    }
}

out_path = os.path.join(ROOT, "web_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(model_data, f)

print("\n  Kaydedildi: %s" % out_path)
print("  Version: v11_league_aware")
print("  Features: %d" % len(F))
print("  Train: %d mac" % len(final_tr))
print("  Sure: %ds" % (time.time() - t0))
