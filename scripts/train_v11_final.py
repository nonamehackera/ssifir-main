"""v11: FINAL — threshold-based draw override + confidence weighting + league-aware."""
import sys, os, warnings, json, time, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 70)
print("  v11: THRESHOLD + LEAGUE-AWARE DRAW OVERRIDE")
print("=" * 70)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold")
feat = pd.read_parquet(os.path.join(DATA, "features.parquet"))
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["is_draw"] = (feat["result"] == "D").astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

# ROLLING LEAGUE STATS
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

WINDOWS = [
    ("2023-01-01","2023-04-01","2023-Q1"), ("2023-04-01","2023-07-01","2023-Q2"),
    ("2023-07-01","2023-10-01","2023-Q3"), ("2023-10-01","2024-01-01","2023-Q4"),
    ("2024-01-01","2024-04-01","2024-Q1"), ("2024-04-01","2024-07-01","2024-Q2"),
    ("2024-07-01","2024-10-01","2024-Q3"), ("2024-10-01","2025-01-01","2024-Q4"),
    ("2025-01-01","2025-04-01","2025-Q1"), ("2025-04-01","2025-07-01","2025-Q2"),
    ("2025-07-01","2025-10-01","2025-Q3"),
]

print("\n  WALK-FORWARD: 5 VERSIYON")
print("=" * 70)

all_res = []
for vs, ve, lb in WINDOWS:
    vs_, ve_ = pd.Timestamp(vs), pd.Timestamp(ve)
    tm = (feat["date"] >= vs_ - pd.DateOffset(years=2)) & (feat["date"] < vs_)
    vm = (feat["date"] >= vs_) & (feat["date"] < ve_)
    tr = feat[tm].copy(); va = feat[vm].copy()
    if len(tr) < 5000 or len(va) < 100: continue

    t_days = (tr["date"].max() - tr["date"]).dt.days.clip(lower=0)
    decay = np.exp(-1.0 * t_days / 365.0).values
    sp = int(len(tr) * 0.85)
    y = tr["result"].map({"H":0,"D":1,"A":2}).values
    yv = va["result"].map({"H":0,"D":1,"A":2}).values

    X_sp = tr[F].fillna(0).iloc[:sp]
    X_va = va[F].fillna(0)

    # 1. 1X2 MODEL (draw_weight=0)
    m1 = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
        learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=42)
    m1.fit(X_sp, y[:sp], sample_weight=decay[:sp])
    m2 = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
        learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=99)
    m2.fit(X_sp, y[:sp], sample_weight=decay[:sp])
    p_main = (m1.predict_proba(X_va) + m2.predict_proba(X_va)) / 2
    s = p_main.sum(axis=1, keepdims=True); s = np.where(s==0,1,s); p_main /= s

    # 2. DRAW MODEL (binary, calibrated)
    y_draw = tr["is_draw"].values
    draw_models = []
    for seed in [42, 123, 456]:
        dm = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=80,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed)
        dm.fit(X_sp, y_draw[:sp], sample_weight=decay[:sp])
        raw_cal = dm.predict_proba(tr[F].fillna(0).iloc[sp:])[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal, y_draw[sp:].astype(float))
        draw_models.append((dm, ir))
    draw_prob = np.mean([ir.predict(dm.predict_proba(X_va)[:, 1]) for dm, ir in draw_models], axis=0)

    # V1: Baseline (argmax)
    pred_v1 = np.argmax(p_main, axis=1)
    
    # V2: draw_prob > 0.40 -> X
    pred_v2 = pred_v1.copy()
    pred_v2[draw_prob > 0.40] = 1
    
    # V3: draw_prob > 0.35 AND close match -> X
    pred_v3 = pred_v1.copy()
    close = np.abs(p_main[:,0] - p_main[:,2]) < 0.15
    pred_v3[(draw_prob > 0.35) & close] = 1
    
    # V4: League-aware threshold
    # high_draw ligde daha dusuk esik, low_draw ligde daha yuksek esik
    lg_draw = va["lg_real_draw_rate"].values
    pred_v4 = pred_v1.copy()
    # high_draw (>0.28): threshold 0.32
    mask_hd = (lg_draw > 0.28) & (draw_prob > 0.32)
    pred_v4[mask_hd] = 1
    # mid_draw (0.24-0.28): threshold 0.38
    mask_md = (lg_draw >= 0.22) & (lg_draw <= 0.28) & (draw_prob > 0.38)
    pred_v4[mask_md] = 1
    # low_draw (<0.22): threshold 0.45
    mask_ld = (lg_draw < 0.22) & (draw_prob > 0.45)
    pred_v4[mask_ld] = 1

    # V5: V4 + close match requirement
    pred_v5 = pred_v1.copy()
    mask_hd5 = (lg_draw > 0.28) & (draw_prob > 0.32) & close
    pred_v5[mask_hd5] = 1
    mask_md5 = (lg_draw >= 0.22) & (lg_draw <= 0.28) & (draw_prob > 0.38) & close
    pred_v5[mask_md5] = 1
    mask_ld5 = (lg_draw < 0.22) & (draw_prob > 0.45) & close
    pred_v5[mask_ld5] = 1

    def eval_(p, yv, lb):
        acc = (p == yv).mean()
        dm = yv == 1; nd = yv != 1; n_draw = dm.sum()
        dh = (p[dm] == 1).sum() if n_draw > 0 else 0
        nda = (p[nd] == yv[nd]).mean() if nd.sum() > 0 else 0
        x_pred = p == 1; nx = x_pred.sum()
        xp = (p[x_pred] == yv[x_pred]).mean() if nx > 0 else 0
        return acc, dh, nda, n_draw, nx, xp

    r1 = eval_(pred_v1, yv, lb)
    r2 = eval_(pred_v2, yv, lb)
    r3 = eval_(pred_v3, yv, lb)
    r4 = eval_(pred_v4, yv, lb)
    r5 = eval_(pred_v5, yv, lb)

    print("  %s n=%d" % (lb, len(va)))
    print("    V1 baseline:      Acc:%.1f%% DH:%.1f%% ND:%.1f%%" % (r1[0]*100, r1[1]/r1[3]*100 if r1[3] else 0, r1[2]*100))
    print("    V2 draw>0.40:     Acc:%.1f%% DH:%.1f%% ND:%.1f%% X:%d(%.0f%%)" % (r2[0]*100, r2[1]/r2[3]*100 if r2[3] else 0, r2[2]*100, r2[4], r2[5]*100))
    print("    V3 draw>0.35+cl:  Acc:%.1f%% DH:%.1f%% ND:%.1f%% X:%d(%.0f%%)" % (r3[0]*100, r3[1]/r3[3]*100 if r3[3] else 0, r3[2]*100, r3[4], r3[5]*100))
    print("    V4 league-aware:  Acc:%.1f%% DH:%.1f%% ND:%.1f%% X:%d(%.0f%%)" % (r4[0]*100, r4[1]/r4[3]*100 if r4[3] else 0, r4[2]*100, r4[4], r4[5]*100))
    print("    V5 league+close:  Acc:%.1f%% DH:%.1f%% ND:%.1f%% X:%d(%.0f%%)" % (r5[0]*100, r5[1]/r5[3]*100 if r5[3] else 0, r5[2]*100, r5[4], r5[5]*100))

    all_res.append({
        "n": len(va), "n_draw": r1[3],
        "acc1": r1[0], "acc2": r2[0], "acc3": r3[0], "acc4": r4[0], "acc5": r5[0],
        "dh1": r1[1], "dh2": r2[1], "dh3": r3[1], "dh4": r4[1], "dh5": r5[1],
        "nda1": r1[2], "nda4": r4[2], "nda5": r5[2],
        "nx4": r2[4], "xp4": r2[5], "nx5": r3[4], "xp5": r3[5],
    })

print("\n" + "=" * 70)
print("  AGGREGATE")
print("=" * 70)
df = pd.DataFrame(all_res)
td = df["n_draw"].sum()
for i, name in [(1,"V1 baseline"), (2,"V2 draw>0.40"), (3,"V3 draw>0.35+close"), (4,"V4 league-aware"), (5,"V5 league+close")]:
    a = df["acc%d"%i].mean()*100
    d = df["dh%d"%i].sum()/td*100 if td else 0
    n = df["nda%d"%i].mean()*100 if i in [1,4,5] else 0
    print("  %-20s Acc:%.1f%% DH:%.1f%% ND:%.1f%%" % (name, a, d, n))

print("\n  Sure: %ds" % (time.time() - t0))
