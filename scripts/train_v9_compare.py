"""Draw-aware 1X2 v3 — dogru strateji.
   1) Ana 1X2 model: class_weight yok veya cok hafif
   2) Ayr draw binary model: sadece draw tespiti
   3) Prediction: draw modeli cok eminse + ana model karissizsa -> draw
"""
import sys, os, warnings, json, time, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 70)
print("  DRAW-AWARE 1X2 v3")
print("=" * 70)

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
feat["is_draw"] = (feat["result"] == "D").astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

# DRAW FEATURES
feat["elo_diff_abs"] = feat["elo_diff"].abs()
feat["form_diff"] = feat.get("home_pts_5", 0).fillna(0) - feat.get("away_pts_5", 0).fillna(0)
feat["form_diff_abs"] = feat["form_diff"].abs()
feat["gf_diff"] = feat.get("home_gf_5", 0).fillna(0) - feat.get("away_gf_5", 0).fillna(0)
feat["gf_diff_abs"] = feat["gf_diff"].abs()
feat["ga_diff"] = feat.get("home_ga_5", 0).fillna(0) - feat.get("away_ga_5", 0).fillna(0)
feat["ga_diff_abs"] = feat["ga_diff"].abs()
feat["teams_very_even"] = (feat["elo_diff_abs"] < 15).astype(int)
feat["teams_even"] = (feat["elo_diff_abs"] < 30).astype(int)
feat["form_similar"] = (feat["form_diff_abs"] < 1.0).astype(int)
feat["attack_similar"] = (feat["gf_diff_abs"] < 0.5).astype(int)
feat["h2h_draw_high"] = (feat.get("h2h_draw", 0).fillna(0) > 0.3).astype(int)
feat["lg_draw_high"] = (feat.get("lg_draw_rate", 0.25).fillna(0.25) > 0.28).astype(int)
feat["draw_signal"] = (
    feat["teams_even"].astype(int) + feat["form_similar"].astype(int) +
    feat["attack_similar"].astype(int) + feat["h2h_draw_high"].astype(int) +
    feat["lg_draw_high"].astype(int)
)

# FEATURES
BASE_FEATURES = [
    "elo_diff", "elo_diff_abs", "home_elo", "away_elo",
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
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate", "lg_corner_avg",
    "home_momentum", "away_momentum",
    "home_wins_last5", "away_wins_last5",
    "home_gdiff5", "away_gdiff5",
    "home_form_std", "away_form_std",
    "home_pts_std", "away_pts_std", "data_completeness",
    # DRAW FEATURES
    "form_diff", "form_diff_abs", "gf_diff", "gf_diff_abs",
    "ga_diff", "ga_diff_abs",
    "teams_even", "teams_very_even", "form_similar", "attack_similar",
    "h2h_draw_high", "lg_draw_high", "draw_signal",
]
F = [c for c in BASE_FEATURES if c in feat.columns]
print("  Feature: %d" % len(F))

WINDOWS = [
    ("2023-01-01", "2023-04-01", "2023-Q1"),
    ("2023-04-01", "2023-07-01", "2023-Q2"),
    ("2023-07-01", "2023-10-01", "2023-Q3"),
    ("2023-10-01", "2024-01-01", "2023-Q4"),
    ("2024-01-01", "2024-04-01", "2024-Q1"),
    ("2024-04-01", "2024-07-01", "2024-Q2"),
    ("2024-07-01", "2024-10-01", "2024-Q3"),
    ("2024-10-01", "2025-01-01", "2024-Q4"),
    ("2025-01-01", "2025-04-01", "2025-Q1"),
    ("2025-04-01", "2025-07-01", "2025-Q2"),
    ("2025-07-01", "2025-10-01", "2025-Q3"),
]

print("\n" + "=" * 70)
print("  WALK-FORWARD: 3 VERSIYON KARSILASTIRMA")
print("  V1: draw_weight=4.0 (mevcut)")
print("  V2: draw_weight=1.5 (hafif)")
print("  V3: draw_weight=0 + draw overlay")
print("=" * 70)

all_results = []
for vs, ve, lb in WINDOWS:
    vs_, ve_ = pd.Timestamp(vs), pd.Timestamp(ve)
    tm = (feat["date"] >= vs_ - pd.DateOffset(years=2)) & (feat["date"] < vs_)
    vm = (feat["date"] >= vs_) & (feat["date"] < ve_)
    tr = feat[tm].copy()
    va = feat[vm].copy()
    if len(tr) < 5000 or len(va) < 100:
        continue

    t_days = (tr["date"].max() - tr["date"]).dt.days.clip(lower=0)
    decay = np.exp(-1.0 * t_days / 365.0).values
    sp = int(len(tr) * 0.85)

    X_tr = tr[F].fillna(0)
    X_va = va[F].fillna(0)
    X_sp = X_tr.iloc[:sp]
    X_cal = X_tr.iloc[sp:]

    y = tr["result"].map({"H": 0, "D": 1, "A": 2}).values
    yv = va["result"].map({"H": 0, "D": 1, "A": 2}).values

    # V1: draw_weight=4.0
    results_v1 = []
    for seed in [42, 99]:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
            learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed, class_weight={0: 1, 1: 4.0, 2: 1})
        m.fit(X_sp, y[:sp], sample_weight=decay[:sp])
        results_v1.append(m.predict_proba(X_va))
    p_v1 = np.mean(results_v1, axis=0)
    s = p_v1.sum(axis=1, keepdims=True); s = np.where(s == 0, 1, s); p_v1 /= s

    # V2: draw_weight=1.5 (hafif)
    results_v2 = []
    for seed in [42, 99]:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
            learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed, class_weight={0: 1, 1: 1.5, 2: 1})
        m.fit(X_sp, y[:sp], sample_weight=decay[:sp])
        results_v2.append(m.predict_proba(X_va))
    p_v2 = np.mean(results_v2, axis=0)
    s = p_v2.sum(axis=1, keepdims=True); s = np.where(s == 0, 1, s); p_v2 /= s

    # V3a: draw_weight=0 (none)
    results_v3 = []
    for seed in [42, 99]:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
            learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed)
        m.fit(X_sp, y[:sp], sample_weight=decay[:sp])
        results_v3.append(m.predict_proba(X_va))
    p_v3 = np.mean(results_v3, axis=0)
    s = p_v3.sum(axis=1, keepdims=True); s = np.where(s == 0, 1, s); p_v3 /= s

    # DRAW MODEL (standalone binary)
    y_draw = tr["is_draw"].values
    draw_probs = []
    for seed in [42, 123, 456]:
        dm = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=80,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed)
        dm.fit(X_sp, y_draw[:sp], sample_weight=decay[:sp])
        raw_cal = dm.predict_proba(X_cal)[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal, y_draw[sp:].astype(float))
        draw_probs.append(ir.predict(dm.predict_proba(X_va)[:, 1]))
    draw_prob = np.mean(draw_probs, axis=0)

    # CALIBRATION SET for draw model
    raw_cal_draw = draw_probs  # already calibrated

    # V3b: draw overlay — sadece draw modeli cok eminse + ana model karissizsa
    p_v3_overlay = p_v3.copy()
    h_prob = p_v3_overlay[:, 0]
    a_prob = p_v3_overlay[:, 2]
    d_prob = p_v3_overlay[:, 1]
    gap = np.abs(h_prob - a_prob)

    # STRATEJI: draw olasiligini sadece belirli kosullarda boost et
    # 1. Ana model cok karissiz (gap < 0.15) + draw modeli guclu (> 0.35)
    s1 = (gap < 0.15) & (draw_prob > 0.35)
    p_v3_overlay[s1, 1] = np.maximum(p_v3_overlay[s1, 1], draw_prob[s1])

    # 2. Takimlar cok esit (elo_diff_abs < 20) + draw modeli > 0.32
    s2 = (va["elo_diff_abs"].values < 20) & (draw_prob > 0.32)
    p_v3_overlay[s2, 1] = np.maximum(p_v3_overlay[s2, 1], draw_prob[s2] * 0.9)

    # 3. draw_signal >= 4 + draw modeli > 0.30
    ds = va["draw_signal"].values if "draw_signal" in va.columns else np.zeros(len(va))
    s3 = (ds >= 4) & (draw_prob > 0.30)
    p_v3_overlay[s3, 1] = np.maximum(p_v3_overlay[s3, 1], draw_prob[s3] * 0.85)

    s = p_v3_overlay.sum(axis=1, keepdims=True); s = np.where(s == 0, 1, s); p_v3_overlay /= s

    # EVALUATE
    pred_v1 = np.argmax(p_v1, axis=1)
    pred_v2 = np.argmax(p_v2, axis=1)
    pred_v3 = np.argmax(p_v3, axis=1)
    pred_v3o = np.argmax(p_v3_overlay, axis=1)

    acc_v1 = (pred_v1 == yv).mean()
    acc_v2 = (pred_v2 == yv).mean()
    acc_v3 = (pred_v3 == yv).mean()
    acc_v3o = (pred_v3o == yv).mean()

    draw_m = yv == 1
    nd_m = yv != 1
    n_draw = draw_m.sum()

    dh_v1 = (pred_v1[draw_m] == 1).sum() if n_draw > 0 else 0
    dh_v2 = (pred_v2[draw_m] == 1).sum() if n_draw > 0 else 0
    dh_v3 = (pred_v3[draw_m] == 1).sum() if n_draw > 0 else 0
    dh_v3o = (pred_v3o[draw_m] == 1).sum() if n_draw > 0 else 0

    nd_v1 = (pred_v1[nd_m] == yv[nd_m]).mean() if nd_m.sum() > 0 else 0
    nd_v2 = (pred_v2[nd_m] == yv[nd_m]).mean() if nd_m.sum() > 0 else 0
    nd_v3 = (pred_v3[nd_m] == yv[nd_m]).mean() if nd_m.sum() > 0 else 0
    nd_v3o = (pred_v3o[nd_m] == yv[nd_m]).mean() if nd_m.sum() > 0 else 0

    print("  %s n=%5d" % (lb, len(va)))
    print("    V1(w=4.0): %5.1f%%  DH:%5.1f%% ND:%5.1f%%" % (acc_v1*100, dh_v1/n_draw*100 if n_draw else 0, nd_v1*100))
    print("    V2(w=1.5): %5.1f%%  DH:%5.1f%% ND:%5.1f%%" % (acc_v2*100, dh_v2/n_draw*100 if n_draw else 0, nd_v2*100))
    print("    V3(w=0):   %5.1f%%  DH:%5.1f%% ND:%5.1f%%" % (acc_v3*100, dh_v3/n_draw*100 if n_draw else 0, nd_v3*100))
    print("    V3+overlay:%5.1f%%  DH:%5.1f%% ND:%5.1f%%" % (acc_v3o*100, dh_v3o/n_draw*100 if n_draw else 0, nd_v3o*100))

    all_results.append({
        "period": lb, "n": len(va), "n_draw": n_draw,
        "acc_v1": acc_v1, "acc_v2": acc_v2, "acc_v3": acc_v3, "acc_v3o": acc_v3o,
        "dh_v1": dh_v1, "dh_v2": dh_v2, "dh_v3": dh_v3, "dh_v3o": dh_v3o,
        "nd_v1": nd_v1, "nd_v2": nd_v2, "nd_v3": nd_v3, "nd_v3o": nd_v3o,
    })

print("\n" + "=" * 70)
print("  AGGREGATE")
print("=" * 70)
df = pd.DataFrame(all_results)
total_draws = df["n_draw"].sum()
print("  V1 (w=4.0):    Acc: %.1f%%  Draw Hit: %.1f%% (%d/%d)  ND: %.1f%%" % (
    df["acc_v1"].mean()*100, df["dh_v1"].sum()/total_draws*100, int(df["dh_v1"].sum()), total_draws, df["nd_v1"].mean()*100))
print("  V2 (w=1.5):    Acc: %.1f%%  Draw Hit: %.1f%% (%d/%d)  ND: %.1f%%" % (
    df["acc_v2"].mean()*100, df["dh_v2"].sum()/total_draws*100, int(df["dh_v2"].sum()), total_draws, df["nd_v2"].mean()*100))
print("  V3 (w=0):      Acc: %.1f%%  Draw Hit: %.1f%% (%d/%d)  ND: %.1f%%" % (
    df["acc_v3"].mean()*100, df["dh_v3"].sum()/total_draws*100, int(df["dh_v3"].sum()), total_draws, df["nd_v3"].mean()*100))
print("  V3+overlay:    Acc: %.1f%%  Draw Hit: %.1f%% (%d/%d)  ND: %.1f%%" % (
    df["acc_v3o"].mean()*100, df["dh_v3o"].sum()/total_draws*100, int(df["dh_v3o"].sum()), total_draws, df["nd_v3o"].mean()*100))

print("\n  Sure: %ds" % (time.time() - t0))
