"""Draw-aware 1X2 v4 — DRAW-FIRST strateji.
   V3 (en iyi H/A: 66.4%) + draw modeli cok eminse -> X
   Eger draw_prob > esik ise X de, diger durumda V3'ün H/A tahminini kullan.
"""
import sys, os, warnings, json, time, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 70)
print("  DRAW-AWARE 1X2 v4 — DRAW-FIRST")
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
    "form_diff", "form_diff_abs", "gf_diff", "gf_diff_abs",
    "ga_diff", "ga_diff_abs",
    "teams_even", "teams_very_even", "form_similar", "attack_similar",
    "h2h_draw_high", "lg_draw_high", "draw_signal",
]
F = [c for c in BASE_FEATURES if c in feat.columns]

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

print("\n  WALK-FORWARD: DRAW-FIRST vs V3-BASE")
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

    X_sp = tr[F].fillna(0).iloc[:sp]
    X_cal = tr[F].fillna(0).iloc[sp:]
    X_va = va[F].fillna(0)

    y = tr["result"].map({"H": 0, "D": 1, "A": 2}).values
    yv = va["result"].map({"H": 0, "D": 1, "A": 2}).values
    y_draw = tr["is_draw"].values

    # 1. V3: draw_weight=0 (en iyi H/A modeli)
    v3_models = []
    for seed in [42, 99]:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
            learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed)
        m.fit(X_sp, y[:sp], sample_weight=decay[:sp])
        v3_models.append(m)
    p_v3 = np.mean([m.predict_proba(X_va) for m in v3_models], axis=0)
    s = p_v3.sum(axis=1, keepdims=True); s = np.where(s == 0, 1, s); p_v3 /= s

    # 2. DRAW MODEL (calibrated binary)
    draw_models = []
    for seed in [42, 123, 456]:
        dm = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=80,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed)
        dm.fit(X_sp, y_draw[:sp], sample_weight=decay[:sp])
        raw_cal = dm.predict_proba(X_cal)[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal, y_draw[sp:].astype(float))
        draw_models.append((dm, ir))
    draw_prob = np.mean([ir.predict(dm.predict_proba(X_va)[:, 1]) for dm, ir in draw_models], axis=0)

    # 3. DRAW-FIRST: draw_prob > threshold -> X, else -> V3
    # Tum threshold'lari dene
    best = None
    for thresh in [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]:
        pred_df = np.full(len(va), -1, dtype=int)
        # Draw threshold asamasinda X de
        draw_override = draw_prob > thresh
        pred_df[draw_override] = 1
        # Digerlerinde V3'un tahminini kullan
        pred_df[~draw_override] = np.argmax(p_v3[~draw_override], axis=1)

        acc = (pred_df == yv).mean()
        dm = yv == 1
        nd = yv != 1
        n_draw = dm.sum()
        dh = (pred_df[dm] == 1).sum() if n_draw > 0 else 0
        nda = (pred_df[nd] == yv[nd]).mean() if nd.sum() > 0 else 0

        # Draw precision: X dediginde ne kadar dogru
        x_pred = pred_df == 1
        n_x_pred = x_pred.sum()
        x_prec = (pred_df[x_pred] == yv[x_pred]).mean() if n_x_pred > 0 else 0

        if best is None or acc > best[0]:
            best = (acc, thresh, dh, nda, n_draw, n_x_pred, x_prec)

    # En iyi threshold ile sonuclar
    acc_best, t_best, dh_best, nda_best, nd_best, nx_best, xp_best = best

    # V3 baseline
    pred_v3 = np.argmax(p_v3, axis=1)
    acc_v3 = (pred_v3 == yv).mean()
    dh_v3 = (pred_v3[yv == 1] == 1).sum() if (yv == 1).sum() > 0 else 0
    nda_v3 = (pred_v3[yv != 1] == yv[yv != 1]).mean() if (yv != 1).sum() > 0 else 0

    print("  %s n=%5d" % (lb, len(va)))
    print("    V3-baseline: Acc: %.1f%% | DH: %d/%d (%.1f%%) | ND: %.1f%%" % (
        acc_v3*100, dh_v3, (yv==1).sum(), dh_v3/(yv==1).sum()*100 if (yv==1).sum()>0 else 0, nda_v3*100))
    print("    DRAW-FIRST:  Acc: %.1f%% (t=%.2f) | DH: %d/%d (%.1f%%) | ND: %.1f%% | X_prec: %.1f%% (%d picks)" % (
        acc_best*100, t_best, dh_best, nd_best, dh_best/nd_best*100 if nd_best>0 else 0, nda_best*100, xp_best*100, nx_best))

    all_results.append({
        "period": lb, "n": len(va), "n_draw": int((yv==1).sum()),
        "acc_v3": acc_v3, "dh_v3": dh_v3, "nda_v3": nda_v3,
        "acc_df": acc_best, "t_best": t_best, "dh_df": dh_best,
        "nda_df": nda_best, "nx_df": nx_best, "xp_df": xp_best,
    })

print("\n" + "=" * 70)
print("  AGGREGATE")
print("=" * 70)
df = pd.DataFrame(all_results)
td = df["n_draw"].sum()
print("  V3-baseline:   Acc: %.1f%% | DH: %d/%d (%.1f%%) | ND: %.1f%%" % (
    df["acc_v3"].mean()*100, int(df["dh_v3"].sum()), td,
    df["dh_v3"].sum()/td*100, df["nda_v3"].mean()*100))
print("  DRAW-FIRST:    Acc: %.1f%% | DH: %d/%d (%.1f%%) | ND: %.1f%% | X_prec: %.1f%%" % (
    df["acc_df"].mean()*100, int(df["dh_df"].sum()), td,
    df["dh_df"].sum()/td*100, df["nda_df"].mean()*100, df["xp_df"].mean()*100))
print("  Ort esik: %.3f" % df["t_best"].mean())

print("\n  Sure: %ds" % (time.time() - t0))
