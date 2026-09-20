"""Draw-aware 1X2 v2 - dogru hybrid strateji."""
import sys, os, warnings, json, time, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 70)
print("  DRAW-AWARE 1X2 v2")
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
feat["home_advantage_weak"] = (feat["elo_diff"].abs() < 50).astype(int)
feat["teams_even"] = (feat["elo_diff_abs"] < 30).astype(int)
feat["teams_very_even"] = (feat["elo_diff_abs"] < 15).astype(int)
feat["form_similar"] = (feat["form_diff_abs"] < 1.0).astype(int)
feat["form_very_similar"] = (feat["form_diff_abs"] < 0.5).astype(int)
feat["attack_similar"] = (feat["gf_diff_abs"] < 0.5).astype(int)
feat["h2h_draw_high"] = (feat.get("h2h_draw", 0).fillna(0) > 0.3).astype(int)
feat["lg_draw_high"] = (feat.get("lg_draw_rate", 0.25).fillna(0.25) > 0.28).astype(int)
feat["draw_signal"] = (
    feat["teams_even"].astype(int) + 
    feat["form_similar"].astype(int) + 
    feat["attack_similar"].astype(int) + 
    feat["h2h_draw_high"].astype(int) + 
    feat["lg_draw_high"].astype(int)
)

# 1X2 FEATURES (mevcut ile ayni)
BASE_FEATURES = [
    "elo_diff", "elo_diff_abs",
    "home_elo", "away_elo",
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
    "home_pts_std", "away_pts_std",
    "data_completeness",
    # DRAW FEATURES
    "form_diff", "form_diff_abs",
    "gf_diff", "gf_diff_abs",
    "ga_diff", "ga_diff_abs",
    "home_advantage_weak", "teams_even", "teams_very_even",
    "form_similar", "form_very_similar",
    "attack_similar",
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
print("  WALK-FORWARD")
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
    X_tr_sp = X_tr.iloc[:sp]
    X_val = X_tr.iloc[sp:]

    y_1x2 = tr["result"].map({"H": 0, "D": 1, "A": 2}).values
    y_1x2_va = va["result"].map({"H": 0, "D": 1, "A": 2}).values
    y_draw = tr["is_draw"].values
    y_draw_va = va["is_draw"].values

    # DRAW MODEL
    draw_probs = []
    for seed in [42, 123, 456]:
        dm = lgb.LGBMClassifier(
            objective="binary", num_leaves=30, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=80,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed
        )
        dm.fit(X_tr_sp, y_draw[:sp], sample_weight=decay[:sp])
        raw_cal = dm.predict_proba(X_val)[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal, y_draw[sp:].astype(float))
        draw_probs.append(ir.predict(dm.predict_proba(X_va)[:, 1]))
    draw_prob = np.mean(draw_probs, axis=0)

    # 1X2 MODEL
    main_probs = []
    for seed in [42, 123]:
        mm = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, num_leaves=35, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed, class_weight={0: 1, 1: 4.0, 2: 1}
        )
        mm.fit(X_tr_sp, y_1x2[:sp], sample_weight=decay[:sp])
        main_probs.append(mm.predict_proba(X_va))
    main_prob = np.mean(main_probs, axis=0)
    s = main_prob.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    main_prob /= s

    # HYBRID STRATEJI:
    # 1. Ana modelin 1 ve 2 olasiliklari cok yakinsa (fark < 0.15) -> draw olasiligini ekle
    # 2. Draw modeli > 0.35 ise ve draw_signal >= 3 -> draw'a switch et
    # 3. Draw modeli > 0.45 ise -> kesin draw
    hybrid = main_prob.copy()
    
    h_prob = main_prob[:, 0]
    a_prob = main_prob[:, 2]
    d_prob = main_prob[:, 1]
    
    # Close match: 1 ve 2 olasiliklari cok yakinsa
    close_match = np.abs(h_prob - a_prob) < 0.15
    
    # Strategy 1: Close match + draw model destekli
    s1 = close_match & (draw_prob > 0.30) & (va["draw_signal"].values >= 3)
    hybrid[s1, 1] = np.maximum(hybrid[s1, 1], draw_prob[s1] * 0.9)
    
    # Strategy 2: Draw modeli cok guclu
    s2 = draw_prob > 0.40
    hybrid[s2, 1] = np.maximum(hybrid[s2, 1], draw_prob[s2])
    
    # Strategy 3: Cok esit takimlar + draw modeli > 0.35
    s3 = (va["teams_very_even"].values == 1) & (draw_prob > 0.35)
    hybrid[s3, 1] = np.maximum(hybrid[s3, 1], draw_prob[s3] * 0.95)
    
    # Renormalize
    s = hybrid.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    hybrid /= s

    pred_main = np.argmax(main_prob, axis=1)
    pred_hybrid = np.argmax(hybrid, axis=1)
    
    acc_main = (pred_main == y_1x2_va).mean()
    acc_hybrid = (pred_hybrid == y_1x2_va).mean()
    
    draw_mask = y_1x2_va == 1
    n_draw = draw_mask.sum()
    dh_main = (pred_main[draw_mask] == 1).sum() if n_draw > 0 else 0
    dh_hybrid = (pred_hybrid[draw_mask] == 1).sum() if n_draw > 0 else 0
    
    # Non-draw accuracy
    nd_mask = y_1x2_va != 1
    nd_main = (pred_main[nd_mask] == y_1x2_va[nd_mask]).mean() if nd_mask.sum() > 0 else 0
    nd_hybrid = (pred_hybrid[nd_mask] == y_1x2_va[nd_mask]).mean() if nd_mask.sum() > 0 else 0
    
    all_results.append({
        "period": lb, "n": len(va),
        "acc_main": acc_main, "acc_hybrid": acc_hybrid,
        "draw_n": n_draw, "dh_main": dh_main, "dh_hybrid": dh_hybrid,
        "nd_main": nd_main, "nd_hybrid": nd_hybrid,
    })
    
    print("  %s n=%5d | Ana:%.1f%% Hybrid:%.1f%% | Draw:%d Hybrid_dh:%.1f%% | ND:%.1f%%/%.1f%%" % (
        lb, len(va), acc_main * 100, acc_hybrid * 100,
        n_draw, dh_hybrid / n_draw * 100 if n_draw > 0 else 0,
        nd_main * 100, nd_hybrid * 100))

print("\n" + "=" * 70)
print("  AGGREGATE")
print("=" * 70)
df = pd.DataFrame(all_results)
print("  Ana Model:      %.1f%% accuracy" % (df["acc_main"].mean() * 100))
print("  Hybrid Model:   %.1f%% accuracy" % (df["acc_hybrid"].mean() * 100))
print("  Draw Hit Ana:   %.1f%% (%d/%d)" % (
    df["dh_main"].sum() / df["draw_n"].sum() * 100 if df["draw_n"].sum() > 0 else 0,
    int(df["dh_main"].sum()), int(df["draw_n"].sum())))
print("  Draw Hit Hybrid:%.1f%% (%d/%d)" % (
    df["dh_hybrid"].sum() / df["draw_n"].sum() * 100 if df["draw_n"].sum() > 0 else 0,
    int(df["dh_hybrid"].sum()), int(df["draw_n"].sum())))
print("  ND Acc Ana:     %.1f%%" % (df["nd_main"].mean() * 100))
print("  ND Acc Hybrid:  %.1f%%" % (df["nd_hybrid"].mean() * 100))

# FINAL MODEL
print("\n" + "=" * 70)
print("  FINAL MODEL")
print("=" * 70)
FINAL_CUTOFF = pd.Timestamp("2024-09-01")
final_tr = feat[feat["date"] >= FINAL_CUTOFF].copy().dropna(subset=["result"])
for c in F: final_tr[c] = final_tr[c].fillna(0)
X_f = final_tr[F]
t_f = (final_tr["date"].max() - final_tr["date"]).dt.days.clip(lower=0)
d_f = np.exp(-1.0 * t_f / 365.0).values
sp = int(len(X_f) * 0.85)

draw_final = []
for seed in [42, 123, 456]:
    dm = lgb.LGBMClassifier(objective="binary", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=80,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=seed)
    dm.fit(X_f.iloc[:sp], final_tr["is_draw"].values[:sp], sample_weight=d_f[:sp])
    raw = dm.predict_proba(X_f.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw, final_tr["is_draw"].values[sp:].astype(float))
    draw_final.append((dm, ir))

main_final = []
y_1x2_f = final_tr["result"].map({"H": 0, "D": 1, "A": 2}).values
for seed in [42, 123]:
    mm = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=seed, class_weight={0: 1, 1: 4.0, 2: 1})
    mm.fit(X_f.iloc[:sp], y_1x2_f[:sp], sample_weight=d_f[:sp])
    main_final.append(mm)

model_data = {
    "m_draw": draw_final,
    "m_main": main_final,
    "features": F,
    "version": "v9_draw_aware_v2",
    "train_date": str(pd.Timestamp.now().date()),
    "train_n": len(final_tr),
    "thresholds": {"close_match_gap": 0.15, "draw_t1": 0.30, "draw_t2": 0.40, "draw_t3": 0.35}
}
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(model_data, f)
print("  Kaydedildi: %s" % out_path)
print("  Sure: %ds" % (time.time() - t0))
