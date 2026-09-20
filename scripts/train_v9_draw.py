"""Draw-aware 1X2 model - beraberlik tespiti icin ozel model."""
import sys, os, warnings, json, time, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 70)
print("  DRAW-AWARE 1X2 MODEL")
print("=" * 70)

# 1. DATA
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

print("  Veri: %d mac" % len(feat))

# 2. DRAW-SPECIFIC FEATURES
# Draw'in en guclu sinyalleri: takimlar esitse draw olur
feat["elo_diff_abs"] = feat["elo_diff"].abs()
feat["form_diff"] = feat.get("home_pts_5", 0).fillna(0) - feat.get("away_pts_5", 0).fillna(0)
feat["form_diff_abs"] = feat["form_diff"].abs()
feat["gf_diff"] = feat.get("home_gf_5", 0).fillna(0) - feat.get("away_gf_5", 0).fillna(0)
feat["gf_diff_abs"] = feat["gf_diff"].abs()
feat["ga_diff"] = feat.get("home_ga_5", 0).fillna(0) - feat.get("away_ga_5", 0).fillna(0)
feat["ga_diff_abs"] = feat["ga_diff"].abs()
feat["pts_diff_abs"] = (feat.get("home_pts_std", 1.4).fillna(1.4) - feat.get("away_pts_std", 1.0).fillna(1.0)).abs()

# Ev avantajı zayıf -> draw olasılığı artar
feat["home_advantage_weak"] = (feat["elo_diff"].abs() < 50).astype(int)
feat["teams_even"] = (feat["elo_diff_abs"] < 30).astype(int)
feat["teams_very_even"] = (feat["elo_diff_abs"] < 15).astype(int)

# Form benzerligi
feat["form_similar"] = (feat["form_diff_abs"] < 1.0).astype(int)
feat["form_very_similar"] = (feat["form_diff_abs"] < 0.5).astype(int)

# Gol farki benzer
feat["attack_similar"] = (feat["gf_diff_abs"] < 0.5).astype(int)
feat["defence_similar"] = (feat["ga_diff_abs"] < 0.5).astype(int)

# H2H draw orani
feat["h2h_draw_high"] = (feat.get("h2h_draw", 0).fillna(0) > 0.3).astype(int)

# Lig draw orani yuksek
feat["lg_draw_high"] = (feat.get("lg_draw_rate", 0.25).fillna(0.25) > 0.28).astype(int)

# Tum bunlarin kombinasyonu
feat["draw_signal"] = (
    feat["teams_even"] + 
    feat["form_similar"] + 
    feat["attack_similar"] + 
    feat["h2h_draw_high"] + 
    feat["lg_draw_high"]
)

# 3. FEATURE SET
DRAW_FEATURES = [
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
    "home_rest_days", "away_rest_days",
    "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg",
    "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "home_momentum", "away_momentum",
    "home_gdiff5", "away_gdiff5",
    # DRAW-SPECIFIC
    "form_diff", "form_diff_abs",
    "gf_diff", "gf_diff_abs",
    "ga_diff", "ga_diff_abs",
    "home_advantage_weak", "teams_even", "teams_very_even",
    "form_similar", "form_very_similar",
    "attack_similar", "defence_similar",
    "h2h_draw_high", "lg_draw_high", "draw_signal",
]

F = [c for c in DRAW_FEATURES if c in feat.columns]
print("  Feature: %d" % len(F))

# 4. WALK-FORWARD
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
print("  WALK-FORWARD (Draw Model + Hybrid 1X2)")
print("=" * 70)

results = []
for vs, ve, lb in WINDOWS:
    vs_, ve_ = pd.Timestamp(vs), pd.Timestamp(ve)
    tm = (feat["date"] >= vs_ - pd.DateOffset(years=2)) & (feat["date"] < vs_)
    vm = (feat["date"] >= vs_) & (feat["date"] < ve_)
    tr = feat[tm].copy()
    va = feat[vm].copy()
    if len(tr) < 5000 or len(va) < 100:
        continue

    # TIME DECAY
    t_days = (tr["date"].max() - tr["date"]).dt.days.clip(lower=0)
    decay = np.exp(-1.0 * t_days / 365.0).values

    X_tr = tr[F].fillna(0)
    X_va = va[F].fillna(0)

    # DRAW MODEL (binary)
    y_draw = tr["is_draw"].values
    y_draw_va = va["is_draw"].values

    draw_models = []
    for seed in [42, 123, 456]:
        dm = lgb.LGBMClassifier(
            objective="binary", num_leaves=30, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=80,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed
        )
        sp = int(len(X_tr) * 0.85)
        dm.fit(X_tr.iloc[:sp], y_draw[:sp], sample_weight=decay[:sp])
        
        raw_cal = dm.predict_proba(X_tr.iloc[sp:])[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal, y_draw[sp:].astype(float))
        
        raw_va = dm.predict_proba(X_va)[:, 1]
        cal_va = ir.predict(raw_va)
        draw_models.append((dm, ir))
    
    draw_prob = np.mean([m.predict_proba(X_va)[:, 1] for m, _ in draw_models], axis=0)
    draw_cal = np.mean([ir.predict(m.predict_proba(X_va)[:, 1]) for m, ir in draw_models], axis=0)

    # 1X2 MODEL (multiclass)
    y_1x2 = tr["result"].map({"H": 0, "D": 1, "A": 2}).values
    y_1x2_va = va["result"].map({"H": 0, "D": 1, "A": 2}).values

    main_models = []
    for seed in [42, 123]:
        mm = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, num_leaves=35, learning_rate=0.02,
            n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed, class_weight={0: 1, 1: 4.0, 2: 1}
        )
        sp = int(len(X_tr) * 0.85)
        mm.fit(X_tr.iloc[:sp], y_1x2[:sp], sample_weight=decay[:sp])
        main_models.append(mm)

    main_prob = np.mean([m.predict_proba(X_va) for m in main_models], axis=0)
    # Renormalize
    s = main_prob.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    main_prob /= s

    # HYBRID: draw model + main model
    # Eger draw modeli > esik ise, draw olasiligini artir
    hybrid = main_prob.copy()
    
    # Dynamic threshold: draw sinyali gucluyse threshold dusur
    draw_signal = va["draw_signal"].values if "draw_signal" in va.columns else np.zeros(len(va))
    
    # Threshold 1: sabit esik
    t1 = 0.28  # draw olasiligi bu degerden buyukse
    boost1 = draw_cal > t1
    hybrid[boost1, 1] = np.maximum(hybrid[boost1, 1], draw_cal[boost1])
    
    # Threshold 2: draw sinyali gucluyse daha dusuk esik
    t2 = 0.22
    boost2 = (draw_cal > t2) & (draw_signal >= 3)
    hybrid[boost2, 1] = np.maximum(hybrid[boost2, 1], draw_cal[boost2] * 1.1)
    
    # Threshold 3: cok esit takimlar
    t3 = 0.20
    boost3 = (draw_cal > t3) & (va["teams_very_even"].values == 1)
    hybrid[boost3, 1] = np.maximum(hybrid[boost3, 1], draw_cal[boost3] * 1.2)

    # Renormalize
    s = hybrid.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    hybrid /= s

    # EVALUATE
    pred_main = np.argmax(main_prob, axis=1)
    pred_hybrid = np.argmax(hybrid, axis=1)
    
    acc_main = (pred_main == y_1x2_va).mean()
    acc_hybrid = (pred_hybrid == y_1x2_va).mean()
    
    # Draw accuracy
    draw_mask = y_1x2_va == 1
    n_draw = draw_mask.sum()
    draw_hit_main = (pred_main[draw_mask] == 1).sum() if n_draw > 0 else 0
    draw_hit_hybrid = (pred_hybrid[draw_mask] == 1).sum() if n_draw > 0 else 0
    
    results.append({
        "period": lb, "n": len(va),
        "acc_main": acc_main, "acc_hybrid": acc_hybrid,
        "draw_n": n_draw,
        "draw_hit_main": draw_hit_main,
        "draw_hit_hybrid": draw_hit_hybrid,
        "draw_rate_main": draw_hit_main / n_draw if n_draw > 0 else 0,
        "draw_rate_hybrid": draw_hit_hybrid / n_draw if n_draw > 0 else 0,
    })
    
    print("  %s n=%5d | Ana: %.1f%% | Hybrid: %.1f%% | Draw: %d/mac, Hybrid: %.1f%%" % (
        lb, len(va), acc_main * 100, acc_hybrid * 100,
        n_draw, draw_hit_hybrid / n_draw * 100 if n_draw > 0 else 0))

# AGGREGATE
print("\n" + "=" * 70)
print("  AGGREGATE")
print("=" * 70)
df = pd.DataFrame(results)
print("  Ana Model:     %.1f%% accuracy" % (df["acc_main"].mean() * 100))
print("  Hybrid Model:  %.1f%% accuracy" % (df["acc_hybrid"].mean() * 100))
print("  Draw Hit (Ana):    %.1f%% (%d/%d)" % (
    df["draw_rate_main"].mean() * 100,
    int(df["draw_hit_main"].sum()),
    int(df["draw_n"].sum())))
print("  Draw Hit (Hybrid): %.1f%% (%d/%d)" % (
    df["draw_rate_hybrid"].mean() * 100,
    int(df["draw_hit_hybrid"].sum()),
    int(df["draw_n"].sum())))

# 5. FINAL MODEL
print("\n" + "=" * 70)
print("  FINAL MODEL EGITIMI")
print("=" * 70)

FINAL_CUTOFF = pd.Timestamp("2024-09-01")
final_tr = feat[feat["date"] >= FINAL_CUTOFF].copy()
final_tr = final_tr.dropna(subset=["result"])
for c in F:
    final_tr[c] = final_tr[c].fillna(0)

X_final = final_tr[F]
t_days_f = (final_tr["date"].max() - final_tr["date"]).dt.days.clip(lower=0)
decay_f = np.exp(-1.0 * t_days_f / 365.0).values
sp = int(len(X_final) * 0.85)

# Draw model
y_draw_f = final_tr["is_draw"].values
draw_models_final = []
for seed in [42, 123, 456]:
    dm = lgb.LGBMClassifier(
        objective="binary", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=80,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=seed
    )
    dm.fit(X_final.iloc[:sp], y_draw_f[:sp], sample_weight=decay_f[:sp])
    raw_cal = dm.predict_proba(X_final.iloc[sp:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, y_draw_f[sp:].astype(float))
    draw_models_final.append((dm, ir))

# 1X2 model
y_1x2_f = final_tr["result"].map({"H": 0, "D": 1, "A": 2}).values
main_models_final = []
for seed in [42, 123]:
    mm = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, num_leaves=35, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
        verbose=-1, random_state=seed, class_weight={0: 1, 1: 4.0, 2: 1}
    )
    mm.fit(X_final.iloc[:sp], y_1x2_f[:sp], sample_weight=decay_f[:sp])
    main_models_final.append(mm)

# 6. SAVE
model_data = {
    "m_draw": draw_models_final,
    "m_main": main_models_final,
    "features": F,
    "version": "v9_draw_aware",
    "train_date": str(pd.Timestamp.now().date()),
    "train_n": len(final_tr),
    "thresholds": {
        "t1": 0.28,  # sabit draw esigi
        "t2": 0.22,  # draw_signal >= 3 icin
        "t3": 0.20,  # teams_very_even icin
    }
}

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(model_data, f)
print("  Kaydedildi: %s" % out_path)
print("  Sure: %ds" % (time.time() - t0))
