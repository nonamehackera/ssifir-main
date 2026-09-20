"""League-Grouped 1X2 Model — her lig grubu icin ayri model."""
import sys, os, warnings, json, time, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 70)
print("  LEAGUE-GROUPED 1X2 MODEL")
print("=" * 70)

# 1. LOAD DATA
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

# 2. LEAGUE STATS (rolling 1 yil)
print("\n  League istatistikleri hesaplaniyor...")
feat["league_stats_draw"] = np.nan
feat["league_stats_home"] = np.nan
feat["league_stats_goals"] = np.nan

# Rolling window: her mac icin son 1 yillik lig istatistigi
for league in feat["league"].unique():
    mask = feat["league"] == league
    lg = feat[mask].sort_values("date").copy()
    if len(lg) < 50:
        continue
    
    # Rolling draw rate
    draw_rolling = lg["is_draw"].rolling(window=200, min_periods=50).mean()
    # Rolling home win rate
    home_rolling = (lg["result"] == "H").rolling(window=200, min_periods=50).mean()
    # Rolling avg goals
    goals_rolling = lg["total_goals"].rolling(window=200, min_periods=50).mean()
    
    feat.loc[mask, "league_stats_draw"] = draw_rolling.values
    feat.loc[mask, "league_stats_home"] = home_rolling.values
    feat.loc[mask, "league_stats_goals"] = goals_rolling.values

feat = feat.dropna(subset=["league_stats_draw"])

# 3. LEAGUE GROUPING
# Draw rate'e gore 4 grup
feat["league_group"] = pd.cut(
    feat["league_stats_draw"],
    bins=[0, 0.22, 0.27, 0.32, 1.0],
    labels=["low_draw", "mid_draw", "high_draw", "very_high_draw"]
)

print("\n  LEAGUE GRUPLARI:")
for g in ["low_draw", "mid_draw", "high_draw", "very_high_draw"]:
    m = feat["league_group"] == g
    n = m.sum()
    dr = feat.loc[m, "is_draw"].mean()
    hw = (feat.loc[m, "result"] == "H").mean()
    ag = feat.loc[m, "total_goals"].mean()
    print("    %-15s %6d mac | Draw: %.1f%% | Home: %.1f%% | Gol: %.2f" % (g, n, dr*100, hw*100, ag))

# 4. FEATURES
BASE_FEATURES = [
    "elo_diff",
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
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "home_momentum", "away_momentum",
    "home_wins_last5", "away_wins_last5",
    "home_gdiff5", "away_gdiff5",
    "home_pts_std", "away_pts_std", "data_completeness",
    # LEAGUE-SPECIFIC
    "league_stats_draw", "league_stats_home", "league_stats_goals",
]
F = [c for c in BASE_FEATURES if c in feat.columns]
print("\n  Feature: %d" % len(F))

# 5. WALK-FORWARD: GLOBAL vs LEAGUE-GROUPED
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
print("  WALK-FORWARD: GLOBAL vs LEAGUE-GROUPED")
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

    y = tr["result"].map({"H": 0, "D": 1, "A": 2}).values
    yv = va["result"].map({"H": 0, "D": 1, "A": 2}).values

    # --- GLOBAL MODEL ---
    X_sp = tr[F].fillna(0).iloc[:sp]
    X_va = va[F].fillna(0)
    
    global_models = []
    for seed in [42, 99]:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
            learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=100,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
            verbose=-1, random_state=seed)
        m.fit(X_sp, y[:sp], sample_weight=decay[:sp])
        global_models.append(m)
    p_global = np.mean([m.predict_proba(X_va) for m in global_models], axis=0)
    s = p_global.sum(axis=1, keepdims=True); s = np.where(s==0,1,s); p_global /= s
    pred_global = np.argmax(p_global, axis=1)

    # --- LEAGUE-GROUPED MODELS ---
    pred_grouped = np.full(len(va), -1, dtype=int)
    p_grouped = np.zeros((len(va), 3))
    
    for grp in ["low_draw", "mid_draw", "high_draw", "very_high_draw"]:
        grp_tr_mask = tr["league_group"] == grp
        grp_va_mask = va["league_group"] == grp
        
        n_tr = grp_tr_mask.sum()
        n_va = grp_va_mask.sum()
        
        if n_tr < 2000 or n_va < 50:
            # Yeterli veri yoksa global modeli kullan
            if n_va > 0:
                pred_grouped[grp_va_mask] = pred_global[grp_va_mask]
                p_grouped[grp_va_mask] = p_global[grp_va_mask]
            continue
        
        grp_tr = tr[grp_tr_mask]
        grp_y = y[grp_tr_mask.values]
        grp_decay = decay[grp_tr_mask.values]
        grp_sp = int(len(grp_tr) * 0.85)
        
        grp_X_sp = grp_tr[F].fillna(0).iloc[:grp_sp]
        grp_X_va = va.loc[grp_va_mask, F].fillna(0)
        
        grp_models = []
        for seed in [42, 99]:
            m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=35,
                learning_rate=0.02, n_estimators=400, max_depth=5, min_child_samples=80,
                subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
                verbose=-1, random_state=seed)
            m.fit(grp_X_sp, grp_y[:grp_sp], sample_weight=grp_decay[:grp_sp])
            grp_models.append(m)
        
        p_grp = np.mean([m.predict_proba(grp_X_va) for m in grp_models], axis=0)
        s = p_grp.sum(axis=1, keepdims=True); s = np.where(s==0,1,s); p_grp /= s
        
        pred_grouped[grp_va_mask] = np.argmax(p_grp, axis=1)
        p_grouped[grp_va_mask] = p_grp

    # --- HYBRID: GROUPED + GLOBAL CONFIDENCE ---
    # Eger grouped model emin degilse (< %50 max prob) global modeli kullan
    grp_conf = np.max(p_grouped, axis=1)
    glo_conf = np.max(p_global, axis=1)
    
    # Grouped modelin kendine guveni dusukse global'i kullan
    low_conf = grp_conf < 0.45
    pred_hybrid = pred_grouped.copy()
    pred_hybrid[low_conf] = pred_global[low_conf]
    
    # Veya: her mac icin hangisi daha eminse onu kullan
    pred_best = np.where(grp_conf > glo_conf, pred_grouped, pred_global)

    # EVALUATE
    acc_global = (pred_global == yv).mean()
    acc_grouped = (pred_grouped == yv).mean()
    acc_hybrid = (pred_hybrid == yv).mean()
    acc_best = (pred_best == yv).mean()
    
    dm = yv == 1
    n_draw = dm.sum()
    
    # Draw hit
    dh_global = (pred_global[dm] == 1).sum() if n_draw > 0 else 0
    dh_grouped = (pred_grouped[dm] == 1).sum() if n_draw > 0 else 0
    dh_hybrid = (pred_hybrid[dm] == 1).sum() if n_draw > 0 else 0
    dh_best = (pred_best[dm] == 1).sum() if n_draw > 0 else 0
    
    # Non-draw acc
    nd = yv != 1
    nda_global = (pred_global[nd] == yv[nd]).mean() if nd.sum() > 0 else 0
    nda_grouped = (pred_grouped[nd] == yv[nd]).mean() if nd.sum() > 0 else 0
    nda_hybrid = (pred_hybrid[nd] == yv[nd]).mean() if nd.sum() > 0 else 0
    
    # GRUPLARA GORE AYRINTI
    print("\n  %s n=%d" % (lb, len(va)))
    print("    Global:   %.1f%% | DH: %.1f%% | ND: %.1f%%" % (acc_global*100, dh_global/n_draw*100 if n_draw else 0, nda_global*100))
    print("    Grouped:  %.1f%% | DH: %.1f%% | ND: %.1f%%" % (acc_grouped*100, dh_grouped/n_draw*100 if n_draw else 0, nda_grouped*100))
    print("    Hybrid:   %.1f%% | DH: %.1f%% | ND: %.1f%%" % (acc_hybrid*100, dh_hybrid/n_draw*100 if n_draw else 0, nda_hybrid*100))
    print("    Best:     %.1f%% | DH: %.1f%%" % (acc_best*100, dh_best/n_draw*100 if n_draw else 0))
    
    # Grup bazli
    for grp in ["low_draw", "mid_draw", "high_draw", "very_high_draw"]:
        gm = va["league_group"] == grp
        gn = gm.sum()
        if gn < 10: continue
        ga = (pred_global[gm] == yv[gm]).mean()
        gg = (pred_grouped[gm] == yv[gm]).mean()
        print("      %-15s n=%4d | Global:%.1f%% Grouped:%.1f%%" % (grp, gn, ga*100, gg*100))

    all_results.append({
        "period": lb, "n": len(va), "n_draw": n_draw,
        "acc_global": acc_global, "acc_grouped": acc_grouped, "acc_hybrid": acc_hybrid, "acc_best": acc_best,
        "dh_global": dh_global, "dh_grouped": dh_grouped, "dh_hybrid": dh_hybrid, "dh_best": dh_best,
        "nda_global": nda_global, "nda_grouped": nda_grouped,
    })

print("\n" + "=" * 70)
print("  AGGREGATE")
print("=" * 70)
df = pd.DataFrame(all_results)
td = df["n_draw"].sum()
print("  Global:    Acc: %.1f%% | DH: %.1f%% (%d/%d) | ND: %.1f%%" % (
    df["acc_global"].mean()*100, df["dh_global"].sum()/td*100, int(df["dh_global"].sum()), td, df["nda_global"].mean()*100))
print("  Grouped:   Acc: %.1f%% | DH: %.1f%% (%d/%d) | ND: %.1f%%" % (
    df["acc_grouped"].mean()*100, df["dh_grouped"].sum()/td*100, int(df["dh_grouped"].sum()), td, df["nda_grouped"].mean()*100))
print("  Hybrid:    Acc: %.1f%% | DH: %.1f%% (%d/%d)" % (
    df["acc_hybrid"].mean()*100, df["dh_hybrid"].sum()/td*100, int(df["dh_hybrid"].sum()), td))
print("  Best:      Acc: %.1f%% | DH: %.1f%% (%d/%d)" % (
    df["acc_best"].mean()*100, df["dh_best"].sum()/td*100, int(df["dh_best"].sum()), td))

print("\n  Sure: %ds" % (time.time() - t0))
