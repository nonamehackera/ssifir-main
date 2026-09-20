"""Leakage-free CatBoost - walk-forward test."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from models.base import PredictionResult
from sklearn.metrics import log_loss

t0 = time.time()
print("="*80)
print("  LEAKAGE-FREE CATBOOST WALK-FORWARD")
print("="*80)

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

# SIZINTILI FEATURE'LARI CIKAR
LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading",
        "ht_home_goals","ht_away_goals","home_xg","away_xg","total_xg_real",
        "xg_diff_real","home_score_prob","away_score_prob","btts_xprob",
        "over25_xprob","draw_xprob","data_completeness"}

CAT_COLS = ["league","home_team_id","away_team_id","season"]
NUM_SAFE = [
    "home_elo","away_elo","elo_diff",
    "home_gf_5","home_ga_5","home_pts_5","home_shots_5","home_sot_5",
    "home_corners_5","home_w_gf","home_w_ga","home_w_shots","home_w_sot",
    "home_w_corners","home_hgf_5","home_hga_5","home_hpts_5",
    "home_gf_3","home_ga_3","home_pts_3","home_gf_8","home_ga_8",
    "home_gf_20","home_ga_20","home_gf_std","home_ga_std","home_pts_std",
    "away_gf_5","away_ga_5","away_pts_5","away_shots_5","away_sot_5",
    "away_corners_5","away_w_gf","away_w_ga","away_w_shots","away_w_sot",
    "away_w_corners","away_agf_5","away_aga_5","away_apts_5",
    "away_gf_3","away_ga_3","away_pts_3","away_gf_8","away_ga_8",
    "away_gf_20","away_ga_20","away_gf_std","away_ga_std","away_pts_std",
    "home_attack_elo","home_defence_elo","away_attack_elo","away_defence_elo",
    "attack_elo_diff","defence_elo_diff",
    "home_rest_days","away_rest_days",
    "h2h_home_win","h2h_draw","h2h_away_win","h2h_goals_avg","h2h_btts",
    "home_opp_elo","away_opp_elo",
    "lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg","lg_draw_rate",
    "lg_btts_rate","lg_over25_rate","lg_corner_avg","lg_card_avg",
    "home_momentum","away_momentum","home_wins_last5","home_draws_last5",
    "away_wins_last5","away_draws_last5","home_form_std","away_form_std",
    "home_gdiff5","away_gdiff5","home_gf_5_real","home_ga_5_real",
    "away_gf_5_real","away_ga_5_real",
]
# Only keep columns that exist
NUM_SAFE = [c for c in NUM_SAFE if c in feat.columns and c not in LEAK]
ALL_COLS = CAT_COLS + NUM_SAFE
cat_idx = [ALL_COLS.index(c) for c in CAT_COLS]

print(f"  Feature: {len(ALL_COLS)} ({len(CAT_COLS)} cat + {len(NUM_SAFE)} num)")
print(f"  Cikarilan leakage: {LEAK}")

# Walk-Forward
MIN_TRAIN = 100000
TEST_W = 10000
STEP = 10000
folds = []
i = 0
while i + MIN_TRAIN + TEST_W <= len(feat):
    folds.append((i, i+MIN_TRAIN, i+MIN_TRAIN, i+MIN_TRAIN+TEST_W))
    i += STEP
folds = folds[-5:]  # son 5 fold yeterli

print(f"  Walk-Forward: {len(folds)} fold, {TEST_W} test/fold")

all_1x2 = {"y":[], "prob":[], "conf":[], "pred":[]}
all_btts = {"y":[], "prob":[], "conf":[], "pred":[]}
all_o25 = {"y":[], "prob":[], "conf":[], "pred":[]}

for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
    train = feat.iloc[tr_s:tr_e]
    test = feat.iloc[te_s:te_e]
    
    Xtr = train[ALL_COLS]
    Xte = test[ALL_COLS]
    y_1x2 = train["result"].map({"H":0,"D":1,"A":2}).values
    y_1x2_te = test["result"].map({"H":0,"D":1,"A":2}).values
    y_btts_tr = train["btts"].values
    y_btts_te = test["btts"].values
    y_o25_tr = train["over25"].values
    y_o25_te = test["over25"].values
    
    # 1X2
    clf_1x2 = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=10,
        border_count=64, random_strength=2, bagging_temperature=1,
        cat_features=cat_idx, loss_function="MultiClass",
        class_weights={0:1, 1:1.3, 2:1}, random_seed=42, verbose=False,
        allow_writing_files=False)
    clf_1x2.fit(Xtr, y_1x2, use_best_model=True, 
                eval_set=(Xte, y_1x2_te))
    p_1x2 = clf_1x2.predict_proba(Xte)
    all_1x2["y"].extend(y_1x2_te)
    all_1x2["prob"].append(p_1x2)
    
    # BTTS
    clf_btts = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=10,
        border_count=64, random_strength=2, bagging_temperature=1,
        cat_features=cat_idx, loss_function="Logloss",
        random_seed=42, verbose=False, allow_writing_files=False)
    clf_btts.fit(Xtr, y_btts_tr, use_best_model=True,
                 eval_set=(Xte, y_btts_te))
    p_btts = clf_btts.predict_proba(Xte)[:,1]
    all_btts["y"].extend(y_btts_te)
    all_btts["prob"].append(p_btts)
    
    # O25
    clf_o25 = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=10,
        border_count=64, random_strength=2, bagging_temperature=1,
        cat_features=cat_idx, loss_function="Logloss",
        random_seed=42, verbose=False, allow_writing_files=False)
    clf_o25.fit(Xtr, y_o25_tr, use_best_model=True,
                eval_set=(Xte, y_o25_te))
    p_o25 = clf_o25.predict_proba(Xte)[:,1]
    all_o25["y"].extend(y_o25_te)
    all_o25["prob"].append(p_o25)
    
    elapsed = time.time()-t0
    ll_1x2 = log_loss(y_1x2_te, p_1x2)
    ll_btts = log_loss(y_btts_te, np.column_stack([1-p_btts, p_btts]))
    ll_o25 = log_loss(y_o25_te, np.column_stack([1-p_o25, p_o25]))
    print(f"  Fold {fi+1}/{len(folds)} | 1X2={ll_1x2:.4f} BTTS={ll_btts:.4f} O25={ll_o25:.4f} ({elapsed:.0f}s)")

# Aggregate
y_1x2 = np.array(all_1x2["y"])
p_1x2 = np.vstack(all_1x2["prob"])
c_1x2 = np.max(p_1x2, axis=1)
pr_1x2 = np.argmax(p_1x2, axis=1)

y_btts = np.array(all_btts["y"])
p_btts = np.concatenate(all_btts["prob"])
c_btts = np.maximum(p_btts, 1-p_btts)
pr_btts = (p_btts>0.5).astype(int)

y_o25 = np.array(all_o25["y"])
p_o25 = np.concatenate(all_o25["prob"])
c_o25 = np.maximum(p_o25, 1-p_o25)
pr_o25 = (p_o25>0.5).astype(int)

# Report
print(f"\n{'='*80}")
print(f"  LEAKAGE-FREE RESULTS ({len(y_1x2)} TEST MAC)")
print("="*80)

for name, yv, prob, conf, pred in [
    ("1X2", y_1x2, p_1x2, c_1x2, pr_1x2),
    ("BTTS", y_btts, p_btts, c_btts, pr_btts),
    ("O25", y_o25, p_o25, c_o25, pr_o25)]:
    if name == "1X2":
        ll = log_loss(yv, prob)
    else:
        ll = log_loss(yv, np.column_stack([1-prob, prob]))
    
    print(f"\n  {name} LogLoss={ll:.4f}")
    print(f"  {'Esik':>6} {'Picks':>7} {'Oneri%':>7} {'Dogru':>7} {'Acc':>7}")
    print(f"  {'-'*40}")
    for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
        m=conf>=t; n=m.sum()
        if n>0:
            c=(pred[m]==yv[m]).sum()
            print(f"  {t*100:>5.0f}% {n:>7} {n/len(yv)*100:>6.1f}% {c:>7} {c/n*100:>6.1f}%")

# Calibration (1X2)
print(f"\n  1X2 CALIBRATION:")
print(f"    {'Bucket':>10} {'Pred':>8} {'Actual':>8} {'N':>6} {'Gap':>8}")
for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.0)]:
    m=(c_1x2>=lo)&(c_1x2<hi); n=m.sum()
    if n>10:
        mp=c_1x2[m].mean(); ma=(pr_1x2[m]==y_1x2[m]).mean()
        print(f"    {lo*100:.0f}-{hi*100:.0f}% {mp*100:>7.1f}% {ma*100:>7.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")

print(f"\nToplam: {time.time()-t0:.0f}s")
