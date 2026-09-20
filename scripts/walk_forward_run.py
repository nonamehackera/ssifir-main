"""WALK-FORWARD ROLLING BACKTEST - STANDALONE
3 split rolling forward, her split'te model yeniden egitilir.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import json

t0 = time.time()
print("=" * 80)
print("  WALK-FORWARD ROLLING BACKTEST (3 splits)")
print("=" * 80)

df = pd.read_parquet("data/gold/features_enhanced_v5.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    df[c] = df[c].fillna(0).astype(int)
df["result"] = df["result"].fillna("D")
df = df.dropna(subset=["result", "elo_diff"])

df["form_draw_pct"] = (df["home_draws_last5"].fillna(0) + df["away_draws_last5"].fillna(0)) / 10.0
df["elo_close_match"] = (df["elo_diff"].abs() < 100).astype(float)
df["both_weak_attack"] = ((df["home_gf_5"].fillna(0) < 1.0) & (df["away_gf_5"].fillna(0) < 1.0)).astype(float)
df["elo_diff_squared"] = df["elo_diff"] ** 2
df["rest_advantage"] = df["home_rest_days"].fillna(3) - df["away_rest_days"].fillna(3)

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading","ht_home_goals","ht_away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners","home_yellow","away_yellow","home_red","away_red","shots_diff","sot_diff","home_goals","away_goals","result","result_H","result_D","result_A","btts","over25","over15","over35","total_goals","match_id","season","date","referee"}
DEAD = {"home_xgot","away_xgot","home_big_chances","away_big_chances","home_xg_flash","away_xg_flash","home_fouls","away_fouls","home_possession","away_possession","home_xgot_5","away_xgot_5","home_big_chances_5","away_big_chances_5","home_xg_5","away_xg_5","home_fouls_5","away_fouls_5","home_possession_5","away_possession_5","home_score_prob","away_score_prob","btts_xprob","over25_xprob","draw_xprob"}
CLOSING = {"mkt_close_home_prob","mkt_close_draw_prob","mkt_close_away_prob","mkt_close_over25_prob","avg_close_home_odds","avg_close_draw_odds","avg_close_away_odds","avg_close_over25_odds"}
exclude = LEAK | DEAD | CLOSING
CAT = ["league", "home_team_id", "away_team_id"]
cats = [c for c in CAT if c in df.columns]
nums = [c for c in df.columns if c not in exclude and c not in cats and df[c].dtype in ["float64", "int64", "float32", "int32"]]
cols = cats + nums
for c in cols:
    if c not in CAT:
        df[c] = df[c].fillna(0)

df_lgb = df.copy()
for c in CAT:
    if c in df_lgb.columns:
        df_lgb[c] = df_lgb[c].astype(str).astype("category").cat.codes

print(f"  Features: {len(cols)}")

splits = [
    ("Split 1: 3.5y train, 6mo val, 6mo test", "2021-01-01", "2024-06-01", "2024-06-01", "2025-01-01", "2025-01-01", "2025-09-01"),
    ("Split 2: 3.5y train, 6mo val, 6mo test", "2021-06-01", "2024-12-01", "2024-12-01", "2025-03-01", "2025-03-01", "2025-09-01"),
    ("Split 3: 3y train, 3mo val, 3mo test",   "2022-01-01", "2025-03-01", "2025-03-01", "2025-06-01", "2025-06-01", "2025-09-01"),
]

results = []
for si, (name, ts, te, vs, ve, tes, tee) in enumerate(splits):
    tr = df_lgb[(df_lgb["date"] >= ts) & (df_lgb["date"] < te)]
    va = df_lgb[(df_lgb["date"] >= vs) & (df_lgb["date"] < ve)]
    te_df = df_lgb[(df_lgb["date"] >= tes) & (df_lgb["date"] < tee)]

    if len(tr) < 5000 or len(va) < 500 or len(te_df) < 500:
        print(f"  Split {si+1}: skip (tr={len(tr)}, va={len(va)}, te={len(te_df)})")
        continue

    print(f"\n  {name}")
    print(f"  Train: {ts}->{te} ({len(tr)}), Val: {vs}->{ve} ({len(va)}), Test: {tes}->{tee} ({len(te_df)})")

    Xtr = tr[cols].astype(np.float32).values
    ytr = tr["result"].map({"H":0,"D":1,"A":2}).values
    Xva = va[cols].astype(np.float32).values
    yva = va["result"].map({"H":0,"D":1,"A":2}).values
    Xte = te_df[cols].astype(np.float32).values
    yte = te_df["result"].map({"H":0,"D":1,"A":2}).values

    models = []
    for s in [42, 123]:
        t1 = time.time()
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, num_leaves=50,
            learning_rate=0.03, n_estimators=500, max_depth=6,
            min_child_samples=80, subsample=0.75, colsample_bytree=0.65,
            reg_alpha=0.5, reg_lambda=5.0, random_seed=s,
            verbose=-1, n_jobs=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        models.append(m)
        print(f"    LGB s={s}: {time.time()-t1:.0f}s")

    raw_val = np.mean([m.predict_proba(Xva) for m in models], axis=0)
    cal = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_val[:, c], (yva == c).astype(float))
        cal.append(ir)

    raw_te = np.mean([m.predict_proba(Xte) for m in models], axis=0)
    cp = np.column_stack([cal[c].predict(raw_te[:, c]) for c in range(3)])
    s = cp.sum(axis=1, keepdims=True); s = np.where(s > 0, s, 1); cp /= s

    preds = np.argmax(cp, axis=1)
    conf = cp.max(axis=1)
    acc = accuracy_score(yte, preds)
    dr = (preds[yte == 1] == 1).mean() if (yte == 1).sum() else 0

    ho = te_df["avg_home_odds"].values
    do_ = te_df["avg_draw_odds"].values
    ao = te_df["avg_away_odds"].values
    valid = ~(np.isnan(ho) | np.isnan(do_) | np.isnan(ao))
    bets = profit = staked = wins = 0
    for i in np.where(valid)[0]:
        for out, ov, mdl, ri in [("H", ho[i], cp[i, 0], 0), ("D", do_[i], cp[i, 1], 1), ("A", ao[i], cp[i, 2], 2)]:
            mp = 1 / ov; edge = mdl - mp
            if edge > 0.03 and ov > 1.01:
                kelly = edge / (ov - 1) * 0.20; kelly = max(0, min(kelly, 0.05))
                if kelly > 0.005:
                    bets += 1; st = kelly * 100; staked += st
                    if yte[i] == ri: profit += st * (ov - 1); wins += 1
                    else: profit -= st
    roi = profit / staked * 100 if staked > 0 else 0

    hc_results = {}
    for thr in [0.50, 0.60, 0.65, 0.70]:
        mask = conf >= thr
        if mask.sum() > 0:
            hc_results[f"HC{int(thr*100)}"] = accuracy_score(yte[mask], preds[mask])

    r = {"split": si+1, "accuracy": acc, "draw_recall": dr, "roi": roi, "n_bets": bets, **hc_results}
    results.append(r)

    print(f"    Accuracy: {acc*100:.1f}%")
    print(f"    Draw Recall: {dr*100:.1f}%")
    for k, v in hc_results.items():
        print(f"    {k}: {v*100:.1f}%")
    print(f"    ROI: {roi:+.2f}% ({bets} bets)")

if results:
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD ORTALAMA ({len(results)} splits)")
    print(f"{'='*80}")
    avg = {}
    for key in results[0]:
        if key in ("split", "n_bets"):
            continue
        vals = [r[key] for r in results if key in r]
        avg[key] = np.mean(vals)
        if key == "accuracy":
            print(f"  Accuracy: {avg[key]*100:.1f}%")
        elif key == "draw_recall":
            print(f"  Draw Recall: {avg[key]*100:.1f}%")
        elif key == "roi":
            print(f"  ROI: {avg[key]:+.2f}%")
        elif key.startswith("HC"):
            print(f"  {key}: {avg[key]*100:.1f}%")

    avg["n_bets_total"] = sum(r["n_bets"] for r in results)
    print(f"  Total bets: {avg['n_bets_total']}")

    with open("tahminler/walk_forward_results.json", "w") as f:
        json.dump({"results": results, "average": {k: float(v) for k, v in avg.items()}}, f, indent=2)

print(f"\n  Total time: {time.time()-t0:.0f}s")
