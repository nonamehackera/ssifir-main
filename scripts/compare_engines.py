"""Multi-engine karsilastirma - optimize."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

t0 = time.time()
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
print(f"  Veri: {len(feat)} mac ({time.time()-t0:.0f}s)")

N = 5000
test = feat.tail(N).copy()
train = feat.iloc[-60000:-N].copy()
print(f"  Train: {len(train)}, Test: {N}")
y = np.array([{"H":0,"D":1,"A":2}[r] for r in test["result"]])

SKIP={"match_id","league","season","date","home_team_id","away_team_id",
    "home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over25","over15","over35","total_goals",
    "home_shots","away_shots","home_sot","away_sot",
    "home_corners","away_corners","home_xg","away_xg","home_yellow","away_yellow",
    "home_red","away_red","referee","data_completeness","home_xg_real","away_xg_real",
    "total_xg_real","xg_diff_real","ht_home_goals","ht_away_goals","ht_total_goals",
    "ht_result_is_draw","ht_home_leading","ht_second_half_goals_expected",
    "second_half_goals","home_score_prob","away_score_prob","btts_xprob","over25_xprob",
    "xg_diff_abs","draw_xprob","avg_home_odds","avg_draw_odds","avg_away_odds",
    "avg_over25_odds","avg_close_home_odds","avg_close_draw_odds",
    "avg_close_away_odds","avg_close_over25_odds","mkt_close_home_prob",
    "mkt_close_draw_prob","mkt_close_away_prob","mkt_close_over25_prob",
    "shots_diff","sot_diff"}
lgb_feats = [c for c in feat.columns if c not in SKIP and feat[c].dtype in ["float64","int64","float32","int32"]]

results = {}

# 1. ELO
print(f"\n  [1/5] EloModel...", end=" ", flush=True)
try:
    from models.elo.model import EloModel
    m = EloModel(); m.fit(train)
    res = m.predict(test)
    p = np.column_stack([res.home_win, res.draw, res.away_win])
    results["EloModel"] = (p, np.max(p,axis=1), np.argmax(p,axis=1), y)
    print(f"OK ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"HATA: {e}")

# 2. CATBOOST (kucuk iterasyon)
print(f"  [2/5] CatBoost(500 iter)...", end=" ", flush=True)
try:
    from models.catboost.model import CatBoostModel
    m = CatBoostModel(with_odds=False, iterations=500, verbose=False)
    m.fit(train)
    res = m.predict(test)
    p = np.column_stack([res.home_win, res.draw, res.away_win])
    results["CatBoost"] = (p, np.max(p,axis=1), np.argmax(p,axis=1), y)
    if res.btts_yes is not None:
        bt=res.btts_yes; results["CB_BTTS"]=(bt,np.maximum(bt,1-bt),(bt>0.5).astype(int),test["btts"].values)
    if res.over25 is not None:
        o25=res.over25; results["CB_O25"]=(o25,np.maximum(o25,1-o25),(o25>0.5).astype(int),test["over25"].values)
    print(f"OK ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"HATA: {e}")

# 3. CATBOOST + ODDS
print(f"  [3/5] CatBoost+Odds(500)...", end=" ", flush=True)
try:
    m = CatBoostModel(with_odds=True, iterations=500, verbose=False)
    m.fit(train)
    res = m.predict(test)
    p = np.column_stack([res.home_win, res.draw, res.away_win])
    results["CB+Odds"] = (p, np.max(p,axis=1), np.argmax(p,axis=1), y)
    print(f"OK ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"HATA: {e}")

# 4. GRADIENT ENSEMBLE
print(f"  [4/5] GradientEnsemble...", end=" ", flush=True)
try:
    from models.improved.gradient_ensemble import GradientEnsembleModel
    m = GradientEnsembleModel(n_models=2); m.fit(train)
    res = m.predict(test)
    p = np.column_stack([res.home_win, res.draw, res.away_win])
    results["GradEns"] = (p, np.max(p,axis=1), np.argmax(p,axis=1), y)
    print(f"OK ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"HATA: {e}")

# 5. LIGHTGBM
print(f"  [5/5] LightGBM...", end=" ", flush=True)
try:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    Xtr = train[lgb_feats].fillna(0); Xte = test[lgb_feats].fillna(0)
    ytr = np.array([{"H":0,"D":1,"A":2}[r] for r in train["result"]])
    sp = int(len(Xtr)*0.85)
    ms = []
    for s in [42,123]:
        m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,
            learning_rate=0.015,n_estimators=500,max_depth=6,min_child_samples=100,
            subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
            random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(Xtr.iloc[:sp], ytr[:sp])
        raw = m.predict_proba(Xtr.iloc[sp:])
        cal = [IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(ytr[sp:]==c).astype(float)) for c in range(3)]
        ms.append((m,cal))
    ps = []
    for m,cal in ms:
        raw = m.predict_proba(Xte)
        cp = np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
        s2=cp.sum(axis=1,keepdims=True); s2=np.where(s2==0,1,s2); cp/=s2; ps.append(cp)
    p = np.mean(ps, axis=0)
    results["LightGBM"] = (p, np.max(p,axis=1), np.argmax(p,axis=1), y)
    print(f"OK ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"HATA: {e}")

# ─── RAPOR ─────────────────────────────────────────────────────────────────
from sklearn.metrics import log_loss as ll_func

print(f"\n{'='*80}")
print(f"  1X2 KARSILASTIRMA ({N} MAC, son {N} mac)")
print("="*80)
print(f"  {'Model':<20} {'LogLoss':>8} | {'T55':>7} {'T60':>7} {'T65':>7} {'T70':>7} {'T75':>7} {'T80':>7}")
print(f"  {'-'*85}")

for name in results:
    prob, conf, pred, yv = results[name]
    if yv.ndim != 1 or len(np.unique(yv)) != 3: continue
    l = ll_func(yv, prob)
    accs = []
    for t in [0.55,0.60,0.65,0.70,0.75,0.80]:
        m = conf>=t; n = m.sum()
        if n > 10: accs.append(f"{(pred[m]==yv[m]).mean()*100:>6.1f}%")
        else: accs.append("    -  ")
    print(f"  {name:<20} {l:>8.4f} | {'  '.join(accs)}")

# BTTS + O25
print(f"\n  BTTS + O25:")
for name in results:
    prob, conf, pred, yv = results[name]
    if yv.ndim == 1 and len(np.unique(yv)) == 2:
        l = ll_func(yv, np.column_stack([1-prob, prob]))
        acc = (pred==yv).mean()
        m55 = conf>=0.55; n55=m55.sum()
        a55 = f"{(pred[m55]==yv[m55]).mean()*100:.1f}%" if n55>10 else "-"
        print(f"  {name:<25} LogLoss={l:.4f} Acc={acc*100:.1f}% Acc55+={a55}")

# SIRALAMA
print(f"\n  SIRALAMA (1X2 LogLoss - dusuk iyi):")
ranked = sorted([(n,v) for n,v in results.items() if v[3].ndim==1 and len(np.unique(v[3]))==3], 
                key=lambda x: ll_func(x[1][3], x[1][0]))
for i,(name,(prob,conf,pred,yv)) in enumerate(ranked,1):
    l = ll_func(yv, prob)
    print(f"    {i}. {name:<20} LogLoss={l:.4f}")

# CALIBRATION (en iyi model)
best = ranked[0][0]
bp, bc, bpred, by = results[best]
print(f"\n  CALIBRATION ({best}):")
print(f"    {'Bucket':>10} {'Pred':>8} {'Actual':>8} {'N':>6} {'Gap':>8}")
for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.0)]:
    m=(bc>=lo)&(bc<hi); n=m.sum()
    if n>10:
        mp=bc[m].mean(); ma=(bpred[m]==by[m]).mean()
        print(f"    {lo*100:.0f}-{hi*100:.0f}% {mp*100:>7.1f}% {ma*100:>7.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")

print(f"\nToplam: {time.time()-t0:.0f}s")
