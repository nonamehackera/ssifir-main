"""DIAGNOSTIC: Model ALT mi UST mi diyor? Kalibrasyon dogru mu?

Tespit eder:
  - Her pazar icin model kac mac 'ust', kac mac 'alt' dedi (dagilim)
  - Gercek dagilim (test setinde ust/alt orani)
  - Kalibrasyon: model P(ust)=x dediginde gercekten x oraninda ust mi cikti?
  - Confusion: ust dedik alt cikti vs

Usage: python scripts/diag_balance.py
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"
import numpy as np
import pandas as pd

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

df = pd.read_parquet("data/gold/features_master.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
N=3000
test = df.tail(N).copy()
train = df.iloc[:-N].copy()
P(f"Train: {len(train):,} | Test: {len(test):,}")

SKIP = {"match_id","index","round","league","league_name","season","date","home_team","away_team",
    "home_team_id","away_team_id","home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over05","over15","over25","over35","over45","total_goals","under05","under15","under25",
    "under35","under45","double_1X","double_X2","double_12","total_corners","home_shots","away_shots",
    "home_sot","away_sot","home_xg","away_xg","home_corners","away_corners","ht_home_goals","ht_away_goals",
    "ht_total_goals","home_fouls","away_fouls","home_yellow","away_yellow","home_red","away_red",
    "referee","stadium","source","_src","attendance","home_minutes","away_minutes","home_assists","away_assists",
    "sot_diff","shot_diff","corner_diff","xg_diff","xg_home","xg_away","imp_home","imp_draw","imp_away",
    "h2h_home_win_rate","home_form_gf_5","home_form_ga_5","home_form_pts_5","home_form_gf_20","home_form_ga_20",
    "home_form_pts_20","away_form_gf_5","away_form_ga_5","away_form_pts_5","away_form_gf_20","away_form_ga_20",
    "away_form_pts_20","home_home_gf_10","home_home_ga_10","home_home_pts_10","away_away_gf_10","away_away_ga_10",
    "away_away_pts_10","lg_gpg","lg_hwin","lg_draw","cor_over_75","cor_over_85","cor_over_95","cor_over_105",
    "cor_under_75","cor_under_85","cor_under_95","cor_under_105"}
FEATS = [c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
for f in FEATS:
    df[f] = pd.to_numeric(df[f], errors="coerce")
sp = int(len(train)*0.85)

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
def tr(X,y):
    ms=[]
    for s in [42,123,456]:
        m=lgb.LGBMClassifier(objective="binary",num_leaves=48,learning_rate=0.02,n_estimators=500,
            max_depth=6,min_child_samples=50,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.3,
            reg_lambda=4.0,random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp],y[:sp])
        raw=m.predict_proba(X.iloc[sp:])
        cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1],y[sp:].astype(float))
        ms.append((m,cal))
    return ms
def pr(ms,X):
    ps=[]
    for m,cal in ms:
        raw=m.predict_proba(X)
        ps.append(cal.predict(raw[:,1]))
    return np.mean(ps,axis=0)

def analyze(name, target_col, over_label, under_label):
    tc=train.dropna(subset=[target_col])
    X=tc[FEATS].fillna(0); y=tc[target_col].values.astype(int)
    ms=tr(X,y)
    prob=pr(ms, test[FEATS].fillna(0))
    pred=(prob>0.5).astype(int)
    # gercek dagilim
    real_over = test[target_col].mean()
    pred_over = pred.mean()
    P(f"\n  {name}")
    P(f"    Gercek UST orani (test): {real_over*100:.1f}%")
    P(f"    Model UST dedi: {pred_over*100:.1f}%  (ALT dedi: {(1-pred_over)*100:.1f}%)")
    # kalibrasyon bucket
    P(f"    {'Prob Araligi':<14}{'Mac':>6}{'Gercek UST%':>12}")
    bins=[(0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.01)]
    for lo,hi in bins:
        m=(prob>=lo)&(prob<hi)
        n=m.sum()
        if n>0:
            g=test[target_col][m].mean()
            P(f"    [{lo:.1f}-{hi:.1f}){'>':<8}{n:>6}{g*100:>11.1f}%")
    # yanlis yon
    over_said_under=((pred==1)&(test[target_col]==0)).sum()
    under_said_over=((pred==0)&(test[target_col]==1)).sum()
    P(f"    UST dedik ALT cikti: {over_said_under}")
    P(f"    ALT dedik UST cikti: {under_said_over}")

P("\n" + "="*70)
P("  ALT/UST DENGESI VE KALIBRASYON TESHISI")
P("="*70)
analyze("Gol Ust 1.5", "over15", "UST", "ALT")
analyze("Gol Ust 2.5", "over25", "UST", "ALT")
analyze("Gol Ust 3.5", "over35", "UST", "ALT")
analyze("Korner Ust 8.5", "cor_over_85", "UST", "ALT")
analyze("Korner Ust 9.5", "cor_over_95", "UST", "ALT")
P(f"\nToplam sure: {time.time()-t0:.0f}s")
