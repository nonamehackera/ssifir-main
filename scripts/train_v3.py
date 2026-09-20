"""Model v4 - mevcut features.parquet kullanarak hizli."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("="*70)

# 1. Veri
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
feat["total_goals"] = feat["home_goals"].fillna(0).astype(int) + feat["away_goals"].fillna(0).astype(int)
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"].fillna(0).astype(int)>0)&(feat["away_goals"].fillna(0).astype(int)>0)).astype(int)
feat["over15"] = (feat["total_goals"]>1.5).astype(int)
feat["over25"] = (feat["total_goals"]>2.5).astype(int)
feat["over35"] = (feat["total_goals"]>3.5).astype(int)
feat = feat.dropna(subset=["result","elo_diff"])
print(f"  Veri: {len(feat)} mac ({time.time()-t0:.0f}s)")

# 2. Feature selection - en iyi 40 feature
SKIP = {"match_id","league","season","date","home_team_id","away_team_id",
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
FEATS = [c for c in feat.columns if c not in SKIP and feat[c].dtype in ["float64","int64","float32","int32"]]
print(f"  Features: {len(FEATS)}")

# 3. Split
N = 3000
test_m = feat.tail(N).copy()
train_m = feat.iloc[:-N].copy()
print(f"  Train: {len(train_m)}, Test: {len(test_m)}")
print(f"  Test: {test_m['date'].min().date()} -> {test_m['date'].max().date()}")

# 4. Model
sp = int(len(train_m)*0.85)

def train_cal(X, y, task="binary"):
    ms = []
    for s in [42,123,456]:
        if task=="multiclass":
            m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,
                learning_rate=0.015,n_estimators=800,max_depth=6,min_child_samples=100,
                subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        else:
            m=lgb.LGBMClassifier(objective="binary",num_leaves=45,
                learning_rate=0.015,n_estimators=800,max_depth=6,min_child_samples=80,
                subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp], y[:sp])
        raw=m.predict_proba(X.iloc[sp:])
        if task=="multiclass":
            cal=[IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y[sp:]==c).astype(float)) for c in range(3)]
        else:
            cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw,y[sp:].astype(float))
        ms.append((m,cal))
    return ms

def pred_cal(ms, X, task="binary"):
    ps=[]
    for m,cal in ms:
        raw=m.predict_proba(X)
        if task=="multiclass":
            cp=np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
            s=cp.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); cp/=s; ps.append(cp)
        else:
            ps.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
    return np.mean(ps, axis=0)

X_tr, X_te = train_m[FEATS].fillna(0), test_m[FEATS].fillna(0)
tc = train_m.dropna(subset=["result","btts","over25"])
X_tr = tc[FEATS].fillna(0)

y_r = np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
print("  1X2..."); m1x2=train_cal(X_tr,y_r,"multiclass"); p1x2=pred_cal(m1x2,X_te,"multiclass")
pw=np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
    np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
c1=np.max(p1x2,axis=1)

print("  BTTS..."); mbt=train_cal(X_tr,tc["btts"].values)
pbt=pred_cal(mbt,X_te); pbt_v=(pbt>0.5).astype(int); cbt=np.maximum(pbt,1-pbt)

print("  O15..."); mo15=train_cal(X_tr,tc["over15"].values)
po15=pred_cal(mo15,X_te); po15_v=(po15>0.5).astype(int); co15=np.maximum(po15,1-po15)

print("  O25..."); mo25=train_cal(X_tr,tc["over25"].values)
po25=pred_cal(mo25,X_te); po25_v=(po25>0.5).astype(int); co25=np.maximum(po25,1-po25)

print("  O35..."); mo35=train_cal(X_tr,tc["over35"].values)
po35=pred_cal(mo35,X_te); po35_v=(po35>0.5).astype(int); co35=np.maximum(po35,1-po35)

yr=test_m["result"].values; ybt=test_m["btts"].values.astype(int)
yo15=test_m["over15"].values; yo25=test_m["over25"].values; yo35=test_m["over35"].values

print(f"\n{'='*70}\n  3000 MAC SONUC\n{'='*70}")
for name,preds,conf,act in [
    ("1X2",pw,c1,yr),("BTTS",pbt_v,cbt,ybt),
    ("GOL UST 1.5",po15_v,co15,yo15),("GOL UST 2.5",po25_v,co25,yo25),
    ("GOL UST 3.5",po35_v,co35,yo35)]:
    print(f"\n  {name}")
    print(f"  {'Esik':>6} {'Picks':>6} {'Oneri%':>7} {'Dogru':>6} {'Acc':>7}")
    print(f"  {'-'*40}")
    for t in [0.50,0.52,0.55,0.58,0.60,0.62,0.65,0.68,0.70,0.75,0.80]:
        m=conf>=t; n=m.sum()
        if n>0:
            c=(preds[m]==act[m]).sum()
            print(f"  {t*100:>5.0f}% {n:>6} {n/len(preds)*100:>6.1f}% {c:>6} {c/n*100:>6.1f}%")

# Feature importance
print(f"\n  1X2 Top 15 Feature:")
imp=m1x2[0][0].feature_importances_
for fn,fi_v in sorted(zip(FEATS,imp),key=lambda x:-x[1])[:15]:
    print(f"    {fn:<35} {fi_v:>6}")

print(f"\nToplam: {time.time()-t0:.0f}s")
