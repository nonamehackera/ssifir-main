"""LOCKED FINAL TEST - Train->Calibration->Lock->Untouched Test."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

t0 = time.time()
print("="*80)
print("  LOCKED FINAL TEST")
print("  Train: 2020-01~2025-12")
print("  Calibration: 2026-01~2026-03")
print("  TEST (locked): 2026-04~2026-07")
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

LEAK={"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading",
    "ht_home_goals","ht_away_goals","home_xg","away_xg","total_xg_real","xg_diff_real",
    "home_score_prob","away_score_prob","btts_xprob","over25_xprob","draw_xprob","data_completeness",
    "home_shots","away_shots","home_sot","away_sot","home_corners","away_corners",
    "home_yellow","away_yellow","home_red","away_red","avg_home_odds","avg_draw_odds",
    "avg_away_odds","avg_over25_odds","avg_close_home_odds","avg_close_draw_odds",
    "avg_close_away_odds","avg_close_over25_odds","mkt_close_home_prob","mkt_close_draw_prob",
    "mkt_close_away_prob","mkt_close_over25_prob","shots_diff","sot_diff",
    "home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over25","over15","over35","total_goals","match_id","season","date",
    "referee","home_xg_real","away_xg_real"}
CAT_COLS=["league","home_team_id","away_team_id"]
ACO=[c for c in feat.columns if c not in LEAK and feat[c].dtype in ["float64","int64","float32","int32"]]
ALL_COLS=CAT_COLS+[c for c in ACO if c not in CAT_COLS]
cat_idx=[ALL_COLS.index(c) for c in CAT_COLS]

train=feat[(feat["date"]>="2020-01-01")&(feat["date"]<"2026-01-01")].copy()
calib=feat[(feat["date"]>="2026-01-01")&(feat["date"]<"2026-04-01")].copy()
test=feat[(feat["date"]>="2026-04-01")&(feat["date"]<"2026-07-07")].copy()

print(f"\n  Train: {len(train):>7} | {train['date'].min().date()} ~ {train['date'].max().date()}")
print(f"  Calib: {len(calib):>7} | {calib['date'].min().date()} ~ {calib['date'].max().date()}")
print(f"  TEST:  {len(test):>7} | {test['date'].min().date()} ~ {test['date'].max().date()}")

# TRAIN
print(f"\n  [1/3] TRAIN...", end=" ", flush=True)
Xtr=train[ALL_COLS]; ytr=train["result"].map({"H":0,"D":1,"A":2}).values

clf_1x2=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,l2_leaf_reg=10,
    border_count=64,random_strength=2,bagging_temperature=1,cat_features=cat_idx,
    loss_function="MultiClass",class_weights={0:1,1:1.3,2:1},random_seed=42,verbose=False,
    allow_writing_files=False)
clf_1x2.fit(Xtr, ytr)

y_bt=train["btts"].values
clf_bt=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,l2_leaf_reg=10,
    cat_features=cat_idx,loss_function="Logloss",random_seed=42,verbose=False,
    allow_writing_files=False)
clf_bt.fit(Xtr, y_bt)

y_o25=train["over25"].values
clf_o25=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,l2_leaf_reg=10,
    cat_features=cat_idx,loss_function="Logloss",random_seed=42,verbose=False,
    allow_writing_files=False)
clf_o25.fit(Xtr, y_o25)
print(f"OK ({time.time()-t0:.0f}s)")

# CALIBRATION
print(f"  [2/3] CALIBRATION...", end=" ", flush=True)
Xcal=calib[ALL_COLS]; ycal=calib["result"].map({"H":0,"D":1,"A":2}).values
raw_cal=clf_1x2.predict_proba(Xcal)
raw_bt_cal=clf_bt.predict_proba(Xcal)[:,1]
raw_o25_cal=clf_o25.predict_proba(Xcal)[:,1]

cal_ir=[]
for c in range(3):
    ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(raw_cal[:,c],(ycal==c).astype(float)); cal_ir.append(ir)
ir_bt=IsotonicRegression(out_of_bounds="clip"); ir_bt.fit(raw_bt_cal,calib["btts"].values.astype(float))
ir_o25=IsotonicRegression(out_of_bounds="clip"); ir_o25.fit(raw_o25_cal,calib["over25"].values.astype(float))

cal_p=np.column_stack([cal_ir[c].predict(raw_cal[:,c]) for c in range(3)])
cs=cal_p.sum(axis=1,keepdims=True); cs=np.where(cs==0,1,cs); cal_p/=cs
cal_acc=(np.argmax(cal_p,axis=1)==ycal).mean()
print(f"OK cal_acc={cal_acc*100:.1f}% ({time.time()-t0:.0f}s)")

# LOCKED TEST
print(f"  [3/3] LOCKED TEST...", end=" ", flush=True)
Xte=test[ALL_COLS]; yte=test["result"].map({"H":0,"D":1,"A":2}).values
y_bt_te=test["btts"].values; y_o25_te=test["over25"].values

raw_te=clf_1x2.predict_proba(Xte)
te_p=np.column_stack([cal_ir[c].predict(raw_te[:,c]) for c in range(3)])
ts=te_p.sum(axis=1,keepdims=True); ts=np.where(ts==0,1,ts); te_p/=ts
te_conf=np.max(te_p,axis=1); te_pred=np.argmax(te_p,axis=1); N=len(yte)

te_bt=ir_bt.predict(clf_bt.predict_proba(Xte)[:,1])
te_o25=ir_o25.predict(clf_o25.predict_proba(Xte)[:,1])
print(f"OK ({time.time()-t0:.0f}s)")

# 1X2 RESULTS
ll=log_loss(yte,te_p)
print(f"\n{'='*80}")
print(f"  1X2 LogLoss={ll:.4f} ({N} TEST MAC)")
print(f"  {'Esik':>6} {'Picks':>7} {'Oneri%':>7} {'Dogru':>7} {'Acc':>7}")
print(f"  {'-'*40}")
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    m=te_conf>=t; n=m.sum()
    if n>0:
        c=(te_pred[m]==yte[m]).sum()
        print(f"  {t*100:>5.0f}% {n:>7} {n/N*100:>6.1f}% {c:>7} {c/n*100:>6.1f}%")

# BTTS
ll_bt=log_loss(y_bt_te,np.column_stack([1-te_bt,te_bt]))
print(f"\n  BTTS LogLoss={ll_bt:.4f}")
for t in [0.55,0.60,0.65]:
    m=np.maximum(te_bt,1-te_bt)>=t; n=m.sum()
    if n>10: print(f"    {t*100:.0f}%+: n={n} Acc={((te_bt[m]>0.5).astype(int)==y_bt_te[m]).mean()*100:.1f}%")

# O25
ll_o=log_loss(y_o25_te,np.column_stack([1-te_o25,te_o25]))
print(f"\n  O25 LogLoss={ll_o:.4f}")
for t in [0.55,0.60,0.65]:
    m=np.maximum(te_o25,1-te_o25)>=t; n=m.sum()
    if n>10: print(f"    {t*100:.0f}%+: n={n} Acc={((te_o25[m]>0.5).astype(int)==y_o25_te[m]).mean()*100:.1f}%")

# CALIBRATION
print(f"\n  CALIBRATION (locked test):")
print(f"    {'Bucket':>10} {'Pred':>8} {'Actual':>8} {'N':>6} {'Gap':>8}")
for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.0)]:
    m=(te_conf>=lo)&(te_conf<hi); n=m.sum()
    if n>10:
        mp=te_conf[m].mean(); ma=(te_pred[m]==yte[m]).mean()
        print(f"    {lo*100:.0f}-{hi*100:.0f}% {mp*100:>7.1f}% {ma*100:>7.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")

# ROI
oh=test["avg_home_odds"].fillna(0).values
od=test["avg_draw_odds"].fillna(0).values
oa=test["avg_away_odds"].fillna(0).values
has_odds=(oh>0)&(od>0)&(oa>0)
print(f"\n  ROI (oran olan: {has_odds.sum()}/{N}):")

edge_h=te_p[:,0]-np.where(has_odds,1/oh,0)
edge_d=te_p[:,1]-np.where(has_odds,1/od,0)
edge_a=te_p[:,2]-np.where(has_odds,1/oa,0)

def strat(name, fn):
    p=0; nb=0; w=0
    for i in range(N):
        if not has_odds[i]: continue
        if fn(i):
            nb+=1; b=te_pred[i]; odds=[oh[i],od[i],oa[i]][b]
            if b==yte[i]: w+=1; p+=odds-1
            else: p-=1
    if nb>0: print(f"    {name:<38} {nb:>5} bahis | ROI={p/nb*100:>+6.1f}% | Win={w/nb*100:.1f}% | P={p:>+7.1f}")

strat("All (calibrated max)", lambda i: True)
strat("60%+ confidence", lambda i: te_conf[i]>=0.60)
strat("65%+ confidence", lambda i: te_conf[i]>=0.65)
strat("70%+ confidence", lambda i: te_conf[i]>=0.70)
strat("75%+ confidence", lambda i: te_conf[i]>=0.75)
strat("80%+ confidence", lambda i: te_conf[i]>=0.80)

for e in [0.02,0.05,0.08,0.10]:
    def vf(i,ee=e): return max(edge_h[i],edge_d[i],edge_a[i])>ee
    strat(f"Value bet (edge>{int(e*100)}%)", vf)

# MAX DRAWDOWN
print(f"\n  MAX DRAWDOWN (60%+ calibrated):")
cum=0; peak=0; mdd=0; wins=0; losses=0
for i in range(N):
    if not has_odds[i] or te_conf[i]<0.60: continue
    b=te_pred[i]; odds=[oh[i],od[i],oa[i]][b]
    if b==yte[i]: cum+=odds-1; wins+=1
    else: cum-=1; losses+=1
    peak=max(peak,cum); mdd=max(mdd,peak-cum)
print(f"    Final: {cum:>+7.1f} | Peak: {peak:>+7.1f} | MaxDD: {mdd:>7.1f} | W/L: {wins}/{losses}")

# LEAGUE
print(f"\n  LEAGUE (locked test, 1X2 calibrated):")
lg_data={}
for i in range(N):
    lg=test.iloc[i]["league"]
    if lg not in lg_data: lg_data[lg]={"n":0,"w":0}
    lg_data[lg]["n"]+=1
    if te_pred[i]==yte[i]: lg_data[lg]["w"]+=1
for lg,d in sorted(lg_data.items(),key=lambda x:-x[1]["n"])[:10]:
    print(f"    {lg:<15} n={d['n']:>4} Acc={d['w']/d['n']*100:.1f}%")

print(f"\nToplam: {time.time()-t0:.0f}s")
