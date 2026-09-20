"""Production Ensemble - SON KAPSAMLI DOGRULAMA RAPORU.

ProductionEnsemble sinifi + enhance features tam pipeline testi.
Varsayilan (buyuk) ligler, orta ve az verili ligler ayri ayri raporlanir.
"""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, mean_absolute_error

from feature_engine.enhance import enhance_features
from models.production_ensemble import ProductionEnsemble, default_feature_set

t0 = time.time()
print("=" * 95)
print("  SON DOGRULAMA - PRODUCTION ENSEMBLE")
print("=" * 95)

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])
feat=enhance_features(feat)

lc=feat["league"].value_counts()
feat["liga_cat"]=feat["league"].map(lc).apply(lambda n:"buyuk" if n>=3000 else ("orta" if n>=1000 else "az"))
feat["has_mkt"]=feat["mkt_home_prob"].notna()&feat["mkt_draw_prob"].notna()&feat["mkt_away_prob"].notna()

# Son 3 ay blind
TEST_START=pd.Timestamp("2026-06-01")
train=feat[feat["date"]<TEST_START].copy()
test=feat[feat["date"]>=TEST_START].copy()
n_cal=max(1000,int(len(train)*0.15))
cal=train.iloc[-n_cal:].copy()
tr=train.iloc[:-n_cal].copy()

feats_all=default_feature_set(tr); feats_all=[f for f in feats_all if f in tr.columns]
print(f"\nTrain: {len(tr)} | Cal: {len(cal)} | Test: {len(test)} | Feature: {len(feats_all)}")

ens=ProductionEnsemble(base_features=feats_all)
ens.fit(tr)
pred=ens.predict(test)

y1=test["result"].map({"H":0,"D":1,"A":2}).to_numpy()
yb=test["btts"].to_numpy(); yo=test["over25"].to_numpy()
ytot=(test["home_goals"]+test["away_goals"]).to_numpy()
ph=np.asarray(pred.home_win); pd_=np.asarray(pred.draw); pa=np.asarray(pred.away_win)
P=np.column_stack([ph,pd_,pa]); P=P/P.sum(axis=1,keepdims=True)
pr=np.argmax(P,axis=1); cf=np.max(P,axis=1)
ll=log_loss(y1,P); acc=accuracy_score(y1,pr); f1=f1_score(y1,pr,average="macro"); bh=(y1==0).mean()
pb=np.asarray(pred.btts_yes); po=np.asarray(pred.over25)
auc_b=roc_auc_score(yb,pb) if len(np.unique(yb))>1 else 0.5
auc_o=roc_auc_score(yo,po) if len(np.unique(yo))>1 else 0.5
hl=np.asarray(pred.home_lambda); al=np.asarray(pred.away_lambda)
mae=mean_absolute_error(ytot,hl+al); nav=train["total_goals"].mean(); mae_n=mean_absolute_error(ytot,np.full(len(ytot),nav))

print("\n"+"="*95)
print("  BLIND TEST (Haziran-Agustos 2026)")
print("="*95)
print(f"\n  1X2:  LogLoss={ll:.4f} Acc={acc:.3f} F1={f1:.3f} base_H={bh:.3f}")
print(f"  BTTS:  AUC={auc_b:.4f}")
print(f"  O25:   AUC={auc_o:.4f}")
print(f"  GOL:   MAE={mae:.4f} naive={mae_n:.4f}")
print(f"\n  Esik bazli 1X2:")
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    m=cf>=t; n=m.sum()
    if n>10: print(f"    @{t:.0%}: {accuracy_score(y1[m],pr[m]):.3f} ({n} picks)")

print(f"\n  LIG KATEGORISINE GORE:")
print(f"  {'Kat':>6s} {'Mac':>6s} {'Acc':>7s} {'LL':>7s} {'BTTS':>6s} {'O25':>6s} {'GOL':>7s}")
print(f"  {'-'*48}")
for cat in ["buyuk","orta","az"]:
    m=test["liga_cat"]==cat; n=m.sum()
    if n<20: continue
    a=accuracy_score(y1[m],pr[m]); l=log_loss(y1[m],P[m])
    b=roc_auc_score(yb[m],pb[m]) if len(np.unique(yb[m]))>1 else 0.5
    o=roc_auc_score(yo[m],po[m]) if len(np.unique(yo[m]))>1 else 0.5
    g=mean_absolute_error(ytot[m],hl[m]+al[m])
    print(f"  {cat:>6s} {n:6d} {a:6.3f} {l:7.4f} {b:6.3f} {o:6.3f} {g:7.4f}")

print(f"\n  MARKET ORANI VAR/YOK:")
for v,lbl in [(True,"oran var"),(False,"oran yok")]:
    m=test["has_mkt"]==v; n=m.sum()
    if n<10: continue
    print(f"    {lbl:>8s}: Acc={accuracy_score(y1[m],pr[m]):.3f} LL={log_loss(y1[m],P[m]):.4f} (n={n})")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("="*95)
