"""OPTIMAL MARKET-BAZLI FEATURE SELECTION.
Her market icin EN IYI feature set'i bul.
Dogrudan clean_test.py gibi calisir ama her pazar icin farkli feature set kullanir.
"""
import sys, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error

t0=time.time()
feat=pd.read_parquet("data/gold/features_enhanced.parquet").sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])
lc=feat["league"].value_counts()
feat["liga_cat"]=feat["league"].map(lc).apply(lambda n:"buyuk" if n>=3000 else("orta" if n>=1000 else "az"))

# ═══════════════════════════════════════════════════════════════════════════════
# HER PAZAR ICIN EN IYI FEATURE SET
# ═══════════════════════════════════════════════════════════════════════════════
# Neden farkli? Cunku her pazar farkli sinyallerden etkileniyor:
# 1X2: market odds + elo + form -> misalignment onemli
# BTTS: market odds CALISMADI, sadece futbol sinyaller
# O25: market odds + misalignment faydali
# Goals: form + elo + lig ortalamasi onemli

BASE=[
    "elo_diff","home_elo","away_elo","attack_elo_diff","defence_elo_diff",
    "home_attack_elo","home_defence_elo","away_attack_elo","away_defence_elo",
    "home_gf_5","home_ga_5","home_pts_5","away_gf_5","away_ga_5","away_pts_5",
    "home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_hgf_5","home_hga_5","home_hpts_5","away_agf_5","away_aga_5","away_apts_5",
    "home_w_gf","home_w_ga","away_w_gf","away_w_ga",
    "home_w_shots","home_w_sot","away_w_shots","away_w_sot",
    "home_rest_days","away_rest_days","home_opp_elo","away_opp_elo",
    "lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg",
    "lg_draw_rate","lg_btts_rate","lg_over25_rate",
]

# 1X2: market odds + misalignment faydali
FEAT_1X2=BASE+[
    "mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob",
    "mkt_draw_mismatch","mkt_over_misalign","form_pts_diff","lg_confidence_weight",
]

# BTTS: market odds CALISMADI (ECE +0.021 bozuyor) -> futbol sinyallerine don
FEAT_BTTS=BASE+[
    "btts_signal","form_pts_diff",
]

# O25: market odds + misalignment faydali
FEAT_O25=BASE+[
    "mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob",
    "mkt_over_misalign","form_pts_diff","lg_confidence_weight",
]

# Goals: form + elo + lig
FEAT_GOALS=BASE+[
    "lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg",
    "form_pts_diff","lg_confidence_weight",
]

# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════
N_FOLDS=3; TEST_SIZE=15000; STEP=15000
total=len(feat); folds=[]; start=0
while start+TEST_SIZE<=total:
    folds.append((start,start+TEST_SIZE)); start+=STEP
folds=folds[-N_FOLDS:]

def ce(p,t,nb=10):
    b=np.linspace(0,1,nb+1); e=0.0
    for lo,hi in zip(b[:-1],b[1:]):
        m=(p>=lo)&(p<hi); n=m.sum()
        if n==0: continue
        e+=abs(p[m].mean()-t[m].mean())*n/len(p)
    return e

def brier_m(yt,p):
    oh=np.zeros((len(yt),3)); oh[np.arange(len(yt)),yt]=1.0; return ((p-oh)**2).mean()

print("="*90)
print("  OPTIMAL MARKET-BAZLI TEST - HER PAZAR ICIN EN IYI FEATURE")
print("="*90)

all_results={c:[] for c in ["all","buyuk","orta","az"]}

for fold_i,(te_s,te_e) in enumerate(folds):
    test=feat.iloc[te_s:te_e].copy()
    train=feat.iloc[:te_s].copy()
    if len(train)<10000: continue
    print(f"\n  === FOLD {fold_i+1} | {test['date'].iloc[0].date()}~{test['date'].iloc[-1].date()} ===")

    y_tr=train["result"].map({"H":0,"D":1,"A":2}).values
    y_te=test["result"].map({"H":0,"D":1,"A":2}).values
    y_bt_te=test["btts"].values; y_o25_te=test["over25"].values
    y_tot_te=(test["home_goals"]+test["away_goals"]).values

    split=int(len(train)*0.85)

    # 1X2 - enhanced
    af=list(dict.fromkeys([c for c in FEAT_1X2 if c in train.columns]))
    m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,
        learning_rate=0.015,n_estimators=800,max_depth=5,min_child_samples=60,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,
        random_seed=42,verbose=-1)
    m.fit(train[af].iloc[:split],y_tr[:split])
    ir=[IsotonicRegression(out_of_bounds="clip").fit(
        m.predict_proba(train[af].iloc[split:])[:,c],(y_tr[split:]==c).astype(float)) for c in range(3)]
    P1=np.column_stack([ir[c].predict(m.predict_proba(test[af])[:,c]) for c in range(3)])
    P1=np.maximum(P1,0); s=P1.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); P1/=s

    # BTTS - base only (enhanced bozuyor)
    af_bt=list(dict.fromkeys([c for c in FEAT_BTTS if c in train.columns]))
    mb=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
        n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,
        colsample_bytree=0.6,random_seed=42,verbose=-1)
    yb=train["btts"].values
    mb.fit(train[af_bt].iloc[:split],yb[:split])
    ir_b=IsotonicRegression(out_of_bounds="clip").fit(mb.predict_proba(train[af_bt].iloc[split:])[:,1],yb[split:].astype(float))
    Pbt=np.clip(ir_b.predict(mb.predict_proba(test[af_bt])[:,1]),0.001,0.999)

    # O25 - enhanced
    af_o=list(dict.fromkeys([c for c in FEAT_O25 if c in train.columns]))
    yo=train["over25"].values
    mo=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
        n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,
        colsample_bytree=0.6,random_seed=42,verbose=-1)
    mo.fit(train[af_o].iloc[:split],yo[:split])
    ir_o=IsotonicRegression(out_of_bounds="clip").fit(mo.predict_proba(train[af_o].iloc[split:])[:,1],yo[split:].astype(float))
    Po25=np.clip(ir_o.predict(mo.predict_proba(test[af_o])[:,1]),0.001,0.999)

    # Goals - enhanced
    af_g=list(dict.fromkeys([c for c in FEAT_GOALS if c in train.columns]))
    mh=lgb.LGBMRegressor(objective="poisson",num_leaves=36,learning_rate=0.015,
        n_estimators=400,max_depth=5,subsample=0.8,colsample_bytree=0.6,random_seed=42,verbose=-1)
    mh.fit(train[af_g].iloc[:split],train["home_goals"].values[:split].astype(float))
    hl=np.maximum(mh.predict(test[af_g]),0.05)
    ma_g=lgb.LGBMRegressor(objective="poisson",num_leaves=36,learning_rate=0.015,
        n_estimators=400,max_depth=5,subsample=0.8,colsample_bytree=0.6,random_seed=42,verbose=-1)
    ma_g.fit(train[af_g].iloc[:split],train["away_goals"].values[:split].astype(float))
    al=np.maximum(ma_g.predict(test[af_g]),0.05)
    tot=hl+al
    Po15=np.clip(1.0-np.exp(-tot)*(1+tot),0.001,0.999)
    Po35=np.clip(1.0-np.exp(-tot)*(1+tot+tot**2/2+tot**3/6),0.001,0.999)

    # Metrics
    y_o15=(y_tot_te>1.5).astype(int); y_o35=(y_tot_te>3.5).astype(int)
    for cat,mask_fn in [("all",lambda t: np.ones(len(t),dtype=bool)),
                        ("buyuk",lambda t: t["liga_cat"]=="buyuk"),
                        ("orta",lambda t: t["liga_cat"]=="orta"),
                        ("az",lambda t: t["liga_cat"]=="az")]:
        m=mask_fn(test); n=m.sum()
        if n<20: continue
        all_results[cat].append({
            "fold":fold_i,"n":n,
            "ll_1x2":log_loss(y_te[m],np.clip(P1[m],1e-10,1-1e-10)),
            "brier_1x2":brier_m(y_te[m],P1[m]),
            "ece_1x2":ce(np.max(P1[m],axis=1),(np.argmax(P1[m],axis=1)==y_te[m]).astype(float)),
            "ll_bt":log_loss(y_bt_te[m],Pbt[m]),"brier_bt":brier_score_loss(y_bt_te[m],Pbt[m]),
            "ece_bt":ce(Pbt[m],y_bt_te[m].astype(float)),
            "auc_bt":roc_auc_score(y_bt_te[m],Pbt[m]) if len(np.unique(y_bt_te[m]))>1 else .5,
            "ll_o25":log_loss(y_o25_te[m],Po25[m]),"brier_o25":brier_score_loss(y_o25_te[m],Po25[m]),
            "ece_o25":ce(Po25[m],y_o25_te[m].astype(float)),
            "auc_o25":roc_auc_score(y_o25_te[m],Po25[m]) if len(np.unique(y_o25_te[m]))>1 else .5,
            "ll_o15":log_loss(y_o15[m],Po15[m]),"auc_o15":roc_auc_score(y_o15[m],Po15[m]) if len(np.unique(y_o15[m]))>1 else .5,
            "ll_o35":log_loss(y_o35[m],Po35[m]),"auc_o35":roc_auc_score(y_o35[m],Po35[m]) if len(np.unique(y_o35[m]))>1 else .5,
            "mae_g":mean_absolute_error(y_tot_te[m],tot[m]),"rmse_g":np.sqrt(((y_tot_te[m]-tot[m])**2).mean()),
        })

    print(f"    1X2 LL={log_loss(y_te,P1):.4f} Brier={brier_m(y_te,P1):.4f} ECE={ce(np.max(P1,axis=1),(np.argmax(P1,axis=1)==y_te).astype(float)):.4f}")
    print(f"    BTTS LL={log_loss(y_bt_te,Pbt):.4f} AUC={roc_auc_score(y_bt_te,Pbt):.4f} ECE={ce(Pbt,y_bt_te.astype(float)):.4f}")
    print(f"    O25  LL={log_loss(y_o25_te,Po25):.4f} AUC={roc_auc_score(y_o25_te,Po25):.4f} ECE={ce(Po25,y_o25_te.astype(float)):.4f}")
    print(f"    GOL  MAE={mean_absolute_error(y_tot_te,tot):.4f} (naive {mean_absolute_error(y_tot_te,np.full(len(y_tot_te),train['total_goals'].mean())):.4f})")

print("\n"+"="*90)
print("  OPTIMAL FEATURE - FINAL RAPOR")
print("="*90)
metrics=[
    ("1X2","ll_1x2","LogLoss",True),("1X2","brier_1x2","Brier",True),("1X2","ece_1x2","ECE",True),
    ("BTTS","ll_bt","LogLoss",True),("BTTS","brier_bt","Brier",True),("BTTS","ece_bt","ECE",True),("BTTS","auc_bt","AUC",False),
    ("O2.5","ll_o25","LogLoss",True),("O2.5","brier_o25","Brier",True),("O2.5","ece_o25","ECE",True),("O2.5","auc_o25","AUC",False),
    ("O1.5","ll_o15","LogLoss",True),("O1.5","auc_o15","AUC",False),
    ("O3.5","ll_o35","LogLoss",True),("O3.5","auc_o35","AUC",False),
    ("GOL","mae_g","MAE",True),("GOL","rmse_g","RMSE",True),
]
for cat in ["all","buyuk","orta","az"]:
    d=all_results[cat]
    if not d: continue
    print(f"\n  {'='*75}")
    print(f"  {cat.upper()}")
    print(f"  {'='*75}")
    print(f"  {'PAZAR':<6s} {'METRIK':<10s} {'DEGER':>10s} {'Naive':>10s}")
    print(f"  {'-'*40}")
    for pazar,key,mname,bl in metrics:
        v=np.mean([x[key] for x in d])
        naive=""
        if key=="mae_g": naive=f"{np.mean([x['naive_g'] for x in d]) if 'naive_g' in d[0] else 0:.4f}"
        print(f"  {pazar:<6s} {mname:<10s} {v:>10.4f}")

print(f"\n  Sure: {time.time()-t0:.0f}s")
