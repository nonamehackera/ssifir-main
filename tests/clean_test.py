"""TEMIZ enhanced feature ile walk-forward test.
Gurultu feature'lari silindi, sadece gercekten yeni bilgi getirenler kaldi.
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

OLD=[
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
    "mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob",
]

# SADECE YENi BILGI GETIREN 5 FEATURE (gurultu silindi)
CLEAN_ENHANCED=[
    "mkt_draw_mismatch","mkt_over_misalign",
    "form_pts_diff","btts_signal","lg_confidence_weight",
]
NEW=OLD+CLEAN_ENHANCED

N_FOLDS=3; TEST_SIZE=15000; STEP=15000
total=len(feat); folds=[]; start=0
while start+TEST_SIZE<=total:
    folds.append((start,start+TEST_SIZE)); start+=STEP
folds=folds[-N_FOLDS:]

def calibration_error(probs,targets,n_bins=10):
    bins=np.linspace(0,1,n_bins+1); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(probs>=lo)&(probs<hi); n=m.sum()
        if n==0: continue
        ece+=abs(probs[m].mean()-targets[m].mean())*n/len(probs)
    return ece

def brier_multi(yt,p):
    oh=np.zeros((len(yt),3)); oh[np.arange(len(yt)),yt]=1.0; return ((p-oh)**2).mean()

def ll_multi(yt,p):
    return log_loss(yt,np.clip(p,1e-10,1-1e-10))

print("="*90)
print("  TEMIZ ENHANCED - WALK-FORWARD (5 feature, gurultu silindi)")
print("="*90)
print(f"  Eski: {len(OLD)} feature | Yeni: {len(NEW)} feature (+5 temiz)")

all_results={"OLD":{c:[] for c in ["all","buyuk","orta","az"]},
             "NEW":{c:[] for c in ["all","buyuk","orta","az"]}}

for fold_i,(te_s,te_e) in enumerate(folds):
    test=feat.iloc[te_s:te_e].copy()
    train=feat.iloc[:te_s].copy()
    if len(train)<10000: continue
    print(f"\n  === FOLD {fold_i+1} | test={len(test)} | {test['date'].iloc[0].date()}~{test['date'].iloc[-1].date()} ===")

    y_tr=train["result"].map({"H":0,"D":1,"A":2}).values
    y_te=test["result"].map({"H":0,"D":1,"A":2}).values
    y_bt_te=test["btts"].values; y_o25_te=test["over25"].values
    y_tot_te=(test["home_goals"]+test["away_goals"]).values

    for tag,features in [("OLD",OLD),("NEW",NEW)]:
        af=[c for c in features if c in train.columns]
        Xtr=train[af]; Xte=test[af]
        split=int(len(Xtr)*0.85)

        # 1X2
        m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,
            learning_rate=0.015,n_estimators=800,max_depth=5,min_child_samples=60,
            subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,
            random_seed=42,verbose=-1)
        m.fit(Xtr.iloc[:split],y_tr[:split])
        ir=[IsotonicRegression(out_of_bounds="clip").fit(
            m.predict_proba(Xtr.iloc[split:])[:,c],(y_tr[split:]==c).astype(float)) for c in range(3)]
        P=np.column_stack([ir[c].predict(m.predict_proba(Xte)[:,c]) for c in range(3)])
        P=np.maximum(P,0); s=P.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); P/=s

        # BTTS
        yb=train["btts"].values
        mb=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
            n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,
            colsample_bytree=0.6,random_seed=42,verbose=-1)
        mb.fit(Xtr.iloc[:split],yb[:split])
        ir_b=IsotonicRegression(out_of_bounds="clip").fit(mb.predict_proba(Xtr.iloc[split:])[:,1],yb[split:].astype(float))
        P_bt=np.clip(ir_b.predict(mb.predict_proba(Xte)[:,1]),0.001,0.999)

        # O25
        yo=train["over25"].values
        mo=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
            n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,
            colsample_bytree=0.6,random_seed=42,verbose=-1)
        mo.fit(Xtr.iloc[:split],yo[:split])
        ir_o=IsotonicRegression(out_of_bounds="clip").fit(mo.predict_proba(Xtr.iloc[split:])[:,1],yo[split:].astype(float))
        P_o25=np.clip(ir_o.predict(mo.predict_proba(Xte)[:,1]),0.001,0.999)

        # Goals
        mh=lgb.LGBMRegressor(objective="poisson",num_leaves=36,learning_rate=0.015,
            n_estimators=400,max_depth=5,subsample=0.8,colsample_bytree=0.6,random_seed=42,verbose=-1)
        mh.fit(Xtr.iloc[:split],train["home_goals"].values[:split].astype(float))
        hl=np.maximum(mh.predict(Xte),0.05)
        ma_g=lgb.LGBMRegressor(objective="poisson",num_leaves=36,learning_rate=0.015,
            n_estimators=400,max_depth=5,subsample=0.8,colsample_bytree=0.6,random_seed=42,verbose=-1)
        ma_g.fit(Xtr.iloc[:split],train["away_goals"].values[:split].astype(float))
        al=np.maximum(ma_g.predict(Xte),0.05)
        tot=hl+al

        P_o15=np.clip(1.0-np.exp(-tot)*(1+tot),0.001,0.999)
        P_o35=np.clip(1.0-np.exp(-tot)*(1+tot+tot**2/2+tot**3/6),0.001,0.999)

        for cat,mask_fn in [("all",lambda t: np.ones(len(t),dtype=bool)),
                            ("buyuk",lambda t: t["liga_cat"]=="buyuk"),
                            ("orta",lambda t: t["liga_cat"]=="orta"),
                            ("az",lambda t: t["liga_cat"]=="az")]:
            m=mask_fn(test); n=m.sum()
            if n<20: continue

            y_o15=(y_tot_te>1.5).astype(int); y_o35=(y_tot_te>3.5).astype(int)
            all_results[tag][cat].append({
                "fold":fold_i,"n":n,
                "ll_1x2":ll_multi(y_te[m],P[m]),"brier_1x2":brier_multi(y_te[m],P[m]),
                "ece_1x2":calibration_error(np.max(P[m],axis=1),(np.argmax(P[m],axis=1)==y_te[m]).astype(float)),
                "ll_bt":log_loss(y_bt_te[m],P_bt[m]),"brier_bt":brier_score_loss(y_bt_te[m],P_bt[m]),
                "ece_bt":calibration_error(P_bt[m],y_bt_te[m].astype(float)),
                "auc_bt":roc_auc_score(y_bt_te[m],P_bt[m]) if len(np.unique(y_bt_te[m]))>1 else .5,
                "ll_o25":log_loss(y_o25_te[m],P_o25[m]),"brier_o25":brier_score_loss(y_o25_te[m],P_o25[m]),
                "ece_o25":calibration_error(P_o25[m],y_o25_te[m].astype(float)),
                "auc_o25":roc_auc_score(y_o25_te[m],P_o25[m]) if len(np.unique(y_o25_te[m]))>1 else .5,
                "ll_o15":log_loss(y_o15[m],P_o15[m]),"auc_o15":roc_auc_score(y_o15[m],P_o15[m]) if len(np.unique(y_o15[m]))>1 else .5,
                "ll_o35":log_loss(y_o35[m],P_o35[m]),"auc_o35":roc_auc_score(y_o35[m],P_o35[m]) if len(np.unique(y_o35[m]))>1 else .5,
                "mae_g":mean_absolute_error(y_tot_te[m],tot[m]),"rmse_g":np.sqrt(((y_tot_te[m]-tot[m])**2).mean()),
            })
        print(f"    {tag}: LL={ll_multi(y_te,P):.4f} Brier={brier_multi(y_te,P):.4f} BTTS_AUC={roc_auc_score(y_bt_te,P_bt):.4f} O25_AUC={roc_auc_score(y_o25_te,P_o25):.4f}")

# RAPOR
print("\n"+"="*90)
print("  FINAL RAPOR - TEMIZ ENHANCED (5 feature)")
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
    od=all_results["OLD"][cat]; nd=all_results["NEW"][cat]
    if not od or not nd: continue
    print(f"\n  {'='*80}")
    print(f"  {cat.upper()}")
    print(f"  {'='*80}")
    print(f"  {'PAZAR':<6s} {'METRIK':<10s} {'ESKI':>10s} {'YENi':>10s} {'DELTA':>10s} {'SONUC':>7s}")
    print(f"  {'-'*60}")
    for pazar,key,mname,bl in metrics:
        ov=np.mean([d[key] for d in od]); nv=np.mean([d[key] for d in nd])
        d=nv-ov
        if bl: ok="IYI" if d<0 else "KOTU" if d>0 else "="
        else: ok="IYI" if d>0 else "KOTU" if d<0 else "="
        print(f"  {pazar:<6s} {mname:<10s} {ov:>10.4f} {nv:>10.4f} {d:>+10.4f} {ok:>7s}")

print(f"\n  Sure: {time.time()-t0:.0f}s")
