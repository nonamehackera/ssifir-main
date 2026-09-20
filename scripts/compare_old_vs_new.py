"""ESKI vs YENI - hizli karsilastirma (kucuk orneklem)."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score
import time

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

OLD=["elo_diff","home_elo","away_elo","attack_elo_diff","defence_elo_diff",
    "home_attack_elo","home_defence_elo","away_attack_elo","away_defence_elo",
    "home_gf_5","home_ga_5","home_pts_5","away_gf_5","away_ga_5","away_pts_5",
    "home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_hgf_5","home_hga_5","home_hpts_5","away_agf_5","away_aga_5","away_apts_5",
    "home_w_gf","home_w_ga","away_w_gf","away_w_ga",
    "home_w_shots","home_w_sot","away_w_shots","away_w_sot",
    "home_rest_days","away_rest_days","home_opp_elo","away_opp_elo",
    "lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg",
    "lg_draw_rate","lg_btts_rate","lg_over25_rate",
    "mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob"]
NEW=OLD+["mkt_home_misalign","mkt_draw_mismatch","mkt_over_misalign",
    "home_gf_per_ga","away_gf_per_ga","home_attack_ratio","away_attack_ratio",
    "home_pts_per_game","away_pts_per_game","form_pts_diff",
    "btts_signal","home_overall","away_overall","overall_diff","lg_confidence_weight"]

# Son 90 gun test
test_start=feat["date"].max()-pd.Timedelta(days=90)
test=feat[feat["date"]>=test_start].copy()
train=feat[feat["date"]<test_start].copy()
f_old=[c for c in OLD if c in train.columns]
f_new=[c for c in NEW if c in train.columns]
print(f"Train: {len(train)} | Test: {len(test)} | Features: old={len(f_old)} new={len(f_new)}")

y_tr=train["result"].map({"H":0,"D":1,"A":2}).values
y_te=test["result"].map({"H":0,"D":1,"A":2}).values
results={}
for tag,feats in [("OLD",f_old),("NEW",f_new)]:
    Xtr=train[feats]; Xte=test[feats]
    # 1X2
    m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,
        learning_rate=0.015,n_estimators=600,max_depth=5,min_child_samples=60,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,
        random_seed=42,verbose=-1)
    split=int(len(Xtr)*0.85)
    m.fit(Xtr.iloc[:split],y_tr[:split])
    ir=[IsotonicRegression(out_of_bounds="clip").fit(
        m.predict_proba(Xtr.iloc[split:])[:,c],(y_tr[split:]==c).astype(float)) for c in range(3)]
    P=np.column_stack([ir[c].predict(m.predict_proba(Xte)[:,c]) for c in range(3)])
    P=np.maximum(P,0); s=P.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); P/=s
    pred=np.argmax(P,axis=1); conf=np.max(P,axis=1)

    yb=train["btts"].values; yo=train["over25"].values
    mb=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
        n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,colsample_bytree=0.6,
        random_seed=42,verbose=-1)
    mb.fit(Xtr.iloc[:split],yb[:split])
    ir_b=IsotonicRegression(out_of_bounds="clip").fit(mb.predict_proba(Xtr.iloc[split:])[:,1],yb[split:].astype(float))
    pb=np.clip(ir_b.predict(mb.predict_proba(Xte)[:,1]),0,1)

    mo=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
        n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,colsample_bytree=0.6,
        random_seed=42,verbose=-1)
    mo.fit(Xtr.iloc[:split],yo[:split])
    ir_o=IsotonicRegression(out_of_bounds="clip").fit(mo.predict_proba(Xtr.iloc[split:])[:,1],yo[split:].astype(float))
    po=np.clip(ir_o.predict(mo.predict_proba(Xte)[:,1]),0,1)

    results[tag]={"P":P,"pred":pred,"conf":conf,
        "ll":log_loss(y_te,P),"acc":accuracy_score(y_te,pred),"f1":f1_score(y_te,pred,average="macro"),
        "bt":roc_auc_score(yb[-len(test):],pb) if len(np.unique(yb[-len(test):]))>1 else .5,
        "o25":roc_auc_score(yo[-len(test):],po) if len(np.unique(yo[-len(test):]))>1 else .5}

E=results["OLD"]; N=results["NEW"]
print(f"\n{'='*70}")
print(f"  ESKI vs YENI - son 90 gun test")
print(f"{'='*70}")
print(f"  {'METRIK':<20} {'ESKI':>9} {'YENI':>9} {'DELTA':>9} {'NETICE':>7}")
print(f"  {'-'*55}")
for l,o,n,b in [("1X2 LogLoss",E["ll"],N["ll"],True),("1X2 Accuracy",E["acc"],N["acc"],False),
    ("1X2 F1",E["f1"],N["f1"],False),("BTTS AUC",E["bt"],N["bt"],False),("O25 AUC",E["o25"],N["o25"],False)]:
    d=n-o; ok="IYI" if (d<0 if b else d>0) else "KOTU"
    print(f"  {l:<20} {o:>9.4f} {n:>9.4f} {d:>+9.4f} {ok:>7}")

print(f"\n  ESIK BAZLI 1X2 ACC:")
for t in [0.55,0.60,0.65,0.70,0.75]:
    om=E["conf"]>=t; nm=N["conf"]>=t
    oa=accuracy_score(y_te[om],E["pred"][om]) if om.sum()>10 else 0
    na=accuracy_score(y_te[nm],N["pred"][nm]) if nm.sum()>10 else 0
    print(f"    @{t:.0%}: OLD={oa:.3f}({om.sum()}) NEW={na:.3f}({nm.sum()}) d={na-oa:+.3f}")

print(f"\n  LIGE GORE 1X2:")
for cat in ["buyuk","orta","az"]:
    m=test["liga_cat"]==cat; n=m.sum()
    if n<10: continue
    oa=accuracy_score(y_te[m],E["pred"][m]); na=accuracy_score(y_te[m],N["pred"][m])
    ol=log_loss(y_te[m],E["P"][m]); nl=log_loss(y_te[m],N["P"][m])
    print(f"  {cat:>5s}: Acc OLD={oa:.3f} NEW={na:.3f} d={na-oa:+.3f} | LL OLD={ol:.4f} NEW={nl:.4f} d={nl-ol:+.4f} (n={n})")

print(f"\n  {time.time()-t0:.0f}s")
