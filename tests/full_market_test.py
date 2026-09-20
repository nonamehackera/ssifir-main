"""TUM PAZARLAR icin ADAM AKILLI dogruluk testi.
Accuracy degil - Log Loss, Brier Score, Calibration kullaniliyor.

Her market icin:
  - Log Loss (ne kadar iyi olasilik uretiyor)
  - Brier Score (ne kadar iyi kalibre)
  - Calibration (ECE - Expected Calibration Error)
  - Walk-forward (3 fold, gercek zaman simülasyonu)
  - Lig kategorisi bazli (buyuk/orta/az)

ESKI vs YENI karsilastirmasi (ayni split, ayni seed, sadece feature farki).
"""
import sys, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# YUKLE
# ═══════════════════════════════════════════════════════════════════════════════
feat = pd.read_parquet("data/gold/features_enhanced.parquet")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

lc=feat["league"].value_counts()
feat["liga_cat"]=feat["league"].map(lc).apply(lambda n:"buyuk" if n>=3000 else("orta" if n>=1000 else "az"))

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE SETLERI
# ═══════════════════════════════════════════════════════════════════════════════
OLD_1X2=[
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
NEW_1X2=OLD_1X2+[
    "mkt_home_misalign","mkt_draw_mismatch","mkt_over_misalign",
    "home_gf_per_ga","away_gf_per_ga","home_attack_ratio","away_attack_ratio",
    "home_pts_per_game","away_pts_per_game","form_pts_diff",
    "btts_signal","home_overall","away_overall","overall_diff","lg_confidence_weight",
]

# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI METRIKLER
# ═══════════════════════════════════════════════════════════════════════════════
def calibration_error(probs, targets, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        if n == 0: continue
        avg_prob = probs[mask].mean()
        avg_true = targets[mask].mean()
        ece += abs(avg_prob - avg_true) * n / len(probs)
    return ece

def brier_multi(y_true_1x2, probs_1x2):
    """Multi-class Brier score (dusuk = iyi)."""
    n = len(y_true_1x2)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), y_true_1x2] = 1.0
    return ((probs_1x2 - onehot) ** 2).mean()

def ll_multi(y_true_1x2, probs_1x2):
    """Multi-class log loss."""
    return log_loss(y_true_1x2, np.clip(probs_1x2, 1e-10, 1-1e-10))

# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD TRAIN + TEST
# ═══════════════════════════════════════════════════════════════════════════════
N_FOLDS=3
TEST_SIZE=15000
STEP=15000
total=len(feat)
folds=[]
start=0
while start+TEST_SIZE<=total:
    folds.append((start, start+TEST_SIZE))
    start+=STEP
folds=folds[-N_FOLDS:]

print("="*90)
print("  TUM PAZARLAR - WALK-FORWARD DOGRULUK TESTI")
print("  Metrikler: LogLoss, Brier Score, Calibration (ECE), AUC")
print("  Accuracy degil - dogru olasilik uretimi test ediliyor")
print("="*90)
print(f"  {N_FOLDS} fold, test={TEST_SIZE} mac/fold")
print(f"  Eski feature: {len(OLD_1X2)} | Yeni feature: {len(NEW_1X2)} (+{len(NEW_1X2)-len(OLD_1X2)})")

all_results={"OLD":{cat:[] for cat in ["all","buyuk","orta","az"]},
             "NEW":{cat:[] for cat in ["all","buyuk","orta","az"]}}

for fold_i,(te_s,te_e) in enumerate(folds):
    test=feat.iloc[te_s:te_e].copy()
    train=feat.iloc[:te_s].copy()
    if len(train)<10000: continue

    print(f"\n  === FOLD {fold_i+1}/{N_FOLDS} | train={len(train)} test={len(test)} | {test['date'].iloc[0].date()}~{test['date'].iloc[-1].date()} ===")

    y_tr=train["result"].map({"H":0,"D":1,"A":2}).values
    y_te=test["result"].map({"H":0,"D":1,"A":2}).values
    y_bt_te=test["btts"].values
    y_o25_te=test["over25"].values
    y_tot_te=(test["home_goals"]+test["away_goals"]).values

    for tag,features in [("OLD",OLD_1X2),("NEW",NEW_1X2)]:
        avail_f=[c for c in features if c in train.columns]
        Xtr=train[avail_f]; Xte=test[avail_f]

        # === 1X2 ===
        split=int(len(Xtr)*0.85)
        m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,
            learning_rate=0.015,n_estimators=800,max_depth=5,min_child_samples=60,
            subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,
            random_seed=42,verbose=-1)
        m.fit(Xtr.iloc[:split],y_tr[:split])
        ir=[IsotonicRegression(out_of_bounds="clip").fit(
            m.predict_proba(Xtr.iloc[split:])[:,c],(y_tr[split:]==c).astype(float)) for c in range(3)]
        P_1x2=np.column_stack([ir[c].predict(m.predict_proba(Xte)[:,c]) for c in range(3)])
        P_1x2=np.maximum(P_1x2,0); s=P_1x2.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); P_1x2/=s

        # === BTTS ===
        yb=train["btts"].values
        mb=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
            n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,
            colsample_bytree=0.6,random_seed=42,verbose=-1)
        mb.fit(Xtr.iloc[:split],yb[:split])
        ir_b=IsotonicRegression(out_of_bounds="clip").fit(mb.predict_proba(Xtr.iloc[split:])[:,1],yb[split:].astype(float))
        P_bt=np.clip(ir_b.predict(mb.predict_proba(Xte)[:,1]),0.001,0.999)

        # === OVER 2.5 ===
        yo=train["over25"].values
        mo=lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.015,
            n_estimators=500,max_depth=5,min_child_samples=60,subsample=0.7,
            colsample_bytree=0.6,random_seed=42,verbose=-1)
        mo.fit(Xtr.iloc[:split],yo[:split])
        ir_o=IsotonicRegression(out_of_bounds="clip").fit(mo.predict_proba(Xtr.iloc[split:])[:,1],yo[split:].astype(float))
        P_o25=np.clip(ir_o.predict(mo.predict_proba(Xte)[:,1]),0.001,0.999)

        # === GOALS (lambda) ===
        mh=lgb.LGBMRegressor(objective="poisson",num_leaves=36,learning_rate=0.015,
            n_estimators=400,max_depth=5,subsample=0.8,colsample_bytree=0.6,random_seed=42,verbose=-1)
        mh.fit(Xtr.iloc[:split],train["home_goals"].values[:split].astype(float))
        hl=np.maximum(mh.predict(Xte),0.05)
        ma_g=lgb.LGBMRegressor(objective="poisson",num_leaves=36,learning_rate=0.015,
            n_estimators=400,max_depth=5,subsample=0.8,colsample_bytree=0.6,random_seed=42,verbose=-1)
        ma_g.fit(Xtr.iloc[:split],train["away_goals"].values[:split].astype(float))
        al=np.maximum(ma_g.predict(Xte),0.05)
        tot_lam=hl+al

        # === OVER 1.5 & 3.5 (Poisson) ===
        P_o15=np.clip(1.0-np.exp(-tot_lam)*(1+tot_lam),0.001,0.999)
        P_o35=np.clip(1.0-np.exp(-tot_lam)*(1+tot_lam+tot_lam**2/2+tot_lam**3/6),0.001,0.999)

        # === METRIKLERI HESAPLA ===
        for cat,mask_fn in [("all",lambda t: np.ones(len(t),dtype=bool)),
                            ("buyuk",lambda t: t["liga_cat"]=="buyuk"),
                            ("orta",lambda t: t["liga_cat"]=="orta"),
                            ("az",lambda t: t["liga_cat"]=="az")]:
            m=mask_fn(test); n=m.sum()
            if n<20: continue

            # 1X2
            ll_1x2=ll_multi(y_te[m],P_1x2[m])
            brier_1x2=brier_multi(y_te[m],P_1x2[m])
            ece_1x2=calibration_error(np.max(P_1x2[m],axis=1),(np.argmax(P_1x2[m],axis=1)==y_te[m]).astype(float))

            # BTTS
            ll_bt=log_loss(y_bt_te[m],P_bt[m])
            brier_bt=brier_score_loss(y_bt_te[m],P_bt[m])
            ece_bt=calibration_error(P_bt[m],y_bt_te[m].astype(float))
            try: auc_bt=roc_auc_score(y_bt_te[m],P_bt[m])
            except: auc_bt=0.5

            # OVER 2.5
            ll_o25=log_loss(y_o25_te[m],P_o25[m])
            brier_o25=brier_score_loss(y_o25_te[m],P_o25[m])
            ece_o25=calibration_error(P_o25[m],y_o25_te[m].astype(float))
            try: auc_o25=roc_auc_score(y_o25_te[m],P_o25[m])
            except: auc_o25=0.5

            # OVER 1.5
            y_o15=(y_tot_te>1.5).astype(int)
            ll_o15=log_loss(y_o15[m],P_o15[m])
            brier_o15=brier_score_loss(y_o15[m],P_o15[m])
            try: auc_o15=roc_auc_score(y_o15[m],P_o15[m])
            except: auc_o15=0.5

            # OVER 3.5
            y_o35=(y_tot_te>3.5).astype(int)
            ll_o35=log_loss(y_o35[m],P_o35[m])
            brier_o35=brier_score_loss(y_o35[m],P_o35[m])
            try: auc_o35=roc_auc_score(y_o35[m],P_o35[m])
            except: auc_o35=0.5

            # GOALS
            mae_g=mean_absolute_error(y_tot_te[m],tot_lam[m])
            rmse_g=np.sqrt(((y_tot_te[m]-tot_lam[m])**2).mean())
            naive_mae=mean_absolute_error(y_tot_te[m],np.full(n,train["total_goals"].mean()))

            all_results[tag][cat].append({
                "fold":fold_i,"n":n,
                "ll_1x2":ll_1x2,"brier_1x2":brier_1x2,"ece_1x2":ece_1x2,
                "ll_bt":ll_bt,"brier_bt":brier_bt,"ece_bt":ece_bt,"auc_bt":auc_bt,
                "ll_o25":ll_o25,"brier_o25":brier_o25,"ece_o25":ece_o25,"auc_o25":auc_o25,
                "ll_o15":ll_o15,"brier_o15":brier_o15,"auc_o15":auc_o15,
                "ll_o35":ll_o35,"brier_o35":brier_o35,"auc_o35":auc_o35,
                "mae_g":mae_g,"rmse_g":rmse_g,"naive_mae":naive_mae,
            })

        print(f"    {tag}: 1X2 LL={ll_multi(y_te,P_1x2):.4f} Brier={brier_multi(y_te,P_1x2):.4f} | BTTS AUC={roc_auc_score(y_bt_te,P_bt):.4f} | O25 AUC={roc_auc_score(y_o25_te,P_o25):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL RAPOR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print("  FINAL RAPOR - TUM PAZARLAR")
print("="*90)

metrics=[
    ("1X2","ll_1x2","LogLoss",True),("1X2","brier_1x2","Brier",True),("1X2","ece_1x2","ECE",True),
    ("BTTS","ll_bt","LogLoss",True),("BTTS","brier_bt","Brier",True),("BTTS","ece_bt","ECE",True),("BTTS","auc_bt","AUC",False),
    ("O2.5","ll_o25","LogLoss",True),("O2.5","brier_o25","Brier",True),("O2.5","ece_o25","ECE",True),("O2.5","auc_o25","AUC",False),
    ("O1.5","ll_o15","LogLoss",True),("O1.5","brier_o15","Brier",True),("O1.5","auc_o15","AUC",False),
    ("O3.5","ll_o35","LogLoss",True),("O3.5","brier_o35","Brier",True),("O3.5","auc_o35","AUC",False),
    ("GOL","mae_g","MAE",True),("GOL","rmse_g","RMSE",True),
]

for cat in ["all","buyuk","orta","az"]:
    old_d=all_results["OLD"][cat]; new_d=all_results["NEW"][cat]
    if not old_d or not new_d: continue
    print(f"\n  {'='*85}")
    print(f"  {cat.upper()} LIGLER")
    print(f"  {'='*85}")
    print(f"  {'PAZAR':<6s} {'METRIK':<10s} {'ESKI':>10s} {'YENI':>10s} {'DELTA':>10s} {'SONUC':>7s}")
    print(f"  {'-'*65}")
    for pazar,key,mname,better_lower in metrics:
        ov=np.mean([d[key] for d in old_d])
        nv=np.mean([d[key] for d in new_d])
        d=nv-ov
        if better_lower: ok="IYI" if d<0 else "KOTU" if d>0 else "="
        else: ok="IYI" if d>0 else "KOTU" if d<0 else "="
        print(f"  {pazar:<6s} {mname:<10s} {ov:>10.4f} {nv:>10.4f} {d:>+10.4f} {ok:>7s}")

    # Naive MAE
    old_naive=np.mean([d["naive_mae"] for d in old_d])
    new_naive=np.mean([d["naive_mae"] for d in new_d])
    print(f"  {'GOL':<6s} {'NaiveMAE':<10s} {old_naive:>10.4f} {new_naive:>10.4f}")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("="*90)
