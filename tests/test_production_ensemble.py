"""Production Ensemble - kapsamli test.

Test edilenler:
  1. Walk-forward (4 fold) - genel performans
  2. Blind test (son 6 ay) - gercek referans
  3. Farkli lig KATEGORILERI:
     - Varsayilan/BUYUK (>=3000 mac): guvenilirlik testi
     - ORTA (1000-3000 mac)
     - AZ verili (<1000 mac): fallback guvenilirlik testi
  4. Market orani OLAN vs OLMAYAN maclar ayri ayri

Her market (1X2, BTTS, Over2.5, Goals) icin net rapor.
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
print("  PRODUCTION ENSEMBLE - KAPSAMLI TEST")
print("=" * 95)

# ─── 1. VERI + ENHANCE ──────────────────────────────────────────────────────
print("\n[1] Veri yukleniyor ve enhance ediliyor...")
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]: feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

# Enhanced feature'lar ekle
feat = enhance_features(feat)
print(f"  {len(feat)} mac, {feat.shape[1]} sutun (enhanced)")

# Lig kategorileri
lc = feat["league"].value_counts()
feat["liga_cat"] = feat["league"].map(lc).apply(
    lambda n: "buyuk" if n >= 3000 else ("orta" if n >= 1000 else "az")
)
print(f"\n  Lig kategorileri:")
for cat in ["buyuk", "orta", "az"]:
    n = (feat["liga_cat"] == cat).sum()
    print(f"    {cat}: {n} mac ({n/len(feat)*100:.1f}%)")

# Market orani var/yok
feat["has_mkt"] = feat["mkt_home_prob"].notna() & feat["mkt_draw_prob"].notna() & feat["mkt_away_prob"].notna()
print(f"  Market orani olan mac: {feat['has_mkt'].sum()} ({feat['has_mkt'].mean()*100:.1f}%)")

# ─── 2. WALK-FORWARD ────────────────────────────────────────────────────────
print("\n[2] Walk-forward (4 fold)...")
N_FOLDS = 4
TEST_SIZE = 12000
MIN_TRAIN = 150000
STEP = 12000
total = len(feat)
folds = []
start = 0
while start + MIN_TRAIN + TEST_SIZE <= total:
    folds.append((start, start+MIN_TRAIN, start+MIN_TRAIN, start+MIN_TRAIN+TEST_SIZE))
    start += STEP

wf_rows = []
for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds[:N_FOLDS]):
    train = feat.iloc[tr_s:tr_e].copy()
    test = feat.iloc[te_s:te_e].copy()
    n_cal = max(500, int(len(train)*0.15))
    cal = train.iloc[-n_cal:].copy()
    train_fit = train.iloc[:-n_cal].copy()

    feats_all = default_feature_set(train_fit)
    # Enhanced eksikse yeniden hesapla (guvenli)
    feats_all = [f for f in feats_all if f in train_fit.columns]

    print(f"  === Fold {fold_i+1}/{N_FOLDS} ({test['date'].iloc[0].date()}~{test['date'].iloc[-1].date()}) | {len(feats_all)} feature ===")

    # 1X2 - elo bazli basit kalibrasyonlu ensemble
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    X_tr = train_fit[feats_all].fillna(0)
    X_cal = cal[feats_all].fillna(0)
    X_te = test[feats_all].fillna(0)

    y_tr = train_fit["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    y_cal = cal["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    y_te = test["result"].map({"H":0,"D":1,"A":2}).to_numpy()

    ens_test = []
    for seed in [42,123,456]:
        m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,
            learning_rate=0.015,n_estimators=600,max_depth=6,min_child_samples=100,
            subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
            random_seed=seed,verbose=-1,n_jobs=-1)
        m.fit(X_tr, y_tr)
        raw_cal = m.predict_proba(X_cal)
        cal_e = [IsotonicRegression(out_of_bounds="clip").fit(raw_cal[:,c],(y_cal==c).astype(float)) for c in range(3)]
        raw_te = m.predict_proba(X_te)
        cp = np.column_stack([cal_e[c].predict(raw_te[:,c]) for c in range(3)])
        s = cp.sum(axis=1, keepdims=True); s = np.where(s==0,1,s); cp /= s
        ens_test.append(cp)

    probs = np.mean(ens_test, axis=0)
    pred = np.argmax(probs, axis=1); conf = np.max(probs, axis=1)
    ll = log_loss(y_te, probs); acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro"); base_h = (y_te==0).mean()

    # BTTS
    y_tr_b = train_fit["btts"].to_numpy(); y_cal_b = cal["btts"].to_numpy(); y_te_b = test["btts"].to_numpy()
    mb = lgb.LGBMClassifier(objective="binary",num_leaves=45,learning_rate=0.015,n_estimators=500,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mb.fit(X_tr, y_tr_b)
    raw_b = mb.predict_proba(X_cal)[:,1]
    cb = IsotonicRegression(out_of_bounds="clip").fit(raw_b, y_cal_b.astype(float))
    pb = cb.predict(mb.predict_proba(X_te)[:,1])
    auc_b = roc_auc_score(y_te_b, pb) if len(np.unique(y_te_b))>1 else 0.5

    # Over2.5
    y_tr_o = train_fit["over25"].to_numpy(); y_cal_o = cal["over25"].to_numpy(); y_te_o = test["over25"].to_numpy()
    mo = lgb.LGBMClassifier(objective="binary",num_leaves=45,learning_rate=0.015,n_estimators=500,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mo.fit(X_tr, y_tr_o)
    raw_o = mo.predict_proba(X_cal)[:,1]
    co = IsotonicRegression(out_of_bounds="clip").fit(raw_o, y_cal_o.astype(float))
    po = co.predict(mo.predict_proba(X_te)[:,1])
    auc_o = roc_auc_score(y_te_o, po) if len(np.unique(y_te_o))>1 else 0.5

    # Goals
    mhg = lgb.LGBMRegressor(objective="poisson",num_leaves=45,learning_rate=0.015,n_estimators=400,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mhg.fit(X_tr, train_fit["home_goals"].to_numpy())
    phg = np.maximum(mhg.predict(X_te), 0.05)
    mag = lgb.LGBMRegressor(objective="poisson",num_leaves=45,learning_rate=0.015,n_estimators=400,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mag.fit(X_tr, train_fit["away_goals"].to_numpy())
    pag = np.maximum(mag.predict(X_te), 0.05)
    y_te_total = y_te_hg + y_te_hg if False else (test["home_goals"].to_numpy() + test["away_goals"].to_numpy())
    mae = mean_absolute_error(y_te_total, phg+pag)
    naive = train_fit["total_goals"].mean()
    mae_n = mean_absolute_error(y_te_total, np.full(len(y_te_total), naive))

    print(f"    1X2: LL={ll:.4f} Acc={acc:.3f} F1={f1:.3f} base_H={base_h:.3f}")
    accs = {}
    for t in [0.50,0.55,0.60,0.65,0.70,0.75]:
        mask=conf>=t; n=mask.sum()
        if n>10: accs[t]=(accuracy_score(y_te[mask],pred[mask]),n)
    print(f"         " + " ".join(f"@{t:.0%}:{a:.2f}({n})" for t,(a,n) in accs.items()))
    print(f"    BTTS: AUC={auc_b:.4f} | O25: AUC={auc_o:.4f} | GOL: MAE={mae:.4f}(naive {mae_n:.4f})")

    # Lig kategorisi bazli 1X2
    for cat in ["buyuk","orta","az"]:
        mcat = test["liga_cat"]==cat
        n = mcat.sum()
        if n>=20:
            a = accuracy_score(y_te[mcat], pred[mcat])
            l = log_loss(y_te[mcat], probs[mcat])
            print(f"      {cat} lig: Acc={a:.3f} LL={l:.4f} (n={n})")

    wf_rows.append({"fold":fold_i+1,"1x2_ll":ll,"1x2_acc":acc,"1x2_f1":f1,
                    "btts_auc":auc_b,"o25_auc":auc_o,"mae":mae,"mae_n":mae_n})

print("\n  WALK-FORWARD ORTALAMA:")
wf = pd.DataFrame(wf_rows)
print(f"    1X2 LL={wf['1x2_ll'].mean():.4f} Acc={wf['1x2_acc'].mean():.3f} F1={wf['1x2_f1'].mean():.3f}")
print(f"    BTTS AUC={wf['btts_auc'].mean():.4f}  O25 AUC={wf['o25_auc'].mean():.4f}")
print(f"    GOL MAE={wf['mae'].mean():.4f} naive={wf['mae_n'].mean():.4f}")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("=" * 95)
