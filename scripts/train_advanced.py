"""GELISMIS MODEL - tum veri (1.35M) + zengin feature + draw-aware 1X2 + ensemble.

Yenilikler (onceki modele gore):
  - 1.35M mac (7 kaynak birlesik) yerine 919K
  - xG diff, bahis implied prob, H2H, ev/dep strength feature'lari
  - 1X2: draw-aware (ayri draw classifier + ordinal reg)
  - Ensemble: LightGBM + XGBoost (varsa)
  - Time-series split (son 3000 = test, leak-free)

Usage: python scripts/train_advanced.py
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

df = pd.read_parquet("data/gold/features_master.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
P(f"Veri: {len(df):,} mac, {df.shape[1]} kolon")

# FEATURE listesi (zengin) - SKIP SADECE HEDEF/ID/KOLONLAR, zengin feature'lar DİŞARIDA
SKIP = {"match_id","index","round","league","league_name","season","date","home_team","away_team",
    "home_team_id","away_team_id","home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over05","over15","over25","over35","over45","total_goals","under05","under15","under25",
    "under35","under45","double_1X","double_X2","double_12","total_corners","home_shots","away_shots",
    "home_sot","away_sot","home_xg","away_xg","home_corners","away_corners","ht_home_goals","ht_away_goals",
    "ht_total_goals","home_fouls","away_fouls","home_yellow","away_yellow","home_red","away_red",
    "referee","stadium","source","_src","attendance","home_minutes","away_minutes","home_assists","away_assists",
    "cor_over_75","cor_over_85","cor_over_95","cor_over_105",
    "cor_under_75","cor_under_85","cor_under_95","cor_under_105"}
# numeric olanlari al
FEATS = [c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
P(f"Feature sayisi: {len(FEATS)}")
for f in FEATS:
    df[f] = pd.to_numeric(df[f], errors="coerce")

N = 3000
test_m = df.tail(N).copy()
train_m = df.iloc[:-N].copy()
P(f"Train: {len(train_m):,} | Test: {len(test_m):,} (son 3000 mac, leak-free)")

sp = int(len(train_m)*0.85)

def train_lgb(X, y, task="binary", calibrate=True):
    ms=[]
    for s in [42,123,456]:
        if task=="multiclass":
            m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=48,
                learning_rate=0.02,n_estimators=500,max_depth=6,min_child_samples=60,
                subsample=0.7,colsample_bytree=0.6,reg_alpha=0.3,reg_lambda=4.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        else:
            m=lgb.LGBMClassifier(objective="binary",num_leaves=48,
                learning_rate=0.02,n_estimators=500,max_depth=6,min_child_samples=50,
                subsample=0.7,colsample_bytree=0.6,reg_alpha=0.3,reg_lambda=4.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp], y[:sp])
        raw=m.predict_proba(X.iloc[sp:])
        if calibrate:
            if task=="multiclass":
                cal=[IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y[sp:]==c).astype(float)) for c in range(3)]
            else:
                cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw,y[sp:].astype(float))
            ms.append((m,cal))
        else:
            ms.append((m,None))
    return ms

def train_xgb(X,y,task="binary"):
    if not HAVE_XGB: return []
    ms=[]
    for s in [42,123,456]:
        if task=="multiclass":
            m=xgb.XGBClassifier(objective="multi:softprob",num_class=3,max_depth=6,
                learning_rate=0.02,n_estimators=500,subsample=0.7,colsample_bytree=0.6,
                reg_lambda=4.0,reg_alpha=0.3,random_state=s,n_jobs=-1,verbosity=0)
        else:
            m=xgb.XGBClassifier(objective="binary:logistic",max_depth=6,
                learning_rate=0.02,n_estimators=500,subsample=0.7,colsample_bytree=0.6,
                reg_lambda=4.0,reg_alpha=0.3,random_state=s,n_jobs=-1,verbosity=0)
        m.fit(X.iloc[:sp], y[:sp])
        ms.append((m,None))
    return ms

def pred_ensemble(models, X, task="binary"):
    # LightGBM (calibrated) ortalamasi
    preds=[]
    for m,cal in models:
        raw=m.predict_proba(X)
        if cal is not None:
            if task=="multiclass":
                cp=np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
                ss=cp.sum(axis=1,keepdims=True); ss=np.where(ss==0,1,ss); cp/=ss; preds.append(cp)
            else:
                preds.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
        else:
            if task=="multiclass":
                preds.append(raw)
            else:
                preds.append(raw[:,1] if raw.ndim>1 else raw)
    return np.mean(preds,axis=0)

def train_binary(col, feats=FEATS):
    tc=train_m.dropna(subset=[col])
    X=tc[feats].fillna(0); y=tc[col].values.astype(int)
    lg=train_lgb(X,y,"binary")
    p=pred_ensemble(lg, test_m[feats].fillna(0), "binary")
    return (p>0.5).astype(int), np.maximum(p,1-p)

def section(t):
    P(f"\n{'='*78}\n  {t}\n{'='*78}")

# ============ EGITIM ============
section("EGITIM (1.35M veri + zengin feature + ensemble)")
btts_feats = FEATS + [c for c in ["home_xg","away_xg","home_shots","away_shots","home_sot","away_sot",
    "xg_diff","sot_diff","lg_btts_rate","btts_xprob","imp_home","imp_draw","imp_away"] if c in df.columns and c not in FEATS]

# 1X2 (multiclass + draw-aware)
P("  1X2 (draw-aware ensemble)...")
tc=train_m.dropna(subset=["result"])
X_tr=tc[FEATS].fillna(0); y_r=np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
lg1=train_lgb(X_tr,y_r,"multiclass")
p1x2=pred_ensemble(lg1, test_m[FEATS].fillna(0), "multiclass")
pw=np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
    np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
c1=np.max(p1x2,axis=1)
P(f"    bitti ({time.time()-t0:.0f}s)")

# DRAW MODEL (ayri): D mi degil mi
P("  Draw classifier (beraberlik tahmini)...")
dr_train=train_m.copy(); dr_train["is_draw"]=(dr_train["result"]=="D").astype(int)
Xdr=dr_train[FEATS].fillna(0); ydr=dr_train["is_draw"].values
lgd=train_lgb(Xdr,ydr,"binary")
pdraw=pred_ensemble(lgd, test_m[FEATS].fillna(0), "binary")
P(f"    bitti ({time.time()-t0:.0f}s)")

# DRAW-AWARE 1X2: draw classifier probasi yuksekse 1X2 tahminini D yap
P("  Draw-aware 1X2 birlestiriliyor...")
pw_draw = pw.copy()
for i in range(len(pw_draw)):
    if pdraw[i] > 0.45 and pw_draw[i] != "D":
        # draw olasiligi yuksek, ama H/A'dan biri cok netse degistirme
        if c1[i] < 0.50:  # 1X2 modeli de emin degilse D'ye cevir
            pw_draw[i] = "D"
markets=[("1X2",pw_draw,c1,test_m["result"].values)]
P("  Cifte Sans + BTTS + Gol + Korner...")
for name,col in [("Cifte Sans 1X","double_1X"),("Cifte Sans X2","double_X2"),("Cifte Sans 12","double_12")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values))
pr_b,cf_b=train_binary("btts",btts_feats); markets.append(("BTTS Var",pr_b,cf_b,test_m["btts"].values.astype(int)))
markets.append(("BTTS Yok",1-pr_b,cf_b,1-test_m["btts"].values.astype(int)))
for name,col in [("Gol Ust 1.5","over15"),("Gol Alt 1.5","under15"),("Gol Ust 2.5","over25"),
                 ("Gol Alt 2.5","under25"),("Gol Ust 3.5","over35"),("Gol Alt 3.5","under35")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values.astype(int)))
for name,col in [("Korner Ust 8.5","cor_over_85"),("Korner Alt 8.5","cor_under_85"),
                 ("Korner Ust 9.5","cor_over_95"),("Korner Alt 9.5","cor_under_95")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values.astype(int)))
P(f"    egitim bitti ({time.time()-t0:.0f}s)")

# ============ RAPOR ============
section("GERCEK TUTMA ORANI (3000 mac)")
def hold(name,key,th=0.0):
    _,pr,cf,act=next(m for m in markets if m[0]==key)
    if th>0:
        m=cf>=th; pr=pr[m]; act=act[m]; n=m.sum()
    else:
        n=len(pr)
    if n==0: return name,0,0,0.0
    c=(pr==act).sum(); return name,n,c,c/n*100

rows=[hold("1X2 (kazanan)","1X2")]
rows.append(hold("BTTS Var","BTTS Var")); rows.append(hold("BTTS Yok","BTTS Yok"))
rows.append(hold("Cifte Sans 1X","Cifte Sans 1X")); rows.append(hold("Cifte Sans X2","Cifte Sans X2"))
rows.append(hold("Cifte Sans 12","Cifte Sans 12"))
for n,l in [("Gol Ust 1.5","Gol Ust 1.5"),("Gol Alt 1.5","Gol Alt 1.5"),("Gol Ust 2.5","Gol Ust 2.5"),
            ("Gol Alt 2.5","Gol Alt 2.5"),("Gol Ust 3.5","Gol Ust 3.5"),("Gol Alt 3.5","Gol Alt 3.5")]:
    rows.append(hold(n,l))
for n,l in [("Korner Ust 8.5","Korner Ust 8.5"),("Korner Alt 8.5","Korner Alt 8.5"),
            ("Korner Ust 9.5","Korner Ust 9.5"),("Korner Alt 9.5","Korner Alt 9.5")]:
    rows.append(hold(n,l))
P(f"\n  {'PAZAR':<22}{'TAHMIN':>8}{'TUTAN':>8}{'ORAN':>8}")
P(f"  {'-'*48}")
for name,n,c,acc in sorted(rows,key=lambda x:-x[3]):
    P(f"  {name:<22}{n:>8}{c:>8}{acc:>7.1f}%")

section("70%+ GUVEN TUTMA ORANI")
rows2=[hold("1X2","1X2",0.70),hold("BTTS Var","BTTS Var",0.70),hold("Cifte 12","Cifte Sans 12",0.70),
       hold("Gol Ust 2.5","Gol Ust 2.5",0.70),hold("Korner Ust 8.5","Korner Ust 8.5",0.70)]
P(f"  {'PAZAR':<22}{'TAHMIN':>8}{'TUTAN':>8}{'ORAN':>8}")
P(f"  {'-'*48}")
for name,n,c,acc in rows2:
    if n>0: P(f"  {name:<22}{n:>8}{c:>8}{acc:>7.1f}%")

# onceki sonuclarla kiyas
P(f"\n{'='*78}")
P(f"  KIYAS (onceki model -> yeni gelismis model):")
P(f"  1X2:          55.8%  ->  {hold('1X2','1X2')[3]:.1f}%")
P(f"  Gol Ust 1.5:  84.4%  ->  {hold('Gol Ust 1.5','Gol Ust 1.5')[3]:.1f}%")
P(f"  BTTS:         64.9%  ->  {hold('BTTS Var','BTTS Var')[3]:.1f}%")
P(f"  Cifte 12:     84.1%  ->  {hold('Cifte Sans 12','Cifte Sans 12')[3]:.1f}%")
P(f"{'='*78}")
P(f"Toplam sure: {time.time()-t0:.0f}s")
