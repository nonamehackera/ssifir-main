"""DETAYLI TEST RAPORU - model ne kadar dogru/yanlis, hangi markette ne kadar sapiyor.

train_with_sofascore.py ile ayni pipeline (features_fast + 3-seed LightGBM + isotonic)
AMA raporlama cok daha derin:
  - Her pazar icin confusion matrix (dogru/yanlis kırılımı)
  - Precision / Recall / F1
  - Guven esigi bazinda dogru-yanlis sayilari (eşik egrisi)
  - En cok saplanan market (en dusuk acc)
  - 1X2 icin H/D/A ayri dogruluk
  - Korener/GoI marketleri icin yanlis yon analizi (over yerine under demiş mi)

Usage: python scripts/deep_test.py
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

feat = pd.read_parquet("data/gold/features_fast.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)

# HEDEFLER
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["btts"] = ((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over15"] = (feat["total_goals"]>1.5).astype(int)
feat["over25"] = (feat["total_goals"]>2.5).astype(int)
feat["over35"] = (feat["total_goals"]>3.5).astype(int)
feat["under15"] = 1-feat["over15"]; feat["under25"] = 1-feat["over25"]; feat["under35"] = 1-feat["over35"]
feat["double_1X"] = ((feat["result"]=="H")|(feat["result"]=="D")).astype(int)
feat["double_X2"] = ((feat["result"]=="D")|(feat["result"]=="A")).astype(int)
feat["double_12"] = ((feat["result"]=="H")|(feat["result"]=="A")).astype(int)
feat["total_corners"] = pd.to_numeric(feat["home_corners"],errors="coerce").fillna(0) + pd.to_numeric(feat["away_corners"],errors="coerce").fillna(0)
feat["corners_85"] = (feat["total_corners"]>=9).astype(int)
feat["corners_95"] = (feat["total_corners"]>=10).astype(int)
feat["corners_under85"] = 1-feat["corners_85"]
feat["corners_under95"] = 1-feat["corners_95"]

N = 3000
test_m = feat.tail(N).copy()
train_m = feat.iloc[:-N].copy()
P(f"Train: {len(train_m):,} | Test: {len(test_m):,} (son 3000 sofascore maci)")

SKIP = {"match_id","league","season","date","home_team_id","away_team_id",
    "home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over25","over15","over35","total_goals","under15","under25","under35",
    "double_1X","double_X2","double_12",
    "corners_85","corners_95","corners_under85","corners_under95","total_corners",
    "home_shots","away_shots","home_sot","away_sot","home_xg","away_xg",
    "home_yellow","away_yellow","home_red","away_red","referee","data_completeness",
    "home_corners","away_corners","ht_home_goals","ht_away_goals","ht_total_goals",
    "sot_diff","shots_diff"}
FEATS = [c for c in feat.columns if c not in SKIP and pd.api.types.is_numeric_dtype(feat[c])]
P(f"Feature: {len(FEATS)}")
sp = int(len(train_m)*0.85)

def quick_train(X, y, task="binary"):
    ms = []
    for s in [42,123,456]:
        if task=="multiclass":
            m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,
                learning_rate=0.02,n_estimators=400,max_depth=5,min_child_samples=80,
                subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        else:
            m = lgb.LGBMClassifier(objective="binary",num_leaves=40,
                learning_rate=0.02,n_estimators=400,max_depth=5,min_child_samples=60,
                subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp], y[:sp])
        raw = m.predict_proba(X.iloc[sp:])
        if task=="multiclass":
            cal=[IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y[sp:]==c).astype(float)) for c in range(3)]
        else:
            cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw,y[sp:].astype(float))
        ms.append((m,cal))
    return ms

def quick_pred(ms, X, task="binary"):
    ps=[]
    for m,cal in ms:
        raw=m.predict_proba(X)
        if task=="multiclass":
            cp=np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
            ss=cp.sum(axis=1,keepdims=True); ss=np.where(ss==0,1,ss); cp/=ss; ps.append(cp)
        else:
            ps.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
    return np.mean(ps,axis=0)

def train_binary(col, feats=FEATS):
    tc=train_m.dropna(subset=[col])
    X=tc[feats].fillna(0); y=tc[col].values.astype(int)
    ms=quick_train(X,y,"binary")
    p=quick_pred(ms,test_m[feats].fillna(0))
    return (p>0.5).astype(int), np.maximum(p,1-p)

def section(t):
    P(f"\n{'='*78}\n  {t}\n{'='*78}")

def confusion(preds, act, labels, name):
    """Confusion matrix + precision/recall her sinif icin."""
    P(f"\n  {name}")
    P(f"  {'':>10}" + "".join(f"{l:>10}" for l in labels) + f"{'':>12}")
    cm = np.zeros((len(labels),len(labels)),dtype=int)
    for p,a in zip(preds,act):
        try:
            cm[labels.index(p), labels.index(a)] += 1
        except Exception:
            pass
    for i,l in enumerate(labels):
        P(f"  {l:>10}" + "".join(f"{cm[i][j]:>10}" for j,_ in enumerate(labels)) + f"{'':>12}")
    P(f"  (satir=tahmin, sutun=gercek)")
    # precision/recall
    for i,l in enumerate(labels):
        tp=cm[i,i]; fp=cm[i,:].sum()-tp; fn=cm[:,i].sum()-tp
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        P(f"    {l}: precision={prec*100:.1f}%  recall={rec*100:.1f}%  (dogru={tp}, yanlis={fp+fn})")
    return cm

def binary_detail(name, preds, conf, act, positive_label="1"):
    """Binary market icin derin analiz: esik egrisi + yanlis yon."""
    P(f"\n  {name}")
    P(f"  {'Esik':>6} {'Oneri':>7} {'Dogru':>6} {'Yanlis':>6} {'Acc':>7} {'Yanlis%%':>8}")
    P(f"  {'-'*48}")
    total_wrong = 0; total_picks = 0
    for th in [0.50,0.52,0.55,0.58,0.60,0.62,0.65,0.68,0.70,0.72,0.75,0.78,0.80,0.85]:
        m=conf>=th; n=m.sum()
        if n>0:
            c=(preds[m]==act[m]).sum(); w=n-c
            total_wrong += w; total_picks += n
            P(f"  {th*100:>5.0f}% {n:>7} {c:>6} {w:>6} {c/n*100:>6.1f}% {w/n*100:>7.1f}%")
    # yanlis yon: over tahmin edip under cikan vs tam tersi
    wrong_mask = (preds!=act)
    n_wrong = wrong_mask.sum()
    if name.startswith("Gol Ust") or name.startswith("Korner Ust"):
        # over=1 tahmin, under=0 gercek
        over_said_under = ((preds==1)&(act==0)).sum()
        under_said_over = ((preds==0)&(act==1)).sum()
        P(f"  Yanlis yon: over dedik under cikti={over_said_under}, under dedik over cikti={under_said_over}")
    P(f"  TOPLAM: {total_picks} oneri, {total_wrong} yanlis ({total_wrong/total_picks*100:.1f}%)")
    return n_wrong, total_picks

# ============ EGITIM ============
section("EGITIM (tum pazarlar - 3 seed ensemble + isotonic)")
P("  1X2...")
tc=train_m.dropna(subset=["result"])
X_tr=tc[FEATS].fillna(0); y_r=np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
m1x2=quick_train(X_tr,y_r,"multiclass")
p1x2=quick_pred(m1x2,test_m[FEATS].fillna(0),"multiclass")
pw=np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
    np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
c1=np.max(p1x2,axis=1)
P(f"    bitti ({time.time()-t0:.0f}s)")

btts_feats = FEATS + [c for c in ["home_xg","away_xg","home_shots","away_shots","home_sot","away_sot","lg_btts_rate","btts_xprob"] if c in feat.columns and c not in FEATS]
corner_feats=[f for f in FEATS if "corner" not in f.lower()]

markets = []
P("  Cifte Sans...")
for name,col in [("Cifte Sans 1X","double_1X"),("Cifte Sans X2","double_X2"),("Cifte Sans 12","double_12")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values))
P("  BTTS...")
pb,cfb=train_binary("btts",btts_feats); markets.append(("BTTS Var",pb,cfb,test_m["btts"].values.astype(int)))
markets.append(("BTTS Yok",1-pb,cfb,1-test_m["btts"].values.astype(int)))
P("  Gol Alt/Ust...")
for name,col in [("Gol Ust 1.5","over15"),("Gol Alt 1.5","under15"),("Gol Ust 2.5","over25"),
                 ("Gol Alt 2.5","under25"),("Gol Ust 3.5","over35"),("Gol Alt 3.5","under35")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values.astype(int)))
P("  Korner...")
for name,col in [("Korner Ust 8.5","corners_85"),("Korner Alt 8.5","corners_under85"),
                 ("Korner Ust 9.5","corners_95"),("Korner Alt 9.5","corners_under95")]:
    pr,cf=train_binary(col,corner_feats); markets.append((name,pr,cf,test_m[col].values.astype(int)))
P(f"    egitim bitti ({time.time()-t0:.0f}s)")

# ============ DERIN RAPOR ============
section("1X2 - CONFUSION MATRIX (H/D/A)")
confusion(pw, test_m["result"].values, ["H","D","A"], "1X2 Mac Sonucu")
# 1X2 guven detay
binary_detail_acc = []
P(f"\n  1X2 guven esigi detayi:")
for th in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]:
    m=c1>=th; n=m.sum()
    if n>0:
        c=(pw[m]==test_m["result"].values[m]).sum()
        P(f"    {th*100:>5.0f}%: {n:>5} oneri, {c} dogru ({c/n*100:.1f}%)")

section("BINARY PAZARLAR - DERIN ANALIZ (dogru/yanlis kırılımı)")
summary=[]
for name,pr,cf,act in markets:
    w,p = binary_detail(name, pr, cf, act)
    summary.append((name, p, w, (p-w)/p*100 if p>0 else 0))

section("OZET - EN KOTU -> EN IYI (75%+ guven acc)")
summary_sorted = sorted(summary, key=lambda x: x[3])
P(f"  {'PAZAR':<20}{'ONERI':>7}{'YANLIS':>8}{'ACC':>8}")
P(f"  {'-'*45}")
for name,p,w,acc in summary_sorted:
    P(f"  {name:<20}{p:>7}{w:>8}{acc:>7.1f}%")

section("GENEL DEGERLENDIRME")
P(f"  Test seti: {N} mac (en yeni sofascore verisi)")
P(f"  Toplam sure: {time.time()-t0:.0f}s")
P(f"  En zayif pazar: {summary_sorted[0][0]} ({summary_sorted[0][3]:.1f}%)")
P(f"  En guclu pazar: {summary_sorted[-1][0]} ({summary_sorted[-1][3]:.1f}%)")
P(f"\n  NOT: 'Yanlis yon' satirlari over/under marketlerinde modelin hangi")
P(f"  yone bias yaptigini gosterir (over dedik under cikti vs).")
