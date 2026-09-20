"""GERCEK MAC TAHMİN RAPORU - kullanici istedigini gorsun:

Her test maci icin TUM pazarlar:
  - Kazanan (1X2)
  - BTTS Var/Yok
  - Cifte Sans (1X/X2/12)
  - Toplam Gol Alt/Ust: 0.5 / 1.5 / 2.5 / 3.5 / 4.5
  - Toplam Korner Alt/Ust: 7.5 / 8.5 / 9.5 / 10.5
Yanina GERCEK sonucu yazilir -> tutup tutmadigi gorulur.
Sonunda her pazar icin TUTMA ORANI.

Usage: python scripts/predict_real.py [--show N]
"""
import sys, os, time, warnings, argparse
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--show", type=int, default=40, help="kac mac detayli gosterilsin")
args = ap.parse_args()

feat = pd.read_parquet("data/gold/features_fast.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)

# HEDEFLER
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["btts"] = ((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over05"] = (feat["total_goals"]>0.5).astype(int)
feat["over15"] = (feat["total_goals"]>1.5).astype(int)
feat["over25"] = (feat["total_goals"]>2.5).astype(int)
feat["over35"] = (feat["total_goals"]>3.5).astype(int)
feat["over45"] = (feat["total_goals"]>4.5).astype(int)
for lo in ["05","15","25","35","45"]:
    feat[f"under{lo}"] = 1-feat[f"over{lo}"]
feat["double_1X"] = ((feat["result"]=="H")|(feat["result"]=="D")).astype(int)
feat["double_X2"] = ((feat["result"]=="D")|(feat["result"]=="A")).astype(int)
feat["double_12"] = ((feat["result"]=="H")|(feat["result"]=="A")).astype(int)
feat["total_corners"] = pd.to_numeric(feat["home_corners"],errors="coerce").fillna(0) + pd.to_numeric(feat["away_corners"],errors="coerce").fillna(0)
for cval,lab in [(7.5,"75"),(8.5,"85"),(9.5,"95"),(10.5,"105")]:
    feat[f"cor_over_{lab}"] = (feat["total_corners"]>= (cval+0.5)).astype(int)
    feat[f"cor_under_{lab}"] = 1-feat[f"cor_over_{lab}"]

N = 3000
test_m = feat.tail(N).copy()
train_m = feat.iloc[:-N].copy()
P(f"Train: {len(train_m):,} | Test (gercek mac): {len(test_m):,}")

SKIP = {"match_id","league","season","date","home_team_id","away_team_id",
    "home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over25","over15","over35","total_goals","under15","under25","under35",
    "double_1X","double_X2","double_12",
    "total_corners","home_shots","away_shots","home_sot","away_sot","home_xg","away_xg",
    "home_yellow","away_yellow","home_red","away_red","referee","data_completeness",
    "home_corners","away_corners","ht_home_goals","ht_away_goals","ht_total_goals",
    "sot_diff","shots_diff"}
FEATS = [c for c in feat.columns if c not in SKIP and pd.api.types.is_numeric_dtype(feat[c])]
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
    P(f"\n{'='*90}\n  {t}\n{'='*90}")

# ============ EGITIM ============
section("EGITIM (tum pazarlar)")
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

def train_ret(col, feats=FEATS):
    pr,cf=train_binary(col,feats); return pr,cf

# tum pazarlar
P("  Tum pazarlar egitiliyor...")
mkt = {}
mkt["1X2"] = (pw, c1, test_m["result"].values)
pr_btts, cf_b = train_ret("btts", btts_feats)
mkt["BTTS_Var"] = (pr_btts, cf_b, test_m["btts"].values.astype(int))
mkt["BTTS_Yok"] = (1-pr_btts, cf_b, 1-test_m["btts"].values.astype(int))
for n,c in [("Cifte_1X","double_1X"),("Cifte_X2","double_X2"),("Cifte_12","double_12")]:
    pr,cf=train_ret(c); mkt[n]=(pr,cf,test_m[c].values)
for n,c in [("Gol_U05","over05"),("Gol_U15","over15"),("Gol_U25","over25"),("Gol_U35","over35"),("Gol_U45","over45"),
            ("Gol_A05","under05"),("Gol_A15","under15"),("Gol_A25","under25"),("Gol_A35","under35"),("Gol_A45","under45")]:
    pr,cf=train_ret(c); mkt[n]=(pr,cf,test_m[c].values.astype(int))
# KORNER GECICI CIKARILDI: test setinde total_corners=0 (veri yok), sahte %100 veriyor
P(f"    egitim bitti ({time.time()-t0:.0f}s)")

# ============ GERCEK MAC TAHMİN RAPORU ============
section(f"GERCEK MAC TAHMİNLERİ (son {args.show} mac - tahmin + GERCEK sonuc)")
P(f"  {'TARIH':<10} {'EV':<18} {'DEP':<18} {'SKOR':>5} | 1X2      BTTS    CIFTE12  G25")
P(f"  {'-'*83}")
# team isimleri icin id->name map (mevcut + sofascore)
try:
    idmap = json.load(open("data/gold/team_id_to_name.json",encoding="utf-8"))
except Exception:
    idmap = {}
try:
    sidmap = json.load(open("data/bronze/sofascore_team_ids.json",encoding="utf-8"))
    idmap.update(sidmap)
except Exception:
    pass
def tname(tid):
    s=str(tid).strip().lower()
    # reverse lookup: idmap value==name, key==id
    for k,v in idmap.items():
        if str(v).strip().lower()==s:
            return str(v)[:18]
    return idmap.get(str(tid), str(tid))[:18]

show_n = min(args.show, len(test_m))
for i in range(show_n):
    row = test_m.iloc[i]
    dt = row["date"]
    dt_s = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "?"
    h = tname(row["home_team_id"]); a = tname(row["away_team_id"])
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    actual_res = row["result"]
    pred_res = mkt["1X2"][0][i]
    btts_p = "Var" if mkt["BTTS_Var"][0][i]==1 else "Yok"
    btts_a = "Var" if row["btts"]==1 else "Yok"
    c12_p = "12" if mkt["Cifte_12"][0][i]==1 else "X2/D"
    g25_p = "U" if mkt["Gol_U25"][0][i]==1 else "A"
    g25_a = "U" if row["over25"]==1 else "A"
    # tutma isaretleri
    ok1 = "✓" if pred_res==actual_res else "✗"
    okb = "✓" if btts_p==btts_a else "✗"
    ok12 = "✓" if (mkt["Cifte_12"][0][i]==1 and actual_res!="D") or (mkt["Cifte_12"][0][i]==0 and actual_res=="D") else "✗"
    ok25 = "✓" if g25_p==g25_a else "✗"
    P(f"  {dt_s} {h:<18} {a:<18} {hg}-{ag:>2} | {pred_res}{ok1}  {btts_p}{okb}  {c12_p}{ok12}   {g25_p}{ok25}")

# ============ TUTMA ORANI ============
section("TUM PAZARLAR - GERCEK TUTMA ORANI (3000 mac uzerinde)")
def hold(name, key, lbl_map=None, threshold=0.0):
    pr,cf,act = mkt[key]
    if threshold>0:
        m=cf>=threshold
    else:
        m=np.ones(len(pr),dtype=bool)
    n=m.sum()
    if n==0:
        return name,0,0,0.0
    c=(pr[m]==act[m]).sum()
    return name,n,c,c/n*100

rows=[]
rows.append(hold("1X2 (kazanan)","1X2"))
rows.append(hold("BTTS Var","BTTS_Var"))
rows.append(hold("BTTS Yok","BTTS_Yok"))
rows.append(hold("Cifte Sans 1X","Cifte_1X"))
rows.append(hold("Cifte Sans X2","Cifte_X2"))
rows.append(hold("Cifte Sans 12","Cifte_12"))
for n,l in [("Gol Ust 0.5","Gol_U05"),("Gol Ust 1.5","Gol_U15"),("Gol Ust 2.5","Gol_U25"),
            ("Gol Ust 3.5","Gol_U35"),("Gol Ust 4.5","Gol_U45")]:
    rows.append(hold(n,l))
for n,l in [("Gol Alt 0.5","Gol_A05"),("Gol Alt 1.5","Gol_A15"),("Gol Alt 2.5","Gol_A25"),
            ("Gol Alt 3.5","Gol_A35"),("Gol Alt 4.5","Gol_A45")]:
    rows.append(hold(n,l))
# KORNER CIKARILDI: test verisinde total_corners=0 (veri yok), sahte %100 veriyor
P(f"  NOT: Korner pazarlari gosterilmiyor - test setinde korner verisi bos (0),")
P(f"        gercek korner verisi eklenince yeniden eklenecek.")

P(f"  {'PAZAR':<22}{'TAHMIN':>8}{'TUTAN':>8}{'ORAN':>8}")
P(f"  {'-'*48}")
for name,n,c,acc in sorted(rows,key=lambda x:-x[3]):
    P(f"  {name:<22}{n:>8}{c:>8}{acc:>7.1f}%")

# 70%+ guven altinda tutma
section("70%+ GUVEN ALTINDA TUTMA ORANI")
rows2=[]
for name,key in [("1X2","1X2"),("BTTS Var","BTTS_Var"),("Cifte 12","Cifte_12"),("Gol Ust 2.5","Gol_U25")]:
    rows2.append(hold(name,key,threshold=0.70))
P(f"  {'PAZAR':<22}{'TAHMIN':>8}{'TUTAN':>8}{'ORAN':>8}")
P(f"  {'-'*48}")
for name,n,c,acc in rows2:
    if n>0:
        P(f"  {name:<22}{n:>8}{c:>8}{acc:>7.1f}%")
    else:
        P(f"  {name:<22}{'0':>8}{'-':>8}{'-':>8}")

P(f"\nToplam sure: {time.time()-t0:.0f}s")
