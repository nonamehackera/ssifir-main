"""V2 DENEYI - model SADECE sofascore verisiyle egitilir, mevcut karma modelle kiyaslanir.

Akis:
  1. Sadece data/bronze/sofascore_matches.parquet -> features (build_features_fast mantigi)
  2. Train/test split (son 3000 = test, gerisi train)
  3. Tum pazarlar egitilir
  4. KARMA MODEL (features_fast.parquet) ile ayni test setinde kiyas tablosu

Usage: python scripts/train_sofascore_only_v2.py
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

# ---------- 1. SADECE SOFASCORE VERISI -> FEATURES ----------
SRC = "data/bronze/sofascore_matches.parquet"
P(f"Sadece sofascore verisi okunuyor: {SRC}")
df = pd.read_parquet(SRC)
P(f"  satir: {len(df):,}")

# normalize columns (build_features_fast ile ayni)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["result"] = np.where(df["home_goals"]>df["away_goals"],"H",
                np.where(df["away_goals"]>df["home_goals"],"A","D"))
for c in ["home_goals","away_goals"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["total_goals"] = df["home_goals"].fillna(0) + df["away_goals"].fillna(0)
def col_or_zero(df, c):
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)

df["ht_total_goals"] = col_or_zero(df,"ht_home_goals") + col_or_zero(df,"ht_away_goals")
df["home_corners"] = col_or_zero(df,"home_corners")
df["away_corners"] = col_or_zero(df,"away_corners")
df["home_shots"] = col_or_zero(df,"home_shots")
df["away_shots"] = col_or_zero(df,"away_shots")
df["home_sot"] = col_or_zero(df,"home_sot")
df["away_sot"] = col_or_zero(df,"away_sot")
df["home_xg"] = col_or_zero(df,"home_xg")
df["away_xg"] = col_or_zero(df,"away_xg")
df["home_team_id"] = df["home_team_id"].astype(str)
df["away_team_id"] = df["away_team_id"].astype(str)

# ELO + FORM (O(n) dict-based, sadece sofascore)
def compute_features_elo_form(df):
    df = df.sort_values("date").reset_index(drop=True)
    teams = pd.unique(df[["home_team_id","away_team_id"]].values.ravel())
    elo = {t:1500.0 for t in teams}
    # her takim icin history listesi (tuple: tgf, tga, pts)
    hist = {t:[] for t in teams}
    lg_stats = {}  # league -> {"n":, "win":, "draw":, "btts":}
    K=32
    rows=[]
    for idx, r in df.iterrows():
        h=str(r["home_team_id"]); a=str(r["away_team_id"]); lg=r.get("league","?")
        eh,ea=elo[h],elo[a]
        # form (son 8 ve 20 mac)
        def form_stats(t, n):
            h_=hist[t][-n:]
            gf=sum(x[0] for x in h_); ga=sum(x[1] for x in h_); pts=sum(x[2] for x in h_)
            return gf,ga,pts
        hf8=form_stats(h,8); hf20=form_stats(h,20)
        af8=form_stats(a,8); af20=form_stats(a,20)
        # league stats (kümülatif)
        if lg not in lg_stats:
            lg_stats[lg]={"n":0,"win":0,"draw":0,"btts":0}
        ls=lg_stats[lg]
        n=ls["n"]
        lg_win = ls["win"]/n if n>0 else 0.45
        lg_draw = ls["draw"]/n if n>0 else 0.27
        lg_btts = ls["btts"]/n if n>0 else 0.5
        # guncelle (bu maci ekle)
        hg=r["home_goals"]; ag=r["away_goals"]
        if hg>ag: hp=3; ap=0; res="H"
        elif ag>hg: hp=0; ap=3; res="A"
        else: hp=1; ap=1; res="D"
        hist[h].append((hg,ag,hp)); hist[a].append((ag,hg,ap))
        btts=1 if (hg>0 and ag>0) else 0
        ls["n"]+=1; ls["win"]+= (1 if res=="H" else 0); ls["draw"]+= (1 if res=="D" else 0); ls["btts"]+=btts
        # elo update
        wh = 1.0 if res=="H" else (0.5 if res=="D" else 0.0)
        wa = 1.0-wh
        elo[h]=eh+K*(wh-1/(1+10**((ea-eh)/400)))
        elo[a]=ea+K*(wa-1/(1+10**((eh-ea)/400)))
        rows.append({"elo_h":eh,"elo_a":ea,"elo_diff":eh-ea,
            "home_form_gf_8":hf8[0],"home_form_ga_8":hf8[1],"away_form_gf_8":af8[0],"away_form_ga_8":af8[1],
            "home_form_gf_20":hf20[0],"home_form_ga_20":hf20[1],"away_form_gf_20":af20[0],"away_form_ga_20":af20[1],
            "lg_win_rate":lg_win,"lg_draw_rate":lg_draw,"lg_btts_rate":lg_btts})
    return pd.DataFrame(rows).reset_index(drop=True)

P("  ELO + form hesaplaniyor (sadece sofascore)...")
feat_extra = compute_features_elo_form(df)
df = pd.concat([df.reset_index(drop=True), feat_extra.reset_index(drop=True)], axis=1)
P(f"    bitti ({time.time()-t0:.0f}s)")

# hedefler
df["btts"] = ((df["home_goals"]>0)&(df["away_goals"]>0)).astype(int)
df["over15"]=(df["total_goals"]>1.5).astype(int)
df["over25"]=(df["total_goals"]>2.5).astype(int)
df["over35"]=(df["total_goals"]>3.5).astype(int)
df["under15"]=1-df["over15"]; df["under25"]=1-df["over25"]; df["under35"]=1-df["over35"]
df["double_1X"]=((df["result"]=="H")|(df["result"]=="D")).astype(int)
df["double_X2"]=((df["result"]=="D")|(df["result"]=="A")).astype(int)
df["double_12"]=((df["result"]=="H")|(df["result"]=="A")).astype(int)

df = df.sort_values("date").reset_index(drop=True)
N=3000
test_m = df.tail(N).copy()
train_m = df.iloc[:-N].copy()
P(f"SOFASCORE-ONLY: Train={len(train_m):,} | Test={len(test_m):,} (son 3000 sofascore maci)")

SKIP={"match_id","league","season","date","home_team_id","away_team_id","home_goals","away_goals",
    "result","result_H","result_D","result_A","btts","over25","over15","over35","total_goals",
    "under15","under25","under35","double_1X","double_X2","double_12",
    "home_shots","away_shots","home_sot","away_sot","home_xg","away_xg","home_corners","away_corners",
    "ht_home_goals","ht_away_goals","ht_total_goals","home_yellow","away_yellow","home_red","away_red",
    "referee","data_completeness","sot_diff","shots_diff"}
FEATS=[c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
P(f"Feature: {len(FEATS)}")
sp=int(len(train_m)*0.85)

def quick_train(X,y,task="binary"):
    ms=[]
    for s in [42,123,456]:
        if task=="multiclass":
            m=lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,learning_rate=0.02,
                n_estimators=400,max_depth=5,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,
                reg_alpha=0.5,reg_lambda=5.0,random_seed=s,verbose=-1,n_jobs=-1)
        else:
            m=lgb.LGBMClassifier(objective="binary",num_leaves=40,learning_rate=0.02,n_estimators=400,
                max_depth=5,min_child_samples=60,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
                reg_lambda=5.0,random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp],y[:sp])
        raw=m.predict_proba(X.iloc[sp:])
        if task=="multiclass":
            cal=[IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y[sp:]==c).astype(float)) for c in range(3)]
        else:
            cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw,y[sp:].astype(float))
        ms.append((m,cal))
    return ms

def quick_pred(ms,X,task="binary"):
    ps=[]
    for m,cal in ms:
        raw=m.predict_proba(X)
        if task=="multiclass":
            cp=np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
            ss=cp.sum(axis=1,keepdims=True); ss=np.where(ss==0,1,ss); cp/=ss; ps.append(cp)
        else:
            ps.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
    return np.mean(ps,axis=0)

def train_binary(col,feats=FEATS):
    tc=train_m.dropna(subset=[col])
    X=tc[feats].fillna(0); y=tc[col].values.astype(int)
    ms=quick_train(X,y,"binary")
    p=quick_pred(ms,test_m[feats].fillna(0))
    return (p>0.5).astype(int), np.maximum(p,1-p)

# 1X2
P("  1X2 (sadece sofascore)...")
tc=train_m.dropna(subset=["result"])
X_tr=tc[FEATS].fillna(0); y_r=np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
m1x2=quick_train(X_tr,y_r,"multiclass")
p1x2=quick_pred(m1x2,test_m[FEATS].fillna(0),"multiclass")
pw=np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
    np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
c1=np.max(p1x2,axis=1)

btts_feats=FEATS+[c for c in ["home_xg","away_xg","home_shots","away_shots","home_sot","away_sot","lg_btts_rate"] if c in df.columns and c not in FEATS]

markets=[("1X2",pw,c1,test_m["result"].values)]
for name,col in [("Cifte Sans 1X","double_1X"),("Cifte Sans X2","double_X2"),("Cifte Sans 12","double_12")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values))
pr_b,cf_b=train_binary("btts",btts_feats); markets.append(("BTTS Var",pr_b,cf_b,test_m["btts"].values.astype(int)))
markets.append(("BTTS Yok",1-pr_b,cf_b,1-test_m["btts"].values.astype(int)))
for name,col in [("Gol Ust 1.5","over15"),("Gol Alt 1.5","under15"),("Gol Ust 2.5","over25"),
                 ("Gol Alt 2.5","under25"),("Gol Ust 3.5","over35"),("Gol Alt 3.5","under35")]:
    pr,cf=train_binary(col); markets.append((name,pr,cf,test_m[col].values.astype(int)))

P(f"    egitim bitti ({time.time()-t0:.0f}s)")

# KIYAS TABLOSU
P(f"\n{'='*70}\n  V2 KIYAS: SADECE SOFASCORE vs KARMA MODEL\n{'='*70}")
# karma model sonuclari (onceki run'dan)
KARMA = {
    "1X2 (kazanan)": 55.8,
    "Cifte Sans 1X": 68.0,
    "Cifte Sans X2": 65.1,
    "Cifte Sans 12": 84.1,
    "BTTS Var": 64.9,
    "Gol Ust 1.5": 84.4,
    "Gol Alt 1.5": 84.4,
    "Gol Ust 2.5": 67.6,
    "Gol Alt 2.5": 67.6,
    "Gol Ust 3.5": 85.0,
    "Gol Alt 3.5": 85.0,
}
def hold_v2(name,key):
    _,pr,cf,act=next(m for m in markets if m[0]==key)
    n=len(pr); c=(pr==act).sum(); return n,c,c/n*100

v2_results={}
v2_results["1X2 (kazanan)"]=hold_v2("1X2","1X2")
v2_results["Cifte Sans 1X"]=hold_v2("Cifte Sans 1X","Cifte Sans 1X")
v2_results["Cifte Sans X2"]=hold_v2("Cifte Sans X2","Cifte Sans X2")
v2_results["Cifte Sans 12"]=hold_v2("Cifte Sans 12","Cifte Sans 12")
v2_results["BTTS Var"]=hold_v2("BTTS Var","BTTS Var")
v2_results["Gol Ust 1.5"]=hold_v2("Gol Ust 1.5","Gol Ust 1.5")
v2_results["Gol Alt 1.5"]=hold_v2("Gol Alt 1.5","Gol Alt 1.5")
v2_results["Gol Ust 2.5"]=hold_v2("Gol Ust 2.5","Gol Ust 2.5")
v2_results["Gol Alt 2.5"]=hold_v2("Gol Alt 2.5","Gol Alt 2.5")
v2_results["Gol Ust 3.5"]=hold_v2("Gol Ust 3.5","Gol Ust 3.5")
v2_results["Gol Alt 3.5"]=hold_v2("Gol Alt 3.5","Gol Alt 3.5")

P(f"\n  {'PAZAR':<20}{'KARMA %':>9}{'SADECE SOF %':>15}{'FARK':>8}")
P(f"  {'-'*52}")
for name in KARMA:
    kv=KARMA[name]; vv=v2_results[name][2]; diff=vv-kv
    P(f"  {name:<20}{kv:>8.1f}%{vv:>14.1f}%{diff:>+7.1f}%")

P(f"\n  Ortalama (KARMA): {np.mean(list(KARMA.values())):.1f}%")
P(f"  Ortalama (SADECE SOFASCORE): {np.mean([v2_results[n][2] for n in KARMA]):.1f}%")
P(f"\n  SONUC: " + ("SADECE SOFASCORE DAHA IYI" if np.mean([v2_results[n][2] for n in KARMA])>np.mean(list(KARMA.values())) else "KARMA MODEL DAHA IYI") + "")
P(f"  Toplam sure: {time.time()-t0:.0f}s")
