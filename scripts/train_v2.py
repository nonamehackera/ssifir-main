"""Gelismis model - tum pazarlar, haftalik guncellenebilir."""
import sys, os, warnings, time, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict

t0 = time.time()
OUT = "data/gold"
os.makedirs("models/lightgbm_v2", exist_ok=True)

###############################################################################
# 1. VERI YUKLE + BIRLESTIR
###############################################################################
print("="*70)
print("  ADIM 1: VERI HAZIRLAMA")
print("="*70)

all_m = pd.read_parquet(f"{OUT}/matches_all.parquet")
all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
all_m = all_m.sort_values("date").reset_index(drop=True)
all_m["total_goals"] = all_m["home_goals"] + all_m["away_goals"]
all_m["btts"] = ((all_m["home_goals"]>0)&(all_m["away_goals"]>0)).astype(int)
all_m["over15"] = (all_m["total_goals"]>1.5).astype(int)
all_m["over25"] = (all_m["total_goals"]>2.5).astype(int)
all_m["over35"] = (all_m["total_goals"]>3.5).astype(int)
print(f"  Toplam: {len(all_m)} mac")

###############################################################################
# 2. FEATURE ENGINE (leakage-free)
###############################################################################
print(f"\n{'='*70}")
print("  ADIM 2: FEATURE ENGINE")
print("="*70)

elo = {}
form = defaultdict(list)
h2h = defaultdict(list)
home_form = defaultdict(list)
away_form = defaultdict(list)
K, HA = 32, 50

def get_elo(t): return elo.get(t, 1500)
def get_form(t, n=5):
    if t not in form or len(form[t])==0: return [0,0,0]
    r = form[t][-n:]
    return [np.mean([x[i] for x in r]) for i in range(3)]

# Pre-compute all features
feats_data = []
for i, r in all_m.iterrows():
    h, a = r["home_team"], r["away_team"]
    he, ae = get_elo(h), get_elo(a)
    
    hf5 = get_form(h, 5); af5 = get_form(a, 5)
    hf3 = get_form(h, 3); af3 = get_form(a, 3)
    
    # H2H
    h2h_key = tuple(sorted([h, a]))
    h2h_list = h2h.get(h2h_key, [])
    h2h_home_wr = np.mean([x[0] for x in h2h_list[-10:]]) if h2h_list else 0.5
    h2h_goals = np.mean([x[1] for x in h2h_list[-10:]]) if h2h_list else 2.5
    h2h_btts = np.mean([x[2] for x in h2h_list[-10:]]) if h2h_list else 0.5
    
    # Home/Away specific form
    h_home = home_form.get(h, [])
    a_away = away_form.get(a, [])
    h_home_pts = np.mean([x[0] for x in h_home[-5:]]) if h_home else 0
    a_away_pts = np.mean([x[0] for x in a_away[-5:]]) if a_away else 0
    h_home_gf = np.mean([x[1] for x in h_home[-5:]]) if h_home else 0
    a_away_gf = np.mean([x[1] for x in a_away[-5:]]) if a_away else 0
    h_home_ga = np.mean([x[2] for x in h_home[-5:]]) if h_home else 0
    a_away_ga = np.mean([x[2] for x in a_away[-5:]]) if a_away else 0
    
    # Momentum (son 3 mac form degisimi)
    h_form3 = get_form(h, 3)
    h_form5 = get_form(h, 5)
    h_momentum = h_form3[2] - h_form5[2] if h_form5[2] > 0 else 0
    a_form3 = get_form(a, 3)
    a_form5 = get_form(a, 5)
    a_momentum = a_form3[2] - a_form5[2] if a_form5[2] > 0 else 0
    
    row = {
        "home_elo": he, "away_elo": ae, "elo_diff": he - ae + HA,
        "home_gf_5": hf5[0], "home_ga_5": hf5[1], "home_pts_5": hf5[2],
        "away_gf_5": af5[0], "away_ga_5": af5[1], "away_pts_5": af5[2],
        "home_gf_3": hf3[0], "home_ga_3": hf3[1], "away_gf_3": af3[0], "away_ga_3": af3[1],
        "home_w_gf": hf5[0], "home_w_ga": hf5[1], "away_w_gf": af5[0], "away_w_ga": af5[1],
        "home_rest_days": 5, "away_rest_days": 5,
        "home_opp_elo": ae, "away_opp_elo": he,
        "h2h_home_wr": h2h_home_wr, "h2h_goals": h2h_goals, "h2h_btts": h2h_btts,
        "home_home_pts": h_home_pts, "away_away_pts": a_away_pts,
        "home_home_gf": h_home_gf, "away_away_gf": a_away_gf,
        "home_home_ga": h_home_ga, "away_away_ga": a_away_ga,
        "home_momentum": h_momentum, "away_momentum": a_momentum,
    }
    feats_data.append(row)
    
    # Update
    is_train = i < len(all_m) - 3000  # Only update from train data
    if is_train:
        pts = 3 if r["result"]=="H" else (1 if r["result"]=="D" else 0)
        pts_a = 3 if r["result"]=="A" else (1 if r["result"]=="D" else 0)
        form[h].append((r["home_goals"], r["away_goals"], pts))
        form[a].append((r["away_goals"], r["home_goals"], pts_a))
        home_form[h].append((pts, r["home_goals"], r["away_goals"]))
        away_form[a].append((pts_a, r["away_goals"], r["home_goals"]))
        
        h2h_list = h2h[h2h_key]
        home_wr = 1 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0)
        h2h_list.append((home_wr, r["home_goals"]+r["away_goals"], 
                         1 if (r["home_goals"]>0 and r["away_goals"]>0) else 0))
        
        if r["result"]=="H":
            ew,el=get_elo(h),get_elo(a); exp=1/(1+10**((el-ew-HA)/400))
            elo[h]=ew+K*(1-exp); elo[a]=el+K*(0-(1-exp))
        elif r["result"]=="A":
            ew,el=get_elo(a),get_elo(h); exp=1/(1+10**((el-ew-HA)/400))
            elo[a]=ew+K*(1-exp); elo[h]=el+K*(0-(1-exp))
        else:
            ew,el=get_elo(h),get_elo(a); exp=1/(1+10**((el-ew-HA)/400))
            elo[h]=ew+K*(0.5-exp); elo[a]=el+K*(0.5-(1-exp))

feats_df = pd.DataFrame(feats_data)
all_m = pd.concat([all_m.reset_index(drop=True), feats_df], axis=1)

# League avg (train only)
train_end = len(all_m) - 3000
lg_s = {}
for i in range(train_end):
    r = all_m.iloc[i]
    lg = r["league"]; t = r["total_goals"]
    lg_s.setdefault(lg, []).append((t, r["home_goals"], r["away_goals"],
        1 if r["result"]=="D" else 0, 1 if (r["home_goals"]>0 and r["away_goals"]>0) else 0,
        1 if t>1.5 else 0, 1 if t>2.5 else 0, 1 if t>3.5 else 0))

lg_a = {}
for lg, s in lg_s.items():
    if len(s)>=5: lg_a[lg]=tuple(np.mean([x[i] for x in s]) for i in range(8))
dl = (2.5,1.3,1.2,0.25,0.5,0.75,0.45,0.25)

for df in [all_m]:
    df["lg_avg_goals"]=df["league"].map(lambda x: lg_a.get(x,dl)[0])
    df["lg_home_goal_avg"]=df["league"].map(lambda x: lg_a.get(x,dl)[1])
    df["lg_away_goal_avg"]=df["league"].map(lambda x: lg_a.get(x,dl)[2])
    df["lg_draw_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[3])
    df["lg_btts_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[4])
    df["lg_o15_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[5])
    df["lg_over25_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[6])
    df["lg_o35_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[7])
    df["mkt_home_prob"]=1/df["lg_home_goal_avg"].clip(lower=0.5)
    df["mkt_draw_prob"]=df["lg_draw_rate"]
    df["mkt_away_prob"]=1/df["lg_away_goal_avg"].clip(lower=0.5)
    df["mkt_over25_prob"]=df["lg_over25_rate"]
    df["home_score_prob"]=df["lg_home_goal_avg"]/df["lg_avg_goals"].clip(lower=1)
    df["away_score_prob"]=df["lg_away_goal_avg"]/df["lg_avg_goals"].clip(lower=1)
    df["btts_xprob"]=df["lg_btts_rate"]; df["over25_xprob"]=df["lg_over25_rate"]
    df["draw_xprob"]=df["lg_draw_rate"]

print(f"  Feature set hazir: {len(all_m)} mac, {len([c for c in all_m.columns if c not in ['match_id','league','season','date','home_team','away_team','home_goals','away_goals','result','total_goals','btts','over15','over25','over35','home_team_id','away_team_id','ht_home_goals','ht_away_goals','home_shots','away_shots','home_sot','away_sot','home_corners','away_corners','home_fouls','away_fouls','home_yellow','away_yellow','home_red','away_red','referee','avg_home_odds','avg_draw_odds','avg_away_odds','b365_home_odds','b365_draw_odds','b365_away_odds','avg_over25_odds','avg_under25_odds','avg_close_home_odds','avg_close_draw_odds','avg_close_away_odds','avg_close_over25_odds','avg_close_under25_odds','home_xg','away_xg','round','source','key','home_assists','away_assists','home_minutes','away_minutes','home_players','away_players','attendance','stadium']])} feature")

###############################################################################
# 3. TRAIN/TEST SPLIT + MODEL
###############################################################################
print(f"\n{'='*70}")
print("  ADIM 3: MODEL EGITIMI")
print("="*70)

test_m = all_m.tail(3000).copy()
train_m = all_m.iloc[:-3000].copy()
print(f"  Train: {len(train_m)}, Test: {len(test_m)}")

BASE_FEATS = ["elo_diff","home_elo","away_elo","home_gf_5","home_ga_5","home_pts_5",
    "away_gf_5","away_ga_5","away_pts_5","home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_w_gf","home_w_ga","away_w_gf","away_w_ga","home_rest_days","away_rest_days",
    "home_opp_elo","away_opp_elo","lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg",
    "lg_draw_rate","lg_btts_rate","lg_over25_rate","mkt_home_prob","mkt_draw_prob",
    "mkt_away_prob","mkt_over25_prob","home_score_prob","away_score_prob","btts_xprob",
    "over25_xprob","draw_xprob","h2h_home_wr","h2h_goals","h2h_btts",
    "home_home_pts","away_away_pts","home_home_gf","away_away_gf",
    "home_home_ga","away_away_ga","home_momentum","away_momentum"]

f = [x for x in BASE_FEATS if x in train_m.columns]
print(f"  Features: {len(f)}")

tc = train_m.dropna(subset=["result","btts","over25"]).copy()
X_tr = tc[f].copy()
sp = int(len(X_tr)*0.8)

# Multi-seed ensemble
def train_ensemble(X, y, task="binary", n_seeds=5, **kwargs):
    seeds = list(range(42, 42+n_seeds))
    models = []
    for s in seeds:
        params = dict(num_leaves=40, learning_rate=0.015, n_estimators=1200,
            max_depth=5, min_child_samples=80, subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1, **kwargs)
        if task == "multiclass":
            params.update(objective="multiclass", num_class=3)
        else:
            params.update(objective="binary")
        
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[:sp], y[:sp])
        
        raw = m.predict_proba(X.iloc[sp:])
        if task == "multiclass":
            cal = [IsotonicRegression(out_of_bounds="clip").fit(raw[:,c], (y[sp:]==c).astype(float)) for c in range(3)]
        else:
            cal = IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw, y[sp:].astype(float))
        models.append((m, cal))
    return models

def predict_ensemble(models, X, task="binary"):
    preds = []
    for m, cal in models:
        raw = m.predict_proba(X)
        if task == "multiclass":
            cal_p = np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
            s = cal_p.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); cal_p/=s
            preds.append(cal_p)
        else:
            preds.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
    return np.mean(preds, axis=0)

# 1X2
print("  1X2...")
y_r = np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
m1x2 = train_ensemble(X_tr, y_r, "multiclass")
p1x2 = predict_ensemble(m1x2, test_m[f], "multiclass")
pw = np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
     np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
c1 = np.max(p1x2, axis=1)

# BTTS
print("  BTTS...")
m_btts = train_ensemble(X_tr, tc["btts"].values, "binary")
pbt = predict_ensemble(m_btts, test_m[f], "binary")
pbt_v = (pbt>0.5).astype(int); cbt = np.maximum(pbt, 1-pbt)

# O15
print("  O15...")
m_o15 = train_ensemble(X_tr, tc["over15"].values, "binary")
po15 = predict_ensemble(m_o15, test_m[f], "binary")
po15_v = (po15>0.5).astype(int); co15 = np.maximum(po15, 1-po15)

# O25
print("  O25...")
m_o25 = train_ensemble(X_tr, tc["over25"].values, "binary")
po25 = predict_ensemble(m_o25, test_m[f], "binary")
po25_v = (po25>0.5).astype(int); co25 = np.maximum(po25, 1-po25)

# O35
print("  O35...")
m_o35 = train_ensemble(X_tr, tc["over35"].values, "binary")
po35 = predict_ensemble(m_o35, test_m[f], "binary")
po35_v = (po35>0.5).astype(int); co35 = np.maximum(po35, 1-po35)

# RESULT
yr = test_m["result"].values
ybt = test_m["btts"].values.astype(int)
yo15 = test_m["over15"].values.astype(int)
yo25 = test_m["over25"].values.astype(int)
yo35 = test_m["over35"].values.astype(int)

###############################################################################
# 4. RAPOR
###############################################################################
print(f"\n{'='*70}")
print(f"  4. SONUCLAR (3000 MAC)")
print(f"{'='*70}")
print(f"  Test: {test_m['date'].min().date()} -> {test_m['date'].max().date()}")
print(f"  Train: {len(train_m)} mac")

for name,preds,conf,act in [
    ("1X2",pw,c1,yr),("BTTS",pbt_v,cbt,ybt),
    ("GOL UST 1.5",po15_v,co15,yo15),("GOL UST 2.5",po25_v,co25,yo25),
    ("GOL UST 3.5",po35_v,co35,yo35)]:
    print(f"\n  {name}")
    print(f"  {'Esik':>6} {'Picks':>6} {'Oneri%':>7} {'Dogru':>6} {'Acc':>7}")
    print(f"  {'-'*40}")
    for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
        m=conf>=t; n=m.sum()
        if n>0:
            c=(preds[m]==act[m]).sum()
            print(f"  {t*100:>5.0f}% {n:>6} {n/len(preds)*100:>6.1f}% {c:>6} {c/n*100:>6.1f}%")

# Feature importance
print(f"\n  1X2 Feature Importance:")
imp = m1x2[0][0].feature_importances_
fi = sorted(zip(f, imp), key=lambda x: -x[1])
for fn, fi_v in fi[:15]:
    print(f"    {fn:<30} {fi_v:>6}")

###############################################################################
# 5. HAFTALIK GUNCELLEME SCRIPTI
###############################################################################
print(f"\n{'='*70}")
print("  5. HAFTALIK GUNCELLEME SCRIPTI")
print("="*70)

refresh_script = '''#!/usr/bin/env python3
"""Haftalik veri guncelleme + model yeniden egitim."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

print("Haftalik guncelleme basliyor...")
t0 = time.time()

# 1. Yeni veri indir (Football-Data.co.uk + Transfermarkt)
# TODO: football_data_ingestion'dan en guncel veriyi cek

# 2. Mevcut matches_all.parquet'e ekle
# all_m = pd.read_parquet("data/gold/matches_all.parquet")
# Yeni verileri ekle, dedup yap

# 3. Feature engine calistir (leakage-free)
# ELO + form + H2H etc.

# 4. Modeli yeniden egit
# python train_v2.py

# 5. Test et
# python test_v2.py

print(f"Tamamlandi: {time.time()-t0:.0f}s")
'''

with open("refresh_weekly.py", "w") as fh:
    fh.write(refresh_script)
print("  refresh_weekly.py yazildi")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
