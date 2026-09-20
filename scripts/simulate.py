"""Hizli simulasyon - vectorized."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()

all_m = pd.read_parquet("data/gold/matches_all.parquet")
all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
all_m = all_m.sort_values("date").reset_index(drop=True)

N = 3000
test_m = all_m.tail(N).copy()
train_m = all_m.iloc[:-N].copy()
print(f"Train: {len(train_m)}, Test: {len(test_m)}")

# ELO + form - only from train data
elo = {}
form = {}
K, HA = 32, 50

def get_elo(t): return elo.get(t, 1500)
def get_form(t, n=5):
    if t not in form or len(form[t])==0: return [0,0,0]
    r = form[t][-n:]
    return [np.mean([x[i] for x in r]) for i in range(3)]

# Compute features as columns
for col in ["home_elo","away_elo","elo_diff","home_gf_5","home_ga_5","home_pts_5",
    "away_gf_5","away_ga_5","away_pts_5","home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_w_gf","home_w_ga","away_w_gf","away_w_ga","home_rest_days","away_rest_days",
    "home_opp_elo","away_opp_elo"]:
    train_m[col] = 0.0
    test_m[col] = 0.0

for df, is_train in [(train_m, True), (test_m, False)]:
    for i in df.index:
        r = df.loc[i]
        h, a = r["home_team"], r["away_team"]
        he, ae = get_elo(h), get_elo(a)
        df.at[i,"home_elo"] = he; df.at[i,"away_elo"] = ae
        df.at[i,"elo_diff"] = he - ae + HA
        hf, af = get_form(h), get_form(a)
        df.at[i,"home_gf_5"]=hf[0]; df.at[i,"home_ga_5"]=hf[1]; df.at[i,"home_pts_5"]=hf[2]
        df.at[i,"away_gf_5"]=af[0]; df.at[i,"away_ga_5"]=af[1]; df.at[i,"away_pts_5"]=af[2]
        h3, a3 = get_form(h,3), get_form(a,3)
        df.at[i,"home_gf_3"]=h3[0]; df.at[i,"home_ga_3"]=h3[1]
        df.at[i,"away_gf_3"]=a3[0]; df.at[i,"away_ga_3"]=a3[1]
        df.at[i,"home_w_gf"]=hf[0]; df.at[i,"home_w_ga"]=hf[1]
        df.at[i,"away_w_gf"]=af[0]; df.at[i,"away_w_ga"]=af[1]
        df.at[i,"home_rest_days"]=5; df.at[i,"away_rest_days"]=5
        df.at[i,"home_opp_elo"]=ae; df.at[i,"away_opp_elo"]=he
        if is_train:
            pts = 3 if r["result"]=="H" else (1 if r["result"]=="D" else 0)
            pts_a = 3 if r["result"]=="A" else (1 if r["result"]=="D" else 0)
            form.setdefault(h,[]).append((r["home_goals"],r["away_goals"],pts))
            form.setdefault(a,[]).append((r["away_goals"],r["home_goals"],pts_a))
            if r["result"]=="H":
                ew,el=get_elo(h),get_elo(a)
                exp=1/(1+10**((el-ew-HA)/400))
                elo[h]=ew+K*(1-exp); elo[a]=el+K*(0-(1-exp))
            elif r["result"]=="A":
                ew,el=get_elo(a),get_elo(h)
                exp=1/(1+10**((el-ew-HA)/400))
                elo[a]=ew+K*(1-exp); elo[h]=el+K*(0-(1-exp))
            else:
                ew,el=get_elo(h),get_elo(a)
                exp=1/(1+10**((el-ew-HA)/400))
                elo[h]=ew+K*(0.5-exp); elo[a]=el+K*(0.5-(1-exp))

# League avg
lg_s = {}
for _,r in train_m.iterrows():
    lg=r["league"]; t=r["home_goals"]+r["away_goals"]
    lg_s.setdefault(lg,[]).append((t,r["home_goals"],r["away_goals"],
        1 if r["result"]=="D" else 0, 1 if (r["home_goals"]>0 and r["away_goals"]>0) else 0, 1 if t>2.5 else 0))
lg_a = {}
for lg,s in lg_s.items():
    if len(s)>=5: lg_a[lg]=tuple(np.mean([x[i] for x in s]) for i in range(6))
dl = (2.5,1.3,1.2,0.25,0.5,0.45)

for df in [train_m,test_m]:
    df["lg_avg_goals"]=df["league"].map(lambda x: lg_a.get(x,dl)[0])
    df["lg_home_goal_avg"]=df["league"].map(lambda x: lg_a.get(x,dl)[1])
    df["lg_away_goal_avg"]=df["league"].map(lambda x: lg_a.get(x,dl)[2])
    df["lg_draw_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[3])
    df["lg_btts_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[4])
    df["lg_over25_rate"]=df["league"].map(lambda x: lg_a.get(x,dl)[5])
    df["mkt_home_prob"]=1/df["lg_home_goal_avg"].clip(lower=0.5)
    df["mkt_draw_prob"]=df["lg_draw_rate"]
    df["mkt_away_prob"]=1/df["lg_away_goal_avg"].clip(lower=0.5)
    df["mkt_over25_prob"]=df["lg_over25_rate"]
    df["home_score_prob"]=df["lg_home_goal_avg"]/df["lg_avg_goals"].clip(lower=1)
    df["away_score_prob"]=df["lg_away_goal_avg"]/df["lg_avg_goals"].clip(lower=1)
    df["btts_xprob"]=df["lg_btts_rate"]; df["over25_xprob"]=df["lg_over25_rate"]
    df["draw_xprob"]=df["lg_draw_rate"]

F = ["elo_diff","home_elo","away_elo","home_gf_5","home_ga_5","home_pts_5",
    "away_gf_5","away_ga_5","away_pts_5","home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_w_gf","home_w_ga","away_w_gf","away_w_ga","home_rest_days","away_rest_days",
    "home_opp_elo","away_opp_elo","lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg",
    "lg_draw_rate","lg_btts_rate","lg_over25_rate","mkt_home_prob","mkt_draw_prob",
    "mkt_away_prob","mkt_over25_prob","home_score_prob","away_score_prob","btts_xprob",
    "over25_xprob","draw_xprob"]
f = [x for x in F if x in train_m.columns]

tc = train_m.dropna(subset=["result","btts","over25"]).copy()
X_tr, X_te = tc[f].copy(), test_m[f].copy()
y_r_enc = np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
sp = int(len(X_tr)*0.8)

# 1X2
print("1X2...")
m1 = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,learning_rate=0.02,
    n_estimators=1000,max_depth=5,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,
    reg_alpha=0.5,reg_lambda=8.0,verbose=-1).fit(X_tr.iloc[:sp],y_r_enc[:sp])
raw=m1.predict_proba(X_tr.iloc[sp:]); cal1=[IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y_r_enc[sp:]==c).astype(float)) for c in range(3)]
p1=np.column_stack([cal1[c].predict(m1.predict_proba(X_te)[:,c]) for c in range(3)])
s=p1.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); p1/=s
pw=np.where((p1[:,0]>p1[:,1])&(p1[:,0]>p1[:,2]),"H",np.where((p1[:,2]>p1[:,1])&(p1[:,2]>p1[:,0]),"A","D"))
c1=np.max(p1,axis=1)

# BTTS
print("BTTS...")
m2 = lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.02,
    n_estimators=800,max_depth=5,min_child_samples=60,subsample=0.7,colsample_bytree=0.6,
    reg_alpha=0.5,reg_lambda=8.0,verbose=-1).fit(X_tr.iloc[:sp],tc["btts"].values[:sp])
raw=m2.predict_proba(X_tr.iloc[sp:])[:,1]; cal2=IsotonicRegression(out_of_bounds="clip").fit(raw,tc["btts"].values[sp:].astype(float))
pbt=cal2.predict(m2.predict_proba(X_te)[:,1]); pbt_v=(pbt>0.5).astype(int); cbt=np.maximum(pbt,1-pbt)

# O25
print("O25...")
m3 = lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.02,
    n_estimators=800,max_depth=5,min_child_samples=60,subsample=0.7,colsample_bytree=0.6,
    reg_alpha=0.5,reg_lambda=8.0,verbose=-1).fit(X_tr.iloc[:sp],tc["over25"].values[:sp])
raw=m3.predict_proba(X_tr.iloc[sp:])[:,1]; cal3=IsotonicRegression(out_of_bounds="clip").fit(raw,tc["over25"].values[sp:].astype(float))
po2=cal3.predict(m3.predict_proba(X_te)[:,1]); po2_v=(po2>0.5).astype(int); co2=np.maximum(po2,1-po2)

# O15
print("O15...")
o15tr=(tc["total_goals"]>1.5).astype(int)
m4 = lgb.LGBMClassifier(objective="binary",num_leaves=36,learning_rate=0.02,
    n_estimators=800,max_depth=5,min_child_samples=60,subsample=0.7,colsample_bytree=0.6,
    reg_alpha=0.5,reg_lambda=8.0,verbose=-1).fit(X_tr.iloc[:sp],o15tr.values[:sp])
raw=m4.predict_proba(X_tr.iloc[sp:])[:,1]; cal4=IsotonicRegression(out_of_bounds="clip").fit(raw,o15tr.values[sp:].astype(float))
po15=cal4.predict(m4.predict_proba(X_te)[:,1]); po15_v=(po15>0.5).astype(int); co15=np.maximum(po15,1-po15)

# RESULT
yr=test_m["result"].values; ybt=test_m["btts"].values.astype(int)
yo25=test_m["over25"].values.astype(int); yo15=(test_m["total_goals"].values>1.5).astype(int)

print(f"\n{'='*70}")
print(f"  3000 MAC SIMULASYON ({test_m['date'].min().date()} -> {test_m['date'].max().date()})")
print(f"{'='*70}")

for name,preds,conf,act in [("1X2",pw,c1,yr),("BTTS",pbt_v,cbt,ybt),("GOL UST 1.5",po15_v,co15,yo15),("GOL UST 2.5",po2_v,co2,yo25)]:
    print(f"\n  {name}")
    print(f"  {'Esik':>6} {'Picks':>6} {'Oneri%':>7} {'Dogru':>6} {'Acc':>7}")
    print(f"  {'-'*40}")
    for t in [0.50,0.52,0.55,0.58,0.60,0.62,0.65,0.68,0.70,0.75,0.80]:
        m=conf>=t; n=m.sum()
        if n>0:
            c=(preds[m]==act[m]).sum()
            print(f"  {t*100:>5.0f}% {n:>6} {n/len(preds)*100:>6.1f}% {c:>6} {c/n*100:>6.1f}%")

print(f"\n  Sure: {time.time()-t0:.0f}s")
