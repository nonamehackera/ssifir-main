"""v6: HIZLI ENSEMBLE (2xLGB) + YENI FEATURE + WALK-FORWARD"""
import sys, time, warnings
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
t0 = time.time()

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"]+feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"] = (feat["total_goals"]>2.5).astype(int)
feat["over15"] = (feat["total_goals"]>1.5).astype(int)
feat = feat.dropna(subset=["result","elo_diff"])
hc = pd.to_numeric(feat["home_corners"],errors="coerce").fillna(0)
ac = pd.to_numeric(feat["away_corners"],errors="coerce").fillna(0)
feat["total_corners"] = hc+ac; feat["corner_over85"] = (feat["total_corners"]>8.5).astype(int)
feat["corner_reliable"] = feat["total_corners"]>=5
oh=pd.to_numeric(feat["avg_home_odds"],errors="coerce"); od=pd.to_numeric(feat["avg_draw_odds"],errors="coerce"); oa=pd.to_numeric(feat["avg_away_odds"],errors="coerce")
feat["has_odds"] = oh.notna()&od.notna()&oa.notna()&(oh>1.01)&(od>1.01)&(oa>1.01)

feat["form_diff"] = feat.get("home_pts_5",0).fillna(0)-feat.get("away_pts_5",0).fillna(0)
feat["gf_diff"] = feat.get("home_gf_5",0).fillna(0)-feat.get("away_gf_5",0).fillna(0)
feat["ga_diff"] = feat.get("home_ga_5",0).fillna(0)-feat.get("away_ga_5",0).fillna(0)
feat["elo_x_hform"] = feat["elo_diff"]*feat.get("home_pts_5",0).fillna(0)
feat["home_strong"] = (feat["elo_diff"]>100).astype(int)
feat["away_strong"] = (feat["elo_diff"]<-100).astype(int)
feat["draw_likely"] = ((feat["elo_diff"].abs()<50)&(feat.get("lg_draw_rate",0.25).fillna(0.25)>0.25)).astype(int)
feat["shots_diff_5"] = feat.get("home_shots_5",0).fillna(0)-feat.get("away_shots_5",0).fillna(0)
feat["sot_diff_5"] = feat.get("home_sot_5",0).fillna(0)-feat.get("away_sot_5",0).fillna(0)
feat["corner_diff_5"] = feat.get("home_corners_5",0).fillna(0)-feat.get("away_corners_5",0).fillna(0)

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading","home_xg","away_xg","total_xg_real","xg_diff_real","home_score_prob","away_score_prob","btts_xprob","over25_xprob","draw_xprob","xg_diff_abs"}
F = [c for c in ["home_elo","away_elo","elo_diff","home_gf_5","home_ga_5","home_pts_5","home_shots_5","home_sot_5","home_corners_5","home_w_gf","home_w_ga","home_w_shots","home_w_sot","home_w_corners","home_hgf_5","home_hga_5","home_hpts_5","home_gf_3","home_ga_3","home_pts_3","home_gf_8","home_ga_8","home_gf_20","home_ga_20","home_gf_std","home_ga_std","home_pts_std","away_gf_5","away_ga_5","away_pts_5","away_shots_5","away_sot_5","away_corners_5","away_w_gf","away_w_ga","away_w_shots","away_w_sot","away_w_corners","away_agf_5","away_aga_5","away_apts_5","away_gf_3","away_ga_3","away_pts_3","away_gf_8","away_ga_8","away_gf_20","away_ga_20","away_gf_std","away_ga_std","away_pts_std","home_attack_elo","home_defence_elo","away_attack_elo","away_defence_elo","attack_elo_diff","defence_elo_diff","home_rest_days","away_rest_days","h2h_home_win","h2h_draw","h2h_away_win","h2h_goals_avg","h2h_btts","home_opp_elo","away_opp_elo","lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg","lg_draw_rate","lg_btts_rate","lg_over25_rate","lg_corner_avg","lg_card_avg","data_completeness","mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob","home_momentum","away_momentum","home_wins_last5","home_draws_last5","away_wins_last5","away_draws_last5","home_form_std","away_form_std","home_gdiff5","away_gdiff5","home_gf_5_real","home_ga_5_real","away_gf_5_real","away_ga_5_real","form_diff","gf_diff","ga_diff","elo_x_hform","home_strong","away_strong","draw_likely","shots_diff_5","sot_diff_5","corner_diff_5"] if c not in LEAK and c in feat.columns]
for c in ["league","home_team_id","away_team_id","season"]:
    if c in feat.columns: feat[c+"_enc"] = feat[c].astype("category").cat.codes
F += [c+"_enc" for c in ["league","home_team_id","away_team_id","season"] if c+"_enc" in feat.columns]
print(f"Feature: {len(F)}")

W = [("2023-01-01","2023-04-01","2023-Q1"),("2023-04-01","2023-07-01","2023-Q2"),("2023-07-01","2023-10-01","2023-Q3"),("2023-10-01","2024-01-01","2023-Q4"),("2024-01-01","2024-04-01","2024-Q1"),("2024-04-01","2024-07-01","2024-Q2"),("2024-07-01","2024-10-01","2024-Q3"),("2024-10-01","2025-01-01","2024-Q4"),("2025-01-01","2025-04-01","2025-Q1"),("2025-04-01","2025-07-01","2025-Q2")]

LC = dict(num_leaves=40,learning_rate=0.02,n_estimators=500,max_depth=5,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,verbose=-1,random_state=42)
LB = dict(num_leaves=30,learning_rate=0.03,n_estimators=350,max_depth=4,min_child_samples=120,subsample=0.65,colsample_bytree=0.5,reg_alpha=1.0,reg_lambda=12.0,verbose=-1,random_state=99)
BA = dict(objective="binary",num_leaves=40,learning_rate=0.02,n_estimators=500,max_depth=5,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,verbose=-1,random_state=42)
BB = dict(objective="binary",num_leaves=30,learning_rate=0.03,n_estimators=350,max_depth=4,min_child_samples=120,subsample=0.65,colsample_bytree=0.5,reg_alpha=1.0,reg_lambda=12.0,verbose=-1,random_state=99)

def c3(pv,pt,yv):
    cal=[IsotonicRegression(out_of_bounds="clip").fit(pv[:,c],(yv==c).astype(float)) for c in range(3)]
    o=np.column_stack([cal[c].predict(pt[:,c]) for c in range(3)])
    s=o.sum(axis=1);s=np.where(s==0,1,s);o/=s[:,None];return o
def cb(pv,pt,yv):
    return IsotonicRegression(out_of_bounds="clip").fit(pv,yv.astype(float)).predict(pt)

R=[];AC=[];AP=[];AY=[]
print("="*72);print("ENSEMBLE WALK-FORWARD (2xLGB, 2y train, 3mo test)");print("="*72)

for vs,ve,lb in W:
    vs_,ve_=pd.Timestamp(vs),pd.Timestamp(ve)
    tm=(feat["date"]>=vs_-pd.DateOffset(years=2))&(feat["date"]<vs_)
    vm=(feat["date"]>=vs_)&(feat["date"]<ve_)
    tr=feat[tm];va=feat[vm]
    if len(tr)<5000 or len(va)<100: print(f"  {lb}: atlandi");continue
    yr=tr["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    sp=int(len(tr)*0.85)
    Xt=tr[F].fillna(0).iloc[:sp];Xv=tr[F].fillna(0).iloc[sp:];Xte=va[F].fillna(0)

    mA=lgb.LGBMClassifier(objective="multiclass",num_class=3,**LC);mA.fit(Xt,yr[:sp])
    mB=lgb.LGBMClassifier(objective="multiclass",num_class=3,**LB);mB.fit(Xt,yr[:sp])
    pA=c3(mA.predict_proba(Xv),mA.predict_proba(Xte),yr[sp:])
    pB=c3(mB.predict_proba(Xv),mB.predict_proba(Xte),yr[sp:])
    p=0.5*pA+0.5*pB;s=p.sum(axis=1);s=np.where(s==0,1,s);p/=s[:,None]

    bA=lgb.LGBMClassifier(**BA);bA.fit(Xt,tr["btts"].values[:sp])
    bB=lgb.LGBMClassifier(**BB);bB.fit(Xt,tr["btts"].values[:sp])
    bp=0.5*cb(bA.predict_proba(Xv)[:,1],bA.predict_proba(Xte)[:,1],tr["btts"].values[sp:])+0.5*cb(bB.predict_proba(Xv)[:,1],bB.predict_proba(Xte)[:,1],tr["btts"].values[sp:])

    oA=lgb.LGBMClassifier(**BA);oA.fit(Xt,tr["over25"].values[:sp])
    oB=lgb.LGBMClassifier(**BB);oB.fit(Xt,tr["over25"].values[:sp])
    op25=0.5*cb(oA.predict_proba(Xv)[:,1],oA.predict_proba(Xte)[:,1],tr["over25"].values[sp:])+0.5*cb(oB.predict_proba(Xv)[:,1],oB.predict_proba(Xte)[:,1],tr["over25"].values[sp:])

    o1m=lgb.LGBMClassifier(**BA);o1m.fit(Xt,tr["over15"].values[:sp])
    o15p=cb(o1m.predict_proba(Xv)[:,1],o1m.predict_proba(Xte)[:,1],tr["over15"].values[sp:])

    ctr=tr["corner_reliable"].values.astype(bool);cte=va["corner_reliable"].values.astype(bool)
    cp=np.full(len(va),0.5)
    if ctr.sum()>500 and cte.sum()>10:
        NF=[c for c in F if "_enc" not in c]
        m5=lgb.LGBMRegressor(objective="poisson",num_leaves=30,learning_rate=0.02,n_estimators=400,max_depth=5,min_child_samples=40,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=8.0,verbose=-1,random_state=42)
        m5.fit(tr.loc[ctr,NF].fillna(0),tr.loc[ctr,"total_corners"].values.astype(float))
        cp[cte]=1-poisson.cdf(8,np.maximum(m5.predict(va.loc[cte,NF].fillna(0)),0.5))

    n=len(va);y2=va["result"].map({"H":0,"D":1,"A":2}).to_numpy();pk=np.argmax(p,axis=1)
    a1=(pk==y2).mean()
    csp=np.column_stack([p[:,0]+p[:,1],p[:,1]+p[:,2],p[:,0]+p[:,2]])
    csy=np.column_stack([(y2==0)|(y2==1),(y2==1)|(y2==2),(y2==0)|(y2==2)])
    acs=csy[np.arange(n),np.argmax(csp,axis=1)].mean()
    abt=((bp>0.5).astype(int)==va["btts"].values).mean()
    ao1=((o15p>0.5).astype(int)==va["over15"].values).mean()
    ao2=((op25>0.5).astype(int)==va["over25"].values).mean()
    om=va["has_odds"].values;no=int(om.sum());vbr=vbn=vbh=0.0
    if no>10:
        od2=np.column_stack([va.loc[om,"avg_home_odds"].values.astype(float),va.loc[om,"avg_draw_odds"].values.astype(float),va.loc[om,"avg_away_odds"].values.astype(float)])
        idx=np.where(om)[0];edge=np.max(p[idx]*od2,axis=1)-1;vb=edge>0.05;vbn=int(vb.sum())
        if vbn>0:
            pf=np.where(y2[idx[vb]]==pk[idx[vb]],od2[vb,pk[idx[vb]]]-1,-1.0)
            vbr=pf.mean();vbh=(y2[idx[vb]]==pk[idx[vb]]).mean()
    ll=-np.mean(np.log(np.clip(p[np.arange(n),y2],1e-9,None)))
    lpA=np.argmax(pA,axis=1);a1l=(lpA==y2).mean()

    print(f"  {lb}: n={n:>5} ENS={a1*100:.1f}% A={a1l*100:.1f}% CS={acs*100:.1f}% BT={abt*100:.1f}% O15={ao1*100:.1f}% O25={ao2*100:.1f}% od={no:>4} VB={vbr*100:+.1f}%({vbn})")
    R.append({"lb":lb,"n":n,"a1":a1,"a1l":a1l,"acs":acs,"abt":abt,"ao1":ao1,"ao2":ao2,"ll":ll,"vbr":vbr,"vbn":vbn,"vbh":vbh,"cn":int(cte.sum()),"acr":((cp[cte]>0.5).astype(int)==va.loc[cte,"corner_over85"].values).mean() if cte.sum()>10 else None})
    AC.append(np.max(p,axis=1));AP.append(pk);AY.append(y2)

print(f"\n{'='*72}");print("GENEL SONUCLAR (WF Agirlikli Ort)");print("="*72)
df=pd.DataFrame(R);w=df["n"]/df["n"].sum()
for co,nm in[("a1","ENSEMBLE 1X2"),("a1l","Tek Model 1X2"),("acs","Cifte Sans"),("abt","BTTS"),("ao1","Ust 1.5"),("ao2","Ust 2.5")]:
    v=(df[co]*w).sum();print(f"  {nm:22s}: {v*100:>5.1f}%  (min={df[co].min()*100:.1f}% max={df[co].max()*100:.1f}%)")
cr=df[df["cn"]>10]
if len(cr)>0:
    cw=cr["cn"]/cr["cn"].sum();vc=(cr["acr"]*cw).sum();print(f"  {'Korner Ust 8.5':22s}: {vc*100:>5.1f}%  (n={int(cr['cn'].sum())})")
print(f"\n  Logloss ort: {df['ll'].mean():.4f}")
vdf=df[df["vbn"]>0]
if len(vdf)>0:
    tv=vdf["vbn"].sum();ro=(vdf["vbr"]*vdf["vbn"]/tv).sum();hi=(vdf["vbh"]*vdf["vbn"]/tv).sum()
    print(f"  BAHIS (edge>5%): {int(tv)} bahis  ROI={ro*100:+.1f}%  isabet={hi*100:.1f}%")

print(f"\n{'='*72}");print("KALIBRASYON - ENSEMBLE 1X2");print("="*72)
ac2=np.concatenate(AC);ap2=np.concatenate(AP);ay2=np.concatenate(AY)
print(f"  {'Guven':>10} {'n':>6} {'Gercek':>8} {'Model':>8}")
for lo in[.35,.40,.45,.50,.55,.60,.65,.70]:
    hi=lo+.05;msk=(ac2>=lo)&(ac2<hi)
    if msk.sum()>10:
        r=(ap2[msk]==ay2[msk]).mean();print(f"  {lo*100:.0f}-{hi*100:.0f}%  {msk.sum():>6} {r*100:>7.1f}%  {ac2[msk].mean()*100:>7.1f}%")

print(f"\nTOPLAM SURE: {time.time()-t0:.0f}s")
