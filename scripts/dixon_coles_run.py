"""DIXON-COLES - STANDALONE
ML ensemble'dan bagimsiz calisir. bootstrap baslangic ile.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import time, json

t0 = time.time()
print("=" * 60)
print("  DIXON-COLES POISSON MODEL")
print("=" * 60)

df = pd.read_parquet("data/gold/features_enhanced_v5.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["home_goals","away_goals","home_team_id","away_team_id"])
df["home_goals"] = df["home_goals"].astype(int)
df["away_goals"] = df["away_goals"].astype(int)

train = df[(df["date"] >= "2021-01-01") & (df["date"] < "2025-09-01")]
if len(train) > 40000:
    train = train.sample(n=40000, random_state=42)
test = df[df["date"] >= "2025-09-01"]
print(f"  Train: {len(train)}, Test: {len(test)}")

# Team mapping
team_counts = train["home_team_id"].value_counts() + train["away_team_id"].value_counts()
top_n = 100
top_teams = team_counts.nlargest(top_n).index.tolist()
other_teams = (set(train["home_team_id"].unique()) | set(train["away_team_id"].unique())) - set(top_teams)
team_ids = {t: i for i, t in enumerate(top_teams)}
other_idx = top_n
for t in other_teams: team_ids[t] = other_idx
n_teams = other_idx + 1

# Vectorized index mapping
h_idx = train["home_team_id"].map(team_ids).fillna(other_idx).astype(int).values
a_idx = train["away_team_id"].map(team_ids).fillna(other_idx).astype(int).values
hg = train["home_goals"].values.astype(np.float64)
ag = train["away_goals"].values.astype(np.float64)

def nll(params):
    home_adv = np.exp(params[0]); rho = params[1]
    att = params[2:2+n_teams]; dfe = params[2+n_teams:2+2*n_teams]
    lh = np.exp(np.clip(att[h_idx]+dfe[a_idx]+np.log(home_adv), -10, 10))
    la = np.exp(np.clip(att[a_idx]+dfe[h_idx], -10, 10))
    ll = hg*np.log(np.maximum(lh,1e-10))-lh-gammaln(hg+1)+ag*np.log(np.maximum(la,1e-10))-la-gammaln(ag+1)
    tau = np.ones(len(hg))
    c00=(hg==0)&(ag==0); c01=(hg==0)&(ag==1); c10=(hg==1)&(ag==0); c11=(hg==1)&(ag==1)
    tau[c00]=1-lh[c00]*la[c00]*rho; tau[c01]=1+lh[c01]*rho
    tau[c10]=1+la[c10]*rho; tau[c11]=1-rho
    tau=np.clip(tau,1e-10,None)
    return -np.sum(ll+np.log(tau))

# Simple init
x0 = np.zeros(2+2*n_teams)
x0[0] = np.log(1.25)
x0[1] = -0.13

print(f"  Optimizing ({n_teams} teams, {len(train)} matches)...", flush=True)
r = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 100})
print(f"  Done: nll={r.fun:.0f}, nit={r.nit} [{time.time()-t0:.0f}s]")

params = r.x
home_adv = np.exp(params[0]); rho = params[1]
att = params[2:2+n_teams]; dfe = params[2+n_teams:2+2*n_teams]

# Predict
h_idx_t = test["home_team_id"].map(team_ids).fillna(other_idx).astype(int).values
a_idx_t = test["away_team_id"].map(team_ids).fillna(other_idx).astype(int).values
lh_t = np.exp(np.clip(att[h_idx_t]+dfe[a_idx_t]+np.log(home_adv), -10, 10))
la_t = np.exp(np.clip(att[a_idx_t]+dfe[h_idx_t], -10, 10))

max_g = 10; g = np.arange(max_g+1)
po_h = np.exp(g[np.newaxis,:]*np.log(np.maximum(lh_t[:,np.newaxis],1e-10))-lh_t[:,np.newaxis]-gammaln(g[np.newaxis,:]+1))
po_a = np.exp(g[np.newaxis,:]*np.log(np.maximum(la_t[:,np.newaxis],1e-10))-la_t[:,np.newaxis]-gammaln(g[np.newaxis,:]+1))
cdf_a = np.cumsum(po_a, axis=1); cdf_h = np.cumsum(po_h, axis=1)

n = len(test)
p_home = np.zeros(n)
for h in range(1, max_g+1): p_home += po_h[:,h]*cdf_a[:,h-1]
p_away = np.zeros(n)
for a in range(1, max_g+1): p_away += po_a[:,a]*cdf_h[:,a-1]
p_draw = np.sum(po_h*po_a, axis=1)

p_exact_00=po_h[:,0]*po_a[:,0]*(1-lh_t*la_t*rho)
p_exact_01=po_h[:,0]*po_a[:,1]*(1+lh_t*rho)
p_exact_10=po_h[:,1]*po_a[:,0]*(1+la_t*rho)
p_exact_11=po_h[:,1]*po_a[:,1]*(1-rho)
p_draw_corr = p_exact_00+p_exact_01+p_exact_10+p_exact_11+(p_draw-po_h[:,0]*po_a[:,0]-po_h[:,1]*po_a[:,1])
p_home_corr = p_home-po_h[:,1]*po_a[:,0]+p_exact_10
p_away_corr = p_away-po_h[:,0]*po_a[:,1]+p_exact_01

probs = np.column_stack([p_home_corr, p_draw_corr, p_away_corr])
probs /= probs.sum(axis=1,keepdims=True)
probs = np.clip(probs, 1e-10, 1)
preds = np.argmax(probs, axis=1)
yte = test["result"].map({"H":0,"D":1,"A":2}).values
conf = probs.max(axis=1)

acc = (preds==yte).mean()
dc_dr = (preds[yte==1]==1).mean() if (yte==1).sum() else 0

print(f"\n  === DIXON-COLES RESULTS ===")
print(f"  Accuracy: {acc*100:.1f}%")
print(f"  Draw Recall: {dc_dr*100:.1f}%")
for thr in [0.50, 0.55, 0.60, 0.65]:
    mask = conf>=thr
    if mask.sum()>0:
        print(f"  HC{int(thr*100)}: {(preds[mask]==yte[mask]).mean()*100:.1f}% ({mask.sum()} mac)")

# Distribution
pred_dist = pd.Series(preds).map({0:"H",1:"D",2:"A"}).value_counts(normalize=True)
real_dist = pd.Series(yte).map({0:"H",1:"D",2:"A"}).value_counts(normalize=True)
for k in ["H","D","A"]:
    print(f"  {k}: pred={pred_dist.get(k,0)*100:.1f}% real={real_dist.get(k,0)*100:.1f}%")

# ROI
ho = test["avg_home_odds"].values; do_ = test["avg_draw_odds"].values; ao = test["avg_away_odds"].values
valid = ~(np.isnan(ho)|np.isnan(do_)|np.isnan(ao))
bets=profit=staked=wins=0
for i in np.where(valid)[0]:
    for out,ov,mdl,ri in [("H",ho[i],probs[i,0],0),("D",do_[i],probs[i,1],1),("A",ao[i],probs[i,2],2)]:
        mp=1/ov; edge=mdl-mp
        if edge>0.03 and ov>1.01:
            kelly=edge/(ov-1)*0.20; kelly=max(0,min(kelly,0.05))
            if kelly>0.005:
                bets+=1; st=kelly*100; staked+=st
                if yte[i]==ri: profit+=st*(ov-1); wins+=1
                else: profit-=st

roi=profit/staked*100 if staked>0 else 0
print(f"\n  Value Bets: {bets}")
print(f"  Win: {wins}/{bets} ({wins/bets*100:.1f}%)" if bets else "  Win: 0/0")
print(f"  ROI: {roi:+.2f}%")
print(f"  100 birim -> {100+profit:.1f}")
print(f"\n  Total: {time.time()-t0:.0f}s")

result = {
    "model": "dixon_coles",
    "accuracy": acc, "draw_recall": dc_dr,
    "roi": roi, "n_bets": bets,
    "home_adv": home_adv, "rho": rho,
}
with open("tahminler/dixon_coles_results.json", "w") as f:
    json.dump(result, f, indent=2)
