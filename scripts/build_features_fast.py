"""Hizli vectorize ELO + form feature builder.

build_features (engine.py) cok yavas (~30+ dk / 900K mac). Bu script onun
yerine ayni feature'lari numpy ile saniyelerde uretir.

CIKTI: data/gold/features_fast.parquet
- ELO (kazanan/kaybeden update, home advantage)
- form windows (3/5/8/20): gf, ga, pts (ev + deplasman ayri)
- league baselines (avg goals, draw, btts, over25, corner)
- market-implied probs (odds varsa)
- HEDEFLER: result, btts, over15/25/35, double_1X/X2/12, corners

Kullanim: python scripts/build_features_fast.py
"""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

# ---------- 1. VERI YUKLE ----------
base = pd.read_parquet("data/gold/matches_all.parquet")
sofa = pd.read_parquet("data/bronze/sofascore_matches.parquet")
common = [c for c in base.columns if c in sofa.columns]
sofa_c = sofa[common].copy()
df = pd.concat([base, sofa_c], ignore_index=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0).astype(int)
df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0).astype(int)
df["result"] = np.where(df["home_goals"]>df["away_goals"],"H",
                np.where(df["away_goals"]>df["home_goals"],"A","D"))
P(f"Toplam mac: {len(df):,} (mevcut {len(base):,} + yeni {len(sofa):,})")
P(f"Tarih: {df['date'].min().date()} -> {df['date'].max().date()}")

N = len(df)
home_id = df["home_team_id"].astype(str).values
away_id = df["away_team_id"].astype(str).values
hg = df["home_goals"].values
ag = df["away_goals"].values
hr = (hg > ag).astype(int)
ar = (ag > hg).astype(int)
dr = (hg == ag).astype(int)

# ---------- 2. ELO (vectorize) ----------
K = 32.0; HOME_ADV = 50.0
elo = {}
elo_h = np.zeros(N); elo_a = np.zeros(N)
for i in range(N):
    h, a = home_id[i], away_id[i]
    eh = elo.get(h, 1500.0); ea = elo.get(a, 1500.0)
    elo_h[i] = eh; elo_a[i] = ea
    exp_h = 1/(1+10**((ea-eh-HOME_ADV)/400))
    sh = hr[i] + 0.5*dr[i]; sa = ar[i] + 0.5*dr[i]
    elo[h] = eh + K*(sh-exp_h)
    elo[a] = ea + K*(sa-(1-exp_h))
P(f"ELO bitti ({time.time()-t0:.0f}s)")

# ---------- 3. FORM (per-team rolling, ev+deplasman ayri) ----------
team_hist = {}
def get_form(t, n):
    h = team_hist.get(t, [])
    if not h: return (0.0, 0.0, 0.0)
    sl = h[-n:]
    return (float(np.mean([x[0] for x in sl])), float(np.mean([x[1] for x in sl])),
            float(np.mean([x[2] for x in sl])))

hg5=np.zeros(N); ga5=np.zeros(N); pt5=np.zeros(N)
hg3=np.zeros(N); ga3=np.zeros(N); pt3=np.zeros(N)
hg8=np.zeros(N); ga8=np.zeros(N)
hg20=np.zeros(N); ga20=np.zeros(N)
ag5=np.zeros(N); aa5=np.zeros(N); ap5=np.zeros(N)
ag3=np.zeros(N); aa3=np.zeros(N); ap3=np.zeros(N)
ag8=np.zeros(N); aa8=np.zeros(N)
ag20=np.zeros(N); aa20=np.zeros(N)
for i in range(N):
    h,a = home_id[i], away_id[i]
    hf=get_form(h,5); af=get_form(a,5)
    hg5[i],ga5[i],pt5[i]=hf; ag5[i],aa5[i],ap5[i]=af
    hf3=get_form(h,3); af3=get_form(a,3)
    hg3[i],ga3[i],pt3[i]=hf3; ag3[i],aa3[i],ap3[i]=af3
    hf8=get_form(h,8); af8=get_form(a,8)
    hg8[i],ga8[i],_=hf8; ag8[i],aa8[i],_=af8
    hf20=get_form(h,20); af20=get_form(a,20)
    hg20[i],ga20[i],_=hf20; ag20[i],aa20[i],_=af20
    pts_h=hr[i]*3+dr[i]; pts_a=ar[i]*3+dr[i]
    team_hist.setdefault(h,[]).append((hg[i],ag[i],pts_h))
    team_hist.setdefault(a,[]).append((ag[i],hg[i],pts_a))
P(f"Form bitti ({time.time()-t0:.0f}s)")

# ---------- 4. LEAGUE BASELINES ----------
lg_stats = {}
league = df["league"].values
home_c = pd.to_numeric(df["home_corners"],errors="coerce").fillna(0).values
away_c = pd.to_numeric(df["away_corners"],errors="coerce").fillna(0).values
lg_avg_goals=np.full(N,2.5); lg_home=np.full(N,1.3); lg_away=np.full(N,1.2)
lg_draw=np.full(N,0.25); lg_btts=np.full(N,0.5); lg_o25=np.full(N,0.45)
for i in range(N):
    lg=league[i]
    s=lg_stats.setdefault(lg,{"n":0,"hg":0.0,"ag":0.0,"tg":0.0,"dr":0,"bt":0,"o25":0,"cr":0.0})
    if s["n"]>=5:
        lg_avg_goals[i]=s["tg"]/s["n"]; lg_home[i]=s["hg"]/s["n"]; lg_away[i]=s["ag"]/s["n"]
        lg_draw[i]=s["dr"]/s["n"]; lg_btts[i]=s["bt"]/s["n"]; lg_o25[i]=s["o25"]/s["n"]
    tot=hg[i]+ag[i]
    s["n"]+=1; s["hg"]+=hg[i]; s["ag"]+=ag[i]; s["tg"]+=tot
    s["dr"]+=dr[i]; s["bt"]+=1 if (hg[i]>0 and ag[i]>0) else 0
    s["o25"]+=1 if tot>=3 else 0
    s["cr"]+=home_c[i]+away_c[i]
P(f"League baselines bitti ({time.time()-t0:.0f}s)")

# ---------- 5. MARKET PROBS ----------
def impl(o):
    o=np.asarray(o,dtype=float)
    return np.where(o>1.01, 1.0/o, np.nan)
mh=impl(df["avg_home_odds"].values); md=impl(df["avg_draw_odds"].values)
ma=impl(df["avg_away_odds"].values); mo=impl(df["avg_under25_odds"].values)

# ---------- 6. HEDEFLER ----------
total=hg+ag
btts=((hg>0)&(ag>0)).astype(int)
over15=(total>1.5).astype(int); over25=(total>2.5).astype(int); over35=(total>3.5).astype(int)
under15=1-over15; under25=1-over25; under35=1-over35
res=df["result"].values
double_1X=((res=="H")|(res=="D")).astype(int)
double_X2=((res=="D")|(res=="A")).astype(int)
double_12=((res=="H")|(res=="A")).astype(int)
total_corners=home_c+away_c
corners_85=(total_corners>=9).astype(int); corners_95=(total_corners>=10).astype(int)
corners_under85=1-corners_85; corners_under95=1-corners_95

# ---------- 7. CIKTI ----------
out = pd.DataFrame({
    "match_id": df["match_id"].values if "match_id" in df else np.arange(N),
    "league": league, "season": df["season"].values if "season" in df else "",
    "date": df["date"].values,
    "home_team_id": df["home_team_id"].astype(str).values, "away_team_id": df["away_team_id"].astype(str).values,
    "home_elo": elo_h, "away_elo": elo_a, "elo_diff": elo_h-elo_a+HOME_ADV,
    "home_gf_5": hg5, "home_ga_5": ga5, "home_pts_5": pt5,
    "away_gf_5": ag5, "away_ga_5": aa5, "away_pts_5": ap5,
    "home_gf_3": hg3, "home_ga_3": ga3, "home_pts_3": pt3,
    "away_gf_3": ag3, "away_ga_3": aa3, "away_pts_3": ap3,
    "home_gf_8": hg8, "home_ga_8": ga8,
    "away_gf_8": ag8, "away_ga_8": aa8,
    "home_gf_20": hg20, "home_ga_20": ga20,
    "away_gf_20": ag20, "away_ga_20": aa20,
    "lg_avg_goals": lg_avg_goals, "lg_home_goal_avg": lg_home, "lg_away_goal_avg": lg_away,
    "lg_draw_rate": lg_draw, "lg_btts_rate": lg_btts, "lg_over25_rate": lg_o25,
    "mkt_home_prob": mh, "mkt_draw_prob": md, "mkt_away_prob": ma, "mkt_over25_prob": mo,
    "home_score_prob": np.where(lg_avg_goals>0, lg_home/lg_avg_goals, 0.5),
    "away_score_prob": np.where(lg_avg_goals>0, lg_away/lg_avg_goals, 0.5),
    "btts_xprob": lg_btts, "over25_xprob": lg_o25, "draw_xprob": lg_draw,
    "home_goals": hg, "away_goals": ag, "result": res,
    "btts": btts, "over15": over15, "over25": over25, "over35": over35,
    "under15": under15, "under25": under25, "under35": under35,
    "double_1X": double_1X, "double_X2": double_X2, "double_12": double_12,
    "total_goals": total,
    "home_corners": home_c, "away_corners": away_c,
    "total_corners": total_corners,
    "corners_85": corners_85, "corners_95": corners_95,
    "corners_under85": corners_under85, "corners_under95": corners_under95,
    "home_xg": pd.to_numeric(df["home_xg"],errors="coerce").values,
    "away_xg": pd.to_numeric(df["away_xg"],errors="coerce").values,
    "home_shots": pd.to_numeric(df["home_shots"],errors="coerce").values,
    "away_shots": pd.to_numeric(df["away_shots"],errors="coerce").values,
    "home_sot": pd.to_numeric(df["home_sot"],errors="coerce").values,
    "away_sot": pd.to_numeric(df["away_sot"],errors="coerce").values,
    "home_yellow": pd.to_numeric(df["home_yellow"],errors="coerce").values,
    "away_yellow": pd.to_numeric(df["away_yellow"],errors="coerce").values,
})
out.to_parquet("data/gold/features_fast.parquet", index=False)
P(f"Yazildi: data/gold/features_fast.parquet ({out.shape}) ({time.time()-t0:.0f}s)")
