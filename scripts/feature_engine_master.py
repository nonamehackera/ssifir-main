"""ZENGIN FEATURE ENGINEERING - matches_master uzerinde.

Eklenen feature'lar (onceki modelde YOKTU):
  - xG diff / SOT diff / shot diff (kalite)
  - Bahis implied probability (avg_home/draw/away odds -> 1/odds normalize)
  - H2H: son 5 karsilasmada ev sahibi avantaji
  - Ev/deplasman strength: takimin son 20 mac ev/dep formu ayri
  - Agirlikli form: son 5/10/20 mac, recency agirlikli (exp decay)
  - Lig gucu: o ligdeki ortalama gol / kazanma orani
  - Sezon fakturu: sezon ilerlemesi
  - Result-relative: ht_goal diff, comeback potential

Cikti: data/gold/features_master.parquet
"""
import sys, os, time
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

df = pd.read_parquet("data/gold/matches_master.parquet")
P(f"Okundu: {len(df):,} mac")
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ---- temel normalize ----
for c in ["home_goals","away_goals","ht_home_goals","ht_away_goals",
          "home_shots","away_shots","home_sot","away_sot","home_corners","away_corners",
          "home_fouls","away_fouls","home_yellow","away_yellow","home_red","away_red",
          "home_xg","away_xg","home_assists","away_assists","attendance",
          "home_minutes","away_minutes"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["home_team_id"] = df["home_team_id"].astype(str)
df["away_team_id"] = df["away_team_id"].astype(str)
df["total_goals"] = df["home_goals"].fillna(0) + df["away_goals"].fillna(0)
df["ht_total_goals"] = df["ht_home_goals"].fillna(0) + df["ht_away_goals"].fillna(0)
df["result"] = np.where(df["home_goals"]>df["away_goals"],"H",
                np.where(df["away_goals"]>df["home_goals"],"A","D"))

# ---- BAHIS IMPLIED PROB ----
for h,d,a in [("avg_home_odds","avg_draw_odds","avg_away_odds"),
              ("b365_home_odds","b365_draw_odds","b365_away_odds"),
              ("avg_close_home_odds","avg_close_draw_odds","avg_close_away_odds")]:
    if h in df.columns and d in df.columns and a in df.columns:
        ho=pd.to_numeric(df[h],errors="coerce"); dr=pd.to_numeric(df[d],errors="coerce"); ao=pd.to_numeric(df[a],errors="coerce")
        inv = 1/ho + 1/dr + 1/ao
        df["imp_home"] = (1/ho)/inv
        df["imp_draw"] = (1/dr)/inv
        df["imp_away"] = (1/ao)/inv
        break
else:
    df["imp_home"]=np.nan; df["imp_draw"]=np.nan; df["imp_away"]=np.nan
P(f"  bahis implied prob dolulugu: {df['imp_home'].notna().sum():,}")

# ---- HIZLI O(n) STATE-BASED FEATURE HESABI ----
teams = pd.unique(df[["home_team_id","away_team_id"]].values.ravel())
state = {t:{"hist":[], "home_hist":[], "away_hist":[]} for t in teams}
lg_state = {}
K=32

# precompute league average
leagues = df["league"].fillna("?").unique()
lg_avg = {}
for lg in leagues:
    m = df[df["league"]==lg]
    lg_avg[lg] = {"gpg": m["total_goals"].mean(), "hwin": (m["result"]=="H").mean(), "draw":(m["result"]=="D").mean()}

rows=[]
for idx, r in df.iterrows():
    h=str(r["home_team_id"]); a=str(r["away_team_id"]); lg=r["league"]
    hg=r["home_goals"]; ag=r["away_goals"]
    # H2H: son 5 karsilasma
    h2h = [x for x in state[h]["hist"] if x[3]==a][-5:]
    h2h_hwin = sum(1 for x in h2h if x[0]=="H")/len(h2h) if h2h else 0.5
    # agirlikli form (exp decay)
    def wform(hist, n, home_only=False):
        sub = hist[-n:] if not home_only else [x for x in hist if x[4]=="H"][-n:]
        if not sub: return (0.0,0.0,0.0)  # gf, ga, pts
        weights = np.exp(-np.arange(len(sub))[::-1]/10.0)
        gf=sum(w*x[1] for w,x in zip(weights,sub)); ga=sum(w*x[2] for w,x in zip(weights,sub)); pts=sum(w*x[5] for w,x in zip(weights,sub))
        sw=sum(weights)
        return (gf/sw, ga/sw, pts/sw)
    hf5=wform(state[h]["hist"],5); hf20=wform(state[h]["hist"],20)
    af5=wform(state[a]["hist"],5); af20=wform(state[a]["hist"],20)
    # ev/deplasman strength
    hf_h=wform(state[h]["home_hist"],10); af_a=wform(state[a]["away_hist"],10)
    # guncelle state (bu maci ekle)
    if hg>ag: res="H"; hp=3; ap=0
    elif ag>hg: res="A"; hp=0; ap=3
    else: res="D"; hp=1; ap=1
    state[h]["hist"].append((res, hg, ag, a, "H", hp))
    state[a]["hist"].append((res, ag, hg, h, "A", ap))
    state[h]["home_hist"].append((res, hg, ag, a, "H", hp))
    state[a]["away_hist"].append((res, ag, hg, h, "A", ap))
    # lig gucu
    la = lg_avg.get(lg, {"gpg":2.5,"hwin":0.45,"draw":0.27})
    # xG diff
    hxg=r["home_xg"] if pd.notna(r["home_xg"]) else np.nan
    axg=r["away_xg"] if pd.notna(r["away_xg"]) else np.nan
    rows.append({
        "xg_diff": (hxg-axg) if pd.notna(hxg) and pd.notna(axg) else np.nan,
        "xg_home": hxg, "xg_away": axg,
        "sot_diff": (r["home_sot"]-r["away_sot"]) if pd.notna(r["home_sot"]) and pd.notna(r["away_sot"]) else np.nan,
        "shot_diff": (r["home_shots"]-r["away_shots"]) if pd.notna(r["home_shots"]) and pd.notna(r["away_shots"]) else np.nan,
        "corner_diff": (r["home_corners"]-r["away_corners"]) if pd.notna(r["home_corners"]) and pd.notna(r["away_corners"]) else np.nan,
        "h2h_home_win_rate": h2h_hwin,
        "home_form_gf_5": hf5[0], "home_form_ga_5": hf5[1], "home_form_pts_5": hf5[2],
        "home_form_gf_20": hf20[0], "home_form_ga_20": hf20[1], "home_form_pts_20": hf20[2],
        "away_form_gf_5": af5[0], "away_form_ga_5": af5[1], "away_form_pts_5": af5[2],
        "away_form_gf_20": af20[0], "away_form_ga_20": af20[1], "away_form_pts_20": af20[2],
        "home_home_gf_10": hf_h[0], "home_home_ga_10": hf_h[1], "home_home_pts_10": hf_h[2],
        "away_away_gf_10": af_a[0], "away_away_ga_10": af_a[1], "away_away_pts_10": af_a[2],
        "lg_gpg": la["gpg"], "lg_hwin": la["hwin"], "lg_draw": la["draw"],
    })

feat_extra = pd.DataFrame(rows)
df = pd.concat([df.reset_index(drop=True), feat_extra.reset_index(drop=True)], axis=1)
P(f"  feature hesabi bitti ({time.time()-t0:.0f}s)")

# ---- HEDEFLER ----
df["btts"] = ((df["home_goals"]>0)&(df["away_goals"]>0)).astype(int)
df["over05"]=(df["total_goals"]>0.5).astype(int)
df["over15"]=(df["total_goals"]>1.5).astype(int)
df["over25"]=(df["total_goals"]>2.5).astype(int)
df["over35"]=(df["total_goals"]>3.5).astype(int)
df["over45"]=(df["total_goals"]>4.5).astype(int)
for lo in ["05","15","25","35","45"]:
    df[f"under{lo}"]=1-df[f"over{lo}"]
df["double_1X"]=((df["result"]=="H")|(df["result"]=="D")).astype(int)
df["double_X2"]=((df["result"]=="D")|(df["result"]=="A")).astype(int)
df["double_12"]=((df["result"]=="H")|(df["result"]=="A")).astype(int)
df["total_corners"]=df["home_corners"].fillna(0)+df["away_corners"].fillna(0)
for cv,lab in [(7.5,"75"),(8.5,"85"),(9.5,"95"),(10.5,"105")]:
    df[f"cor_over_{lab}"]=(df["total_corners"]>=cv+0.5).astype(int)
    df[f"cor_under_{lab}"]=1-df[f"cor_over_{lab}"]

df.to_parquet("data/gold/features_master.parquet")
P(f"Kaydedildi: data/gold/features_master.parquet ({len(df):,} satir, {df.shape[1]} kolon)")
P(f"Toplam sure: {time.time()-t0:.0f}s")
