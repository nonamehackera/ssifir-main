"""TUM VERI KAYNAKLARINI BIRLESTIR - tek şema, leak-free, dedup.

Kaynaklar:
  - matches_all_ext (782K)  : temel, bahis oranlari var
  - sofascore (319K)        : xg/shots/corners (raw'dan)
  - xgabora (230K)          : xg/shots/sot/corners/faul/kart/odds
  - understat (21K)         : xg (premier league)
  - transfermarkt (88K)     : assists/players/attendance
  - openfootball (10K), statsbomb (4K)

Cikti: data/gold/matches_master.parquet  (tum kaynaklar birlesik, normalize)
"""
import sys, os, time
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

def load(p):
    try:
        return pd.read_parquet(p)
    except Exception as e:
        P(f"  ! {p}: {e}")
        return None

files = {
    "matches_all_ext": "data/gold/matches_all_ext.parquet",
    "sofascore": "data/bronze/sofascore_matches.parquet",
    "xgabora": "data/gold/xgabora_matches.parquet",
    "understat": "data/gold/understat_matches.parquet",
    "transfermarkt": "data/gold/transfermarkt_matches.parquet",
    "openfootball": "data/gold/openfootball_matches.parquet",
    "statsbomb": "data/gold/statsbomb_matches.parquet",
}

frames = []
for name, path in files.items():
    d = load(path)
    if d is None:
        continue
    d = d.copy()
    d["_src"] = name
    # normalize column names to canonical
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce")
    for c in ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
              "home_shots", "away_shots", "home_sot", "away_sot",
              "home_corners", "away_corners", "home_fouls", "away_fouls",
              "home_yellow", "away_yellow", "home_red", "away_red",
              "home_xg", "away_xg", "home_assists", "away_assists",
              "attendance", "home_minutes", "away_minutes"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # team ids -> string
    for c in ["home_team_id", "away_team_id"]:
        if c in d.columns:
            d[c] = d[c].astype(str)
    # lig
    if "league" not in d.columns and "league_name" in d.columns:
        d["league"] = d["league_name"]
    # bahis oranlari
    for base in ["avg_home_odds", "avg_draw_odds", "avg_away_odds",
                 "b365_home_odds", "b365_draw_odds", "b365_away_odds",
                 "avg_over25_odds", "avg_under25_odds"]:
        if base in d.columns:
            d[base] = pd.to_numeric(d[base], errors="coerce")
    # result
    if "result" not in d.columns and "home_goals" in d.columns and "away_goals" in d.columns:
        d["result"] = np.where(d["home_goals"] > d["away_goals"], "H",
                       np.where(d["away_goals"] > d["home_goals"], "A", "D"))
    # total
    if "total_goals" not in d.columns and "home_goals" in d.columns:
        d["total_goals"] = d["home_goals"].fillna(0) + d["away_goals"].fillna(0)
    frames.append(d)
    P(f"  {name:<14} {len(d):>8} satir, kolon={len(d.columns)}")

P(f"\nToplam kaynak: {len(frames)}")
all_df = pd.concat(frames, ignore_index=True, sort=False)
P(f"Birlestirildi: {len(all_df):,} satir")

# ---- DEDUP ----
#ayni (home_team_id, away_team_id, date) olanlari tek tut (ilk kaynagi koru)
key = all_df["home_team_id"].astype(str) + "_" + all_df["away_team_id"].astype(str) + "_" + all_df["date"].astype(str)
before = len(all_df)
all_df = all_df.drop_duplicates(subset=["home_team", "away_team", "date"], keep="first")
after_dedup = len(all_df)
P(f"Dedup: {before:,} -> {after_dedup:,} (silinen {before-after_dedup:,})")

# tarih normalize (tz uyumsuzlugu gider)
all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce", utc=True).dt.tz_localize(None)
# tarih gecersiz olanlari at
all_df = all_df[all_df["date"].notna()]
all_df = all_df[all_df["date"] >= pd.Timestamp("2000-01-01")]
P(f"Gecerli tarihli: {len(all_df):,}")

# skor gecersiz olanlari at (NaN skor = antrenman icin uygun degil)
all_df = all_df[all_df["home_goals"].notna() & all_df["away_goals"].notna()]
P(f"Skorlu (train icin): {len(all_df):,}")

all_df = all_df.sort_values("date").reset_index(drop=True)
# gereksiz/karisik tip column'lari at
for drop_c in ["match_id", "index", "round"]:
    if drop_c in all_df.columns:
        all_df = all_df.drop(columns=[drop_c])
all_df.to_parquet("data/gold/matches_master.parquet")
P(f"\nKaydedildi: data/gold/matches_master.parquet ({len(all_df):,} satir)")
P(f"Sure: {time.time()-t0:.0f}s")

# ozet
P(f"\nKaynak dagilimi:")
P(all_df["_src"].value_counts().to_string())
P(f"\nLig sayisi: {all_df['league'].nunique()}")
P(f"Tarih araligi: {all_df['date'].min()} -> {all_df['date'].max()}")
P(f"xG dolulugu: {all_df['home_xg'].notna().sum():,} / {len(all_df):,}")
P(f"Korner dolulugu: {all_df['home_corners'].notna().sum():,} / {len(all_df):,}")
