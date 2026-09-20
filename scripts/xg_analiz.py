import pandas as pd, numpy as np

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"])

# xG durumu
xg = pd.to_numeric(feat["home_xg"], errors="coerce")
xg_real = pd.to_numeric(feat["home_xg_real"], errors="coerce")
print("xG DURUMU:")
print(f"  home_xg (model): {xg.notna().sum()}/{len(feat)} ({xg.notna().mean():.1%})")
print(f"  home_xg_real: {xg_real.notna().sum()}/{len(feat)} ({xg_real.notna().mean():.1%})")

# Understat xG features.parquet'de var mi?
us_xg = feat[feat["league"].str.startswith("US_")]
print(f"\nUnderstat ligleri features.parquet'de: {len(us_xg)} mac")
if len(us_xg) > 0:
    print(f"  xG dolu: {pd.to_numeric(us_xg['home_xg'], errors='coerce').notna().sum()}")

# HF liglerinde xG
hf = feat[feat["league"].str.startswith("HF_")]
hf_xg = pd.to_numeric(hf["home_xg"], errors="coerce")
print(f"\nHF ligleri: {len(hf)} mac, xG olan: {hf_xg.notna().sum()} ({hf_xg.notna().mean():.1%})")

# understat parquet
us = pd.read_parquet("data/gold/understat_matches.parquet")
us["date"] = pd.to_datetime(us["date"])
print(f"\nUnderstat parquet: {len(us)} mac")
print(f"  xG: {us['home_xg'].notna().sum()}")
print(f"  Tarih: {us['date'].min().date()} -> {us['date'].max().date()}")

# understat -> features.parquet eslesme kontrolu
# Takim isimleri uusuyor mu?
hf_sample = feat[feat["league"].str.startswith("US_")].head(5)
print(f"\nOrnek Understat satir:")
for _, r in hf_sample.iterrows():
    hx = pd.to_numeric(r.get("home_xg"), errors="coerce")
    print(f"  {r['date'].date()} {r.get('home_team_id','?')} vs {r.get('away_team_id','?')} xG={hx}")
