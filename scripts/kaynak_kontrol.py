import pandas as pd
us = pd.read_parquet("data/gold/understat_matches.parquet")
us["date"] = pd.to_datetime(us["date"])
us = us.sort_values("date")
print("Understat ozet:")
print(f"  Ligler: {list(us['league'].unique())}")
print(f"  Tarih: {us['date'].min().date()} -> {us['date'].max().date()}")
print(f"  Mac: {len(us)}")
print(f"  Kolonlar: {list(us.columns)}")
for c in us.columns:
    if "xg" in c.lower():
        print(f"  {c}: {us[c].notna().sum()} non-null")

# FBRef kontrol
import os, glob
fb = glob.glob("data/raw/fbref/**/*", recursive=True)
print(f"\nFBRef dosyalari: {len(fb)}")
for f in fb[:10]:
    print(f"  {f}")

# api-football kontrol
af = glob.glob("data/raw/**/api*", recursive=True)
print(f"\nAPI-Football: {len(af)}")
for f in af[:5]:
    print(f"  {f}")

# TheSportsDB
ts = glob.glob("data/raw/**/thesports*", recursive=True)
print(f"\nTheSportsDB: {len(ts)}")
