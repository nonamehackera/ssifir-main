"""matches.parquet durumu - hangi kaynaktan kac mac, oran/istatistik dolulugu"""
import pandas as pd, numpy as np

m = pd.read_parquet("data/gold/matches.parquet")
m["date"] = pd.to_datetime(m["date"])
m = m.sort_values("date").reset_index(drop=True)

# Kaynak tespiti
def detect_source(league):
    if str(league).startswith("XG_"): return "xgabora"
    if str(league).startswith("FB_"): return "fbref"
    if str(league).startswith("SB_"): return "statsbomb"
    if str(league).startswith("US_"): return "understat"
    if str(league).startswith("HF_"): return "huggingface"
    if str(league).startswith("OF_"): return "openfootball"
    if str(league).startswith("TS_"): return "thesportsdb"
    return "football_data"  # E0, SP1, T1, etc.

m["source"] = m["league"].apply(detect_source)

# Her kaynak icin istatistik
print("KAYNAK ANALIZI:")
print(f"{'Kaynak':<15} {'Mac':>7} {'Oran%':>7} {'Sut%':>7} {'Korner%':>7} {'xG%':>7} {'Tarih':>25}")
print("-" * 80)

for src in ["football_data", "xgabora", "fbref", "statsbomb", "understat", "huggingface", "openfootball", "thesportsdb"]:
    sub = m[m["source"] == src]
    if len(sub) == 0:
        continue
    odds_pct = pd.to_numeric(sub["avg_home_odds"], errors="coerce").notna().mean() * 100
    shots_pct = pd.to_numeric(sub["home_shots"], errors="coerce").notna().mean() * 100
    corner_pct = pd.to_numeric(sub["home_corners"], errors="coerce").notna().mean() * 100
    xg_pct = pd.to_numeric(sub["home_xg"], errors="coerce").notna().mean() * 100
    date_range = f"{sub['date'].min().date()} -> {sub['date'].max().date()}"
    print(f"{src:<15} {len(sub):>7} {odds_pct:>6.1f}% {shots_pct:>6.1f}% {corner_pct:>6.1f}% {xg_pct:>6.1f}% {date_range:>25}")

print(f"\nTOPLAM: {len(m)} mac")

# HF detay - istatistik dolulugu
hf = m[m["source"] == "huggingface"]
print(f"\nHF DETAY ({len(hf)} mac):")
for c in ["home_shots", "home_sot", "home_corners", "home_fouls", "home_xg", "avg_home_odds"]:
    if c in hf.columns:
        pct = pd.to_numeric(hf[c], errors="coerce").notna().mean() * 100
        print(f"  {c}: {pct:.1f}%")

# Son 2000 test penceresi
test = m.iloc[-2000:]
print(f"\nSON 2000 TEST PENCERESI:")
for src in test["source"].value_counts().index:
    sub = test[test["source"] == src]
    print(f"  {src}: {len(sub)} mac")
