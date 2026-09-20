"""FD ligleri vs HF ligleri - features.parquet analiz"""
import pandas as pd, numpy as np

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"])
feat = feat.sort_values("date").reset_index(drop=True)

# FD ligleri (football-data.co.uk)
FD = {"E0","E1","E2","E3","EC","SP1","SP2","D1","D2","I1","I2","F1","F2",
      "N1","T1","P1","P2","B1","G1","SC0","SC1","SC2","SC3","DK1","DK2",
      "SE1","NO1","PL1","IR1","IS1"}

feat["is_fd"] = feat["league"].isin(FD)
feat["is_hf"] = feat["league"].str.startswith("HF_")

oh = pd.to_numeric(feat["avg_home_odds"], errors="coerce")
feat["has_odds"] = oh.notna()

print("FD vs HF KARSILASTIRMASI")
print("="*60)
print(f"  FD mac sayisi: {feat['is_fd'].sum()} ({feat['is_fd'].mean():.1%})")
print(f"  HF mac sayisi: {feat['is_hf'].sum()} ({feat['is_hf'].mean():.1%})")
print(f"  Diger: {len(feat)-feat['is_fd'].sum()-feat['is_hf'].sum()}")

# FD icin oran durumu
fd = feat[feat["is_fd"]]
hf = feat[feat["is_hf"]]
print(f"\nFD: {len(fd)} mac, oran olan: {fd['has_odds'].sum()} ({fd['has_odds'].mean():.1%})")
print(f"HF: {len(hf)} mac, oran olan: {hf['has_odds'].sum()} ({hf['has_odds'].mean():.1%})")

# FD lig bazinda
print("\nFD LIG BAZINDA:")
for lg in sorted(fd["league"].unique()):
    sub = fd[fd["league"]==lg]
    odds = sub["has_odds"].sum()
    print(f"  {lg:5s}: {len(sub):>5} mac  oran:{odds:>5} ({odds/len(sub)*100:.0f}%)")

# Son 2000'de FD
test = feat.iloc[-2000:]
print(f"\nSON 2000 MAC:")
print(f"  FD: {test['is_fd'].sum()}")
print(f"  HF: {test['is_hf'].sum()}")
print(f"  Oran olan: {test['has_odds'].sum()}")
fd_test = test[test["is_fd"]]
print(f"  FD testte oran olan: {fd_test['has_odds'].sum()}/{len(fd_test)}")

# Tarih araliklari
print("\nTARIH ARALIKLARI:")
print(f"  FD: {fd['date'].min().date()} -> {fd['date'].max().date()}")
print(f"  HF: {hf['date'].min().date()} -> {hf['date'].max().date()}")
