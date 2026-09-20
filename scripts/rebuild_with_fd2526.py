"""FD 2025/26 ekleyerek tam rebuild."""
import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

import pandas as pd
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features

t0 = time.time()
print("1/2: matches.parquet yeniden insa ediliyor...")
matches = build_matches(
    use_huggingface=True,
    use_openfootball=True,
    use_thesportsdb=False,
    use_fbref=False,
    min_year=2015,
)
t1 = time.time()
print(f"  Matches tamam: {len(matches)} mac ({t1-t0:.0f}s)")

# FD 2025/26 kac mac eklendi?
fd_2526 = matches[(matches["league"].isin(["E0","SP1","D1","I1","F1"])) & (matches["season"]=="2025")]
print(f"  FD 2025/26: {len(fd_2526)} mac")

# Test seti kaynak dagilimi
test = matches.iloc[-1000:]
print(f"\n  Test seti (son 1000):")
for prefix in ["FD", "HF_", "XG_", "SB_", "US_", "OF_"]:
    if prefix == "FD":
        n = test[~test["league"].str.startswith(("HF_","OF_","TS_","FB_","XG_","SB_","US_"), na=False)]
    else:
        n = test[test["league"].str.startswith(prefix, na=False)]
    if len(n) > 0:
        print(f"    {prefix:>4}: {len(n):>4} mac")

# Test seti veri kalitesi
print(f"\n  Test seti veri kalitesi:")
print(f"    xG: {test['home_xg'].notna().sum()}/1000")
print(f"    Corner: {test['home_corners'].notna().sum()}/1000")
print(f"    Shot: {test['home_shots'].notna().sum()}/1000")
print(f"    Odds: {test['avg_home_odds'].notna().sum()}/1000")
print(f"    Close Odds: {test.get('avg_close_home_odds', pd.Series()).notna().sum()}/1000")

print(f"\n2/2: features.parquet yeniden insa ediliyor...")
feats = build_features(matches)
t2 = time.time()
print(f"  Features tamam: {len(feats)} satir, {feats.shape[1]} kolon ({t2-t1:.0f}s)")

# Test seti ozeti
test_f = feats.iloc[-1000:]
print(f"\n  Test seti ozeti:")
print(f"    xG: {test_f['home_xg'].notna().sum()}/1000")
print(f"    Corner: {test_f['home_corners'].notna().sum()}/1000")
print(f"    Odds: {test_f['avg_home_odds'].notna().sum()}/1000")
print(f"    Close Odds: {test_f.get('mkt_close_home_prob', pd.Series()).notna().sum()}/1000")
print(f"  Toplam sure: {t2-t0:.0f}s")
