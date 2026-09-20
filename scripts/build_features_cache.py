"""Tek seferlik feature build + cache (bet_eval_full bunu okur).

Veri: FD + HF[odds_only] + xgabora. Feature engine'i bir kez calistir,
data/gold/features.parquet'a yazar. Sonra bet_eval sadece okur (hizli).
"""
import sys, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
sys.path.insert(0, ".")

import pandas as pd
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features

def main():
    t0 = time.time()
    m = build_matches(use_thesportsdb=False, use_openfootball=False, hf_odds_only=True, min_year=2021)
    print(f"matches: {len(m)}  lig: {m['league'].nunique()}")
    f = build_features(m)
    print(f"features: {f.shape[0]} x {f.shape[1]}  -> {time.time()-t0:.0f}s")
    print("ilk 5 kolon ornek:", list(f.columns)[:5])

if __name__ == "__main__":
    main()
