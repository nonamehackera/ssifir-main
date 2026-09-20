#!/usr/bin/env python3
"""Haftalik veri guncelleme + model yeniden egitim."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

print("Haftalik guncelleme basliyor...")
t0 = time.time()

# 1. Yeni veri indir (Football-Data.co.uk + Transfermarkt)
# TODO: football_data_ingestion'dan en guncel veriyi cek

# 2. Mevcut matches_all.parquet'e ekle
# all_m = pd.read_parquet("data/gold/matches_all.parquet")
# Yeni verileri ekle, dedup yap

# 3. Feature engine calistir (leakage-free)
# ELO + form + H2H etc.

# 4. Modeli yeniden egit
# python train_v2.py

# 5. Test et
# python test_v2.py

print(f"Tamamlandi: {time.time()-t0:.0f}s")
