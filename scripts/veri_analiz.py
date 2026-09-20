import pandas as pd, numpy as np
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"])
feat = feat.sort_values("date").reset_index(drop=True)

print("MEVCUT VERI KALITESI (tum veri seti, 514K mac)")
print("="*60)

cols_check = [
    ("Sonuc (H/D/A)", "result"),
    ("Elo", "elo_diff"),
    ("Form (son 5)", "home_gf_5"),
    ("Ort. oranlar", "avg_home_odds"),
    ("Kapanis oranlari", "avg_close_home_odds"),
    ("xG (gercek)", "home_xg_real"),
    ("xG (model)", "home_xg"),
    ("Korner (gercek)", "home_corners"),
    ("Kart (gercek)", "home_yellow"),
    ("Sut (gercek)", "home_shots"),
    ("SOT (gercek)", "home_sot"),
    ("HT gol", "ht_home_goals"),
    ("H2H", "h2h_goals_avg"),
    ("Momentum", "home_momentum"),
    ("Veri kalitesi", "data_completeness"),
    ("Mac sonu xG", "total_xg_real"),
]
for label, col in cols_check:
    if col in feat.columns:
        s = pd.to_numeric(feat[col], errors="coerce")
        nn = s.notna().sum()
        print(f"  {label:25s}: {nn:>7}/{len(feat)} ({nn/len(feat)*100:>5.1f}%)")
    else:
        print(f"  {label:25s}: YOK")

print(f"\nKAYNAK DAGILIMI:")
vc = feat["league"].value_counts()
print(f"  {vc.shape[0]} farkli lig")
for lg, cnt in vc.head(10).items():
    print(f"    {lg}: {cnt}")

print(f"\nTARIH: {feat['date'].min().date()} -> {feat['date'].max().date()}")
yr = feat["date"].dt.year
for y in range(2018, 2027):
    n = (yr==y).sum()
    if n > 0:
        print(f"  {y}: {n}")
