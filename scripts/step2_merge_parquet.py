"""
SofaScore → Parquet ADIM 2: CSV merge → features_combined.parquet
Step1'de parse edilen CSV'yi features_combined.parquet ile birleştirir.
"""
import pandas as pd
import numpy as np
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
start = time.time()

CSV_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\sofascore_parsed.csv"
PARQUET_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_combined.parquet"
OUTPUT_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_combined.parquet"

print("CSV yükleniyor...")
sofa = pd.read_csv(CSV_PATH)
print(f"SofaScore CSV: {len(sofa)} satır")
print(f"Benzersiz takım çifti: {sofa[['home_team_id','away_team_id']].drop_duplicates().shape[0]}")

print("\nfeatures_combined yükleniyor...")
df = pd.read_parquet(PARQUET_PATH)
print(f"features_combined: {len(df)} satır, {df.shape[1]} sütun")

# Prepare df
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for c in ["home_team_id", "away_team_id"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["home_team_id", "away_team_id"])
df["home_team_id"] = df["home_team_id"].astype(int)
df["away_team_id"] = df["away_team_id"].astype(int)

# Ensure all sf_ columns exist
sf_cols = [
    "sf_home_possession", "sf_away_possession",
    "sf_home_xg", "sf_away_xg", "sf_home_xg_on_target", "sf_away_xg_on_target",
    "sf_home_big_chances_scored", "sf_away_big_chances_scored",
    "sf_home_big_chances_missed", "sf_away_big_chances_missed",
    "sf_home_touches_pen", "sf_away_touches_pen",
    "sf_home_tackles", "sf_away_tackles",
    "sf_home_interceptions", "sf_away_interceptions",
    "sf_home_recoveries", "sf_away_recoveries",
    "sf_home_saves", "sf_away_saves",
    "sf_home_goals_prevented", "sf_away_goals_prevented",
]
stat_cols = [
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "home_fouls", "away_fouls",
]
update_cols = sf_cols + stat_cols
for c in update_cols:
    if c not in df.columns:
        df[c] = np.nan

# Prepare sofa
sofa["date"] = pd.to_datetime(sofa["date"], errors="coerce")

# Build lookup from sofa
print("\nSofaScore lookup tablosu oluşturuluyor...")
# Key: (date_key, home_id, away_id) → row dict
# Use date rounded to hour for fuzzy matching
sofa["date_key精确"] = sofa["date"].dt.strftime("%Y-%m-%d %H")
sofa["date_key小时"] = sofa["date"].dt.strftime("%Y-%m-%d %H:00")

# Build dict: (date_hour, home_id, away_id) → sofa row
lookup = {}
for _, row in sofa.iterrows():
    d = row["date"]
    if pd.isna(d): continue
    key = (d.year, d.month, d.day, d.hour, int(row["home_team_id"]), int(row["away_team_id"]))
    key_h = (d.year, d.month, d.day, d.hour, int(row["home_team_id"]), int(row["away_team_id"]))
    vals = {c: row[c] for c in update_cols if c in row.index}
    lookup[key] = vals

print(f"Lookup boyutu: {len(lookup)} benzersiz maç")

# Merge
print("\nMerge başlıyor...")
matched = 0
updated = 0
not_found = 0

for idx in range(len(df)):
    row = df.iloc[idx]
    d = row["date"]
    if pd.isna(d):
        not_found += 1; continue
    hid = int(row["home_team_id"])
    aid = int(row["away_team_id"])
    key = (d.year, d.month, d.day, d.hour, hid, aid)

    vals = lookup.get(key)
    if vals is None:
        not_found += 1; continue
    matched += 1

    # Update only NaN/0 values
    for c in update_cols:
        v = vals.get(c)
        if v is not None and v != 0:
            current = row.get(c)
            if pd.isna(current) or current == 0:
                df.at[df.index[idx], c] = v
                updated += 1

    if matched % 50000 == 0:
        print(f"  ... {matched} eşleşti, {updated} güncellendi")

print(f"\nMerge tamam:")
print(f"  Eşleşen: {matched}")
print(f"  Güncellenen hücre: {updated}")
print(f"  Bulunamayan: {not_found}")

# Save
print(f"\nKaydediliyor: {OUTPUT_PATH}")
df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

# Report
print(f"\nGüncelleme sonrası:")
for c in update_cols:
    non_null = df[c].notna().sum()
    nonzero = (df[c] > 0).sum()
    pct = 100 * non_null / len(df)
    print(f"  {c:40s}: {non_null:>8} dolu ({pct:.1f}%) | {nonzero:>8} sıfırdan farklı")

elapsed = time.time() - start
print(f"\nToplam süre: {elapsed:.0f} saniye")
