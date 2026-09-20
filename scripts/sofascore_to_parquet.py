"""
SofaScore JSON → features_combined.parquet Dönüştürücü
Takimlar/ altındaki tüm takım JSON'larını parse edip
features_combined.parquet'teki sf_* sütunlarını doldurur.
"""
import json, os, sys, glob, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# === CONFIG ===
TAKIMLAR_DIR = r"C:\Users\furka\Desktop\sofascore-scraper-main\Takimlar"
PARQUET_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_combined.parquet"
NAME_MAP_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\team_id_to_name.json"
OUTPUT_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_combined.parquet"
STATS_KEY_VARIANTS = ["İstatistikler", "Istatistikler", "istatistikler"]

# === LOAD name map ===
with open(NAME_MAP_PATH, encoding="utf-8") as f:
    name_map_raw = json.load(f)
name_to_id = {}
for k, v in name_map_raw.items():
    name_to_id[v.lower().strip()] = int(k)
    name_to_id[v.strip()] = int(k)

def find_team_id(sofa_name):
    """SofaScore takım ismini → parquet team_id'ye çevir"""
    name_clean = sofa_name.strip()
    if name_clean in name_to_id:
        return name_to_id[name_clean]
    lower = name_clean.lower()
    if lower in name_to_id:
        return name_to_id[lower]
    for n, tid in name_to_id.items():
        if lower in n.lower() or n.lower() in lower:
            return tid
    return None

def safe_int(val, default=0):
    if val is None: return default
    s = str(val).strip().replace("%", "").replace(" km", "")
    s = re.sub(r'[^\d\-\.]', '', s.split("/")[0].split(" ")[0])
    try: return int(float(s))
    except: return default

def safe_float(val, default=0.0):
    if val is None: return default
    s = str(val).strip().replace("%", "").replace(" km", "")
    s = re.sub(r'[^\d\-\.]', '', s.split("/")[0].split(" ")[0])
    try: return float(s)
    except: return default

def get_stats(match_data, team_name):
    """Takım istatistiklerini al (Türkçe/İngilizce anahtar desteği)"""
    team_section = match_data.get(team_name, {})
    if not team_section:
        return {}
    for key in STATS_KEY_VARIANTS:
        stats = team_section.get(key)
        if stats and isinstance(stats, dict) and len(stats) > 0:
            return stats
    return {}

def parse_match(match):
    """Bir maç JSON'unu → dict (parquet satırına eklenecek değerler)"""
    md = match.get("match_data", {})
    info = md.get("Match_Info", {})
    if not info:
        return None

    # Temel bilgiler
    home_name = info.get("Home", "")
    away_name = info.get("Away", "")
    score = info.get("Score", "0 - 0")
    timestamp = info.get("Timestamp", 0)
    tournament = info.get("Tournament", "")

    try:
        parts = score.split(" - ")
        home_goals = int(parts[0].strip())
        away_goals = int(parts[1].strip())
    except:
        return None

    match_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
    home_id = find_team_id(home_name)
    away_id = find_team_id(away_name)

    if not home_id or not away_id:
        return None

    # İstatistikler
    h_stats = get_stats(md, home_name)
    a_stats = get_stats(md, away_name)

    h_overview = h_stats.get("Match overview", {}) if h_stats else {}
    a_overview = a_stats.get("Match overview", {}) if a_stats else {}
    h_shots = h_stats.get("Shots", {}) if h_stats else {}
    a_shots = a_stats.get("Shots", {}) if a_stats else {}
    h_attack = h_stats.get("Attack", {}) if h_stats else {}
    a_attack = a_stats.get("Attack", {}) if a_stats else {}
    h_defend = h_stats.get("Defending", {}) if h_stats else {}
    a_defend = a_stats.get("Defending", {}) if a_stats else {}
    h_gk = h_stats.get("Goalkeeping", {}) if h_stats else {}
    a_gk = a_stats.get("Goalkeeping", {}) if a_stats else {}

    # Possession parse: "68%" → 0.68
    h_poss = safe_float(h_overview.get("Ball possession", None)) / 100.0 if h_overview.get("Ball possession") else None
    a_poss = safe_float(a_overview.get("Ball possession", None)) / 100.0 if a_overview.get("Ball possession") else None

    return {
        "date": match_date,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_goals": home_goals,
        "away_goals": away_goals,
        # Match overview
        "home_shots": safe_int(h_overview.get("Total shots")),
        "away_shots": safe_int(a_overview.get("Total shots")),
        "home_corners": safe_int(h_overview.get("Corner kicks")),
        "away_corners": safe_int(a_overview.get("Corner kicks")),
        "home_yellow": safe_int(h_overview.get("Yellow cards")),
        "away_yellow": safe_int(a_overview.get("Yellow cards")),
        "home_red": safe_int(h_overview.get("Red cards", "0")),
        "away_red": safe_int(a_overview.get("Red cards", "0")),
        # SofaScore specific
        "sf_home_possession": h_poss,
        "sf_away_possession": a_poss,
        "sf_home_xg": safe_float(h_overview.get("Expected goals")) or safe_float(h_shots.get("Expected goals")),
        "sf_away_xg": safe_float(a_overview.get("Expected goals")) or safe_float(a_shots.get("Expected goals")),
        "sf_home_xg_on_target": safe_float(h_shots.get("Expected goals on target")),
        "sf_away_xg_on_target": safe_float(a_shots.get("Expected goals on target")),
        # Shots detail
        "home_sot": safe_int(h_shots.get("Shots on target")),
        "away_sot": safe_int(a_shots.get("Shots on target")),
        # Attack
        "sf_home_big_chances_scored": safe_int(h_attack.get("Big chances scored")),
        "sf_away_big_chances_scored": safe_int(a_attack.get("Big chances scored")),
        "sf_home_big_chances_missed": safe_int(h_attack.get("Big chances missed")),
        "sf_away_big_chances_missed": safe_int(a_attack.get("Big chances missed")),
        "sf_home_touches_pen": safe_int(h_attack.get("Touches in penalty area")),
        "sf_away_touches_pen": safe_int(a_attack.get("Touches in penalty area")),
        # Defending
        "sf_home_tackles": safe_int(h_defend.get("Total tackles") or h_defend.get("Tackles won")),
        "sf_away_tackles": safe_int(a_defend.get("Total tackles") or a_defend.get("Tackles won")),
        "sf_home_interceptions": safe_int(h_defend.get("Interceptions")),
        "sf_away_interceptions": safe_int(a_defend.get("Interceptions")),
        "sf_home_recoveries": safe_int(h_defend.get("Recoveries")),
        "sf_away_recoveries": safe_int(a_defend.get("Recoveries")),
        # Goalkeeping
        "sf_home_saves": safe_int(h_gk.get("Total saves")),
        "sf_away_saves": safe_int(a_gk.get("Total saves")),
        "sf_home_goals_prevented": safe_float(h_gk.get("Goals prevented")),
        "sf_away_goals_prevented": safe_float(a_gk.get("Goals prevented")),
        # Fouls
        "home_fouls": safe_int(h_overview.get("Fouls")),
        "away_fouls": safe_int(a_overview.get("Fouls")),
    }

# === PHASE 1: Parse all JSON files ===
print("=" * 70)
print("  SofaScore JSON → features_combined.parquet DÖNÜŞTÜRÜCÜ")
print("=" * 70)

all_matches = []
json_files = glob.glob(os.path.join(TAKIMLAR_DIR, "**", "*.json"), recursive=True)
json_files = [f for f in json_files if "progress" not in f.lower() and "ligveri" not in f.lower()
              and "target" not in f.lower() and "preset" not in f.lower() and "cache" not in f.lower()]

print(f"\n  Toplam JSON dosyası: {len(json_files)}")

# Build a set of all parsed matches: {(date, home_id, away_id): stats_dict}
match_lookup = {}
processed = 0
errors = 0
skipped_no_id = 0
skipped_no_stats = 0

for i, jf in enumerate(json_files):
    if i % 500 == 0 and i > 0:
        print(f"  ... {i}/{len(json_files)} dosya işlendi ({len(match_lookup)} maç, {skipped_no_id} ID bulunamadı)")
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        matches = data.get("matches", [])
        for m in matches:
            parsed = parse_match(m)
            if parsed is None:
                errors += 1
                continue
            # Key: (date, home_id, away_id) - minute precision
            d = parsed["date"]
            key = (d.year, d.month, d.day, d.hour, d.minute, parsed["home_team_id"], parsed["away_team_id"])
            # Also try rounded to hour
            key_h = (d.year, d.month, d.day, d.hour, 0, parsed["home_team_id"], parsed["away_team_id"])
            # Store with both keys for matching flexibility
            match_lookup[key] = parsed
            match_lookup[key_h] = parsed
            processed += 1
    except Exception as e:
        errors += 1

print(f"\n  Parse tamamlandı:")
print(f"    İşlenen maç: {processed}")
print(f"    Hata: {errors}")
print(f"    Benzersiz maç anahtarı: {len(set((k[0],k[1],k[2],k[3],k[5],k[6]) for k in match_lookup.keys()))}")

# === PHASE 2: Merge with features_combined ===
print(f"\n  features_combined.parquet yükleniyor...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  Mevcut satır sayısı: {len(df)}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
for c in ["home_team_id", "away_team_id"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["home_team_id", "away_team_id"])
df["home_team_id"] = df["home_team_id"].astype(int)
df["away_team_id"] = df["away_team_id"].astype(int)

# SF columns to update
sf_cols = [
    "sf_home_possession", "sf_away_possession",
    "sf_home_xg", "sf_away_xg",
    "sf_home_xg_on_target", "sf_away_xg_on_target",
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
all_update_cols = sf_cols + stat_cols

# Ensure columns exist
for c in all_update_cols:
    if c not in df.columns:
        df[c] = np.nan

# === PHASE 3: Match and update ===
matched = 0
not_found = 0
already_has_data = 0

# Sample first 10000 rows to test
sample_size = min(len(df), 10000)
print(f"\n  İlk {sample_size} satırda eşleştirme test ediliyor...")

for idx in range(sample_size):
    row = df.iloc[idx]
    d = row["date"]
    if pd.isna(d):
        continue
    hid = int(row["home_team_id"])
    aid = int(row["away_team_id"])

    key = (d.year, d.month, d.day, d.hour, d.minute, hid, aid)
    key_h = (d.year, d.month, d.day, d.hour, 0, hid, aid)

    parsed = match_lookup.get(key) or match_lookup.get(key_h)
    if parsed is None:
        not_found += 1
        continue

    # Check if already has data
    has_any = False
    for c in sf_cols:
        if pd.notna(row.get(c)) and row.get(c) != 0:
            has_any = True
            break
    if has_any:
        already_has_data += 1
        continue

    # Update columns
    for c in all_update_cols:
        if c in parsed and parsed[c] is not None:
            df.at[df.index[idx], c] = parsed[c]
    matched += 1

print(f"    Eşleşen: {matched}")
print(f"    Bulunamayan: {not_found}")
print(f"    Zaten veri var: {already_has_data}")

# === PHASE 4: Full merge ===
print(f"\n  Tam merge başlatılıyor...")
full_matched = 0
full_not_found = 0

for idx in range(len(df)):
    row = df.iloc[idx]
    d = row["date"]
    if pd.isna(d):
        continue
    hid = int(row["home_team_id"])
    aid = int(row["away_team_id"])

    key = (d.year, d.month, d.day, d.hour, d.minute, hid, aid)
    key_h = (d.year, d.month, d.day, d.hour, 0, hid, aid)

    parsed = match_lookup.get(key) or match_lookup.get(key_h)
    if parsed is None:
        full_not_found += 1
        continue

    # Update columns (only if currently NaN or 0)
    for c in all_update_cols:
        if c in parsed and parsed[c] is not None:
            current = row.get(c)
            if pd.isna(current) or current == 0:
                df.at[df.index[idx], c] = parsed[c]
    full_matched += 1

    if full_matched % 50000 == 0:
        print(f"    ... {full_matched} satır güncellendi")

print(f"\n  Merge tamamlandı:")
print(f"    Güncellenen satır: {full_matched}")
print(f"    Bulunamayan: {full_not_found}")

# === PHASE 5: Save ===
print(f"\n  Kaydediliyor: {OUTPUT_PATH}")
df.to_parquet(OUTPUT_PATH, index=False)

# Report
print(f"\n  SONUÇ:")
for c in sf_cols + ["home_shots", "home_sot", "home_corners", "home_fouls"]:
    non_null = df[c].notna().sum()
    pct = 100 * non_null / len(df)
    print(f"    {c:40s}: {non_null:>8} ({pct:.1f}%)")

print(f"\n  Tamamlandı! {OUTPUT_PATH}")
