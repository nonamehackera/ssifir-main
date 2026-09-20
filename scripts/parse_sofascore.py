"""Sofascore 'Takimlar' JSON verisini standart matches şemasına çevir.

ÇIKTI: data/bronze/sofascore_matches.parquet
- Her mac iki takim dosyasinda da var -> (home,away,ts) ile dedup + istatistik birlestirme
- league: 'SOF|' on ekli lig adi
- team_id: global isim->id map (yeni takimlara yeni id)
- Eksik istatistik -> NaN (sonradan mevcut veriyle birlesir)
"""
import sys, os, json, glob, re, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import datetime, timezone

SRC = r"C:/Users/furka/Desktop/sofascore-scraper-main/Takimlar"
OUT = "data/bronze/sofascore_matches.parquet"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

idmap = {}
try:
    j = json.load(open("data/gold/team_id_to_name.json", encoding="utf-8"))
    for k, v in j.items():
        idmap[str(v).strip().lower()] = int(k)
except Exception as e:
    print("team map yuklenemedi:", e)

_next_id = (max(idmap.values()) + 1) if idmap else 100000

def get_id(name):
    global _next_id
    key = str(name).strip().lower()
    if key in idmap:
        return idmap[key]
    idmap[key] = _next_id
    _next_id += 1
    return idmap[key]

def num(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in ("", "-"):
        return np.nan
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else np.nan

def season_of(ts):
    try:
        ts = float(ts)
        if ts <= 0:
            raise ValueError
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        # gecersiz timestamp -> 1970 fallback (tarih siralamasi icin en eski)
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    y = dt.year
    s, e = (y, y + 1) if dt.month >= 7 else (y - 1, y)
    return f"{s%100:02d}{e%100:02d}", dt

def parse_team_stats(block):
    out = {}
    if not isinstance(block, dict):
        return out
    ist = block.get("Istatistikler", {})
    if not isinstance(ist, dict):
        return out
    mo = ist.get("Match overview", {})
    sh = ist.get("Shots", {})
    if isinstance(mo, dict):
        out["xg"] = num(mo.get("Expected goals"))
        out["shots"] = num(mo.get("Total shots"))
        out["corners"] = num(mo.get("Corner kicks"))
        out["yellow"] = num(mo.get("Yellow cards"))
        out["fouls"] = num(mo.get("Fouls"))
        out["poss"] = num(mo.get("Ball possession"))
        out["bigch"] = num(mo.get("Big chances"))
    if isinstance(sh, dict):
        if np.isnan(out.get("shots", np.nan)):
            out["shots"] = num(sh.get("Total shots"))
        out["sot"] = num(sh.get("Shots on target"))
        if np.isnan(out.get("xg", np.nan)):
            out["xg"] = num(sh.get("Expected goals"))
        out["xg_ot"] = num(sh.get("Expected goals on target"))
        out["bc_scored"] = num(sh.get("Big chances scored"))
        out["bc_missed"] = num(sh.get("Big chances missed"))
    return out

t0 = time.time()
files = [f for f in glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True)
         if os.path.basename(f) != "_progress.json"]
print("Dosya sayisi:", len(files))

rows = []
seen = {}

for fp in files:
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    lg = d.get("league_name", "")
    league = "SOF|" + lg
    for mt in d.get("matches", []):
        md = mt.get("match_data", {})
        mi = md.get("Match_Info", {})
        home, away = mi.get("Home"), mi.get("Away")
        score, ts = mi.get("Score"), mi.get("Timestamp")
        if not home or not away or ts is None or " - " not in str(score):
            continue
        try:
            hg, ag = str(score).split(" - ")
            hg, ag = int(float(hg)), int(float(ag))
        except Exception:
            continue
        season, dt = season_of(ts)
        key = (home, away, int(ts))
        hs, as_ = parse_team_stats(md.get(home, {})), parse_team_stats(md.get(away, {}))
        row = {
            "league": league, "league_name": lg, "season": season, "date": dt,
            "home_team": home, "away_team": away,
            "home_team_id": int(get_id(home)), "away_team_id": int(get_id(away)),
            "home_goals": hg, "away_goals": ag,
            "result": "H" if hg > ag else ("A" if ag > hg else "D"),
            "home_shots": hs.get("shots"), "away_shots": as_.get("shots"),
            "home_sot": hs.get("sot"), "away_sot": as_.get("sot"),
            "home_corners": hs.get("corners"), "away_corners": as_.get("corners"),
            "home_yellow": hs.get("yellow"), "away_yellow": as_.get("yellow"),
            "home_fouls": hs.get("fouls"), "away_fouls": as_.get("fouls"),
            "home_possession": hs.get("poss"), "away_possession": as_.get("poss"),
            "home_xg": hs.get("xg"), "away_xg": as_.get("xg"),
            "home_bigch": hs.get("bigch"), "away_bigch": as_.get("bigch"),
            "home_xg_ot": hs.get("xg_ot"), "away_xg_ot": as_.get("xg_ot"),
            "match_id": None,
        }
        if key in seen:
            idx = seen[key]
            for c, v in row.items():
                if c in ("league","league_name","season","date","home_team","away_team",
                         "home_team_id","away_team_id","home_goals","away_goals","result","match_id"):
                    continue
                if pd.isna(rows[idx].get(c)) and not pd.isna(v):
                    rows[idx][c] = v
        else:
            seen[key] = len(rows)
            rows.append(row)

df = pd.DataFrame(rows)
print("Benzersiz mac:", len(df), f"({time.time()-t0:.0f}s)")
print("Istatistik kapsami:")
for c in ["home_xg","home_corners","home_shots","home_sot","home_possession"]:
    print(f"  {c}: {df[c].notna().sum()}/{len(df)}")
df.to_parquet(OUT, index=False)
json.dump({str(v): k for k, v in idmap.items()},
          open("data/bronze/sofascore_team_ids.json", "w", encoding="utf-8"), ensure_ascii=False)
print("Yazildi:", OUT, "| team_ids:", len(idmap))
