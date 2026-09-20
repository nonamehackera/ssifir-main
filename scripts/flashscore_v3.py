"""Flashscore enrichment - AKILLI ESLESTIRME (v5).
Black list + manual override ile yanlis eslestirmeleri engeller.
"""
import sqlite3, json, re, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

t0 = time.time()

CACHE_DB = r"C:\Users\furka\Desktop\FlashscoreScraping-main\data\cache.db"
MATCH_TABLE_PATH = Path(r"C:\Users\furka\Desktop\ssifir-main\data\gold\flashscore_team_match_v5.json")
FEATURES_PATH = Path(r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_enhanced.parquet")
OUTPUT = Path(r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_enhanced_v5.parquet")


def build_lookup():
    with open(MATCH_TABLE_PATH, encoding="utf-8") as f:
        match_table = json.load(f)
    lookup = {}
    for fs_name, info in match_table.items():
        lookup[fs_name.lower().strip()] = info["tid"]
    return lookup


def ps(val):
    if val is None: return None
    val = str(val).strip()
    m = re.match(r"([\d.]+)%\s*\((\d+)/(\d+)\)", val)
    if m: return int(m.group(2)) / max(int(m.group(3)), 1)
    m = re.match(r"([\d.]+)%", val)
    if m: return float(m.group(1)) / 100.0
    m = re.match(r"([\d.]+)", val)
    if m: return float(m.group(1))
    return None


def main():
    lookup = build_lookup()
    print(f"Lookup table: {len(lookup)} keys")

    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute("SELECT data FROM match_cache").fetchall()
    conn.close()
    print(f"Cache rows: {len(rows)}")

    records = []
    matched = 0

    for r in rows:
        d = json.loads(r[0])
        hn = ((d.get("home") or {}).get("name") or "").strip()
        an = ((d.get("away") or {}).get("name") or "").strip()
        ds = d.get("date") or ""
        if not isinstance(ds, str): continue
        ds = ds.strip()
        res = d.get("result") or {}

        try:
            dt = datetime.strptime(ds, "%d.%m.%Y %H:%M")
        except:
            continue

        hg, ag = res.get("home"), res.get("away")
        if not hg or not ag or hg == "" or ag == "":
            continue
        try:
            hg, ag = int(hg), int(ag)
        except:
            continue

        h_tid = lookup.get(hn.lower().strip())
        a_tid = lookup.get(an.lower().strip())

        if h_tid is None or a_tid is None:
            continue

        matched += 1
        stats = {}
        for s in d.get("statistics", []):
            cat = s.get("category", "")
            hv, av = ps(s.get("homeValue")), ps(s.get("awayValue"))
            if hv is not None and av is not None:
                stats[cat] = (hv, av)

        records.append({
            "fs_date": dt.strftime("%Y-%m-%d"),
            "home_tid": h_tid, "away_tid": a_tid,
            "fs_poss_h": stats.get("Ball possession", (None,None))[0],
            "fs_poss_a": stats.get("Ball possession", (None,None))[1],
            "fs_fouls_h": stats.get("Fouls", (None,None))[0],
            "fs_fouls_a": stats.get("Fouls", (None,None))[1],
            "fs_bc_h": stats.get("Big chances", (None,None))[0],
            "fs_bc_a": stats.get("Big chances", (None,None))[1],
            "fs_xgot_h": stats.get("xG on target (xGOT)", (None,None))[0],
            "fs_xgot_a": stats.get("xGOT faced", (None,None))[0],
            "fs_xg_h": stats.get("Expected goals (xG)", (None,None))[0],
            "fs_xg_a": stats.get("Expected goals (xG)", (None,None))[1],
        })

    df = pd.DataFrame(records)
    print(f"Matched records: {matched}/{len(rows)} ({matched*100//max(len(rows),1)}%)")

    if len(df) == 0:
        print("ERROR: No matched records!")
        return

    for c in ["fs_poss_h","fs_fouls_h","fs_bc_h","fs_xgot_h","fs_xg_h"]:
        if c in df.columns:
            nn = df[c].notna().sum()
            print(f"  {c}: {nn}/{len(df)} ({nn*100//max(len(df),1)}%)")

    feat = pd.read_parquet(FEATURES_PATH)
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    feat["dstr"] = feat["date"].dt.strftime("%Y-%m-%d")

    rn = {"fs_poss_h":"home_possession","fs_poss_a":"away_possession",
          "fs_fouls_h":"home_fouls","fs_fouls_a":"away_fouls",
          "fs_bc_h":"home_big_chances","fs_bc_a":"away_big_chances",
          "fs_xgot_h":"home_xgot","fs_xgot_a":"away_xgot",
          "fs_xg_h":"home_xg_flash","fs_xg_a":"away_xg_flash"}

    mcols = ["fs_date","home_tid","away_tid"] + list(rn.keys())
    merged = feat.merge(
        df[mcols].rename(columns={"fs_date":"dstr","home_tid":"home_team_id","away_tid":"away_team_id"}),
        on=["dstr","home_team_id","away_team_id"], how="left"
    )
    merged.rename(columns=rn, inplace=True)

    has_fs = merged["home_possession"].notna().sum() if "home_possession" in merged.columns else 0
    print(f"Merge: {has_fs}/{len(merged)} ({has_fs*100//max(len(merged),1)}%) eslesen mac")

    print("Rolling...")
    for tid_col, pfx in [("home_team_id","home"),("away_team_id","away")]:
        for stat, nm in [("home_possession","possession"),("home_fouls","fouls"),
                         ("home_big_chances","big_chances"),("home_xgot","xgot"),
                         ("home_xg_flash","xg")]:
            if stat not in merged.columns: continue
            out = f"{pfx}_{nm}_5"
            merged = merged.sort_values(["date", tid_col])
            merged[out] = merged.groupby(tid_col)[stat].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

    new = [c for c in merged.columns if any(x in c for x in
           ["possession_5","fouls_5","big_chances_5","xgot_5"]) or c.endswith("_xg_5")]
    for c in new:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    merged.drop(columns=["dstr"], inplace=True, errors="ignore")
    merged.to_parquet(OUTPUT, index=False)
    print(f"\nKaydedildi: {OUTPUT} ({len(merged)} rows, {len(merged.columns)} cols)")

    for c in sorted(new):
        nz = (merged[c]!=0).sum()
        print(f"  {c}: {nz}/{len(merged)} ({nz*100//max(len(merged),1)}%)")

    print(f"\nSure: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
