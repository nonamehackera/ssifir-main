"""Flashscore team matching: soyad + harf eslesme ile hizli eslestirme.

Cache.db 76K mac, 4942 takim var. Modelde 7793 takim var.
Strateji: normalization + Levenshtein-like matching.
"""
import sqlite3, json, re, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

t0 = time.time()

CACHE_DB = r"C:\Users\furka\Desktop\FlashscoreScraping-main\data\cache.db"
TEAM_MAP_PATH = Path(r"C:\Users\furka\Desktop\ssifir-main\data\gold\team_id_to_name.json")
FEATURES_PATH = Path(r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_enhanced.parquet")
OUTPUT = Path(r"C:\Users\furka\Desktop\ssifir-main\data\gold\features_enhanced_v2.parquet")


def normalize(name):
    """Takim ismini normalize et."""
    name = name.lower().strip()
    name = re.sub(r'\s*\(.*?\)\s*$', '', name)  # parantez temizle
    # yaygin on ek/kisaltmalar
    for pat in [r'\bssc\b', r'\bosc\b', r'\baspc\b', r'\bafc\b', r'\brcd\b',
                r'\brc\b', r'\bsc\b', r'\bfc\b', r'\bfc\b', r'\bfc\b',
                r'\bca\b', r'\bac\b', r'\buc\b', r'\bcp\b', r'\bcf\b',
                r'\bfk\b', r'\bksk\b', r'\bts\b', r'\bsk\b']:
        name = re.sub(pat, '', name)
    name = re.sub(r'^f\.\s*', '', name)
    name = re.sub(r'\.+$', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def char_set(name):
    """Karakter seti (harfler)."""
    return set(re.findall(r'[a-z]', name))


def similarity(a, b):
    """Hizli benzerlik skoru (0-100)."""
    a_n, b_n = normalize(a), normalize(b)
    if a_n == b_n:
        return 100
    if a_n in b_n or b_n in a_n:
        return 90

    a_words = set(a_n.split())
    b_words = set(b_n.split())

    # tam kelime eslesmesi
    common = a_words & b_words
    if common:
        score = 80 + 20 * len(common) / max(len(a_words), len(b_words))
        return min(score, 95)

    # ilk 4+ harf eslesme
    if len(a_n) >= 4 and len(b_n) >= 4:
        prefix_match = sum(1 for i in range(min(6, len(a_n), len(b_n))) if a_n[i] == b_n[i])
        if prefix_match >= 4:
            return 60 + prefix_match * 5

    # karakter seti benzerligi
    a_chars, b_chars = char_set(a_n), char_set(b_n)
    if a_chars and b_chars:
        jaccard = len(a_chars & b_chars) / len(a_chars | b_chars)
        if jaccard > 0.6:
            return int(jaccard * 70)

    return 0


def build_mapping():
    """Model takimlariyla Flashscore takimlarini eslestir."""
    with open(TEAM_MAP_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    model_teams = {}
    for tid, name in raw.items():
        n = normalize(name)
        model_teams[n] = (int(tid), name)

    return model_teams


def match_fs_to_model(fs_name, model_teams):
    """FS takim ismini model takimina eslestir."""
    fs_n = normalize(fs_name)

    # 1. Tam eslesme
    if fs_n in model_teams:
        return model_teams[fs_n][0], 100

    # 2. En iyi eslesmeyi bul
    best_tid, best_score = None, 0
    for mn, (tid, orig) in model_teams.items():
        s = similarity(fs_name, orig)
        if s > best_score:
            best_score = s
            best_tid = tid
        # Erken dur: cok iyi eslesme
        if best_score >= 90:
            break

    return best_tid, best_score


def extract_and_match():
    """Flashscore cikar ve esles."""
    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute("SELECT data FROM match_cache").fetchall()
    conn.close()

    model_teams = build_mapping()
    print(f"Model teams: {len(model_teams)}")

    records = []
    matched = 0
    total = 0

    for i, r in enumerate(rows):
        if i % 20000 == 0 and i > 0:
            print(f"  {i}/{len(rows)} isleniyor... ({matched}/{total} eslesti)")

        d = json.loads(r[0])
        hn = d.get("home", {}).get("name", "").strip()
        an = d.get("away", {}).get("name", "").strip()
        ds = d.get("date", "")
        res = d.get("result", {})

        try:
            dt = datetime.strptime(ds.strip(), "%d.%m.%Y %H:%M")
        except:
            continue

        hg, ag = res.get("home"), res.get("away")
        if not hg or not ag or hg == "" or ag == "":
            continue
        try:
            hg, ag = int(hg), int(ag)
        except:
            continue

        h_tid, h_score = match_fs_to_model(hn, model_teams)
        a_tid, a_score = match_fs_to_model(an, model_teams)

        total += 1
        h_ok = h_tid is not None and h_score >= 55
        a_ok = a_tid is not None and a_score >= 55
        if h_ok and a_ok:
            matched += 1

        # Stats
        stats = {}
        for s in d.get("statistics", []):
            cat = s.get("category", "")
            hv, av = s.get("homeValue"), s.get("awayValue")
            if hv is not None and av is not None:
                hv_f, av_f = _ps(hv), _ps(av)
                if hv_f is not None and av_f is not None:
                    stats[cat] = (hv_f, av_f)

        records.append({
            "fs_date": dt.strftime("%Y-%m-%d"),
            "home_tid": h_tid if h_ok else None,
            "away_tid": a_tid if a_ok else None,
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
    print(f"\nToplam: {total}, Eslesen: {matched} (%{matched*100//max(total,1)})")
    return df


def _ps(val):
    if val is None:
        return None
    val = str(val).strip()
    m = re.match(r"([\d.]+)%\s*\((\d+)/(\d+)\)", val)
    if m:
        return int(m.group(2)) / max(int(m.group(3)), 1)
    m = re.match(r"([\d.]+)%", val)
    if m:
        return float(m.group(1)) / 100.0
    m = re.match(r"([\d.]+)", val)
    if m:
        return float(m.group(1))
    return None


def enrich(feat, fs_df):
    """Flashscore stats ekle + rolling."""
    feat = feat.copy()
    feat["dstr"] = pd.to_datetime(feat["date"]).dt.strftime("%Y-%m-%d")

    v = fs_df.dropna(subset=["home_tid","away_tid"]).copy()
    v["home_tid"] = v["home_tid"].astype(int)
    v["away_tid"] = v["away_tid"].astype(int)

    rn = {"fs_poss_h":"home_possession","fs_poss_a":"away_possession",
          "fs_fouls_h":"home_fouls","fs_fouls_a":"away_fouls",
          "fs_bc_h":"home_big_chances","fs_bc_a":"away_big_chances",
          "fs_xgot_h":"home_xgot","fs_xgot_a":"away_xgot",
          "fs_xg_h":"home_xg_flash","fs_xg_a":"away_xg_flash"}

    mcols = ["fs_date","home_tid","away_tid"] + list(rn.keys())
    merged = feat.merge(
        v[mcols].rename(columns={"fs_date":"dstr","home_tid":"home_team_id","away_tid":"away_team_id"}),
        on=["dstr","home_team_id","away_team_id"], how="left"
    )
    merged.rename(columns=rn, inplace=True)

    print("Rolling features...")
    for tid_col, pfx in [("home_team_id","home"),("away_team_id","away")]:
        for stat, nm in [("home_possession","possession"),("home_fouls","fouls"),
                         ("home_big_chances","big_chances"),("home_xgot","xgot"),
                         ("home_xg_flash","xg")]:
            out = f"{pfx}_{nm}_5"
            merged = merged.sort_values(["date", tid_col])
            merged[out] = merged.groupby(tid_col)[stat].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

    # Fill 0
    new = [c for c in merged.columns if any(x in c for x in
           ["possession_5","fouls_5","big_chances_5","xgot_5"]) or c.endswith("_xg_5")]
    for c in new:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    merged.drop(columns=["dstr"], inplace=True, errors="ignore")
    print(f"Final: {len(merged)} rows, {len(merged.columns)} cols")
    return merged


if __name__ == "__main__":
    fs = extract_and_match()
    feat = pd.read_parquet(FEATURES_PATH)
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    enriched = enrich(feat, fs)
    enriched.to_parquet(OUTPUT, index=False)
    print(f"Kaydedildi: {OUTPUT}")

    new = [c for c in enriched.columns if any(x in c for x in
           ["possession_5","fouls_5","big_chances_5","xgot_5"]) or c.endswith("_xg_5")]
    for c in sorted(new):
        nz = (enriched[c]!=0).sum()
        print(f"  {c}: {nz}/{len(enriched)} non-zero ({nz*100//len(enriched)}%)")

    print(f"\nSure: {time.time()-t0:.0f}s")
