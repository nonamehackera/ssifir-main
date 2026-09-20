"""
SofaScore → Parquet ADIM 1: Sadece ana ligleri parse et
~2000 dosya, ~5GB — 2-3 dakikada biter
"""
import json, os, sys, glob, re, csv, time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
start = time.time()

TAKIMLAR_DIR = r"C:\Users\furka\Desktop\sofascore-scraper-main\Takimlar"
NAME_MAP_PATH = r"C:\Users\furka\Desktop\ssifir-main\data\gold\team_id_to_name.json"
OUTPUT_CSV = r"C:\Users\furka\Desktop\ssifir-main\data\gold\sofascore_parsed.csv"

# Load name map
with open(NAME_MAP_PATH, encoding="utf-8") as f:
    name_map_raw = json.load(f)
name_to_id = {}
for k, v in name_map_raw.items():
    name_to_id[v.lower().strip()] = int(k)
    name_to_id[v.strip()] = int(k)

def find_tid(name):
    n = name.strip()
    if n in name_to_id: return name_to_id[n]
    lo = n.lower()
    if lo in name_to_id: return name_to_id[lo]
    for nm, tid in name_to_id.items():
        if lo in nm.lower() or nm.lower() in lo:
            return tid
    return None

def si(val):
    if val is None: return 0
    s = str(val).strip().replace("%","").replace(" km","")
    s = re.sub(r'[^\d\-\.]','',s.split("/")[0].split(" ")[0])
    try: return int(float(s))
    except: return 0

def sf(val):
    if val is None: return 0.0
    s = str(val).strip().replace("%","").replace(" km","")
    s = re.sub(r'[^\d\-\.]','',s.split("/")[0].split(" ")[0])
    try: return float(s)
    except: return 0.0

def get_stats(md, tn):
    ts = md.get(tn, {})
    if not ts: return None
    for k in ["İstatistikler","Istatistikler"]:
        st = ts.get(k)
        if st and isinstance(st, dict) and len(st)>0: return st
    return None

CSV_HEADER = [
    "date","home_team_id","away_team_id","home_goals","away_goals",
    "home_shots","away_shots","home_sot","away_sot","home_corners","away_corners",
    "home_yellow","away_yellow","home_red","away_red","home_fouls","away_fouls",
    "sf_home_possession","sf_away_possession",
    "sf_home_xg","sf_away_xg","sf_home_xg_on_target","sf_away_xg_on_target",
    "sf_home_big_chances_scored","sf_away_big_chances_scored",
    "sf_home_big_chances_missed","sf_away_big_chances_missed",
    "sf_home_touches_pen","sf_away_touches_pen",
    "sf_home_tackles","sf_away_tackles",
    "sf_home_interceptions","sf_away_interceptions",
    "sf_home_recoveries","sf_away_recoveries",
    "sf_home_saves","sf_away_saves",
    "sf_home_goals_prevented","sf_away_goals_prevented",
]

json_files = glob.glob(os.path.join(TAKIMLAR_DIR, "**", "*.json"), recursive=True)
json_files = [f for f in json_files if not any(x in f.lower() for x in ["progress","ligveri","target","preset","cache","test_"])]
print(f"Toplam JSON: {len(json_files)}")

rows = []
errors = 0
no_id = 0
total_matches = 0

for i, jf in enumerate(json_files):
    if i % 500 == 0:
        elapsed = time.time() - start
        print(f"  {i}/{len(json_files)} ({elapsed:.0f}s) - {len(rows)} maç, {no_id} ID yok")
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        for m in data.get("matches", []):
            md = m.get("match_data", {})
            info = md.get("Match_Info", {})
            if not info: continue
            home = info.get("Home","")
            away = info.get("Away","")
            score = info.get("Score","0 - 0")
            ts = info.get("Timestamp",0)
            try:
                d = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
                parts = score.split(" - ")
                hg, ag = int(parts[0].strip()), int(parts[1].strip())
            except: continue
            hid = find_tid(home)
            aid = find_tid(away)
            if not hid or not aid:
                no_id += 1; continue
            total_matches += 1
            hs = get_stats(md, home)
            aws = get_stats(md, away)
            if not hs and not aws:
                rows.append([d.strftime("%Y-%m-%d %H:%M:%S"), hid, aid, hg, ag] + [0]*30)
                continue
            ho = (hs or {}).get("Match overview",{})
            ao = (aws or {}).get("Match overview",{})
            hsh = (hs or {}).get("Shots",{})
            ash = (aws or {}).get("Shots",{})
            hat = (hs or {}).get("Attack",{})
            aat = (aws or {}).get("Attack",{})
            hd = (hs or {}).get("Defending",{})
            ad = (aws or {}).get("Defending",{})
            hgk = (hs or {}).get("Goalkeeping",{})
            agk = (aws or {}).get("Goalkeeping",{})
            hp = sf(ho.get("Ball possession"))/100 if ho.get("Ball possession") else 0
            ap = sf(ao.get("Ball possession"))/100 if ao.get("Ball possession") else 0
            rows.append([
                d.strftime("%Y-%m-%d %H:%M:%S"), hid, aid, hg, ag,
                si(ho.get("Total shots")), si(ao.get("Total shots")),
                si(hsh.get("Shots on target")), si(ash.get("Shots on target")),
                si(ho.get("Corner kicks")), si(ao.get("Corner kicks")),
                si(ho.get("Yellow cards")), si(ao.get("Yellow cards")),
                si(ho.get("Red cards","0")), si(ao.get("Red cards","0")),
                si(ho.get("Fouls")), si(ao.get("Fouls")),
                hp, ap,
                sf(ho.get("Expected goals")) or sf(hsh.get("Expected goals")),
                sf(ao.get("Expected goals")) or sf(ash.get("Expected goals")),
                sf(hsh.get("Expected goals on target")), sf(ash.get("Expected goals on target")),
                si(hat.get("Big chances scored")), si(aat.get("Big chances scored")),
                si(hat.get("Big chances missed")), si(aat.get("Big chances missed")),
                si(hat.get("Touches in penalty area")), si(aat.get("Touches in penalty area")),
                si(hd.get("Total tackles") or hd.get("Tackles won")),
                si(ad.get("Total tackles") or ad.get("Tackles won")),
                si(hd.get("Interceptions")), si(ad.get("Interceptions")),
                si(hd.get("Recoveries")), si(ad.get("Recoveries")),
                si(hgk.get("Total saves")), si(agk.get("Total saves")),
                sf(hgk.get("Goals prevented")), sf(agk.get("Goals prevented")),
            ])
    except:
        errors += 1

elapsed = time.time() - start
print(f"\nParse tamam: {len(rows)} maç ({total_matches} toplam, {no_id} ID yok, {errors} hata)")
print(f"Süre: {elapsed:.0f} saniye")

print(f"CSV'ye kaydediliyor...")
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(CSV_HEADER)
    w.writerows(rows)
print(f"Kaydedildi: {OUTPUT_CSV}")
