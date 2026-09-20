"""Quick dogrulama: predict.py enhanced feature + playable flag calisiyor mu?"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import time

t0 = time.time()
print("Loading predict.py...")

# predict.py modul olarak import edip test edelim
import importlib
import predict as p
importlib.reload(p)

print(f"\n  Train: {len(p.train)} | Enhanced model: {p.HAS_ENHANCED}")
print(f"  Feature (base): {len(p.ALL_COLS)} | Feature (enhanced): {len(p.ALL_COLS_ENH)}")

# Test tahminleri
test_cases = [
    (1023, 1024, "Buyuk lig takimiornegi"),
    (5000, 5001, "Orta/az verili ligornegi"),
]

print(f"\n  Test tahminleri:")
for h, a, label in test_cases:
    if h in p.latest and a in p.latest:
        r, err = p.tahmin(h, a)
        if r:
            status = "PLAYABLE" if r.get("playable") else "SKIP"
            print(f"    {label}: {r['home_name']} vs {r['away_name']}")
            print(f"      1X2: {r['home_win']*100:.1f}% / {r['draw']*100:.1f}% / {r['away_win']*100:.1f}%")
            print(f"      Status: {status} | Conf: {r.get('confidence',0)}% | Model: {r.get('model_used','?')}")
        else:
            print(f"    {label}: {err}")
    else:
        print(f"    {label}: team not in data (h={h in p.latest}, a={a in p.latest})")

# Rastgele 5 buyuk lig maci test
import pandas as pd
lc = p.LEAGUE_MATCH_COUNTS
big_leagues = [lg for lg, n in lc.items() if n >= 3000][:5]
print(f"\n  Buyuk lig ornekleri:")
for lg in big_leagues:
    lg_teams = []
    for tid, row in p.latest.items():
        if row.get("league") == lg:
            lg_teams.append((tid, row.get("home_elo", 0)))
    lg_teams.sort(key=lambda x: -x[1])
    if len(lg_teams) >= 2:
        h_id, a_id = lg_teams[0][0], lg_teams[1][0]
        r, err = p.tahmin(h_id, a_id)
        if r:
            status = "PLAYABLE" if r.get("playable") else "SKIP"
            print(f"    {lg}: {r['home_name']} vs {r['away_name']} | 1X2: {r['home_win']*100:.1f}/{r['draw']*100:.1f}/{r['away_win']*100:.1f} | {status}")

print(f"\n  Sure: {time.time()-t0:.1f}s")
