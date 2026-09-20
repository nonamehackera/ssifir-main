import requests, json

matches = [
    (145, 250, "Millwall vs Wrexham"),
    (8000008, 73, "West Brom vs Charlton"),
]

results = []
for h_id, a_id, name in matches:
    r = requests.post('http://localhost:5011/api/predict', json={'home_id': h_id, 'away_id': a_id})
    if r.status_code == 200:
        d = r.json()
        print(f"\n=== {name} ===")
        print(f"  1X2: {d['home_win']}/{d['draw']}/{d['away_win']}")
        print(f"  Winner: {d['winner']} ({d['winner_prob']}%)")
        tg = d['total_goals']
        print(f"  Goals: O1.5={tg['over15']} O2.5={tg['over25']} O3.5={tg['over35']}")
        print(f"  BTTS: {d['btts_yes']}/{d['btts_no']}")
        dc = d['double_chance']
        print(f"  DC: 1X={dc['1X']} X2={dc['X2']} 12={dc['12']}")
        cr = d['corners']
        print(f"  Corners: avg={cr['predicted_total']} O7.5={cr['over75']} O8.5={cr['over85']}")
        results.append(d)
    else:
        print(f"  HATA: {r.status_code}")

# Also generate for yesterday's matches that are already played
yesterday = [
    (121, 74, "Sheffield Utd vs Bolton"),
    (3000051, 3000020, "Lincoln vs Blackburn"),
    (7000280, 32, "Swansea vs Watford"),
    (7000295, 8000028, "West Ham vs Wolves"),
    (58, 44, "Preston vs Bristol City"),
    (6001159, 3000021, "Portsmouth vs Derby"),
    (68, 25, "Birmingham vs Southampton"),
    (7000293, 7000292, "Stoke vs Norwich"),
]

print("\n\n=== YESTERDAY (Sept 1) ===")
for h_id, a_id, name in yesterday:
    r = requests.post('http://localhost:5011/api/predict', json={'home_id': h_id, 'away_id': a_id})
    if r.status_code == 200:
        d = r.json()
        print(f"\n{name}: 1X2={d['home_win']}/{d['draw']}/{d['away_win']} | O2.5={d['total_goals']['over25']} | BTTS={d['btts_yes']}/{d['btts_no']} | DC: 1X={d['double_chance']['1X']} X2={d['double_chance']['X2']}")
    else:
        print(f"  HATA: {r.status_code}")
