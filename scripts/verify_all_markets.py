#!/usr/bin/env python3
"""Verify ALL prediction markets against actual results."""
import json, sys, os, codecs
if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")

# Actual results from internet search (ALL 28 matches)
# Format: match_key -> { score, home_goals, away_goals, result(H/D/A), btts, first_goal(H/A/N), corners }
actual = {
    "İstanbul Başakşehir vs Galatasaray": {"score": "2-3", "hg": 2, "ag": 3, "result": "A", "btts": True, "first": "A", "corners": None},
    "Ipswich Town FC vs Liverpool": {"score": "0-2", "hg": 0, "ag": 2, "result": "A", "btts": False, "first": "A", "corners": None},
    "Stuttgart vs 1. FC Köln": {"score": "4-1", "hg": 4, "ag": 1, "result": "H", "btts": True, "first": "H", "corners": None},
    "Newcastle United vs Bournemouth": {"score": "2-2", "hg": 2, "ag": 2, "result": "D", "btts": True, "first": "A", "corners": None},
    "Erzurum BB vs Konyaspor": {"score": "1-0", "hg": 1, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": 7},
    "Manchester City vs Coventry": {"score": "1-0", "hg": 1, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},
    "Nottingham Forest vs Tottenham": {"score": "0-0", "hg": 0, "ag": 0, "result": "D", "btts": False, "first": None, "corners": None},
    "Fulham vs Crystal Palace": {"score": "2-3", "hg": 2, "ag": 3, "result": "A", "btts": True, "first": "H", "corners": None},
    "Brentford vs Sunderland": {"score": "1-1", "hg": 1, "ag": 1, "result": "D", "btts": True, "first": "H", "corners": 7},
    "Brighton vs Leeds United FC": {"score": "1-1", "hg": 1, "ag": 1, "result": "D", "btts": True, "first": "A", "corners": 13},
    "Leverkusen vs 1. FC Union Berlin": {"score": "4-0", "hg": 4, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},
    "Hoffenheim vs Dortmund": {"score": "2-3", "hg": 2, "ag": 3, "result": "A", "btts": True, "first": "H", "corners": None},
    "Werder Bremen vs RB Leipzig": {"score": "3-1", "hg": 3, "ag": 1, "result": "H", "btts": True, "first": "H", "corners": None},
    "M'gladbach vs Elversberg": {"score": "3-4", "hg": 3, "ag": 4, "result": "A", "btts": True, "first": "H", "corners": 17},
    "Paderborn vs Freiburg": {"score": "0-1", "hg": 0, "ag": 1, "result": "A", "btts": False, "first": "A", "corners": None},
    "Hull City AFC vs Aston Villa": {"score": "0-0", "hg": 0, "ag": 0, "result": "D", "btts": False, "first": None, "corners": None},
    "Schalke 04 vs Bayern Munich": {"score": "0-0", "hg": 0, "ag": 0, "result": "D", "btts": False, "first": None, "corners": None},
    "Fenerbahçe vs Beşiktaş": {"score": "1-2", "hg": 1, "ag": 2, "result": "A", "btts": True, "first": "H", "corners": 9},
    "Everton vs Man United": {"score": "2-2", "hg": 2, "ag": 2, "result": "D", "btts": True, "first": "A", "corners": 7},
    "Arsenal vs Chelsea": {"score": "2-1", "hg": 2, "ag": 1, "result": "H", "btts": True, "first": "A", "corners": 8},
    # Turkish league matches (Sep 6-7)
    "Çaykur Rizespor vs Alanyaspor": {"score": "0-1", "hg": 0, "ag": 1, "result": "A", "btts": False, "first": "A", "corners": None},
    "Göztepe vs Gaziantepspor": {"score": "2-4", "hg": 2, "ag": 4, "result": "A", "btts": True, "first": "A", "corners": None},
    "Trabzonspor vs Gençlerbirliği": {"score": "5-0", "hg": 5, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},
    "Kocaelispor vs Samsunspor": {"score": "1-0", "hg": 1, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},
    "Çorum FK vs Eyüpspor": {"score": "3-0", "hg": 3, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},
    "Kasımpaşa vs Amed": {"score": "2-2", "hg": 2, "ag": 2, "result": "D", "btts": True, "first": "H", "corners": None},
}

# Load predictions
with open("tahminler/ftms_tahminler_2026-09-07.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Track results per market
market_stats = {}
match_details = []

for pred in preds:
    mk = pred["match"]
    a = actual.get(mk)
    if not a:
        continue

    p = pred["predictions"]
    recs = pred.get("recommendations", {}).get("all_tips", [])
    total_goals = a["hg"] + a["ag"]

    details = {"match": mk, "score": a["score"], "result": a["result"]}

    # 1X2
    pred_result = "H" if p["1X2"]["winner"] == "Ev Sahibi" else ("D" if p["1X2"]["winner"] == "Beraberlik" else "A")
    details["1X2_correct"] = pred_result == a["result"]
    market_stats.setdefault("1X2", []).append(pred_result == a["result"])

    # Double Chance
    dc = p["double_chance"]
    # 1X: home or draw
    dc_1x = a["result"] in ("H", "D")
    # 12: home or away (no draw)
    dc_12 = a["result"] in ("H", "A")
    # X2: draw or away
    dc_x2 = a["result"] in ("D", "A")

    details["DC_1X_correct"] = dc_1x
    details["DC_12_correct"] = dc_12
    details["DC_X2_correct"] = dc_x2
    market_stats.setdefault("DC_1X", []).append(dc_1x)
    market_stats.setdefault("DC_12", []).append(dc_12)
    market_stats.setdefault("DC_X2", []).append(dc_x2)

    # BTTS
    btts_pred = p["btts"]["yes"] > p["btts"]["no"]
    details["BTTS_correct"] = btts_pred == a["btts"]
    market_stats.setdefault("BTTS", []).append(btts_pred == a["btts"])

    # Over/Under
    ou = p["goals"]["over_under"]
    details["O1.5_correct"] = total_goals > 1.5
    details["O2.5_correct"] = total_goals > 2.5
    details["O3.5_correct"] = total_goals > 3.5
    market_stats.setdefault("O1.5", []).append(total_goals > 1.5)
    market_stats.setdefault("O2.5", []).append(total_goals > 2.5)
    market_stats.setdefault("O3.5", []).append(total_goals > 3.5)

    # xG accuracy
    xg = p["goals"]["expected"]
    details["xG_total"] = xg["total_xg"]
    details["actual_total"] = total_goals
    details["xG_diff"] = abs(xg["total_xg"] - total_goals)

    # Corners (if available)
    if a["corners"] is not None:
        corners_over75 = a["corners"] > 7.5
        corners_over85 = a["corners"] > 8.5
        details["corners"] = a["corners"]
        details["Corners_O7.5_correct"] = corners_over75 == (p["corners"]["over75"] > 50)
        details["Corners_O8.5_correct"] = corners_over85 == (p["corners"]["over85"] > 50)
        market_stats.setdefault("Corners_O7.5", []).append(corners_over75 == (p["corners"]["over75"] > 50))
        market_stats.setdefault("Corners_O8.5", []).append(corners_over85 == (p["corners"]["over85"] > 50))

    # First goal
    if a["first"] is not None:
        first_pred_home = any(r["market"] == "Ilk Gol Ev Atar" and r["pick"] == "1" for r in recs)
        details["FirstGoal_correct"] = (a["first"] == "H") == first_pred_home
        market_stats.setdefault("FirstGoal", []).append((a["first"] == "H") == first_pred_home)

    match_details.append(details)

# Print results
print("=" * 140)
print(f"{'Maç':<40} {'Skor':>6} {'Sonuç':>5} {'1X2':>5} {'BTTS':>5} {'O2.5':>5} {'Çift':>5} {'İlk Gol':>7} {'xG':>5} {'xG Fark':>7}")
print("=" * 140)

for d in match_details:
    m1x2 = "OK" if d["1X2_correct"] else "XX"
    btts = "OK" if d["BTTS_correct"] else "XX"
    o25 = "OK" if d["O2.5_correct"] else "XX"
    dc12 = "OK" if d["DC_12_correct"] else "XX"
    fg = "OK" if d.get("FirstGoal_correct") else ("N/A" if d.get("FirstGoal_correct") is None else "XX")
    xg = d.get("xG_total", 0)
    xg_diff = d.get("xG_diff", 0)
    print(f"{d['match']:<40} {d['score']:>6} {d['result']:>5} {m1x2:>5} {btts:>5} {o25:>5} {dc12:>5} {fg:>7} {xg:>5.1f} {xg_diff:>7.1f}")

print("=" * 140)

# Market accuracy summary
print("\n" + "=" * 80)
print("PİYASA DOĞRULUK ORANLARI")
print("=" * 80)
for market, results in sorted(market_stats.items()):
    correct = sum(results)
    total = len(results)
    pct = correct / total * 100 if total > 0 else 0
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"  {market:<20} {correct:>2}/{total:<2} = %{pct:>5.1f}  {bar}")

print("=" * 80)

# Specific market analysis
print("\n" + "=" * 80)
print("DETAYLI ANALİZ")
print("=" * 80)

# BTTS analysis
btts_matches = [d for d in match_details if d["BTTS_correct"]]
btts_wrong = [d for d in match_details if not d["BTTS_correct"]]
print(f"\nBTTS Doğru: {len(btts_matches)}/20")
for d in btts_wrong:
    print(f"  Yanlış: {d['match']} ({d['score']})")

# O2.5 analysis
o25_wrong = [d for d in match_details if not d["O2.5_correct"]]
print(f"\nOver 2.5 Yanlış: {len(o25_wrong)}/20")
for d in o25_wrong:
    print(f"  {d['match']} ({d['score']}) - Toplam: {d['actual_total']} gol")

# xG accuracy
xg_diffs = [d["xG_diff"] for d in match_details if "xG_diff" in d]
avg_diff = sum(xg_diffs) / len(xg_diffs) if xg_diffs else 0
print(f"\nxG Ortalama Hata: {avg_diff:.2f} gol")
print(f"xG En İyi Tahmin: {min(match_details, key=lambda x: x.get('xG_diff', 99))['match']} ({min(xg_diffs):.2f} fark)")
print(f"xG En Kötü Tahmin: {max(match_details, key=lambda x: x.get('xG_diff', 0))['match']} ({max(xg_diffs):.2f} fark)")
