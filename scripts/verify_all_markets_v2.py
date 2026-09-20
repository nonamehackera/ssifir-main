#!/usr/bin/env python3
"""Verify ALL prediction markets against actual results - index-based matching."""
import json, sys, os, codecs
if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")

# Actual results indexed to match predictions.json order (28 matches)
# Index 0-27 matching predictions.json
actual_by_index = {
    0:  {"score": "2-3", "hg": 2, "ag": 3, "result": "A", "btts": True, "first": "A", "corners": None},  # Basaksehir-Galatasaray
    1:  {"score": "0-2", "hg": 0, "ag": 2, "result": "A", "btts": False, "first": "A", "corners": None},  # Ipswich-Liverpool
    2:  {"score": "4-1", "hg": 4, "ag": 1, "result": "H", "btts": True, "first": "H", "corners": None},  # Stuttgart-Koln
    3:  {"score": "2-2", "hg": 2, "ag": 2, "result": "D", "btts": True, "first": "A", "corners": None},  # Newcastle-Bournemouth
    4:  {"score": "1-0", "hg": 1, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": 7},   # Erzurum-Konyaspor
    5:  {"score": "1-0", "hg": 1, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},  # Man City-Coventry
    6:  {"score": "0-0", "hg": 0, "ag": 0, "result": "D", "btts": False, "first": None, "corners": None},  # Nottingham-Tottenham
    7:  {"score": "2-3", "hg": 2, "ag": 3, "result": "A", "btts": True, "first": "H", "corners": None},  # Fulham-Crystal Palace
    8:  {"score": "1-1", "hg": 1, "ag": 1, "result": "D", "btts": True, "first": "H", "corners": 7},   # Brentford-Sunderland
    9:  {"score": "1-1", "hg": 1, "ag": 1, "result": "D", "btts": True, "first": "A", "corners": 13},  # Brighton-Leeds
    10: {"score": "4-0", "hg": 4, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},  # Leverkusen-Union
    11: {"score": "2-3", "hg": 2, "ag": 3, "result": "A", "btts": True, "first": "H", "corners": None},  # Hoffenheim-Dortmund
    12: {"score": "3-1", "hg": 3, "ag": 1, "result": "H", "btts": True, "first": "H", "corners": None},  # Bremen-Leipzig
    13: {"score": "3-4", "hg": 3, "ag": 4, "result": "A", "btts": True, "first": "H", "corners": 17},  # Gladbach-Elversberg
    14: {"score": "0-1", "hg": 0, "ag": 1, "result": "A", "btts": False, "first": "A", "corners": None},  # Paderborn-Freiburg
    15: {"score": "0-0", "hg": 0, "ag": 0, "result": "D", "btts": False, "first": None, "corners": None},  # Hull-Aston Villa
    16: {"score": "0-0", "hg": 0, "ag": 0, "result": "D", "btts": False, "first": None, "corners": None},  # Schalke-Bayern
    17: {"score": "1-2", "hg": 1, "ag": 2, "result": "A", "btts": True, "first": "H", "corners": 9},   # Fenerbahce-Besiktas
    18: {"score": "2-2", "hg": 2, "ag": 2, "result": "D", "btts": True, "first": "A", "corners": 7},   # Everton-ManUtd
    19: {"score": "3-0", "hg": 3, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},  # Corum-Eyupspor
    20: {"score": "2-2", "hg": 2, "ag": 2, "result": "D", "btts": True, "first": "H", "corners": None},  # Kasimpasa-Amed
    21: {"score": "2-1", "hg": 2, "ag": 1, "result": "H", "btts": True, "first": "A", "corners": 8},   # Arsenal-Chelsea
    22: {"score": "1-0", "hg": 1, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},  # Kocaeli-Samsun
    23: {"score": "5-0", "hg": 5, "ag": 0, "result": "H", "btts": False, "first": "H", "corners": None},  # Trabzon-Genclerbirligi
    24: {"score": "1-4", "hg": 1, "ag": 4, "result": "A", "btts": True, "first": "A", "corners": None},  # Frankfurt-Augsburg
    25: {"score": "0-5", "hg": 0, "ag": 5, "result": "A", "btts": True, "first": "A", "corners": None},  # Hamburg-Mainz
    26: {"score": "2-4", "hg": 2, "ag": 4, "result": "A", "btts": True, "first": "A", "corners": None},  # Goztepe-Gaziantep
    27: {"score": "0-1", "hg": 0, "ag": 1, "result": "A", "btts": False, "first": "A", "corners": None},  # Rizespor-Alanyaspor
}

# Load predictions
with open("tahminler/ftms_tahminler_2026-09-07.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Track results per market
market_stats = {}
match_details = []

for i, pred in enumerate(preds):
    a = actual_by_index.get(i)
    if not a:
        continue

    p = pred["predictions"]
    recs = pred.get("recommendations", {}).get("all_tips", [])
    total_goals = a["hg"] + a["ag"]

    details = {"match": pred["match"], "score": a["score"], "result": a["result"]}

    # 1X2
    pred_result = "H" if p["1X2"]["winner"] == "Ev Sahibi" else ("D" if p["1X2"]["winner"] == "Beraberlik" else "A")
    details["1X2_correct"] = pred_result == a["result"]
    market_stats.setdefault("1X2", []).append(pred_result == a["result"])

    # Double Chance
    dc = p["double_chance"]
    dc_1x = a["result"] in ("H", "D")
    dc_12 = a["result"] in ("H", "A")
    dc_x2 = a["result"] in ("D", "A")

    details["DC_1X"] = dc["1X"]
    details["DC_12"] = dc["12"]
    details["DC_X2"] = dc["X2"]
    details["DC_1X_correct"] = dc_1x
    details["DC_12_correct"] = dc_12
    details["DC_X2_correct"] = dc_x2
    market_stats.setdefault("DC_1X", []).append(dc_1x)
    market_stats.setdefault("DC_12", []).append(dc_12)
    market_stats.setdefault("DC_X2", []).append(dc_x2)

    # BTTS
    btts_pred = p["btts"]["yes"] > p["btts"]["no"]
    details["BTTS_pred"] = "Yes" if btts_pred else "No"
    details["BTTS_actual"] = "Yes" if a["btts"] else "No"
    details["BTTS_correct"] = btts_pred == a["btts"]
    market_stats.setdefault("BTTS", []).append(btts_pred == a["btts"])

    # Over/Under
    ou = p["goals"]["over_under"]
    details["O1.5_pred"] = f"{ou['1.5']:.0f}%"
    details["O2.5_pred"] = f"{ou['2.5']:.0f}%"
    details["O3.5_pred"] = f"{ou['3.5']:.0f}%"
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
        details["FirstGoal_pred"] = "H" if first_pred_home else "A"
        details["FirstGoal_actual"] = a["first"]
        details["FirstGoal_correct"] = (a["first"] == "H") == first_pred_home
        market_stats.setdefault("FirstGoal", []).append((a["first"] == "H") == first_pred_home)

    match_details.append(details)

# Print results
print("=" * 150)
print(f"{'#':>3} {'Maç':<40} {'Skor':>6} {'Sonuç':>5} {'1X2':>5} {'BTTS':>7} {'O2.5':>7} {'Çift12':>7} {'İlk Gol':>8} {'xG':>5} {'Fark':>5}")
print("=" * 150)

for idx, d in enumerate(match_details, 1):
    m1x2 = "OK" if d["1X2_correct"] else "XX"
    btts = "OK" if d["BTTS_correct"] else "XX"
    o25 = "OK" if d["O2.5_correct"] else "XX"
    dc12 = "OK" if d["DC_12_correct"] else "XX"
    fg = "OK" if d.get("FirstGoal_correct") else ("N/A" if d.get("FirstGoal_correct") is None else "XX")
    xg = d.get("xG_total", 0)
    xg_diff = d.get("xG_diff", 0)
    m = d["match"][:40]
    print(f"{idx:>3} {m:<40} {d['score']:>6} {d['result']:>5} {m1x2:>5} {d['BTTS_pred']:>3}/{d['BTTS_actual']:<3} {d['O2.5_pred']:>3}/{d['O2.5_actual'] if 'O2.5_actual' in d else ('>2.5' if d['O2.5_correct'] else '<2.5'):<3} {dc12:>7} {fg:>8} {xg:>5.1f} {xg_diff:>5.1f}")

print("=" * 150)

# Market accuracy summary
print("\n" + "=" * 80)
print("PİYASA DOĞRULUK ORANLARI (28 MAÇ)")
print("=" * 80)
for market, results in sorted(market_stats.items()):
    correct = sum(results)
    total = len(results)
    pct = correct / total * 100 if total > 0 else 0
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"  {market:<20} {correct:>2}/{total:<2} = %{pct:>5.1f}  {bar}")

print("=" * 80)

# Actual result distribution
results_count = {"H": 0, "D": 0, "A": 0}
for d in match_details:
    results_count[d["result"]] += 1
print(f"\nGerçek Sonuç Dağılımı: G={results_count['H']}, B={results_count['D']}, D={results_count['A']}")

# BTTS distribution
btts_yes = sum(1 for d in match_details if d["BTTS_actual"] == "Yes")
print(f"BTTS Dağılımı: Evet={btts_yes}, Hayır={28-btts_yes}")

# Total goals distribution
goals = [d["actual_total"] for d in match_details]
print(f"Gol Dağılımı: Ort={sum(goals)/len(goals):.1f}, Min={min(goals)}, Max={max(goals)}")
print(f"Over 2.5: {sum(1 for g in goals if g > 2.5)}/28 = %{sum(1 for g in goals if g > 2.5)/28*100:.0f}")
print(f"Over 1.5: {sum(1 for g in goals if g > 1.5)}/28 = %{sum(1 for g in goals if g > 1.5)/28*100:.0f}")
print(f"Under 2.5: {sum(1 for g in goals if g < 2.5)}/28 = %{sum(1 for g in goals if g < 2.5)/28*100:.0f}")

# xG analysis
xg_diffs = [d["xG_diff"] for d in match_details if "xG_diff" in d]
avg_diff = sum(xg_diffs) / len(xg_diffs) if xg_diffs else 0
print(f"\nxG Ortalama Hata: {avg_diff:.2f} gol")
best = min(match_details, key=lambda x: x.get("xG_diff", 99))
worst = max(match_details, key=lambda x: x.get("xG_diff", 0))
print(f"xG En İyi: {best['match']} ({best['xG_diff']:.2f} fark)")
print(f"xG En Kötü: {worst['match']} ({worst['xG_diff']:.2f} fark)")
