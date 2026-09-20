#!/usr/bin/env python3
"""Regenerate all predictions with fixed model and print to console."""
import sys, os, json, time, codecs
if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from web.app import _predict_match, _load_model, TEAM_ID_MAP, STATE

def main():
    _load_model()

    preds = json.load(open("tahminler/predictions.json", "r", encoding="utf-8"))
    pairs = [(p["home_id"], p["away_id"]) for p in preds]

    print("=" * 120)
    print(f"{'#':>3} {'Maç':<45} {'Ev%':>6} {'Ber%':>6} {'Dep%':>6} {'Tahmin':<12} {'xG Ev':>6} {'xG Dep':>6} {'Toplam':>6}")
    print("=" * 120)

    correct = 0
    total = 0
    results_map = {
        "İstanbul Başakşehir vs Galatasaray": ("A", "2-3"),
        "Ipswich Town FC vs Liverpool": ("A", "0-2"),
        "Stuttgart vs 1. FC Köln": ("H", "4-1"),
        "Newcastle United vs Bournemouth": ("D", "2-2"),
        "Erzurum BB vs Konyaspor": ("H", "1-0"),
        "Manchester City vs Coventry": ("H", "1-0"),
        "Nottingham Forest vs Tottenham": ("D", "0-0"),
        "Fulham vs Crystal Palace": ("A", "2-3"),
        "Brentford vs Sunderland": ("D", "1-1"),
        "Brighton vs Leeds United FC": ("D", "1-1"),
        "Leverkusen vs 1. FC Union Berlin": ("H", "4-0"),
        "Hoffenheim vs Dortmund": ("A", "2-3"),
        "Werder Bremen vs RB Leipzig": ("H", "3-1"),
        "M'gladbach vs Elversberg": ("A", "3-4"),
        "Paderborn vs Freiburg": ("A", "0-1"),
        "Hull City AFC vs Aston Villa": ("D", "0-0"),
        "Schalke 04 vs Bayern Munich": ("D", "0-0"),
        "Fenerbahçe vs Beşiktaş": ("A", "1-2"),
        "Everton vs Man United": ("D", "2-2"),
        "Arsenal vs Chelsea": ("H", "2-1"),
        "Eintracht Frankfurt vs Augsburg": ("A", "1-4"),
        "Hamburg vs Mainz": ("A", "0-5"),
    }

    for i, (hid, aid) in enumerate(pairs):
        res = _predict_match(hid, aid)
        if res is None:
            print(f"{i+1:>3} {hid}-{aid} TAKIM BULUNAMADI")
            continue

        h_name = res.get("home_name", "?")
        a_name = res.get("away_name", "?")
        match_key = f"{h_name} vs {a_name}"

        hw = res["home_win"]
        dr = res["draw"]
        aw = res["away_win"]
        winner = res["winner"]
        hxg = res.get("goals", {}).get("home_lambda", 0)
        axg = res.get("goals", {}).get("away_lambda", 0)
        total_xg = res.get("goals", {}).get("expected_total", 0)

        actual = results_map.get(match_key)
        if actual:
            actual_result, actual_score = actual
            pred_result = "H" if winner == "Ev Sahibi" else ("D" if winner == "Beraberlik" else "A")
            is_correct = pred_result == actual_result
            if is_correct:
                correct += 1
            total += 1
            mark = "OK" if is_correct else "XX"
            print(f"{i+1:>3} {match_key:<45} {hw:>5.1f}% {dr:>5.1f}% {aw:>5.1f}% {winner:<12} {hxg:>5.2f} {axg:>5.2f} {total_xg:>5.2f}  [{actual_result}] {actual_score} {mark}")
        else:
            print(f"{i+1:>3} {match_key:<45} {hw:>5.1f}% {dr:>5.1f}% {aw:>5.1f}% {winner:<12} {hxg:>5.2f} {axg:>5.2f} {total_xg:>5.2f}")

    print("=" * 120)
    if total > 0:
        print(f"Doğruluk: {correct}/{total} = %{correct/total*100:.1f}")
    print()

    # Also save to JSON
    out = []
    for hid, aid in pairs:
        res = _predict_match(hid, aid)
        if res is None:
            continue
        h_name = res.get("home_name", "?")
        a_name = res.get("away_name", "?")
        match_key = f"{h_name} vs {a_name}"
        actual = results_map.get(match_key)
        rec = {
            "match": match_key,
            "home_win": round(res["home_win"], 1),
            "draw": round(res["draw"], 1),
            "away_win": round(res["away_win"], 1),
            "winner": res["winner"],
            "winner_prob": round(res["winner_prob"], 1),
            "home_xg": round(res.get("goals", {}).get("home_lambda", 0), 2),
            "away_xg": round(res.get("goals", {}).get("away_lambda", 0), 2),
            "total_xg": round(res.get("goals", {}).get("expected_total", 0), 2),
        }
        if actual:
            rec["actual_result"] = actual[0]
            rec["actual_score"] = actual[1]
            pred_result = "H" if res["winner"] == "Ev Sahibi" else ("D" if res["winner"] == "Beraberlik" else "A")
            rec["correct"] = pred_result == actual[0]
        out.append(rec)

    with open("tahminler/v9_predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Sonuçlar tahminler/v9_predictions.json dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
