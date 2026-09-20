"""Kupon durum kontrolcü — Tahminlerin maç sonuçlarını web ve önbellekten çeker, kupon kazanç durumunu gösterir."""
import os
import json
import sys
from datetime import datetime

import prediction.match_score_fetcher as score_fetcher

PREDS_FILE = os.path.join("tahminler", "predictions.json")

def load_predictions():
    if os.path.exists(PREDS_FILE):
        with open(PREDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def evaluate_coupon(pred, actual):
    if not actual:
        return None

    hg, ag = actual["home_goals"], actual["away_goals"]
    total = actual["total_goals"]

    markets = []

    # 1X2
    r = pred.get("result", {})
    home_w, draw, away_w = r.get("home_win", 0), r.get("draw", 0), r.get("away_win", 0)
    if home_w >= draw and home_w >= away_w:
        pred_1x2 = "1"
    elif draw >= home_w and draw >= away_w:
        pred_1x2 = "X"
    else:
        pred_1x2 = "2"

    actual_1x2 = "1" if hg > ag else ("X" if hg == ag else "2")
    markets.append({
        "name": "1X2",
        "prediction": pred_1x2,
        "actual": actual_1x2,
        "won": pred_1x2 == actual_1x2,
    })

    # Gol Üst/Alt 2.5
    tg = pred.get("total_goals", {})
    over25_pred = tg.get("over25", 50)
    pred_goal_market = "Üst 2.5" if over25_pred >= 50 else "Alt 2.5"
    actual_goal_market = "Üst 2.5" if total >= 3 else "Alt 2.5"
    markets.append({
        "name": "Gol Üst/Alt 2.5",
        "prediction": pred_goal_market,
        "actual": actual_goal_market,
        "won": pred_goal_market == actual_goal_market,
    })

    # BTTS
    btts_yes = pred.get("btts_yes", 50)
    pred_btts = "Var" if btts_yes >= 50 else "Yok"
    actual_btts = "Var" if (hg > 0 and ag > 0) else "Yok"
    markets.append({
        "name": "BTTS",
        "prediction": pred_btts,
        "actual": actual_btts,
        "won": pred_btts == actual_btts,
    })

    won_count = sum(1 for m in markets if m["won"])
    return {
        "markets": markets,
        "won_count": won_count,
        "total_markets": len(markets),
        "score": f"{won_count}/{len(markets)}",
    }

preds = load_predictions()

matched = []
unmatched_no_data = []
unmatched_future = []
unmatched_unknown = []

print(f"Toplam prediction: {len(preds)}", flush=True)

for i, p in enumerate(preds):
    hname = p.get("home_name")
    aname = p.get("away_name")
    hid = p.get("home_id")
    aid = p.get("away_id")
    match_date = p.get("match_date", "")

    res = score_fetcher.get_match_result(hid, aid, hname, aname, match_date)

    if res:
        ev = evaluate_coupon(p, res)
        matched.append((i + 1, hname, aname, res['home_goals'], res['away_goals'], res['source'], ev['score'], ev))
        print(f"  {i+1:2d}. {hname} vs {aname} -> {res['home_goals']}-{res['away_goals']} ({res['source']}) | Başarı: {ev['score']}", flush=True)
    else:
        if match_date:
            try:
                md = datetime.fromisoformat(match_date)
                if md.date() > datetime.now().date():
                    unmatched_future.append((i + 1, hname, aname, match_date, "Gelecek tarihi - henuz oynanmadi"))
                    print(f"  {i+1:2d}. {hname} vs {aname} -> Gelecek Maç ({match_date})", flush=True)
                else:
                    unmatched_no_data.append((i + 1, hname, aname, match_date, "Veri kaynaginda sonuc bulunamadi"))
                    print(f"  {i+1:2d}. {hname} vs {aname} -> Sonuç Bulunamadı", flush=True)
            except Exception:
                unmatched_unknown.append((i + 1, hname, aname, match_date, "Tarih parse edilemedi"))
        else:
            unmatched_unknown.append((i + 1, hname, aname, "N/A", "match_date yok"))

print(f"\n--- ÖZET ---", flush=True)
print(f"Eşleşen Maç: {len(matched)}", flush=True)
print(f"Gelecek Maç: {len(unmatched_future)}", flush=True)
print(f"Sonuçsuz Maç: {len(unmatched_no_data)}", flush=True)
print(f"Belirsiz: {len(unmatched_unknown)}", flush=True)
