"""Tahminlerin tutup tutmadigini kontrol et."""
import json
import pandas as pd
import os

# Tahminleri yukle
with open("tahminler/predictions.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Features yukle (gercek sonuclar)
feat = pd.read_parquet("data/gold/features_combined.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")

# Son 30 gun icindeki mac sonuclarini topla
recent = feat[feat["date"] >= feat["date"].max() - pd.Timedelta(days=30)]
recent = recent[["home_team_id", "away_team_id", "home_goals", "away_goals", "date", "result", "home_corners", "away_corners"]].copy()
recent["total_corners"] = recent["home_corners"].fillna(0) + recent["away_corners"].fillna(0)

# Team name mapping
name_path = "data/gold/team_id_to_name.json"
with open(name_path) as fh:
    name_map = {str(k): str(v) for k, v in json.load(fh).items()}

# Tahminleri eslestir
matches = {}
for p in preds:
    key = f"{p['home_name']} vs {p['away_name']}"
    ts = p.get("timestamp", "")
    if key not in matches or ts > matches[key]["timestamp"]:
        matches[key] = p

# Her tahmini degerlendir
results = []
for key, p in matches.items():
    home = p["home_name"]
    away = p["away_name"]

    # Gercek sonucu bul (yaklasik isim eslesmesi)
    real = None
    for _, row in recent.iterrows():
        hn = name_map.get(str(int(row["home_team_id"])), "").lower()
        an = name_map.get(str(int(row["away_team_id"])), "").lower()
        hp = home.lower().strip()
        ap = away.lower().strip()
        if (hp in hn or hn in hp) and (ap in an or an in ap):
            real = row
            break

    if real is None:
        continue

    hg = int(real["home_goals"])
    ag = int(real["away_goals"])
    total = hg + ag
    corners = int(real.get("total_corners", 0))

    # Tahmin sonuclari
    tg = p.get("total_goals", {})
    cr = p.get("corners", {})
    dc = p.get("double_chance", {})
    btts_y = p.get("btts_yes", 0)
    btts_n = p.get("btts_no", 0)

    # Gol: over2.5
    pred_over25 = "U2.5" if tg.get("over25", 0) > tg.get("under25", 0) else "A2.5"
    real_over25 = "U2.5" if total > 2.5 else "A2.5"
    gol_tutmus = pred_over25 == real_over25

    # Korner: over7.5
    pred_cor = "U7.5" if cr.get("over75", 0) > cr.get("under75", 0) else "A7.5"
    real_cor = "U7.5" if corners > 7.5 else "A7.5"
    korner_tutmus = pred_cor == real_cor

    # BTTS
    pred_btts = "Var" if btts_y > btts_n else "Yok"
    real_btts = "Var" if hg > 0 and ag > 0 else "Yok"
    btts_tutmus = pred_btts == real_btts

    # Cift sans
    d1x = dc.get("1X", 0)
    dx2 = dc.get("X2", 0)
    d12 = dc.get("12", 0)
    best_dc = max(d1x, dx2, d12)
    if best_dc == d12:
        pred_dc = "12"
    elif best_dc == d1x:
        pred_dc = "1X"
    else:
        pred_dc = "X2"
    if hg > ag:
        real_dc = "1X"
    elif hg == ag:
        real_dc = "X2"
    else:
        real_dc = "X2"
    # Cift sans dogru mu?
    if hg > ag:
        dc_tutmus = pred_dc in ["1X", "12"]
    elif hg == ag:
        dc_tutmus = pred_dc in ["1X", "X2"]
    else:
        dc_tutmus = pred_dc in ["X2", "12"]

    # Kazanan
    pred_winner = p.get("result", {}).get("winner", "?")
    if hg > ag:
        real_winner = "Ev Sahibi"
    elif hg < ag:
        real_winner = "Deplasman"
    else:
        real_winner = "Beraberlik"
    kazanan_tutmus = pred_winner == real_winner

    results.append({
        "mac": key,
        "skor": f"{hg}-{ag}",
        "korner": corners,
        "kazanan": f"{'TUTTU' if kazanan_tutmus else 'TUTMADI'} ({pred_winner} vs {real_winner})",
        "gol": f"{'TUTTU' if gol_tutmus else 'TUTMADI'} ({pred_over25} vs {real_over25}, {total} gol)",
        "korner_r": f"{'TUTTU' if korner_tutmus else 'TUTMADI'} ({pred_cor} vs {real_cor}, {corners} korner)",
        "btts": f"{'TUTTU' if btts_tutmus else 'TUTMADI'} ({pred_btts} vs {real_btts})",
        "cift_sans": f"{'TUTTU' if dc_tutmus else 'TUTMADI'} ({pred_dc})",
    })

# Sonuclari yazdir
print(f"\n=== {len(results)} macin sonuclari ===\n")
tutar = {"kazanan": 0, "gol": 0, "korner": 0, "btts": 0, "cift_sans": 0}
for r in results:
    print(f"{r['mac']} ({r['skor']}, {r['korner']} korner)")
    print(f"  Kazanan: {r['kazanan']}")
    print(f"  Gol:     {r['gol']}")
    print(f"  Korner:  {r['korner_r']}")
    print(f"  BTTS:    {r['btts']}")
    print(f"  CiftSans: {r['cift_sans']}")
    print()
    if "TUTTU" in r["kazanan"]: tutar["kazanan"] += 1
    if "TUTTU" in r["gol"]: tutar["gol"] += 1
    if "TUTTU" in r["korner_r"]: tutar["korner"] += 1
    if "TUTTU" in r["btts"]: tutar["btts"] += 1
    if "TUTTU" in r["cift_sans"]: tutar["cift_sans"] += 1

n = len(results)
print("=== OZET ===")
for k, v in tutar.items():
    print(f"  {k}: {v}/{n} ({round(v/n*100, 1)}%)")
