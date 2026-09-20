"""Fast validation: directly test model on recent matches from features_combined."""
import pickle, sys, os
import pandas as pd
import numpy as np

# Load everything
with open("web_model.pkl", "rb") as f:
    wm = pickle.load(f)

feat = pd.read_parquet("data/gold/features_combined.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)

with open("team_index.pkl", "rb") as f:
    ti = pickle.load(f)
id_map = ti["team_id_map"]

feat_cols = wm["features"]
print(f"Model features: {len(feat_cols)}")

# Get last 200 matches from 2026
test = feat[feat["date"] >= "2026-01-01"].tail(500).copy()

# Build feature row like _predict_match does (simplified - using available data)
def build_features(row):
    """Build 68-feature row from features_combined row."""
    fr = {}
    
    # Elo
    for col in ["home_elo", "away_elo", "elo_diff", "home_attack_elo", "away_attack_elo",
                "home_defence_elo", "away_defence_elo", "attack_elo_diff", "defence_elo_diff"]:
        fr[col] = float(row.get(col, 0)) if pd.notna(row.get(col)) else 0
    
    # League
    for col in ["lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
                "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg"]:
        fr[col] = float(row.get(col, 0)) if pd.notna(row.get(col)) else 0
    
    # H2H (use defaults - we don't have pre-computed H2H per match)
    fr["h2h_home_win"] = 0.45
    fr["h2h_draw"] = 0.25
    fr["h2h_away_win"] = 0.30
    fr["h2h_goals_avg"] = 2.5
    fr["h2h_btts"] = 0.5
    
    # Rolling - use features_combined's sf_ data where available
    for src, dst in [("h_xg", "h_xg_r"), ("a_xg", "a_xg_r"),
                     ("home_shots", "h_shots_r"), ("away_shots", "a_shots_r"),
                     ("home_sot", "h_sot_r"), ("away_sot", "a_sot_r")]:
        fr[dst] = float(row.get(src, 0)) if pd.notna(row.get(src)) else 0
    
    # Rolling for big_ch, passes, fouls, tackles, yellows, dribbles, crosses - defaults
    for col in ["h_big_ch_r", "a_big_ch_r", "h_passes_r", "a_passes_r",
                "h_fouls_r", "a_fouls_r", "h_tackles_r", "a_tackles_r",
                "h_yellows_r", "a_yellows_r", "h_dribbles_r", "a_dribbles_r",
                "h_crosses_r", "a_crosses_r"]:
        if col not in fr:
            fr[col] = 0
    
    # Form
    for prefix, side in [("hf_", "home"), ("af_", "away")]:
        for n in [5, 10, 20]:
            gf = row.get(f"{side}_gf_{n}", 0)
            ga = row.get(f"{side}_ga_{n}", 0)
            pts = row.get(f"{side}_pts_{n}", 0)
            fr[f"{prefix}pts_{n}"] = float(pts) if pd.notna(pts) else 0
            fr[f"{prefix}gf_{n}"] = float(gf) if pd.notna(gf) else 0
            fr[f"{prefix}ga_{n}"] = float(ga) if pd.notna(ga) else 0
        fr[f"{prefix}rest_days"] = float(row.get(f"{side}_rest_days", 5)) if pd.notna(row.get(f"{side}_rest_days")) else 5
        
        # xg/sot/shots rolling - use sf_ if available
        for src, dst in [("home_xg", "hf_xg_r5"), ("away_xg", "af_xg_r5"),
                         ("home_sot", "hf_sot_r5"), ("away_sot", "af_sot_r5"),
                         ("home_shots", "hf_shots_r5"), ("away_shots", "af_shots_r5")]:
            fr[dst] = float(row.get(src, 0)) if pd.notna(row.get(src)) else 0
    
    return fr


correct_1x2 = 0
correct_btts = 0
correct_o25 = 0
total = 0
prob_h_correct = []
prob_a_correct = []

m1x2_model, m1x2_cal, _ = wm["m_1x2"]
mbtts_model, mbtts_cal, _ = wm["m_btts"]
mo25_model, mo25_cal, _ = wm["m_over25"]

print(f"\nTesting {len(test)} matches...")
for _, row in test.iterrows():
    try:
        fr = build_features(row)
        X = pd.DataFrame([fr])[feat_cols].fillna(0)
        
        # Predict 1X2
        prob = m1x2_cal.predict_proba(X)[0]
        hp, dp, ap = float(prob[0]), float(prob[1]), float(prob[2])
        
        if hp >= dp and hp >= ap:
            pred = "H"
        elif ap >= hp and ap >= dp:
            pred = "A"
        else:
            pred = "D"
        
        actual = row.get("result", "?")
        if actual == "?":
            continue
        
        total += 1
        if pred == actual:
            correct_1x2 += 1
        
        # BTTS
        btts_pred = float(mbtts_cal.predict_proba(X)[:, 1][0])
        btts_actual = int(row.get("btts", 0))
        if (btts_pred >= 0.5) == (btts_actual == 1):
            correct_btts += 1
        
        # Over 2.5
        o25_pred = float(mo25_cal.predict_proba(X)[:, 1][0])
        o25_actual = int(row.get("over25", 0))
        if (o25_pred >= 0.5) == (o25_actual == 1):
            correct_o25 += 1
        
    except Exception as e:
        continue

if total > 0:
    print(f"\n{'='*50}")
    print(f"RESULTS ({total} matches)")
    print(f"{'='*50}")
    print(f"1X2 Accuracy:  {correct_1x2}/{total} = {correct_1x2/total*100:.1f}%")
    print(f"BTTS Accuracy: {correct_btts}/{total} = {correct_btts/total*100:.1f}%")
    print(f"O2.5 Accuracy: {correct_o25}/{total} = {correct_o25/total*100:.1f}%")
    
    # Show some examples
    print(f"\nSample predictions (last 15):")
    for _, row in test.tail(15).iterrows():
        try:
            fr = build_features(row)
            X = pd.DataFrame([fr])[feat_cols].fillna(0)
            prob = m1x2_cal.predict_proba(X)[0]
            hp, dp, ap = float(prob[0]), float(prob[1]), float(prob[2])
            if hp >= dp and hp >= ap:
                pred = "H"
            elif ap >= hp and ap >= dp:
                pred = "A"
            else:
                pred = "D"
            actual = row.get("result", "?")
            flag = "✓" if pred == actual else "✗"
            
            h_name = id_map.get(row["home_team_id"], {}).get("name", str(row["home_team_id"]))
            a_name = id_map.get(row["away_team_id"], {}).get("name", str(row["away_team_id"]))
            print(f"  {flag} {h_name[:20]} vs {a_name[:20]}: "
                  f"{hp*100:.0f}/{dp*100:.0f}/{ap*100:.0f} -> {pred} (actual={actual})")
        except:
            pass
else:
    print("No matches tested!")
