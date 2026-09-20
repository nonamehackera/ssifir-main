import re

path = "web/app.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 1208-1321 arasi (1-indexed) silinecek -> 0-indexed 1207..1321
# 1207 = "" (bos satir), 1208 = "            feats = pd.DataFrame([row])"
start = 1207   # 0-indexed of line 1208
end = 1321     # 0-indexed of line 1322 (exclusive)

clean = '''            feats = pd.DataFrame([row])
            wm = STATE.get("web_model")
            if wm is not None:
                # xG (ev/deplasman)
                xg_h = max(float(np.mean([m.predict(feats[wm["feats"]].fillna(0)) for m in wm["m_h"]])), 0.1)
                xg_a = max(float(np.mean([m.predict(feats[wm["feats"]].fillna(0)) for m in wm["m_a"]])), 0.1)
                # korner
                kc_h = max(float(np.mean([m.predict(feats[wm["feats"]].fillna(0)) for m in wm["m_ch"]])), 0.1)
                kc_a = max(float(np.mean([m.predict(feats[wm["feats"]].fillna(0)) for m in wm["m_ca"]])), 0.1)
                # BTTS
                btts_p = float(np.mean([cal.predict(m.predict_proba(feats[wm["feats"]].fillna(0))[:,1]) for m,cal in wm["m_btts"]]))
                # DRAW
                draw_p = float(np.mean([cal.predict(m.predict_proba(feats[wm["feats"]].fillna(0))[:,1]) for m,cal in wm["m_draw"]]))

                hxg, axg = xg_h, xg_a
                # 1X2: Poisson tabanli (xG ile)
                from scipy.stats import poisson as _pois
                maxg = 8
                ph = [np.exp(-hxg)*(hxg**k)/math.factorial(k) for k in range(maxg)]
                pa = [np.exp(-axg)*(axg**k)/math.factorial(k) for k in range(maxg)]
                mat = np.outer(ph, pa); mat /= mat.sum()
                home_prob = float(np.trace(mat, offset=1))
                draw_prob = float(np.trace(mat))
                away_prob = float(np.trace(mat, offset=-1))
                s = home_prob + draw_prob + away_prob
                if s <= 0: s = 1.0
                home_prob, draw_prob, away_prob = home_prob/s, draw_prob/s, away_prob/s

                total_lam = hxg + axg
                over15 = 1.0 - (_pois.pmf(0, total_lam) + _pois.pmf(1, total_lam))
                over25 = 1.0 - sum(_pois.pmf(k, total_lam) for k in range(3))
                over35 = 1.0 - sum(_pois.pmf(k, total_lam) for k in range(4))

                # DRAW-AWARE kazanan
                if draw_p > wm["m_draw_th"]:
                    winner, win_prob = "Beraberlik", draw_prob
                elif home_prob >= away_prob:
                    winner, win_prob = "Ev Sahibi", home_prob
                else:
                    winner, win_prob = "Deplasman", away_prob

                double_chance = {
                    "1X": round((home_prob + draw_prob) * 100, 1),
                    "12": round((home_prob + away_prob) * 100, 1),
                    "X2": round((draw_prob + away_prob) * 100, 1),
                }

                corner_avg = kc_h + kc_a
                corners_over85 = round(float(1.0 - _pois.cdf(8, corner_avg)) * 100, 1)

                grid = []
                for h in range(0, 6):
                    for a in range(0, 6):
                        p = _pois.pmf(h, hxg) * _pois.pmf(a, axg)
                        grid.append((p, h, a))
                grid.sort(reverse=True)
                scores = [{"score": f"{h}-{a}", "probability": round(p, 4)} for p, h, a in grid[:5]]

                return {
                    "home": hd, "away": ad,
                    "home_win": round(home_prob * 100, 1),
                    "draw": round(draw_prob * 100, 1),
                    "away_win": round(away_prob * 100, 1),
                    "winner": winner,
                    "winner_prob": round(win_prob * 100, 1),
                    "btts_yes": round(btts_p * 100, 1),
                    "btts_no": round((1 - btts_p) * 100, 1),
                    "goals": {
                        "home_lambda": round(hxg, 3),
                        "away_lambda": round(axg, 3),
                        "expected_total": round(hxg + axg, 3),
                    },
                    "total_goals": {
                        "over25": round(over25 * 100, 1),
                        "under25": round((1 - over25) * 100, 1),
                        "over15": round(over15 * 100, 1),
                        "under15": round((1 - over15) * 100, 1),
                        "over35": round(over35 * 100, 1),
                        "under35": round((1 - over35) * 100, 1),
                    },
                    "double_chance": double_chance,
                    "corners": {
                        "avg": round(corner_avg, 1),
                        "over85": corners_over85,
                        "under85": round(100.0 - corners_over85, 1),
                    },
                    "top_scores": scores,
                }
            else:
                # Model yok (wm None), fallback sabit
                from scipy.stats import poisson as _pois
                hxg, axg = 1.3, 1.1
                total_lam = hxg + axg
                over15 = 1.0 - (_pois.pmf(0, total_lam) + _pois.pmf(1, total_lam))
                over25 = 1.0 - sum(_pois.pmf(k, total_lam) for k in range(3))
                return {
                    "home": hd, "away": ad,
                    "home_win": 45.0, "draw": 25.0, "away_win": 30.0,
                    "winner": "Ev Sahibi", "winner_prob": 45.0,
                    "btts_yes": 55.0, "btts_no": 45.0,
                    "goals": {"home_lambda": 1.3, "away_lambda": 1.1, "expected_total": 2.4},
                    "total_goals": {"over25": round(over25*100,1), "under25": round((1-over25)*100,1),
                                    "over15": round(over15*100,1), "under15": round((1-over15)*100,1),
                                    "over35": 0.0, "under35": 100.0},
                    "double_chance": {"1X": 70.0, "12": 75.0, "X2": 55.0},
                    "corners": {"avg": 10.0, "over85": 50.0, "under85": 50.0},
                    "top_scores": [],
                }
'''

new_lines = lines[:start] + [clean] + lines[end:]
with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("done")
