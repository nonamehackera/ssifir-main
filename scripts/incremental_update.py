"""HIZLI Incremental feature update - TUM feature'lari dogru hesapla.
Duzeltmeler:
  - lg_corner_avg / lg_card_avg dogru toplanir
  - home_w_corners/away_w_corners: 10-mac exp-weighted (engine ile ayni)
  - team_history kayitlarinda corner/shot/card verisi saklanir
  - home_form8, home_form20, weighted form eklendi
"""
import sys, os, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

t0 = time.time()

print("1. Veri yukleniyor...", flush=True)
old_feats = pd.read_parquet("data/gold/features.parquet")
old_feats["date"] = pd.to_datetime(old_feats["date"], errors="coerce")
last_date = old_feats["date"].max()
print(f"   features.parquet: {len(old_feats):,} satir, son: {last_date.date()}", flush=True)

mm = pd.read_parquet("data/gold/matches_master.parquet")
mm["date"] = pd.to_datetime(mm["date"], errors="coerce")
for c in ["home_team_id", "away_team_id"]:
    mm[c] = pd.to_numeric(mm[c], errors="coerce").astype("Int64")
mm = mm.dropna(subset=["home_team_id", "away_team_id", "date", "home_goals", "away_goals"])
if "result" not in mm.columns:
    mm["result"] = np.where(mm["home_goals"] > mm["away_goals"], "H",
                   np.where(mm["away_goals"] > mm["home_goals"], "A", "D"))
if "season" not in mm.columns:
    y = mm["date"].dt.year
    mm["season"] = np.where(mm["date"].dt.month >= 7, y.astype(str) + (y+1).astype(str), (y-1).astype(str) + y.astype(str))
if "total_goals" not in mm.columns:
    mm["total_goals"] = mm["home_goals"] + mm["away_goals"]
if "match_id" not in mm.columns:
    mm["match_id"] = range(len(mm))

new_matches = mm[mm["date"] > last_date].copy()
print(f"   Yeni mac: {len(new_matches):,}", flush=True)

if len(new_matches) == 0:
    print("Guncel!", flush=True)
    sys.exit(0)

print(f"\n2. Takim son durumlari hazirlaniyor ({time.time()-t0:.0f}s)...", flush=True)

# Her takim icin son 30 feature satiri - TUM istatistiklerle
team_history = []
for _, row in old_feats.iterrows():
    hid = int(row["home_team_id"])
    aid = int(row["away_team_id"])
    rec = {
        "date": row["date"], "home_team_id": hid, "away_team_id": aid,
        "home_goals": int(row.get("home_goals", 0) or 0),
        "away_goals": int(row.get("away_goals", 0) or 0),
        "result": row.get("result", "D"),
        "home_elo": float(row.get("home_elo", 1500) or 1500),
        "away_elo": float(row.get("away_elo", 1500) or 1500),
        "home_attack_elo": float(row.get("home_attack_elo", 1500) or 1500),
        "home_defence_elo": float(row.get("home_defence_elo", 1500) or 1500),
        "away_attack_elo": float(row.get("away_attack_elo", 1500) or 1500),
        "away_defence_elo": float(row.get("away_defence_elo", 1500) or 1500),
        "home_shots": row.get("home_shots"), "away_shots": row.get("away_shots"),
        "home_sot": row.get("home_sot"), "away_sot": row.get("away_sot"),
        "home_corners": row.get("home_corners"), "away_corners": row.get("away_corners"),
        "home_yellow": row.get("home_yellow"), "away_yellow": row.get("away_yellow"),
        "home_red": row.get("home_red", 0), "away_red": row.get("away_red", 0),
        "home_fouls": row.get("home_fouls"), "away_fouls": row.get("away_fouls"),
    }
    team_history.append((hid, aid, rec))

# Son 30 mac per-team
team_hist_map = {}
for hid, aid, rec in team_history:
    team_hist_map.setdefault(hid, []).append(rec)
    team_hist_map.setdefault(aid, []).append(rec)
for tid in team_hist_map:
    team_hist_map[tid] = team_hist_map[tid][-30:]

# ELO state (son bilinen)
team_elo = {}
for hid, aid, rec in team_history:
    team_elo[hid] = {"elo": rec["home_elo"], "attack": rec["home_attack_elo"], "defence": rec["home_defence_elo"]}
    team_elo[aid] = {"elo": rec["away_elo"], "attack": rec["away_attack_elo"], "defence": rec["away_defence_elo"]}

# League baselines - TUM stats
league_stats = {}
for _, row in old_feats.iterrows():
    lg = row["league"]
    if lg not in league_stats:
        league_stats[lg] = {"goals": 0, "home_g": 0, "away_g": 0, "n": 0,
                            "draws": 0, "btts": 0, "o25": 0, "corners": 0, "cards": 0}
    s = league_stats[lg]
    hg = int(row.get("home_goals", 0) or 0)
    ag = int(row.get("away_goals", 0) or 0)
    s["goals"] += hg + ag
    s["home_g"] += hg
    s["away_g"] += ag
    s["n"] += 1
    s["draws"] += 1 if hg == ag else 0
    s["btts"] += 1 if (hg > 0 and ag > 0) else 0
    s["o25"] += 1 if (hg + ag) >= 3 else 0
    hc = row.get("home_corners")
    ac = row.get("away_corners")
    if pd.notna(hc) and pd.notna(ac):
        s["corners"] += float(hc) + float(ac)
    hy = row.get("home_yellow", 0) or 0
    ay = row.get("away_yellow", 0) or 0
    s["cards"] += float(hy) + float(ay)

print(f"   {len(team_elo)} takim, {len(league_stats)} lig", flush=True)

print(f"\n3. Yeni maclar isleniyor ({len(new_matches)})...", flush=True)

ELO_K = 20

def elo_expected(ra, rb, home=True):
    diff = rb - ra if home else ra - rb
    return 1.0 / (1 + 10 ** (-diff / 400))

def get_history(team_id):
    return team_hist_map.get(team_id, [])

def _last_hist(hist, kick, n):
    return [r for r in hist[-n:] if r["date"] < kick] if hist else []

def _safe_float(v):
    if v is None:
        return np.nan
    try:
        f = float(v)
        return f if f == f else np.nan  # NaN check
    except:
        return np.nan

def calc_form(team_id, kick, n=10):
    hist = _last_hist(get_history(team_id), kick, n)
    if not hist:
        return {"gf": 0, "ga": 0, "pts": 0, "shots": 0, "sot": 0, "corners": 0, "cards": 0}
    is_h = [int(r["home_team_id"]) == team_id for r in hist]
    gf = sum(float(r["home_goals"] if h else r["away_goals"]) for r, h in zip(hist, is_h))
    ga = sum(float(r["away_goals"] if h else r["home_goals"]) for r, h in zip(hist, is_h))
    pts = sum(
        3 if (h and r["result"] == "H") or (not h and r["result"] == "A") else (1 if r["result"] == "D" else 0)
        for r, h in zip(hist, is_h)
    )
    n_real = max(len(hist), 1)
    shots = np.nanmean([_safe_float(r.get("home_shots" if h else "away_shots")) for r, h in zip(hist, is_h)])
    sot = np.nanmean([_safe_float(r.get("home_sot" if h else "away_sot")) for r, h in zip(hist, is_h)])
    corners = np.nanmean([_safe_float(r.get("home_corners" if h else "away_corners")) for r, h in zip(hist, is_h)])
    cards = np.nanmean([_safe_float(r.get("home_yellow" if h else "away_yellow")) for r, h in zip(hist, is_h)])
    return {"gf": gf/n_real, "ga": ga/n_real, "pts": pts,
            "shots": float(shots) if not np.isnan(shots) else 0,
            "sot": float(sot) if not np.isnan(sot) else 0,
            "corners": float(corners) if not np.isnan(corners) else 0,
            "cards": float(cards) if not np.isnan(cards) else 0}

def calc_form_home(team_id, kick, n=10):
    hist = [r for r in _last_hist(get_history(team_id), kick, n) if int(r["home_team_id"]) == team_id]
    if not hist:
        return {"gf": 0, "ga": 0, "pts": 0}
    gf = sum(float(r["home_goals"]) for r in hist)
    ga = sum(float(r["away_goals"]) for r in hist)
    pts = sum(3 if r["result"] == "H" else (1 if r["result"] == "D" else 0) for r in hist)
    n_real = max(len(hist), 1)
    return {"gf": gf/n_real, "ga": ga/n_real, "pts": pts}

def calc_form_away(team_id, kick, n=10):
    hist = [r for r in _last_hist(get_history(team_id), kick, n) if int(r["home_team_id"]) != team_id]
    if not hist:
        return {"gf": 0, "ga": 0, "pts": 0}
    gf = sum(float(r["away_goals"]) for r in hist)
    ga = sum(float(r["home_goals"]) for r in hist)
    pts = sum(3 if r["result"] == "A" else (1 if r["result"] == "D" else 0) for r in hist)
    n_real = max(len(hist), 1)
    return {"gf": gf/n_real, "ga": ga/n_real, "pts": pts}

def calc_weighted_form(team_id, kick, n=10):
    hist = _last_hist(get_history(team_id), kick, n)
    if not hist:
        return {"gf": 0, "ga": 0, "pts": 0, "shots": 0, "sot": 0, "corners": 0, "cards": 0}
    days = [(kick - r["date"]).days for r in hist]
    weights = np.exp(-np.array(days, dtype=float) / 60.0)
    ws = weights.sum()
    if ws <= 0:
        return calc_form(team_id, kick, n)
    is_h = [int(r["home_team_id"]) == team_id for r in hist]
    gf = sum(float(r["home_goals"] if h else r["away_goals"]) * w for r, h, w in zip(hist, is_h, weights)) / ws
    ga = sum(float(r["away_goals"] if h else r["home_goals"]) * w for r, h, w in zip(hist, is_h, weights)) / ws
    pts = sum(
        (3 if (h and r["result"] == "H") or (not h and r["result"] == "A") else (1 if r["result"] == "D" else 0)) * w
        for r, h, w in zip(hist, is_h, weights)
    ) / ws
    shots_vals = [_safe_float(r.get("home_shots" if h else "away_shots")) for r, h in zip(hist, is_h)]
    sot_vals = [_safe_float(r.get("home_sot" if h else "away_sot")) for r, h in zip(hist, is_h)]
    corner_vals = [_safe_float(r.get("home_corners" if h else "away_corners")) for r, h in zip(hist, is_h)]
    card_vals = [_safe_float(r.get("home_yellow" if h else "away_yellow")) for r, h in zip(hist, is_h)]
    def _wmean(vals, ws_arr):
        pairs = [(v, w) for v, w in zip(vals, ws_arr) if not np.isnan(v)]
        if not pairs:
            return 0.0
        vs, ws_s = zip(*pairs)
        sw = sum(ws_s)
        return sum(v * w for v, w in zip(vs, ws_s)) / sw if sw > 0 else 0.0
    shots = _wmean(shots_vals, weights)
    sot = _wmean(sot_vals, weights)
    corners = _wmean(corner_vals, weights)
    cards = _wmean(card_vals, weights)
    return {"gf": gf, "ga": ga, "pts": pts, "shots": shots, "sot": sot, "corners": corners, "cards": cards}

def count_wins_last5(team_id, kick):
    hist = _last_hist(get_history(team_id), kick, 5)
    is_h = [int(r["home_team_id"]) == team_id for r in hist]
    return sum(1 for r, h in zip(hist, is_h) if (h and r["result"] == "H") or (not h and r["result"] == "A"))

def count_draws_last5(team_id, kick):
    return sum(1 for r in _last_hist(get_history(team_id), kick, 5) if r["result"] == "D")

def form_std(team_id, kick):
    hist = _last_hist(get_history(team_id), kick, 5)
    is_h = [int(r["home_team_id"]) == team_id for r in hist]
    pts_list = [
        3 if (h and r["result"] == "H") or (not h and r["result"] == "A") else (1 if r["result"] == "D" else 0)
        for r, h in zip(hist, is_h)
    ]
    return float(np.std(pts_list)) if len(pts_list) >= 2 else 0.0

def implied(odds):
    try:
        o = float(odds)
        return 1/o if o > 1 else None
    except:
        return None

new_rows = []
for idx, m in new_matches.iterrows():
    hid = int(m["home_team_id"])
    aid = int(m["away_team_id"])
    lg = m["league"]
    kick = m["date"]
    
    he = team_elo.get(hid, {"elo": 1500, "attack": 1500, "defence": 1500})
    ae = team_elo.get(aid, {"elo": 1500, "attack": 1500, "defence": 1500})
    
    h_elo_b, a_elo_b = he["elo"], ae["elo"]
    h_att_b, h_def_b = he["attack"], he["defence"]
    a_att_b, a_def_b = ae["attack"], ae["defence"]
    
    h_form = calc_form(hid, kick, 5)
    a_form = calc_form(aid, kick, 5)
    h_form3 = calc_form(hid, kick, 3)
    a_form3 = calc_form(aid, kick, 3)
    h_form8 = calc_form(hid, kick, 8)
    a_form8 = calc_form(aid, kick, 8)
    h_form20 = calc_form(hid, kick, 20)
    a_form20 = calc_form(aid, kick, 20)
    h_form_h = calc_form_home(hid, kick, 5)
    a_form_a = calc_form_away(aid, kick, 5)
    h_form_h20 = calc_form_home(hid, kick, 20)
    a_form_a20 = calc_form_away(aid, kick, 20)
    h_std = calc_form(hid, kick, 10)
    a_std = calc_form(aid, kick, 10)
    h_w = calc_weighted_form(hid, kick, 10)
    a_w = calc_weighted_form(aid, kick, 10)
    
    hg = int(m["home_goals"])
    ag = int(m["away_goals"])
    result = m["result"]
    
    mkt_h = implied(m.get("avg_home_odds"))
    mkt_d = implied(m.get("avg_draw_odds"))
    mkt_a = implied(m.get("avg_away_odds"))
    mkt_o25 = implied(m.get("avg_under25_odds"))
    
    ls = league_stats.get(lg, {"goals": 0, "home_g": 0, "away_g": 0, "n": 1, "draws": 0, "btts": 0, "o25": 0, "corners": 90, "cards": 40})
    ln = max(ls["n"], 1)
    
    # H2H
    h2h_hist = [r for r in get_history(hid) if (int(r["away_team_id"]) == aid or int(r["home_team_id"]) == aid) and r["date"] < kick]
    h2h_last = h2h_hist[-5:] if len(h2h_hist) >= 5 else h2h_hist
    h2h_n = max(len(h2h_last), 1)
    h2h_hw = sum(1 for r in h2h_last if r["result"] == "H")
    h2h_d = sum(1 for r in h2h_last if r["result"] == "D")
    h2h_aw = sum(1 for r in h2h_last if r["result"] == "A")
    h2h_goals = np.mean([int(r.get("home_goals", 0) or 0) + int(r.get("away_goals", 0) or 0) for r in h2h_last]) if h2h_last else 2.5
    h2h_btts = np.mean([1 if (int(r.get("home_goals", 0) or 0) > 0 and int(r.get("away_goals", 0) or 0) > 0) else 0 for r in h2h_last]) if h2h_last else 0.5
    
    # Opp elo
    h_opp_hist = _last_hist(get_history(hid), kick, 10)
    h_opp = np.mean([team_elo.get(int(r["away_team_id"]), {"elo": 1500})["elo"] if int(r["home_team_id"]) == hid else team_elo.get(int(r["home_team_id"]), {"elo": 1500})["elo"] for r in h_opp_hist]) if h_opp_hist else 1500
    a_opp_hist = _last_hist(get_history(aid), kick, 10)
    a_opp = np.mean([team_elo.get(int(r["away_team_id"]), {"elo": 1500})["elo"] if int(r["home_team_id"]) == aid else team_elo.get(int(r["home_team_id"]), {"elo": 1500})["elo"] for r in a_opp_hist]) if a_opp_hist else 1500
    
    # Rest days
    h_rest_data = _last_hist(get_history(hid), kick, 1)
    a_rest_data = _last_hist(get_history(aid), kick, 1)
    h_rest = min((kick - h_rest_data[-1]["date"]).days, 30) if h_rest_data else 7
    a_rest = min((kick - a_rest_data[-1]["date"]).days, 30) if a_rest_data else 7
    
    rec = {
        "match_id": m["match_id"], "league": lg, "season": m["season"], "date": kick,
        "home_team_id": hid, "away_team_id": aid,
        "home_elo": h_elo_b, "away_elo": a_elo_b, "elo_diff": h_elo_b - a_elo_b,
        "home_attack_elo": h_att_b, "home_defence_elo": h_def_b,
        "away_attack_elo": a_att_b, "away_defence_elo": a_def_b,
        "attack_elo_diff": h_att_b - a_att_b, "defence_elo_diff": h_def_b - a_def_b,
        "home_gf_5": h_form["gf"], "home_ga_5": h_form["ga"], "home_pts_5": h_form["pts"],
        "home_shots_5": h_form["shots"], "home_sot_5": h_form["sot"],
        "home_corners_5": h_form["corners"], "home_cards_5": h_form["cards"],
        "home_w_gf": h_w["gf"], "home_w_ga": h_w["ga"],
        "home_w_shots": h_w["shots"], "home_w_sot": h_w["sot"],
        "home_w_corners": h_w["corners"], "home_w_cards": h_w["cards"],
        "home_momentum": count_wins_last5(hid, kick) / 5.0,
        "home_wins_last5": count_wins_last5(hid, kick),
        "home_draws_last5": count_draws_last5(hid, kick),
        "home_form_std": form_std(hid, kick),
        "home_gdiff5": h_form["gf"] - h_form["ga"],
        "home_gf_5_real": h_form["gf"], "home_ga_5_real": h_form["ga"],
        "away_gf_5_real": a_form["gf"], "away_ga_5_real": a_form["ga"],
        "home_hgf_5": h_form_h["gf"], "home_hga_5": h_form_h["ga"], "home_hpts_5": h_form_h["pts"],
        "home_gf_3": h_form3["gf"], "home_ga_3": h_form3["ga"], "home_pts_3": h_form3["pts"],
        "home_gf_8": h_form8["gf"], "home_ga_8": h_form8["ga"],
        "home_gf_20": h_form20["gf"], "home_ga_20": h_form20["ga"],
        "home_gf_std": h_std["gf"], "home_ga_std": h_std["ga"], "home_pts_std": h_std["pts"],
        "away_gf_5": a_form["gf"], "away_ga_5": a_form["ga"], "away_pts_5": a_form["pts"],
        "away_shots_5": a_form["shots"], "away_sot_5": a_form["sot"],
        "away_corners_5": a_form["corners"], "away_cards_5": a_form["cards"],
        "away_w_gf": a_w["gf"], "away_w_ga": a_w["ga"],
        "away_w_shots": a_w["shots"], "away_w_sot": a_w["sot"],
        "away_w_corners": a_w["corners"], "away_w_cards": a_w["cards"],
        "away_momentum": count_wins_last5(aid, kick) / 5.0,
        "away_wins_last5": count_wins_last5(aid, kick),
        "away_draws_last5": count_draws_last5(aid, kick),
        "away_form_std": form_std(aid, kick),
        "away_gdiff5": a_form["gf"] - a_form["ga"],
        "away_agf_5": a_form_a["gf"], "away_aga_5": a_form_a["ga"], "away_apts_5": a_form_a["pts"],
        "away_gf_3": a_form3["gf"], "away_ga_3": a_form3["ga"], "away_pts_3": a_form3["pts"],
        "away_gf_8": a_form8["gf"], "away_ga_8": a_form8["ga"],
        "away_gf_20": a_form20["gf"], "away_ga_20": a_form20["ga"],
        "away_gf_std": a_std["gf"], "away_ga_std": a_std["ga"], "away_pts_std": a_std["pts"],
        "home_rest_days": h_rest, "away_rest_days": a_rest,
        "h2h_home_win": h2h_hw/h2h_n, "h2h_draw": h2h_d/h2h_n, "h2h_away_win": h2h_aw/h2h_n,
        "h2h_goals_avg": h2h_goals, "h2h_btts": h2h_btts,
        "home_opp_elo": h_opp, "away_opp_elo": a_opp,
        "lg_avg_goals": ls["goals"]/ln, "lg_home_goal_avg": ls["home_g"]/ln,
        "lg_away_goal_avg": ls["away_g"]/ln,
        "lg_draw_rate": ls["draws"]/ln, "lg_btts_rate": ls["btts"]/ln,
        "lg_over25_rate": ls["o25"]/ln,
        "lg_corner_avg": ls["corners"]/ln,
        "lg_card_avg": ls["cards"]/ln,
        "mkt_home_prob": mkt_h, "mkt_draw_prob": mkt_d, "mkt_away_prob": mkt_a,
        "mkt_over25_prob": mkt_o25,
        "avg_home_odds": float(m.get("avg_home_odds") or 0) if pd.notna(m.get("avg_home_odds")) else None,
        "avg_draw_odds": float(m.get("avg_draw_odds") or 0) if pd.notna(m.get("avg_draw_odds")) else None,
        "avg_away_odds": float(m.get("avg_away_odds") or 0) if pd.notna(m.get("avg_away_odds")) else None,
        "avg_over25_odds": float(m.get("avg_over25_odds") or 0) if pd.notna(m.get("avg_over25_odds")) else None,
        "mkt_close_home_prob": mkt_h, "mkt_close_draw_prob": mkt_d, "mkt_close_away_prob": mkt_a,
        "mkt_close_over25_prob": mkt_o25,
        "avg_close_home_odds": float(m.get("avg_home_odds") or 0) if pd.notna(m.get("avg_home_odds")) else None,
        "avg_close_draw_odds": float(m.get("avg_draw_odds") or 0) if pd.notna(m.get("avg_draw_odds")) else None,
        "avg_close_away_odds": float(m.get("avg_away_odds") or 0) if pd.notna(m.get("avg_away_odds")) else None,
        "avg_close_over25_odds": float(m.get("avg_over25_odds") or 0) if pd.notna(m.get("avg_over25_odds")) else None,
        "referee": m.get("referee", ""),
        "home_goals": hg, "away_goals": ag, "total_goals": hg + ag,
        "result": result,
        "result_H": 1 if result == "H" else 0, "result_D": 1 if result == "D" else 0, "result_A": 1 if result == "A" else 0,
        "btts": 1 if (hg > 0 and ag > 0) else 0,
        "over05": 1 if hg + ag > 0.5 else 0, "over15": 1 if hg + ag > 1.5 else 0,
        "over25": 1 if hg + ag > 2.5 else 0, "over35": 1 if hg + ag > 3.5 else 0,
        "over45": 1 if hg + ag > 4.5 else 0,
        "home_shots": m.get("home_shots"), "away_shots": m.get("away_shots"),
        "home_sot": m.get("home_sot"), "away_sot": m.get("away_sot"),
        "home_corners": m.get("home_corners"), "away_corners": m.get("away_corners"),
        "home_yellow": m.get("home_yellow"), "away_yellow": m.get("away_yellow"),
        "home_red": m.get("home_red"), "away_red": m.get("away_red"),
        "home_fouls": m.get("home_fouls"), "away_fouls": m.get("away_fouls"),
        "ht_home_goals": m.get("ht_home_goals"), "ht_away_goals": m.get("ht_away_goals"),
    }
    new_rows.append(rec)
    
    # ELO GUNCELLE
    pts_h = 3 if result == "H" else (1 if result == "D" else 0)
    pts_a = 3 if result == "A" else (1 if result == "D" else 0)
    eh = elo_expected(h_elo_b, a_elo_b, True)
    ea = elo_expected(a_elo_b, h_elo_b, False)
    he["elo"] = h_elo_b + ELO_K * (pts_h / 3.0 - eh)
    ae["elo"] = a_elo_b + ELO_K * (pts_a / 3.0 - ea)
    h_att_exp = elo_expected(h_att_b, a_def_b, True)
    a_att_exp = elo_expected(a_att_b, h_def_b, False)
    he["attack"] = h_att_b + ELO_K * (hg / max(hg + ag, 1) - h_att_exp)
    ae["attack"] = a_att_b + ELO_K * (ag / max(hg + ag, 1) - a_att_exp)
    he["defence"] = h_def_b + ELO_K * (ag / max(hg + ag, 1) - (1 - h_att_exp))
    ae["defence"] = a_def_b + ELO_K * (hg / max(hg + ag, 1) - (1 - a_att_exp))
    
    # TUM verilerle guncelle
    new_rec = {
        "date": kick, "home_team_id": hid, "away_team_id": aid,
        "home_goals": hg, "away_goals": ag, "result": result,
        "home_elo": he["elo"], "away_elo": ae["elo"],
        "home_attack_elo": he["attack"], "home_defence_elo": he["defence"],
        "away_attack_elo": ae["attack"], "away_defence_elo": ae["defence"],
        "home_shots": m.get("home_shots"), "away_shots": m.get("away_shots"),
        "home_sot": m.get("home_sot"), "away_sot": m.get("away_sot"),
        "home_corners": m.get("home_corners"), "away_corners": m.get("away_corners"),
        "home_yellow": m.get("home_yellow"), "away_yellow": m.get("away_yellow"),
        "home_red": m.get("home_red", 0), "away_red": m.get("away_red", 0),
    }
    team_hist_map.setdefault(hid, []).append(new_rec)
    team_hist_map.setdefault(aid, []).append(new_rec)
    
    # League stats guncelle
    ls["goals"] += hg + ag
    ls["home_g"] += hg
    ls["away_g"] += ag
    ls["n"] += 1
    ls["draws"] += 1 if hg == ag else 0
    ls["btts"] += 1 if (hg > 0 and ag > 0) else 0
    ls["o25"] += 1 if (hg + ag) >= 3 else 0
    hc = m.get("home_corners")
    ac = m.get("away_corners")
    if pd.notna(hc) and pd.notna(ac):
        ls["corners"] += float(hc) + float(ac)
    hy = m.get("home_yellow", 0) or 0
    ay = m.get("away_yellow", 0) or 0
    ls["cards"] += float(hy) + float(ay)
    
    if len(new_rows) % 5000 == 0:
        print(f"   {len(new_rows):,}/{len(new_matches):,} ({time.time()-t0:.0f}s)", flush=True)

print(f"   {len(new_rows)} satir islendi ({time.time()-t0:.0f}s)", flush=True)

print(f"\n4. Birlestir + kaydet...", flush=True)
new_df = pd.DataFrame(new_rows)
for c in old_feats.columns:
    if c not in new_df.columns:
        new_df[c] = None
new_df = new_df[old_feats.columns]
final = pd.concat([old_feats, new_df], ignore_index=True)
final = final.sort_values("date").reset_index(drop=True)
final.to_parquet("data/gold/features.parquet", index=False)
print(f"   features.parquet: {len(final):,} satir", flush=True)
print(f"   Tarih: {final['date'].min()} -> {final['date'].max()}", flush=True)
print(f"\nToplam: {time.time()-t0:.0f}s", flush=True)
print("Bitti!", flush=True)
