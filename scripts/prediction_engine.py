"""
CALISKAN MOTOR v8: Gerçek form cache, xG entegrasyonu, tam feature seti
Her model tipi için doğru feature seti (MAIN=70, GOAL=69, CORNER=74)
"""
import pickle, json, sys, os, math, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === LOAD DATA ===
print("Veriler yükleniyor...")
with open('data/web_model.pkl', 'rb') as f:
    wm = pickle.load(f)

feat = pd.read_parquet("data/gold/features_combined.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
for c in ["home_team_id","away_team_id"]:
    feat[c] = pd.to_numeric(feat[c], errors="coerce")
feat = feat.dropna(subset=["home_team_id","away_team_id"])
feat["home_team_id"] = feat["home_team_id"].astype(int)
feat["away_team_id"] = feat["away_team_id"].astype(int)
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"]>0) & (feat["away_goals"]>0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)

with open("data/gold/team_id_to_name.json") as fh:
    name_map = {int(k): v for k, v in json.load(fh).items()}
    name_map_rev = {v.lower(): int(k) for k, v in name_map.items()}

FEAT_MAIN = wm["features"]
FEAT_GOAL = wm["goal_features"]
FEAT_CORNER = wm["corner_features"]

print(f"Feature sets: MAIN={len(FEAT_MAIN)} GOAL={len(FEAT_GOAL)} CORNER={len(FEAT_CORNER)}")
print(f"Toplam maç: {len(feat)}, Tarih: {feat['date'].min()} - {feat['date'].max()}")

# === BUILD REAL FORM CACHE ===
print("Gerçek form cache oluşturuluyor (son 5 maç rolling ortalama)...")

all_ids = sorted(set(feat["home_team_id"].unique()) | set(feat["away_team_id"].unique()))

# Her takım için hem home hem away maçlarını topla
team_matches = {}
for tid in all_ids:
    home_matches = feat[feat["home_team_id"] == tid].copy()
    home_matches["is_home"] = True
    home_matches["goals_for"] = home_matches["home_goals"]
    home_matches["goals_against"] = home_matches["away_goals"]
    home_matches["opp_id"] = home_matches["away_team_id"]

    away_matches = feat[feat["away_team_id"] == tid].copy()
    away_matches["is_home"] = False
    away_matches["goals_for"] = away_matches["away_goals"]
    away_matches["goals_against"] = away_matches["home_goals"]
    away_matches["opp_id"] = away_matches["home_team_id"]

    all_m = pd.concat([home_matches, away_matches]).sort_values("date").reset_index(drop=True)
    team_matches[tid] = all_m

def calc_form(tid, n=5):
    """Son N maçın formunu hesapla"""
    matches = team_matches.get(tid)
    if matches is None or len(matches) == 0:
        return None

    last_n = matches.tail(n)
    if len(last_n) == 0:
        return None

    # Temel form
    gf = last_n["goals_for"].mean()
    ga = last_n["goals_against"].mean()
    pts_series = last_n.apply(lambda r: 3 if r["goals_for"]>r["goals_against"] else (1 if r["goals_for"]==r["goals_against"] else 0), axis=1)
    pts = pts_series.sum()
    wins = int((pts_series == 3).sum())
    draws = int((pts_series == 1).sum())
    gdiff = gf - ga
    momentum = pts / (3 * len(last_n))

    # Ev sahibi olarak son 5
    home_matches = last_n[last_n["is_home"] == True].tail(5)
    hgf = home_matches["goals_for"].mean() if len(home_matches) > 0 else gf
    hga = home_matches["goals_against"].mean() if len(home_matches) > 0 else ga
    hpts = home_matches.apply(lambda r: 3 if r["goals_for"]>r["goals_against"] else (1 if r["goals_for"]==r["goals_against"] else 0), axis=1).sum() if len(home_matches) > 0 else pts

    # Deplasman olarak son 5
    away_matches = last_n[last_n["is_home"] == False].tail(5)
    agf = away_matches["goals_for"].mean() if len(away_matches) > 0 else gf
    aga = away_matches["goals_against"].mean() if len(away_matches) > 0 else ga
    apts = away_matches.apply(lambda r: 3 if r["goals_for"]>r["goals_against"] else (1 if r["goals_for"]==r["goals_against"] else 0), axis=1).sum() if len(away_matches) > 0 else pts

    # Son 3 maç
    last3 = matches.tail(3)
    gf3 = last3["goals_for"].mean()
    ga3 = last3["goals_against"].mean()

    #shots/sofa columns
    home_shots_avg = last_n["home_shots"].mean() if "home_shots" in last_n.columns else 0
    home_sot_avg = last_n["home_sot"].mean() if "home_sot" in last_n.columns else 0
    away_shots_avg = last_n["away_shots"].mean() if "away_shots" in last_n.columns else 0
    away_sot_avg = last_n["away_sot"].mean() if "away_sot" in last_n.columns else 0

    # SofaScore specific
    sf_xg = last_n["sf_home_xg"].mean() if "sf_home_xg" in last_n.columns and last_n["sf_home_xg"].notna().any() else 0
    sf_poss = last_n["sf_home_possession"].mean() if "sf_home_possession" in last_n.columns and last_n["sf_home_possession"].notna().any() else 0
    sf_tackles = last_n["sf_home_tackles"].mean() if "sf_home_tackles" in last_n.columns and last_n["sf_home_tackles"].notna().any() else 0
    sf_interc = last_n["sf_home_interceptions"].mean() if "sf_home_interceptions" in last_n.columns and last_n["sf_home_interceptions"].notna().any() else 0
    sf_saves = last_n["sf_home_saves"].mean() if "sf_home_saves" in last_n.columns and last_n["sf_home_saves"].notna().any() else 0
    sf_bigch = 0
    if "sf_home_big_chances_scored" in last_n.columns and last_n["sf_home_big_chances_scored"].notna().any():
        sf_bigch = last_n["sf_home_big_chances_scored"].mean()
    if "sf_home_big_chances_missed" in last_n.columns and last_n["sf_home_big_chances_missed"].notna().any():
        sf_bigch += last_n["sf_home_big_chances_missed"].mean()

    # Rest days
    rest_days = 7
    if len(matches) > 1:
        diff = (matches.iloc[-1]["date"] - matches.iloc[-2]["date"]).days
        rest_days = max(1, min(14, diff)) if diff > 0 else 7

    # Opponent ELO
    opp_elo = 1500
    if len(last_n) > 0:
        last_opp = last_n.iloc[-1]["opp_id"]
        last_opp_matches = feat[(feat["home_team_id"] == last_opp) | (feat["away_team_id"] == last_opp)]
        if len(last_opp_matches) > 0:
            opp_elo = float(last_opp_matches.iloc[-1].get("home_elo", 1500) or 1500)

    # Std
    pts_list = pts_series.tolist()
    pts_std = np.std(pts_list) if len(pts_list) > 1 else 0

    return {
        "gf_5": gf, "ga_5": ga, "pts_5": pts, "gf_3": gf3, "ga_3": ga3,
        "hgf_5": hgf, "hga_5": hga, "hpts_5": hpts,
        "w_gf": gf, "w_ga": ga, "w_shots": home_shots_avg, "w_sot": home_sot_avg,
        "rest_days": rest_days, "momentum": momentum, "wins_last5": wins,
        "draws_last5": draws, "gdiff5": gdiff, "opp_elo": opp_elo,
        "pts_std": pts_std,
        # SofaScore
        "sf_xg": sf_xg, "sf_poss": sf_poss, "sf_tackles": sf_tackles,
        "sf_interc": sf_interc, "sf_saves": sf_saves, "sf_bigch": sf_bigch,
        "home_shots": home_shots_avg, "home_sot": home_sot_avg,
        "away_shots": away_shots_avg, "away_sot": away_sot_avg,
    }

# Precompute form cache
form_cache = {}
for i, tid in enumerate(all_ids):
    if i % 500 == 0:
        print(f"  {i}/{len(all_ids)} takım formu hesaplandı")
    form_cache[tid] = calc_form(tid)

# Also build H2H cache
print("H2H cache oluşturuluyor...")
h2h_cache = {}
for idx in range(len(feat)):
    row = feat.iloc[idx]
    hid = int(row["home_team_id"])
    aid = int(row["away_team_id"])
    key = (hid, aid)
    if key not in h2h_cache:
        h2h_cache[key] = []
    h2h_cache[key].append(row)

print(f"Form cache: {len(form_cache)} takım")
print(f"H2H cache: {len(h2h_cache)} çift")

# === HELPERS ===
def find_id(name):
    nl = name.lower().strip()
    if nl in name_map_rev: return name_map_rev[nl]
    for n, tid in name_map_rev.items():
        if nl in n or n in nl: return tid
    return None

def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        if hasattr(model, 'booster_'):
            arr = X.values if hasattr(X, 'values') else np.array(X)
            raw = model.booster_.predict(arr, predict_disable_shape_check=True)
            if raw.ndim == 2: return raw
            return np.column_stack([1-raw, raw])
        raise

# === BUILD FEATURE ROW ===
def build_feat_row(home_id, away_id):
    hf = form_cache.get(home_id) or {"gf_5":1.2,"ga_5":1.1,"pts_5":6,"gf_3":1.2,"ga_3":1.1,
        "hgf_5":1.4,"hga_5":1.0,"hpts_5":7,"w_gf":1.3,"w_ga":1.1,"w_shots":12,"w_sot":4,
        "rest_days":7,"momentum":0.5,"wins_last5":2,"draws_last5":1,"gdiff5":0.2,"opp_elo":1500,
        "pts_std":1,"sf_xg":0,"sf_poss":0,"sf_tackles":0,"sf_interc":0,"sf_saves":0,"sf_bigch":0,
        "home_shots":12,"home_sot":4,"away_shots":10,"away_sot":3}
    af = form_cache.get(away_id) or dict(hf)

    # ELO from features
    he, ae = 1500, 1500
    hm = feat[feat["home_team_id"]==home_id]
    if len(hm)>0: he = float(hm.iloc[-1].get("home_elo",1500) or 1500)
    am = feat[feat["home_team_id"]==away_id]
    if len(am)>0: ae = float(am.iloc[-1].get("home_elo",1500) or 1500)
    aa = feat[feat["away_team_id"]==away_id]
    if len(aa)>0:
        a2 = aa.iloc[-1]
        if pd.notna(a2.get("away_elo")): ae = float(a2["away_elo"])

    # H2H
    h2h_key = (home_id, away_id)
    h2h_matches = h2h_cache.get(h2h_key, [])
    h2h_hw, h2h_dr, h2h_aw, h2h_ga, h2h_bt = 0.45, 0.25, 0.30, 2.5, 0.5
    if len(h2h_matches) > 0:
        h2h_df = pd.DataFrame(h2h_matches)
        h2h_hw = float((h2h_df["result"]=="H").mean())
        h2h_dr = float((h2h_df["result"]=="D").mean())
        h2h_aw = float((h2h_df["result"]=="A").mean())
        h2h_ga = float(h2h_df["total_goals"].mean())
        h2h_bt = float(h2h_df["btts"].mean())

    # League averages (from features, last 1000 matches)
    lg_d, lg_g, lg_h, lg_a, lg_b, lg_o = 0.25, 2.6, 1.4, 1.1, 0.52, 0.50
    recent = feat.tail(5000)
    if len(recent) > 0:
        lg_d = float((recent["result"]=="D").mean())
        lg_g = float(recent["total_goals"].mean())
        lg_h = float(recent["home_goals"].mean())
        lg_a = float(recent["away_goals"].mean())
        lg_b = float(recent["btts"].mean())
        lg_o = float(recent["over25"].mean())

    ed = he - ae
    h_atk = hf["gf_5"] * he / 1500
    a_atk = af["gf_5"] * ae / 1500
    h_def = hf["ga_5"] * he / 1500
    a_def = af["ga_5"] * ae / 1500

    return {
        "elo_diff": ed, "home_elo": he, "away_elo": ae,
        "home_attack_elo": he, "home_defence_elo": he,
        "away_attack_elo": ae, "away_defence_elo": ae,
        "attack_elo_diff": 0, "defence_elo_diff": 0,
        "home_gf_5": hf["gf_5"], "home_ga_5": hf["ga_5"], "home_pts_5": hf["pts_5"],
        "home_gf_3": hf["gf_3"], "home_ga_3": hf.get("ga_3", hf["ga_5"]),
        "home_hgf_5": hf["hgf_5"], "home_hga_5": hf["hga_5"], "home_hpts_5": hf["hpts_5"],
        "home_w_gf": hf["w_gf"], "home_w_ga": hf["w_ga"],
        "home_w_shots": hf.get("w_shots",12), "home_w_sot": hf.get("w_sot",4),
        "home_rest_days": hf["rest_days"],
        "away_gf_5": af["gf_5"], "away_ga_5": af["ga_5"], "away_pts_5": af["pts_5"],
        "away_gf_3": af.get("gf_3", af["gf_5"]), "away_ga_3": af.get("ga_3", af["ga_5"]),
        "away_agf_5": af["hgf_5"], "away_aga_5": af["hga_5"], "away_apts_5": af["hpts_5"],
        "away_w_gf": af["w_gf"], "away_w_ga": af["w_ga"],
        "away_w_shots": af.get("w_shots",12), "away_w_sot": af.get("w_sot",4),
        "away_rest_days": af["rest_days"],
        "h2h_home_win": h2h_hw, "h2h_draw": h2h_dr, "h2h_away_win": h2h_aw,
        "h2h_goals_avg": h2h_ga, "h2h_btts": h2h_bt,
        "home_opp_elo": ae, "away_opp_elo": he,
        "lg_avg_goals": lg_g, "lg_home_goal_avg": lg_h, "lg_away_goal_avg": lg_a,
        "lg_draw_rate": lg_d, "lg_btts_rate": lg_b, "lg_over25_rate": lg_o,
        "home_momentum": hf["momentum"], "away_momentum": af["momentum"],
        "home_wins_last5": hf["wins_last5"], "away_wins_last5": af["wins_last5"],
        "home_gdiff5": hf["gdiff5"], "away_gdiff5": af["gdiff5"],
        "home_pts_std": hf["pts_std"], "away_pts_std": af["pts_std"],
        "data_completeness": 0.9,
        "form_diff": hf["pts_5"]-af["pts_5"], "gf_diff": hf["gf_5"]-af["gf_5"],
        "ga_diff": hf["ga_5"]-af["ga_5"],
        "form_diff_abs": abs(hf["pts_5"]-af["pts_5"]),
        "gf_diff_abs": abs(hf["gf_5"]-af["gf_5"]),
        "ga_diff_abs": abs(hf["ga_5"]-af["ga_5"]),
        "teams_even": 1 if abs(ed)<30 else 0,
        "teams_very_even": 1 if abs(ed)<15 else 0,
        "form_similar": 1 if abs(hf["pts_5"]-af["pts_5"])<1.0 else 0,
        "attack_similar": 1 if abs(hf["gf_5"]-af["gf_5"])<0.5 else 0,
        "h2h_draw_high": 1 if h2h_dr>0.3 else 0,
        "lg_draw_signal": 1 if lg_d>0.28 else 0,
        "lg_real_draw_rate": lg_d, "lg_real_home_rate": 0.44, "lg_real_goal_avg": lg_g,
        # Goal-specific (69 features)
        "home_attack_power": h_atk, "away_attack_power": a_atk,
        "home_defence_power": h_def, "away_defence_power": a_def,
        "total_attack": h_atk + a_atk, "total_defence": h_def + a_def,
        "attack_defence_ratio": (h_atk+a_atk)/((h_def+a_def)+0.1),
        "home_scoring_rate": hf["gf_5"]/(hf["ga_5"]+0.1),
        "away_scoring_rate": af["gf_5"]/(af["ga_5"]+0.1),
        "scoring_rate_diff": hf["gf_5"]/(hf["ga_5"]+0.1) - af["gf_5"]/(af["ga_5"]+0.1),
        "both_score_potential": 1 if (hf["gf_5"]>0.8 and af["gf_5"]>0.8) else 0,
        "high_scoring_match": 1 if (hf["gf_5"]+af["gf_5"])>2.5 else 0,
        "low_scoring_match": 1 if (hf["gf_5"]+af["gf_5"])<1.5 else 0,
        "home_clean_sheets": 1 if hf["ga_5"]<0.5 else 0,
        "away_clean_sheets": 1 if af["ga_5"]<0.5 else 0,
        # Corner features
        "home_corners_5": 0, "away_corners_5": 0, "corner_diff": 0, "lg_corner_avg": 9.0,
    }

# === PREDICTION ===
def blend_ml(key, X, pois, weight=0.6):
    models = wm.get(key, [])
    if models and isinstance(models, list) and len(models) > 0 and isinstance(models[0], tuple):
        vals = []
        for m, ir in models:
            try:
                raw = safe_predict_proba(m, X)[:, 1]
                cal_val = ir.predict(raw)
                v = float(np.clip(cal_val[0] if hasattr(cal_val, '__len__') else cal_val, 0.05, 0.85))
                vals.append(v)
            except: pass
        if vals:
            ml = float(np.mean(vals))
            return weight * ml + (1 - weight) * pois
    return pois

def predict(home_id, away_id):
    r = build_feat_row(home_id, away_id)
    X = pd.DataFrame([r])[FEAT_MAIN].fillna(0)
    X_goal = pd.DataFrame([r])[FEAT_GOAL].fillna(0)

    # --- 1X2 (70 features) ---
    mA, mB, ca, cb = wm["m_1x2"]
    pA = np.column_stack([ca[c].predict(safe_predict_proba(mA, X)[:,c]) for c in range(3)])
    pB = np.column_stack([cb[c].predict(safe_predict_proba(mB, X)[:,c]) for c in range(3)])
    sA=pA.sum(axis=1,keepdims=True); sA=np.where(sA==0,1,sA); pA/=sA
    sB=pB.sum(axis=1,keepdims=True); sB=np.where(sB==0,1,sB); pB/=sB
    prob = 0.5*pA+0.5*pB
    s=prob.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); prob/=s
    hp,dp,ap = float(prob[0][0]),float(prob[0][1]),float(prob[0][2])
    raw_h,raw_d,raw_a = hp,dp,ap

    # --- DRAW OVERRIDE ---
    try:
        draw_models = wm.get("m_draw", [])
        draw_probs = []
        for dm_tuple in draw_models:
            if isinstance(dm_tuple, tuple) and len(dm_tuple) == 2:
                dm, ir = dm_tuple
                raw = safe_predict_proba(dm, X)[:, 1][0]
                cal = ir.predict([raw])[0]
                draw_probs.append(cal)
        if draw_probs:
            dml = float(np.mean(draw_probs))
            gap = abs(hp - ap)
            if dml > 0.45 or (gap < 0.20 and dml > 0.35):
                dp = min(0.40, max(dp, dml) + max(0.0, 0.30 - dp) * 0.3)
                tot = hp + dp + ap
                if tot > 0: hp /= tot; dp /= tot; ap /= tot
    except: pass

    # --- GOALS xG ---
    hf = form_cache.get(home_id) or {}
    af = form_cache.get(away_id) or {}
    hxg_raw = max(hf.get("gf_5", 1.2), 0.3)
    axg_raw = max(af.get("gf_5", 1.1), 0.3)

    # SofaScore xG varsa kullan
    sf_hxg = hf.get("sf_xg", 0)
    sf_axg = af.get("sf_xg", 0)
    if sf_hxg > 0: hxg_raw = 0.5 * hxg_raw + 0.5 * sf_hxg
    if sf_axg > 0: axg_raw = 0.5 * axg_raw + 0.5 * sf_axg

    mh = wm.get("m_home_goals")
    ma = wm.get("m_away_goals")
    if mh and ma:
        try:
            hxg_p = float(max(safe_predict_proba(mh, X_goal)[0][0], 0.3))
            axg_p = float(max(safe_predict_proba(ma, X_goal)[0][0], 0.3))
            hxg = 0.6 * hxg_raw + 0.4 * hxg_p
            axg = 0.6 * axg_raw + 0.4 * axg_p
        except:
            hxg, axg = hxg_raw, axg_raw
    else:
        hxg, axg = hxg_raw, axg_raw
    hxg = min(max(hxg, 0.3), 3.5)
    axg = min(max(axg, 0.3), 3.5)

    # Poisson
    pmf = lambda lam, k: math.exp(-lam) * (lam ** k) / math.factorial(k)
    def p2(lh, la, th):
        return max(0.0, min(1.0, 1.0 - sum(pmf(lh, gh) * pmf(la, ga) for gh in range(th+1) for ga in range(th+1-gh))))
    o15p = p2(hxg, axg, 1); o25p = p2(hxg, axg, 2); o35p = p2(hxg, axg, 3)
    btpois = max(0.05, min(0.85, 1.0 - pmf(hxg,0)*pmf(axg,0) - pmf(hxg,0)*(1-pmf(axg,0)) - (1-pmf(hxg,0))*pmf(axg,0)))

    o15 = blend_ml("m_over15", X_goal, o15p, 0.5)
    o25 = blend_ml("m_over25", X_goal, o25p, 0.6)
    o35 = blend_ml("m_over35", X_goal, o35p, 0.6)

    btm = wm.get("m_btts", [])
    if btm and isinstance(btm, list) and len(btm) > 0 and isinstance(btm[0], tuple):
        bt_vals = []
        for m, ir in btm:
            try:
                raw = safe_predict_proba(m, X_goal)[:, 1]
                cal = ir.predict(raw)
                v = float(np.clip(cal[0] if hasattr(cal, '__len__') else cal, 0.05, 0.85))
                bt_vals.append(v)
            except: pass
        if bt_vals:
            btts = 0.6 * float(np.mean(bt_vals)) + 0.4 * btpois
        else:
            btts = btpois
    else:
        btts = btpois

    pred = ["H", "D", "A"][np.argmax([hp, dp, ap])]
    return {
        "home_name": name_map.get(home_id, str(home_id)),
        "away_name": name_map.get(away_id, str(away_id)),
        "pred": pred, "hp": hp, "dp": dp, "ap": ap,
        "raw_h": raw_h, "raw_d": raw_d, "raw_a": raw_a,
        "hxg": hxg, "axg": axg,
        "btts": btts, "o15": o15, "o25": o25, "o35": o35,
    }

# === VERIFY ===
with open("tahminler/ftms_tahminler_2026-09-07.json", encoding="utf-8") as f:
    raw_json = json.load(f)
ftms_preds = raw_json.get("predictions", raw_json) if isinstance(raw_json, dict) and "predictions" in raw_json else raw_json

with open("tahminler/predictions.json", encoding="utf-8") as f:
    preds_orig = json.load(f)

result_lookup = {}
for p in preds_orig:
    hid = p.get("home_team_id")
    aid = p.get("away_team_id")
    res = p.get("result", {})
    tg = p.get("total_goals", {})
    if res.get("home_win") is not None:
        result_lookup[(hid, aid)] = {
            "result": res.get("winner", "?"),
            "home_goals": tg.get("home_goals"),
            "away_goals": tg.get("away_goals"),
        }

def get_actual(hid, aid):
    if (hid, aid) in result_lookup: return result_lookup[(hid, aid)]
    hm = feat[(feat["home_team_id"]==hid) & (feat["away_team_id"]==aid)]
    if len(hm) > 0:
        a = hm.iloc[-1]
        return {"result": a["result"], "home_goals": int(a["home_goals"]), "away_goals": int(a["away_goals"])}
    return None

print(f"\n{'='*95}")
print(f"  CALISKAN MOTOR v8 — GERCEK FORM CACHE + xG + DOGRU FEATURE SETLERI")
print(f"{'='*95}")

correct_1x2=0; correct_btts=0; correct_o25=0; total=0
results = []

for pred in ftms_preds:
    home_name = pred.get("home_team","?")
    away_name = pred.get("away_team","?")
    hid = find_id(home_name); aid = find_id(away_name)
    if not hid or not aid:
        print(f"  ID BULUNAMADI: {home_name} vs {away_name}"); continue
    actual = get_actual(hid, aid)
    if actual is None:
        try:
            r = predict(hid, aid)
            print(f"\n  {r['home_name'][:22]:22s} vs {r['away_name'][:22]:22s} | SONUC YOK")
            print(f"    MODEL: H={r['hp']*100:5.1f}% D={r['dp']*100:5.1f}% A={r['ap']*100:5.1f}% -> {r['pred']}")
            print(f"    xG: {r['hxg']:.2f}-{r['axg']:.2f} | O1.5={r['o15']*100:.0f}% O2.5={r['o25']*100:.0f}% O3.5={r['o35']*100:.0f}% BTTS={r['btts']*100:.0f}%")
        except Exception as e:
            import traceback; traceback.print_exc()
        continue
    ar = actual["result"]
    hg = actual.get("home_goals", 0)
    ag = actual.get("away_goals", 0)
    score = f"{hg}-{ag}" if hg is not None else "?"
    total += 1
    try:
        r = predict(hid, aid)
        ok_1x2 = r["pred"] == ar
        act_btts = int(hg>0 and ag>0) if hg is not None and ag is not None else -1
        act_o25 = int((hg or 0)+(ag or 0) > 2.5) if hg is not None else -1
        ok_btts = (r["btts"]>=0.5)==(act_btts==1) if act_btts>=0 else False
        ok_o25 = (r["o25"]>=0.5)==(act_o25==1) if act_o25>=0 else False
        if ok_1x2: correct_1x2+=1
        if ok_btts: correct_btts+=1
        if ok_o25: correct_o25+=1
        s1="OK" if ok_1x2 else " X"; s2="OK" if ok_btts else " -"; s3="OK" if ok_o25 else " -"
        print(f"\n  {r['home_name'][:22]:22s} vs {r['away_name'][:22]:22s} | {score:5s} | Gercek={ar}")
        print(f"    1X2: H={r['hp']*100:5.1f}% D={r['dp']*100:5.1f}% A={r['ap']*100:5.1f}% -> {r['pred']} [{s1}]")
        print(f"    xG: {r['hxg']:.2f}-{r['axg']:.2f} | O1.5={r['o15']*100:.0f}% O2.5={r['o25']*100:.0f}% O3.5={r['o35']*100:.0f}% BTTS={r['btts']*100:.0f}% [{s2}] [{s3}]")
        results.append({"home":r["home_name"],"away":r["away_name"],"score":score,
                         "actual":ar,"pred":r["pred"],"ok":ok_1x2,
                         "hp":r["hp"],"dp":r["dp"],"ap":r["ap"]})
    except Exception as e:
        import traceback; traceback.print_exc()

print(f"\n{'='*95}")
print(f"  SONUCLAR ({total} mac dogrulanabildi)")
print(f"{'='*95}")
if total > 0:
    print(f"  1X2:   {correct_1x2}/{total} = {100*correct_1x2/total:.1f}%")
    print(f"  BTTS:  {correct_btts}/{total} = {100*correct_btts/total:.1f}%")
    print(f"  O/U2.5:{correct_o25}/{total} = {100*correct_o25/total:.1f}%")
if results:
    print(f"\n  TUM DOGRULAMA:")
    for r in results:
        m="OK" if r["ok"] else " X"
        print(f"    {m} {r['home']:22s} vs {r['away']:22s} | {r['score']} | T={r['pred']} G={r['actual']} | H={r['hp']*100:.0f}% D={r['dp']*100:.0f}% A={r['ap']*100:.0f}%")
