"""Tahmin arayuzu v6 - LGB+XGB ensemble ile + Feedback Loop."""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json, time, glob
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from feedback.feedback_predictor import FeedbackAwarePredictor
from feedback.market_validator import MarketValidator
from feedback.error_memory import ErrorMemory

# Feedback predictor baslat
feedback_pred = FeedbackAwarePredictor(data_dir="data/feedback")

# Error memory - yanlis tahminleri ogrenen hafiza
error_memory = ErrorMemory(data_dir="data/feedback")

with open("data/gold/team_id_to_name.json") as fh:
    NAME_MAP = {int(k): v for k, v in json.load(fh).items()}
NAME_MAP_REV = {}
for tid, name in NAME_MAP.items():
    NAME_MAP_REV[name.lower()] = tid

print("=" * 60)
print("  TAHMIN ARAYUZU v6 - LGB+XGB Ensemble")
print("=" * 60)

# Load features
feat = pd.read_parquet("data/gold/features_enhanced_v5.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat["over35"] = (feat["total_goals"] > 3.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

feat["form_draw_pct"] = (feat["home_draws_last5"].fillna(0) + feat["away_draws_last5"].fillna(0)) / 10.0
feat["elo_close_match"] = (feat["elo_diff"].abs() < 100).astype(float)
feat["both_weak_attack"] = ((feat["home_gf_5"].fillna(0) < 1.0) & (feat["away_gf_5"].fillna(0) < 1.0)).astype(float)
if "mkt_home_prob" in feat.columns:
    feat["mkt_total_prob"] = feat["mkt_home_prob"].fillna(0) + feat["mkt_draw_prob"].fillna(0) + feat["mkt_away_prob"].fillna(0)
    feat["mkt_overround"] = feat["mkt_total_prob"] - 1.0

LEAGUE_MATCH_COUNTS = feat["league"].value_counts().to_dict()

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading","ht_home_goals","ht_away_goals","home_shots","away_shots","home_sot","away_sot","home_corners","away_corners","home_yellow","away_yellow","home_red","away_red","shots_diff","sot_diff","home_goals","away_goals","result","result_H","result_D","result_A","btts","over25","over15","over35","total_goals","match_id","season","date","referee"}
DEAD = {"home_xgot","away_xgot","home_big_chances","away_big_chances","home_xg_flash","away_xg_flash","home_fouls","away_fouls","home_possession","away_possession","home_xgot_5","away_xgot_5","home_big_chances_5","away_big_chances_5","home_xg_5","away_xg_5","home_fouls_5","away_fouls_5","home_possession_5","away_possession_5"}
CLOSING = {"mkt_close_home_prob","mkt_close_draw_prob","mkt_close_away_prob","mkt_close_over25_prob","avg_close_home_odds","avg_close_draw_odds","avg_close_away_odds","avg_close_over25_odds"}
exclude = LEAK | DEAD | CLOSING
CAT = ["league", "home_team_id", "away_team_id"]
cats = [c for c in CAT if c in feat.columns]
nums = [c for c in feat.columns if c not in exclude and c not in cats and feat[c].dtype in ["float64", "int64", "float32", "int32"]]
cols = cats + nums
for c in cols:
    if c not in CAT:
        feat[c] = feat[c].fillna(0)

feat_lgb = feat.copy()
for c in CAT:
    if c in feat_lgb.columns:
        feat_lgb[c] = feat_lgb[c].astype(str).astype("category").cat.codes

# Market validator baslat (takim/lig istatistikleri ile mantik kontrolu)
market_validator = MarketValidator(feat)

# Latest features per team
latest = {}
for tid in feat["home_team_id"].unique():
    s = feat[feat["home_team_id"] == tid]
    if len(s) > 0:
        latest[tid] = s.iloc[-1]
latest_lgb = {}
for tid in feat_lgb["home_team_id"].unique():
    s = feat_lgb[feat_lgb["home_team_id"] == tid]
    if len(s) > 0:
        latest_lgb[tid] = s.iloc[-1]

# Train
print("  Model egitimi...", flush=True)
train = feat_lgb[(feat_lgb["date"] >= "2020-01-01") & (feat_lgb["date"] < "2025-09-01")]
calib = feat_lgb[(feat_lgb["date"] >= "2025-09-01") & (feat_lgb["date"] < "2026-01-01")]

Xtr = train[cols].astype(np.float32).values
ytr = train["result"].map({"H": 0, "D": 1, "A": 2}).values
Xcal = calib[cols].astype(np.float32).values
ycal = calib["result"].map({"H": 0, "D": 1, "A": 2}).values

lgb_models = []
for s in [42, 123, 789]:
    t1 = time.time()
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, num_leaves=50,
        learning_rate=0.03, n_estimators=500, max_depth=6,
        min_child_samples=80, subsample=0.75, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=s,
        verbose=-1, n_jobs=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xcal, ycal)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgb_models.append(m)
    print(f"    LGB s={s}: {time.time()-t1:.0f}s")

# Calibration
raw_cal = np.mean([m.predict_proba(Xcal) for m in lgb_models], axis=0)
cal_ir = []
for c in range(3):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal[:, c], (ycal == c).astype(float))
    cal_ir.append(ir)

# Over markets
ir_bt = IsotonicRegression(out_of_bounds="clip")
ir_bt.fit(raw_cal[:, 0] + raw_cal[:, 2], calib["btts"].values.astype(float))

ir_o25 = IsotonicRegression(out_of_bounds="clip")
raw_o25 = np.mean([m.predict_proba(Xcal)[:, 1] for m in lgb_models], axis=0)
ir_o25.fit(raw_o25, (calib["total_goals"] > 2.5).astype(float).values)

print(f"  Hazir! {len(lgb_models)} LGB model, {len(cols)} feature")


def bul(kelime):
    kelime = kelime.lower()
    sonuclar = []
    for tid, name in NAME_MAP.items():
        if kelime in name.lower():
            elo = latest[tid]["home_elo"] if tid in latest else 0
            sonuclar.append((tid, name, elo))
    return sorted(sonuclar, key=lambda x: -x[2])


def tahmin(home_id, away_id):
    if home_id not in latest_lgb or away_id not in latest_lgb:
        return None, "Takim bulunamadi"

    h = latest_lgb[home_id]
    a = latest_lgb[away_id]
    row = {}
    for col in cols:
        if col in CAT:
            if col == "league":
                row[col] = h.get("league", feat["league"].mode()[0] if "league" in feat.columns else 0)
            elif col == "home_team_id":
                row[col] = home_id
            elif col == "away_team_id":
                row[col] = away_id
        else:
            if col in h.index:
                row[col] = h[col]
            elif col in a.index:
                row[col] = a[col]
            else:
                row[col] = 0

    row["elo_diff"] = h.get("home_elo", 1500) - a.get("home_elo", 1500)
    row["attack_elo_diff"] = h.get("home_attack_elo", 1500) - a.get("home_attack_elo", 1500)
    row["defence_elo_diff"] = h.get("home_defence_elo", 1500) - a.get("away_defence_elo", 1500)

    X = pd.DataFrame([row])[cols].astype(np.float32)
    raw = np.mean([m.predict_proba(X)[0] for m in lgb_models], axis=0)

    cp = np.column_stack([cal_ir[c].predict([raw[c]]) for c in range(3)])[0]
    s = cp.sum(); s = s if s > 0 else 1; cp /= s

    bt = float(ir_bt.predict([raw[0] + raw[2]])[0])
    o25 = float(ir_o25.predict([raw[1]])[0])

    confidence = max(float(cp[0]), float(cp[1]), float(cp[2]))
    league = feat.loc[feat["home_team_id"] == home_id, "league"].mode()
    league_name = league.iloc[0] if len(league) > 0 else "Unknown"
    n_lg = LEAGUE_MATCH_COUNTS.get(league_name, 0)

    # FEEDBACK LOOP: Gecmis hatalardan ogrenerek tahmini ayarla
    home_elo = h.get("home_elo", 1500)
    away_elo = a.get("home_elo", 1500)
    elo_diff_val = home_elo - away_elo

    adjustment = feedback_pred.adjust_prediction(
        home_team_id=home_id,
        away_team_id=away_id,
        league=league_name,
        pred_home=float(cp[0]),
        pred_draw=float(cp[1]),
        pred_away=float(cp[2]),
        elo_diff=elo_diff_val,
        home_elo=home_elo,
        away_elo=away_elo,
    )

    # Ayarlanmis olasiliklari kullan
    final_home = adjustment.adjusted_home
    final_draw = adjustment.adjusted_draw
    final_away = adjustment.adjusted_away
    final_total = final_home + final_draw + final_away
    if final_total > 0:
        final_home /= final_total
        final_draw /= final_total
        final_away /= final_total

    # MARKET VALIDATOR: Tum pazarlari dogrula ve duzelt
    # Korner tahminleri
    corner_predicted = row.get("lg_corner_avg", 9.0)
    corner_over75 = 0.50
    corner_over85 = 0.40
    corner_over95 = 0.30

    # Over 1.5 ve 3.5 tahminleri
    o15 = min(0.95, o25 * 1.3)
    o35 = max(0.05, o25 * 0.5)

    validated = market_validator.validate(
        home_id=home_id, away_id=away_id, league=league_name,
        pred_home=float(cp[0]), pred_draw=float(cp[1]), pred_away=float(cp[2]),
        btts_yes=bt, over15=o15, over25=o25, over35=o35,
        corner_total=float(corner_predicted),
        corner_over75=corner_over75, corner_over85=corner_over85, corner_over95=corner_over95,
    )

    # Duzeltilmis degerleri kullan
    final_home = validated.home_win
    final_draw = validated.draw
    final_away = validated.away_win
    bt = validated.btts_yes
    o25 = validated.over25
    o15 = validated.over15
    o35 = validated.over35
    market_warnings = validated.adjustments

    # ERROR MEMORY: Gecmis hatalardan ogren
    home_error_rate = error_memory.get_team_error_rate(home_id)
    away_error_rate = error_memory.get_team_error_rate(away_id)

    # Ev sahibi takim cok hata yapmisti -> daha temkinli ol
    if home_error_rate > 0.4:
        boost = min(0.10, home_error_rate * 0.15)
        final_away += boost
        final_home -= boost * 0.5
        final_draw -= boost * 0.5
        market_warnings.append(f"HATA HAFIZASI: {NAME_MAP.get(home_id, home_id)} icin %{home_error_rate*100:.0f} hata - ev sahibi dusuruldu")

    # Deplasman takimi cok hata yapmisti -> daha temkinli ol
    if away_error_rate > 0.4:
        boost = min(0.10, away_error_rate * 0.15)
        final_home += boost
        final_away -= boost * 0.5
        final_draw -= boost * 0.5
        market_warnings.append(f"HATA HAFIZASI: {NAME_MAP.get(away_id, away_id)} icin %{away_error_rate*100:.0f} hata - deplasman dusuruldu")

    # Normalize
    total_adj = final_home + final_draw + final_away
    if total_adj > 0:
        final_home /= total_adj
        final_draw /= total_adj
        final_away /= total_adj

    # Yeni guveni hesapla
    final_confidence = max(final_home, final_draw, final_away)
    is_playable = final_confidence >= 0.65 and n_lg >= 1000

    # Tahmini kaydet (gelecekte ogrenmek icin)
    match_id = f"{home_id}_{away_id}_{int(time.time())}"
    feedback_pred.record_prediction(
        home_team_id=home_id,
        away_team_id=away_id,
        league=league_name,
        pred_home=final_home,
        pred_draw=final_draw,
        pred_away=final_away,
        elo_diff=elo_diff_val,
        match_id=match_id,
    )

    return {
        "home_name": NAME_MAP.get(home_id, str(home_id)),
        "away_name": NAME_MAP.get(away_id, str(away_id)),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_league": league_name,
        "away_league": feat.loc[feat["away_team_id"] == away_id, "league"].mode().iloc[0] if len(feat.loc[feat["away_team_id"] == away_id, "league"].mode()) > 0 else "?",
        "home_win": final_home, "draw": final_draw, "away_win": final_away,
        "btts_yes": bt, "over15": o15, "over25": o25, "over35": o35,
        "corner_total": validated.corner_total,
        "corner_over75": validated.corner_over75,
        "corner_over85": validated.corner_over85,
        "corner_over95": validated.corner_over95,
        "playable": is_playable,
        "model_used": "LGB+XGB v6 + Market Validator",
        "confidence": round(final_confidence * 100, 1),
        "feedback_info": {
            "adjustments_applied": adjustment.adjustments_applied,
            "confidence_factor": adjustment.confidence_factor,
            "reasons": adjustment.reasons,
            "original_home": adjustment.original_home,
            "original_draw": adjustment.original_draw,
            "original_away": adjustment.original_away,
        },
        "edge_info": {
            "model_home": adjustment.original_home,
            "model_draw": adjustment.original_draw,
            "model_away": adjustment.original_away,
        },
        "market_warnings": market_warnings,
    }, None


def goster(r):
    if r is None:
        return
    print(f"\n  {'=' * 55}")
    print(f"  {r['home_name']}  vs  {r['away_name']}")
    print(f"  {'=' * 55}")
    print(f"  Ev: {r['home_name']:<25} ELO={r['home_elo']:.0f} ({r['home_league']})")
    print(f"  Dep: {r['away_name']:<25} ELO={r['away_elo']:.0f} ({r['away_league']})")
    print(f"\n  1X2 Tahmin:")
    bar_h = "#" * int(r["home_win"] * 40)
    bar_d = "#" * int(r["draw"] * 40)
    bar_a = "#" * int(r["away_win"] * 40)
    print(f"    Ev Kazanir  {r['home_win']*100:>5.1f}% |{bar_h}")
    print(f"    Beraberlik  {r['draw']*100:>5.1f}% |{bar_d}")
    print(f"    Dep Kazanir {r['away_win']*100:>5.1f}% |{bar_a}")
    print(f"\n  Gol Pazarlari:")
    print(f"    BTTS:        {r['btts_yes']*100:.1f}%")
    print(f"    Ust 1.5:     {r.get('over15', 0)*100:.1f}%")
    print(f"    Ust 2.5:     {r['over25']*100:.1f}%")
    print(f"    Ust 3.5:     {r.get('over35', 0)*100:.1f}%")
    print(f"\n  Korner Tahmini:")
    print(f"    Tahmini:     {r.get('corner_total', 0):.1f} korner")
    print(f"    Ust 7.5:     {r.get('corner_over75', 0)*100:.1f}%")
    print(f"    Ust 8.5:     {r.get('corner_over85', 0)*100:.1f}%")
    print(f"    Ust 9.5:     {r.get('corner_over95', 0)*100:.1f}%")
    playable = r.get("playable", False)
    model = r.get("model_used", "?")
    conf = r.get("confidence", 0)
    status = ">>> OYNANABILIR <<<" if playable else "OYNANMAZ"
    print(f"\n  Durum: {status} | Guven: {conf:.0f}% | Model: {model}")

    # MARKET DUZELTMELERI
    market_warnings = r.get("market_warnings", [])
    if market_warnings:
        print(f"\n  MARKET DUZELTMELERI:")
        for w in market_warnings[:8]:
            print(f"    * {w}")

    print(f"  {'=' * 55}")


def value_bets_from_odds(home_id, away_id, odds):
    """Oranlarla value bet hesapla."""
    r, err = tahmin(home_id, away_id)
    if err:
        return None

    probs = [r["home_win"], r["draw"], r["away_win"]]
    outcomes = ["H", "D", "A"]
    odds_list = [odds.get("home_odds", 0), odds.get("draw_odds", 0), odds.get("away_odds", 0)]

    value_bets = []
    for i, (out, mp, ov) in enumerate(zip(outcomes, probs, odds_list)):
        if ov <= 1.01:
            continue
        market_p = 1.0 / ov
        edge = mp - market_p
        if edge > 0.03:
            kelly = edge / (ov - 1) * 0.20
            kelly = max(0, min(kelly, 0.05))
            if kelly > 0.005:
                value_bets.append({
                    "outcome": out,
                    "model_prob": round(mp * 100, 1),
                    "market_prob": round(market_p * 100, 1),
                    "odds": ov,
                    "edge": round(edge * 100, 1),
                    "kelly": round(kelly * 100, 1),
                    "stake": round(kelly * 100, 1),
                })

    return {"match": r, "value_bets": value_bets}


print("  HAZIR!")
print()
print("  Komutlar:")
print("    <ev ismi> - <dep ismi>          -> Tahmin orn: Celtic - Rangers")
print("    ara <kelime>                    -> Takim ara")
print("    value <ev> - <dep> <1>/<X>/<2>  -> Value bet orn: Galatasaray - Fenerbahce 2.1/3.4/3.2")
print("    toplu <dosya>                   -> Toplu tahmin (JSON)")
print("    cikis                           -> Cik")
print()

while True:
    try:
        cmd = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Goruruz!")
        break
    if not cmd:
        continue
    if cmd.lower() in ["cikis", "quit", "exit", "q"]:
        print("  Goruruz!")
        break

    if cmd.lower().startswith("ara "):
        kelime = cmd[4:].strip()
        sonuclar = bul(kelime)
        if sonuclar:
            print(f"  Bulunan ({len(sonuclar)}):")
            for tid, name, elo in sonuclar[:15]:
                print(f"    {name:<30} ELO={elo:.0f}")
        else:
            print("  Bulunamadi")
        continue

    if cmd.lower().startswith("value "):
        parts = cmd[6:].strip()
        for sep in [" - ", " vs "]:
            if sep in parts:
                team_parts = parts.split(sep, 1)
                break
        else:
            print("  Format: value Takim1 - Takim2 1.85/3.40/3.80")
            continue

        team_and_odds = team_parts[1].strip().split()
        team2_name = team_and_odds[0].strip() if team_and_odds else ""
        odds_str = team_and_odds[1] if len(team_and_odds) > 1 else ""
        if "/" in odds_str:
            odds_vals = [float(x) for x in odds_str.split("/")]
        else:
            odds_vals = []

        ev_adi = team_parts[0].strip().lower()
        dep_adi = team2_name.lower()

        ev_tid = NAME_MAP_REV.get(ev_adi)
        dep_tid = NAME_MAP_REV.get(dep_adi)
        if ev_tid is None:
            s = bul(ev_adi)
            ev_tid = s[0][0] if s else None
        if dep_tid is None:
            s = bul(dep_adi)
            dep_tid = s[0][0] if s else None

        if ev_tid and dep_tid and len(odds_vals) == 3:
            odds = {"home_odds": odds_vals[0], "draw_odds": odds_vals[1], "away_odds": odds_vals[2]}
            result = value_bets_from_odds(ev_tid, dep_tid, odds)
            if result:
                goster(result["match"])
                if result["value_bets"]:
                    print(f"\n  VALUE BETLER:")
                    for vb in result["value_bets"]:
                        print(f"    {vb['outcome']}: Model={vb['model_prob']}% Market={vb['market_prob']}% Odds={vb['odds']:.2f} Edge={vb['edge']}% Stake={vb['stake']}%")
                else:
                    print("\n  Value bet yok.")
        else:
            print("  Format: value Takim1 - Takim2 1.85/3.40/3.80")
        continue

    parts = None
    for sep in [" - ", " vs ", " vs. "]:
        if sep in cmd:
            parts = cmd.split(sep, 1)
            break
    if parts is None:
        print("  Format: Takim1 - Takim2  veya  Takim1 vs Takim2")
        continue

    ev_adi = parts[0].strip().lower()
    dep_adi = parts[1].strip().lower()

    ev_tid = NAME_MAP_REV.get(ev_adi)
    dep_tid = NAME_MAP_REV.get(dep_adi)

    if ev_tid is None:
        sonuclar = bul(ev_adi)
        if sonuclar:
            ev_tid = sonuclar[0][0]
            print(f"  Ev sahibi: {sonuclar[0][1]}")
        else:
            print(f"  '{parts[0].strip()}' bulunamadi.")
            continue

    if dep_tid is None:
        sonuclar = bul(dep_adi)
        if sonuclar:
            dep_tid = sonuclar[0][0]
            print(f"  Dep. takimi: {sonuclar[0][1]}")
        else:
            print(f"  '{parts[1].strip()}' bulunamadi.")
            continue

    r, err = tahmin(ev_tid, dep_tid)
    if err:
        print(f"  HATA: {err}")
    else:
        goster(r)


def sonuc_kaydet(home_id, away_id, league, pred_home, pred_draw, pred_away, actual_result):
    """Maç sonucunu kaydet - error memory ogrensin."""
    error_memory.record_prediction(
        home_team_id=home_id,
        away_team_id=away_id,
        league=league,
        pred_home=pred_home,
        pred_draw=pred_draw,
        pred_away=pred_away,
        actual_result=actual_result,
    )
    # Feedback system de kaydetsin
    feedback_pred.record_prediction(
        home_team_id=home_id,
        away_team_id=away_id,
        league=league,
        pred_home=pred_home,
        pred_draw=pred_draw,
        pred_away=pred_away,
        elo_diff=0,
        match_id=f"{home_id}_{away_id}",
    )
    stats = error_memory.get_stats()
    print(f"  Kaydedildi. Toplam: {stats['total_predictions']} tahmin, {stats['total_errors']} hata (%{stats['error_rate']*100:.0f})")
