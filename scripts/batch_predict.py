"""BATCH PREDICT - Football betting ensemble prediction from JSON input.

Usage: python scripts/batch_predict.py matches.json

Input JSON format:
[
  {"home": "Galatasaray", "away": "Fenerbahce", "home_odds": 2.10, "draw_odds": 3.40, "away_odds": 3.20},
  ...
]

Output: JSON with predictions + value bets to stdout.
"""
import sys, os, json, warnings, argparse, time
warnings.filterwarnings("ignore")
os.environ["LIGHTGBM_WARNINGS"] = "0"

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from difflib import get_close_matches

# ===========================
# CONFIG
# ===========================
LEAK = {
    "second_half_goals", "ht_total_goals", "ht_result_is_draw", "ht_home_leading",
    "ht_home_goals", "ht_away_goals", "home_shots", "away_shots", "home_sot",
    "away_sot", "home_corners", "away_corners", "home_yellow", "away_yellow",
    "home_red", "away_red", "shots_diff", "sot_diff", "home_goals", "away_goals",
    "result", "result_H", "result_D", "result_A", "btts", "over25", "over15",
    "over35", "total_goals", "match_id", "season", "date", "referee",
}
DEAD = {
    "home_xgot", "away_xgot", "home_big_chances", "away_big_chances",
    "home_xg_flash", "away_xg_flash", "home_fouls", "away_fouls",
    "home_possession", "away_possession", "home_xgot_5", "away_xgot_5",
    "home_big_chances_5", "away_big_chances_5", "home_xg_5", "away_xg_5",
    "home_fouls_5", "away_fouls_5", "home_possession_5", "away_possession_5",
    "home_score_prob", "away_score_prob", "btts_xprob", "over25_xprob", "draw_xprob",
}
CLOSING = {
    "mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob",
    "mkt_close_over25_prob", "avg_close_home_odds", "avg_close_draw_odds",
    "avg_close_away_odds", "avg_close_over25_odds",
}
EXCLUDE = LEAK | DEAD | CLOSING
CAT = ["league", "home_team_id", "away_team_id"]

LGB_SEEDS = [42, 123, 789]
XGB_SEEDS = [42, 123, 789]

EDGE_THRESHOLD = 0.03
KELLY_FRACTION = 0.20
MAX_BET_FRAC = 0.05
MIN_BET_FRAC = 0.005
HC_THRESHOLD = 0.65


# ===========================
# UTILS
# ===========================
def load_team_map(path="data/gold/team_id_to_name.json"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    id_to_name = {int(k): v for k, v in raw.items()}
    name_to_id = {}
    for tid, name in id_to_name.items():
        name_to_id[name.lower()] = tid
    return id_to_name, name_to_id


def fuzzy_find_team(query, name_to_id, threshold=0.6):
    q = query.lower().strip()
    if q in name_to_id:
        return name_to_id[q]
    candidates = list(name_to_id.keys())
    matches = get_close_matches(q, candidates, n=1, cutoff=threshold)
    if matches:
        return name_to_id[matches[0]]
    for name, tid in name_to_id.items():
        if q in name or name in q:
            return tid
    return None


def build_features_for_match(df, home_tid, away_tid, cols):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest = df.sort_values("date").tail(1).copy()
    latest["home_team_id"] = home_tid
    latest["away_team_id"] = away_tid
    return latest


def get_team_latest_features(df, team_id, side="home"):
    prefix = "home" if side == "home" else "away"
    mask = (df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)
    team_matches = df[mask].sort_values("date")
    if len(team_matches) == 0:
        return {}
    last = team_matches.iloc[-1]
    feats = {}
    for col in df.columns:
        if col.startswith(f"{prefix}_") and col in last.index:
            val = last[col]
            if pd.notna(val):
                feats[col] = val
    return feats


def train_ensemble(Xtr, ytr, Xva, yva):
    lgb_models, lgb_ws = [], []
    for s in LGB_SEEDS:
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, num_leaves=50,
            learning_rate=0.03, n_estimators=500, max_depth=6,
            min_child_samples=80, subsample=0.75, colsample_bytree=0.65,
            reg_alpha=0.5, reg_lambda=5.0, random_seed=s,
            verbose=-1, n_jobs=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        va_acc = (m.predict(Xva) == yva).mean()
        lgb_models.append(m)
        lgb_ws.append(va_acc)

    xgb_models, xgb_ws = [], []
    for s in XGB_SEEDS:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, max_depth=6,
            learning_rate=0.03, n_estimators=500, subsample=0.75,
            colsample_bytree=0.65, reg_alpha=0.5, reg_lambda=5.0,
            random_state=s, eval_metric="mlogloss", use_label_encoder=False,
            n_jobs=-1, verbosity=0,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        va_acc = (m.predict(Xva) == yva).mean()
        xgb_models.append(m)
        xgb_ws.append(va_acc)

    all_models = lgb_models + xgb_models
    all_ws = np.array(lgb_ws + xgb_ws)
    all_ws = all_ws / all_ws.sum()

    all_val = [m.predict_proba(Xva) for m in all_models]
    raw_val = np.average(all_val, axis=0, weights=all_ws)

    cal = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_val[:, c], (yva == c).astype(float))
        cal.append(ir)

    return all_models, all_ws, cal


def predict_ensemble(models, weights, calibrators, X):
    all_p = [m.predict_proba(X) for m in models]
    raw = np.average(all_p, axis=0, weights=weights)
    cp = np.column_stack([calibrators[c].predict(raw[:, c]) for c in range(3)])
    s = cp.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1)
    cp /= s
    return cp


def find_value_bets(probs, home_odds, draw_odds, away_odds):
    bets = []
    labels = ["H", "D", "A"]
    odds_list = [home_odds, draw_odds, away_odds]

    for i in range(len(probs)):
        for j, (label, odds) in enumerate(zip(labels, odds_list)):
            if odds is None or odds <= 1.01:
                continue
            model_prob = probs[i, j]
            market_prob = 1.0 / odds
            edge = model_prob - market_prob
            if edge > EDGE_THRESHOLD:
                kelly = edge / (odds - 1) * KELLY_FRACTION
                kelly = max(0, min(kelly, MAX_BET_FRAC))
                if kelly > MIN_BET_FRAC:
                    bets.append({
                        "match_idx": i,
                        "selection": label,
                        "model_prob": round(model_prob, 4),
                        "market_prob": round(market_prob, 4),
                        "edge": round(edge, 4),
                        "kelly_fraction": round(kelly, 4),
                        "odds": odds,
                    })
    return bets


# ===========================
# MAIN
# ===========================
def main():
    parser = argparse.ArgumentParser(description="Batch predict football matches from JSON")
    parser.add_argument("input", help="Path to JSON file with matches")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Bankroll for Kelly sizing")
    args = parser.parse_args()

    t0 = time.time()

    with open(args.input, encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loading features...", file=sys.stderr)
    df = pd.read_parquet("data/gold/features_enhanced_v5.parquet")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["home_goals", "away_goals"]:
        df[c] = df[c].fillna(0).astype(int)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["result"] = df["result"].fillna("D")
    df = df.dropna(subset=["result", "elo_diff"])

    df["form_draw_pct"] = (df["home_draws_last5"].fillna(0) + df["away_draws_last5"].fillna(0)) / 10.0
    df["elo_close_match"] = (df["elo_diff"].abs() < 100).astype(float)
    df["both_weak_attack"] = ((df["home_gf_5"].fillna(0) < 1.0) & (df["away_gf_5"].fillna(0) < 1.0)).astype(float)
    if "mkt_home_prob" in df.columns:
        df["mkt_total_prob"] = df["mkt_home_prob"].fillna(0) + df["mkt_draw_prob"].fillna(0) + df["mkt_away_prob"].fillna(0)
        df["mkt_overround"] = df["mkt_total_prob"] - 1.0
    df["elo_diff_squared"] = df["elo_diff"] ** 2
    df["rest_advantage"] = df["home_rest_days"].fillna(3) - df["away_rest_days"].fillna(3)

    cats = [c for c in CAT if c in df.columns]
    nums = [c for c in df.columns if c not in EXCLUDE and c not in cats
            and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    cols = cats + nums
    for c in cols:
        if c not in CAT:
            df[c] = df[c].fillna(0)

    df_lgb = df.copy()
    for c in CAT:
        if c in df_lgb.columns:
            df_lgb[c] = df_lgb[c].astype(str).astype("category").cat.codes

    print(f"Features: {len(cols)}", file=sys.stderr)

    tr = df_lgb[(df_lgb["date"] >= "2020-01-01") & (df_lgb["date"] < "2025-03-01")]
    va = df_lgb[(df_lgb["date"] >= "2025-03-01") & (df_lgb["date"] < "2025-09-01")]

    Xtr = tr[cols].astype(np.float32).values
    ytr = tr["result"].map({"H": 0, "D": 1, "A": 2}).values
    Xva = va[cols].astype(np.float32).values
    yva = va["result"].map({"H": 0, "D": 1, "A": 2}).values

    print(f"Training ensemble...", file=sys.stderr)
    models, weights, calibrators = train_ensemble(Xtr, ytr, Xva, yva)
    print(f"Ensemble ready ({len(models)} models)", file=sys.stderr)

    id_to_name, name_to_id = load_team_map()

    results = []
    for idx, match in enumerate(matches):
        home_name = match["home"]
        away_name = match["away"]
        home_odds = match.get("home_odds")
        draw_odds = match.get("draw_odds")
        away_odds = match.get("away_odds")

        home_tid = fuzzy_find_team(home_name, name_to_id)
        away_tid = fuzzy_find_team(away_name, name_to_id)

        if home_tid is None or away_tid is None:
            results.append({
                "home": home_name,
                "away": away_name,
                "error": f"Team not found: {'home' if home_tid is None else 'away'}",
                "home_tid": home_tid,
                "away_tid": away_tid,
            })
            continue

        row = build_features_for_match(df, home_tid, away_tid, cols)
        for c in CAT:
            if c in row.columns:
                cat_map = {v: i for i, v in enumerate(sorted(df[c].unique()))}
                row[c] = row[c].map(cat_map).fillna(0).astype(int)

        X_pred = row[cols].astype(np.float32).values
        probs = predict_ensemble(models, weights, calibrators, X_pred)

        probs_row = probs[0]
        pred_idx = int(np.argmax(probs_row))
        pred_label = ["H", "D", "A"][pred_idx]
        confidence = float(probs_row[pred_idx])

        result_entry = {
            "home": home_name,
            "away": away_name,
            "home_tid": home_tid,
            "away_tid": away_tid,
            "prob_home": round(float(probs_row[0]), 4),
            "prob_draw": round(float(probs_row[1]), 4),
            "prob_away": round(float(probs_row[2]), 4),
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "playable": confidence >= HC_THRESHOLD,
        }

        if home_odds is not None and draw_odds is not None and away_odds is not None:
            vb = find_value_bets(probs, home_odds, draw_odds, away_odds)
            result_entry["value_bets"] = vb
            result_entry["total_value_bets"] = len(vb)

            for bet in vb:
                bet["stake_frac"] = round(bet["kelly_fraction"] * args.bankroll, 2)
                if pred_label == bet["selection"]:
                    potential_profit = bet["stake_frac"] * (bet["odds"] - 1)
                    result_entry["potential_profit"] = round(potential_profit, 2)

        results.append(result_entry)

    output = {
        "n_matches": len(matches),
        "n_predicted": sum(1 for r in results if "error" not in r),
        "n_errors": sum(1 for r in results if "error" in r),
        "n_playable": sum(1 for r in results if r.get("playable", False)),
        "n_value_bets": sum(r.get("total_value_bets", 0) for r in results),
        "bankroll": args.bankroll,
        "predictions": results,
    }

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
