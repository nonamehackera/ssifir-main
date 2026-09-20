"""Train v7: NO rolling features. Only elo + league + h2h + form. Reliable at prediction time."""
import pickle, time, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss

warnings.filterwarnings("ignore")
DATA = r"C:\Users\furka\Desktop\ssifir-main\data\gold"

SS_STATS = [
    "xg", "shots", "sot", "possession", "big_ch", "passes", "fouls", "yellows",
    "corners", "tackles", "interceptions", "gk_saves",
    "big_ch_scored", "big_ch_missed", "shots_inbox", "shots_outbox",
    "shots_off", "shots_blocked", "woodwork", "xg_ot",
    "through_balls", "pen_touches", "offsides",
    "accurate_passes", "final3_entries", "long_balls", "crosses",
    "duels_pct", "dispossessed", "dribbles",
    "total_tackles", "recoveries", "clearances", "errors_shot", "errors_goal",
    "total_saves", "goals_prevented",
]


def load_data():
    sf = pd.read_parquet(f"{DATA}/sofascore_full.parquet")
    sf["date"] = pd.to_datetime(sf["date"]).dt.normalize()
    sf = sf.sort_values("date").reset_index(drop=True)

    sf["result"] = sf.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"] else ("D" if r["home_goals"] == r["away_goals"] else "A"), axis=1)
    sf["btts"] = ((sf["home_goals"] > 0) & (sf["away_goals"] > 0)).astype(int)
    sf["over25"] = ((sf["home_goals"] + sf["away_goals"]) > 2.5).astype(int)
    sf["over35"] = ((sf["home_goals"] + sf["away_goals"]) > 3.5).astype(int)

    print("  Computing form...")
    t0 = time.time()

    home_records = sf[["date", "home_team", "home_goals", "away_goals", "result"]].copy()
    home_records.columns = ["date", "team", "gf", "ga", "result"]
    home_records["points"] = home_records["result"].map({"H": 3, "D": 1, "A": 0}).fillna(0)
    home_records["is_home"] = 1

    away_records = sf[["date", "away_team", "away_goals", "home_goals", "result"]].copy()
    away_records.columns = ["date", "team", "gf", "ga", "result"]
    away_records["points"] = away_records["result"].map({"A": 3, "D": 1, "H": 0}).fillna(0)
    away_records["is_home"] = 0

    tdf = pd.concat([home_records, away_records], ignore_index=True)
    tdf = tdf.sort_values(["team", "date"]).reset_index(drop=True)

    g = tdf.groupby("team")
    for n in [5, 10, 20]:
        tdf[f"pts_{n}"] = g["points"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
        tdf[f"gf_{n}"] = g["gf"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        tdf[f"ga_{n}"] = g["ga"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    tdf["rest_days"] = g["date"].transform(lambda x: x.diff().dt.days)

    for stat in ["xg", "sot", "shots"]:
        col = f"h_{stat}"
        if col in sf.columns:
            hx = sf[["date", "home_team", col]].copy()
            hx.columns = ["date", "team", "val"]
            ax = sf[["date", "away_team", f"a_{stat}"]].copy()
            ax.columns = ["date", "team", "val"]
            ts = pd.concat([hx, ax], ignore_index=True).sort_values(["team", "date"])
            g2 = ts.groupby("team")
            tdf[f"{stat}_r5"] = g2["val"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    hf = tdf[tdf["is_home"] == 1].copy()
    form_cols = [c for c in hf.columns if c not in ["date", "team", "gf", "ga", "result", "points", "is_home"]]
    hf = hf[["date", "team"] + form_cols].copy()
    hf.columns = ["date", "home_team"] + [f"h_{c}" if c not in ["date", "team"] else c for c in form_cols]

    af = tdf[tdf["is_home"] == 0].copy()
    af = af[["date", "team"] + form_cols].copy()
    af.columns = ["date", "away_team"] + [f"a_{c}" if c not in ["date", "team"] else c for c in form_cols]

    sf = sf.merge(hf, on=["date", "home_team"], how="left")
    sf = sf.merge(af, on=["date", "away_team"], how="left")

    for c in sf.columns:
        if c.startswith("h_pts_") or c.startswith("a_pts_") or c.startswith("h_gf_") or c.startswith("a_gf_") or \
           c.startswith("h_ga_") or c.startswith("a_ga_") or c.startswith("h_rest") or c.startswith("a_rest") or \
           c.startswith("h_xg_r") or c.startswith("a_xg_r") or c.startswith("h_sot_r") or c.startswith("a_sot_r") or \
           c.startswith("h_shots_r") or c.startswith("a_shots_r"):
            sf[c] = sf[c].fillna(0)

    print(f"  Done: {time.time()-t0:.0f}s")
    return sf


def train(df):
    print("\n  Preparing features...")
    t0 = time.time()

    elo = ["home_elo", "away_elo", "elo_diff", "home_attack_elo", "away_attack_elo",
           "home_defence_elo", "away_defence_elo", "attack_elo_diff", "defence_elo_diff"]
    league = ["lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
              "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg"]
    h2h = ["h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts"]

    form = [c for c in df.columns if c.startswith("h_pts_") or c.startswith("a_pts_") or
            c.startswith("h_gf_") or c.startswith("a_gf_") or
            c.startswith("h_ga_") or c.startswith("a_ga_") or
            c.startswith("h_rest") or c.startswith("a_rest") or
            c.startswith("h_xg_r") or c.startswith("a_xg_r") or
            c.startswith("h_sot_r") or c.startswith("a_sot_r") or
            c.startswith("h_shots_r") or c.startswith("a_shots_r")]

    feature_cols = elo + league + h2h + form
    feature_cols = [c for c in feature_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if df[c].dtype in ["float64", "int64", "float32", "int32", "bool"]]
    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"  Features: {len(feature_cols)}")

    df = df.sort_values("date").reset_index(drop=True)
    train_df = df[df["date"] >= "2020-01-01"].copy()
    train_df = train_df.dropna(subset=["result"])
    for c in feature_cols:
        train_df[c] = train_df[c].fillna(0)
    print(f"  Data: {len(train_df)}")

    split = int(len(train_df) * 0.8)
    tp = train_df.iloc[:split]
    te = train_df.iloc[split:]
    X_tr = tp[feature_cols]
    X_te = te[feature_cols]
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

    P = dict(n_estimators=400, max_depth=7, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.7,
             class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)

    # 1X2
    y_tr = tp["result"].map({"H": 0, "D": 1, "A": 2}).values
    y_te = te["result"].map({"H": 0, "D": 1, "A": 2}).values
    print("\n  === 1X2 ===")
    m1x2 = LGBMClassifier(**P)
    m1x2.fit(X_tr, y_tr)
    m1x2_cal = CalibratedClassifierCV(m1x2, cv=3, method="isotonic")
    m1x2_cal.fit(X_tr, y_tr)
    prob = m1x2_cal.predict_proba(X_te)
    pred = prob.argmax(axis=1)
    acc = accuracy_score(y_te, pred)
    print(f"  Accuracy: {acc*100:.1f}%")
    mask = prob.max(axis=1) >= 0.55
    if mask.sum() > 0:
        print(f"  >=55%: {mask.sum()}, {accuracy_score(y_te[mask], pred[mask])*100:.1f}%")

    # Check draw detection
    draw_mask = y_te == 1
    draw_pred = pred[draw_mask]
    print(f"  Draw recall: {(draw_pred == 1).sum()}/{draw_mask.sum()} = {(draw_pred == 1).sum()/max(draw_mask.sum(),1)*100:.1f}%")

    # BTTS
    y_tr = tp["btts"].values
    y_te = te["btts"].values
    print("\n  === BTTS ===")
    mbtts = LGBMClassifier(**P)
    mbtts.fit(X_tr, y_tr)
    mbtts_cal = CalibratedClassifierCV(mbtts, cv=3, method="isotonic")
    mbtts_cal.fit(X_tr, y_tr)
    prob_btts = mbtts_cal.predict_proba(X_te)[:, 1]
    print(f"  Acc: {accuracy_score(y_te, (prob_btts>=0.5).astype(int))*100:.1f}%")

    # Over 2.5
    y_tr = tp["over25"].values
    y_te = te["over25"].values
    print("\n  === Over 2.5 ===")
    mo25 = LGBMClassifier(**P)
    mo25.fit(X_tr, y_tr)
    mo25_cal = CalibratedClassifierCV(mo25, cv=3, method="isotonic")
    mo25_cal.fit(X_tr, y_tr)
    prob_o25 = mo25_cal.predict_proba(X_te)[:, 1]
    print(f"  Acc: {accuracy_score(y_te, (prob_o25>=0.5).astype(int))*100:.1f}%")

    # Over 3.5
    y_tr = tp["over35"].values
    y_te = te["over35"].values
    mo35 = LGBMClassifier(**P)
    mo35.fit(X_tr, y_tr)
    mo35_cal = CalibratedClassifierCV(mo35, cv=3, method="isotonic")
    mo35_cal.fit(X_tr, y_tr)
    print(f"  Over 3.5 Acc: {accuracy_score(y_te, mo35_cal.predict(X_te))*100:.1f}%")

    # Under 2.5
    y_tr = (tp["over25"] == 0).astype(int).values
    y_te = (te["over25"] == 0).astype(int).values
    mu25 = LGBMClassifier(**P)
    mu25.fit(X_tr, y_tr)
    mu25_cal = CalibratedClassifierCV(mu25, cv=3, method="isotonic")
    mu25_cal.fit(X_tr, y_tr)
    print(f"  Under 2.5 Acc: {accuracy_score(y_te, mu25_cal.predict(X_te))*100:.1f}%")

    # Double chance
    dcm = {}
    for name, vals in [("1X", ["H", "D"]), ("X2", ["D", "A"]), ("12", ["H", "A"])]:
        y_tr = tp["result"].isin(vals).astype(int).values
        y_te = te["result"].isin(vals).astype(int).values
        m = LGBMClassifier(**P)
        m.fit(X_tr, y_tr)
        mc = CalibratedClassifierCV(m, cv=3, method="isotonic")
        mc.fit(X_tr, y_tr)
        print(f"  {name} Acc: {accuracy_score(y_te, mc.predict(X_te))*100:.1f}%")
        dcm[name] = (m, mc)

    # BTTS No
    y_tr = (tp["btts"] == 0).astype(int).values
    y_te = (te["btts"] == 0).astype(int).values
    mbtno = LGBMClassifier(**P)
    mbtno.fit(X_tr, y_tr)
    mbtno_cal = CalibratedClassifierCV(mbtno, cv=3, method="isotonic")
    mbtno_cal.fit(X_tr, y_tr)
    print(f"  BTTS No Acc: {accuracy_score(y_te, mbtno_cal.predict(X_te))*100:.1f}%")

    # Feature importance
    imp = m1x2.feature_importances_
    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:20]
    print("\n  TOP 20 FEATURES:")
    for name, val in pairs:
        print(f"    {val:5d} {name}")

    # Save
    model_data = {
        "m_1x2": (m1x2, m1x2_cal, prob),
        "m_btts": (mbtts, mbtts_cal, prob_btts),
        "m_over25": (mo25, mo25_cal, prob_o25),
        "m_over35": (mo35, mo35_cal, None),
        "m_under25": (mu25, mu25_cal, None),
        "m_1x": dcm["1X"],
        "m_x2": dcm["X2"],
        "m_12": dcm["12"],
        "m_btno": (mbtno, mbtno_cal, None),
        "features": feature_cols,
    }

    out = r"C:\Users\furka\Desktop\ssifir-main\web_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Saved: {out}")
    print(f"  Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    df = load_data()
    train(df)
