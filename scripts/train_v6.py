"""
Train v6: sofascore_full + rolling merge, no odds dependency.
"""
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


def load_and_merge():
    print("Loading...")
    sf = pd.read_parquet(f"{DATA}/sofascore_full.parquet")
    sr = pd.read_parquet(f"{DATA}/sofascore_rolling.parquet")

    sf['date'] = pd.to_datetime(sf['date']).dt.normalize()
    sr['date'] = pd.to_datetime(sr['date']).dt.normalize()

    # Dedup rolling (keep home perspective)
    sr_dedup = sr.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    print(f"  SF: {len(sf)}, Rolling: {len(sr_dedup)}")

    # Merge
    df = sf.merge(sr_dedup, on=['date', 'home_team', 'away_team'],
                  how='left', suffixes=('', '_r'))

    matched = df['h_xg_r'].notna().sum()
    print(f"  Merged: {len(df)}, rolling matched: {matched} ({matched/len(df)*100:.0f}%)")

    # Compute form features from sofascore_full itself
    print("  Computing form features...")
    df = df.sort_values(['home_team', 'date']).reset_index(drop=True)

    # Create team-match records for form computation
    home_records = df[['date', 'home_team', 'home_goals', 'away_goals', 'result']].copy()
    home_records.columns = ['date', 'team', 'gf', 'ga', 'result']
    home_records['points'] = home_records['result'].map({'H': 3, 'D': 1, 'A': 0}).fillna(0)
    home_records['is_home'] = 1

    away_records = df[['date', 'away_team', 'away_goals', 'home_goals', 'result']].copy()
    away_records.columns = ['date', 'team', 'gf', 'ga', 'result']
    away_records['points'] = away_records['result'].map({'A': 3, 'D': 1, 'H': 0}).fillna(0)
    away_records['is_home'] = 0

    team_df = pd.concat([home_records, away_records], ignore_index=True)
    team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

    g = team_df.groupby('team')
    for n in [5, 10, 20]:
        team_df[f'pts_{n}'] = g['points'].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
        team_df[f'gf_{n}'] = g['gf'].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        team_df[f'ga_{n}'] = g['ga'].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    team_df['rest_days'] = g['date'].transform(lambda x: x.diff().dt.days)

    # xg form
    for stat in ['xg', 'sot', 'shots']:
        col_h = f'h_{stat}'
        col_a = f'a_{stat}'
        if col_h in df.columns:
            # Get xg values per team from both home and away perspective
            hx = df[['date', 'home_team', col_h]].copy()
            hx.columns = ['date', 'team', 'stat_val']
            ax = df[['date', 'away_team', col_a]].copy()
            ax.columns = ['date', 'team', 'stat_val']
            team_stat = pd.concat([hx, ax], ignore_index=True).sort_values(['team', 'date'])
            g2 = team_stat.groupby('team')
            team_df[f'{stat}_r5'] = g2['stat_val'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    # Merge form back to home matches
    home_form = team_df[team_df['is_home'] == 1].copy()
    form_cols = [c for c in home_form.columns if c not in ['date', 'team', 'gf', 'ga', 'result', 'points', 'is_home']]
    home_form = home_form[['date', 'team'] + form_cols].copy()
    home_form.columns = ['date', 'home_team'] + [f'hf_{c}' if c != 'home_team' else c for c in form_cols]

    # Also away form
    away_form = team_df[team_df['is_home'] == 0].copy()
    away_form = away_form[['date', 'team'] + form_cols].copy()
    away_form.columns = ['date', 'away_team'] + [f'af_{c}' if c != 'away_team' else c for c in form_cols]

    df = df.merge(home_form, on=['date', 'home_team'], how='left')
    df = df.merge(away_form, on=['date', 'away_team'], how='left')

    # Fill NaN
    for c in df.columns:
        if c.startswith('hf_') or c.startswith('af_') or c.endswith('_r'):
            df[c] = df[c].fillna(0)

    # Derived
    df['result'] = df.apply(lambda r: 'H' if r['home_goals'] > r['away_goals'] else ('D' if r['home_goals'] == r['away_goals'] else 'A'), axis=1)
    df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
    df['over25'] = ((df['home_goals'] + df['away_goals']) > 2.5).astype(int)
    df['over35'] = ((df['home_goals'] + df['away_goals']) > 3.5).astype(int)

    print(f"  Final: {len(df)} rows, {len(df.columns)} columns")
    return df


def train(df):
    print("\n  Preparing features...")
    t0 = time.time()

    # Elo features
    elo = ["home_elo", "away_elo", "elo_diff", "home_attack_elo", "away_attack_elo",
           "home_defence_elo", "away_defence_elo", "attack_elo_diff", "defence_elo_diff"]

    # League features
    league = ["lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
              "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg"]

    # H2H features
    h2h = ["h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts"]

    # Per-match SofaScore stats (raw for THIS match)
    sf_match = [f"h_{s}" for s in SS_STATS] + [f"a_{s}" for s in SS_STATS]

    # Rolling SofaScore features (from sofascore_rolling.parquet)
    # Exclude current-match results that leak!
    LEAK_COLS = {'home_goals_r', 'away_goals_r', 'total_goals_r', 'btts_r',
                 'result_r', 'over15_r', 'over25_r', 'over35_r'}
    rolling = [c for c in df.columns if c.endswith('_r') and c not in sf_match and c not in LEAK_COLS
               and df[c].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']]

    # Form features
    form = [c for c in df.columns if c.startswith('hf_') or c.startswith('af_')]

    feature_cols = []
    for name, cols in [("elo", elo), ("league", league), ("h2h", h2h),
                        ("rolling", rolling), ("form", form)]:
        existing = [c for c in cols if c in df.columns]
        feature_cols.extend(existing)
        if existing:
            print(f"    {name}: {len(existing)}")

    feature_cols = list(dict.fromkeys(feature_cols))
    # Ensure numeric only
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']]
    print(f"  Total: {len(feature_cols)}")

    # Filter: 2020+ with at least some form history
    df = df.sort_values('date').reset_index(drop=True)
    train_df = df[df['date'] >= '2020-01-01'].copy()
    train_df = train_df.dropna(subset=['result'])

    # Remove rows with excessive NaN
    for c in feature_cols:
        train_df[c] = train_df[c].fillna(0)

    print(f"  Data: {len(train_df)} matches")

    # Time split
    split = int(len(train_df) * 0.8)
    tp = train_df.iloc[:split]
    te = train_df.iloc[split:]
    X_tr = tp[feature_cols]
    X_te = te[feature_cols]
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

    P = dict(n_estimators=400, max_depth=7, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.7,
             class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)

    def eval_model(name, y_tr, y_te, params=P, cal=True):
        m = LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        if cal:
            mc = CalibratedClassifierCV(m, cv=3, method="isotonic")
            mc.fit(X_tr, y_tr)
            prob = mc.predict_proba(X_te)
            pred = prob.argmax(axis=1) if prob.shape[1] > 1 else (prob[:, 1] >= 0.5).astype(int)
            acc = accuracy_score(y_te, pred)
            return m, mc, prob, acc, pred
        else:
            pred = m.predict(X_te)
            acc = accuracy_score(y_te, pred)
            return m, None, None, acc, pred

    # 1X2
    y_tr = tp["result"].map({"H": 0, "D": 1, "A": 2}).values
    y_te = te["result"].map({"H": 0, "D": 1, "A": 2}).values
    print("\n  === 1X2 ===")
    m, mc, prob, acc, pred = eval_model("1X2", y_tr, y_te)
    print(f"  Accuracy: {acc*100:.1f}%")
    mask = prob.max(axis=1) >= 0.55
    if mask.sum() > 0:
        print(f"  >=55%: {mask.sum()}, {accuracy_score(y_te[mask], pred[mask])*100:.1f}%")
    m1x2 = (m, mc, prob)

    # BTTS
    y_tr = tp["btts"].values
    y_te = te["btts"].values
    print("\n  === BTTS ===")
    m, mc, prob, acc, pred = eval_model("BTTS", y_tr, y_te)
    print(f"  Acc: {acc*100:.1f}%, Brier: {brier_score_loss(y_te, prob[:, 1]):.4f}")
    mbtts = (m, mc, prob)

    # Over 2.5
    y_tr = tp["over25"].values
    y_te = te["over25"].values
    print("\n  === Over 2.5 ===")
    m, mc, prob, acc, pred = eval_model("O25", y_tr, y_te)
    print(f"  Acc: {acc*100:.1f}%, Brier: {brier_score_loss(y_te, prob[:, 1]):.4f}")
    mo25 = (m, mc, prob)

    # Over 3.5
    y_tr = tp["over35"].values
    y_te = te["over35"].values
    print("\n  === Over 3.5 ===")
    m, mc, prob, acc, pred = eval_model("O35", y_tr, y_te)
    print(f"  Acc: {acc*100:.1f}%")
    mo35 = (m, mc, prob)

    # Under 2.5
    y_tr = (tp["over25"] == 0).astype(int).values
    y_te = (te["over25"] == 0).astype(int).values
    print("\n  === Under 2.5 ===")
    m, mc, prob, acc, pred = eval_model("U25", y_tr, y_te)
    print(f"  Acc: {acc*100:.1f}%")
    mu25 = (m, mc, prob)

    # Double chance
    dcm = {}
    for name, vals in [("1X", ["H", "D"]), ("X2", ["D", "A"]), ("12", ["H", "A"])]:
        y_tr = tp["result"].isin(vals).astype(int).values
        y_te = te["result"].isin(vals).astype(int).values
        print(f"\n  === {name} ===")
        m, mc, prob, acc, pred = eval_model(name, y_tr, y_te)
        print(f"  Acc: {acc*100:.1f}%")
        dcm[name] = (m, mc)

    # BTTS No
    y_tr = (tp["btts"] == 0).astype(int).values
    y_te = (te["btts"] == 0).astype(int).values
    print("\n  === BTTS No ===")
    m, mc, prob, acc, pred = eval_model("BTN", y_tr, y_te)
    print(f"  Acc: {acc*100:.1f}%")
    mbtno = (m, mc, prob)

    # Feature importance
    imp = m1x2[0].feature_importances_
    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:25]
    print("\n  TOP 25 FEATURES:")
    for name, val in pairs:
        print(f"    {val:5d} {name}")

    # Save
    model_data = {
        "m_1x2": m1x2, "m_btts": mbtts, "m_over25": mo25, "m_over35": mo35,
        "m_under25": mu25, "m_1x": dcm["1X"], "m_x2": dcm["X2"], "m_12": dcm["12"],
        "m_btno": mbtno, "features": feature_cols,
    }

    out = r"C:\Users\furka\Desktop\ssifir-main\web_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Saved: {out}, time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    df = load_and_merge()
    train(df)
