"""
Full SofaScore JSON parser -> parquet.
Extracts ALL stats from Istatistikler for both teams per match.
Also calculates Elo from scratch.
"""
import json, os, glob, time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

TAKIMLAR_DIR = r"C:\Users\furka\Desktop\sofascore-scraper-main\Takimlar"
OUT_DIR = r"C:\Users\furka\Desktop\ssifir-main\data\gold"

# All stats to extract from Istatistikler -> Match overview, Shots, Attack, Passes, Duels, Defending, Goalkeeping
STAT_MAP = {
    # Match overview
    "Ball possession": "possession",
    "Expected goals": "xg",
    "Big chances": "big_ch",
    "Total shots": "shots",
    "Goalkeeper saves": "gk_saves",
    "Corner kicks": "corners",
    "Fouls": "fouls",
    "Passes": "passes",
    "Tackles": "tackles",
    "Yellow cards": "yellows",
    "Red cards": "reds",
    "Free kicks": "free_kicks",
    "Distance covered": "distance_km",
    # Shots
    "Shots on target": "sot",
    "Expected goals on target": "xg_ot",
    "Hit woodwork": "woodwork",
    "Shots off target": "shots_off",
    "Blocked shots": "shots_blocked",
    "Shots inside box": "shots_inbox",
    "Shots outside box": "shots_outbox",
    # Attack
    "Big chances scored": "big_ch_scored",
    "Big chances missed": "big_ch_missed",
    "Through balls": "through_balls",
    "Touches in penalty area": "pen_touches",
    "Offsides": "offsides",
    # Passes
    "Accurate passes": "accurate_passes",
    "Final third entries": "final3_entries",
    "Long balls": "long_balls",
    "Crosses": "crosses",
    "Throw-ins": "throw_ins",
    # Duels
    "Duels": "duels_pct",
    "Dispossessed": "dispossessed",
    "Ground duels": "ground_duels",
    "Aerial duels": "aerial_duels",
    "Dribbles": "dribbles",
    # Defending
    "Total tackles": "total_tackles",
    "Interceptions": "interceptions",
    "Recoveries": "recoveries",
    "Clearances": "clearances",
    "Errors lead to a shot": "errors_shot",
    "Errors lead to a goal": "errors_goal",
    # Goalkeeping
    "Total saves": "total_saves",
    "Goals prevented": "goals_prevented",
    "High claims": "high_claims",
}

ELO_INIT = 1500
ELO_K = 32.0
HOME_ADV = 50.0


def _parse_pct(val):
    """Parse '59%' -> 59.0, '384' -> 384.0"""
    if val is None:
        return np.nan
    s = str(val).strip()
    if s.endswith('%'):
        try:
            return float(s[:-1])
        except ValueError:
            return np.nan
    # Handle "33/65 (50%)/65" format
    if '/' in s:
        parts = s.split('/')
        try:
            return float(parts[0])
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_score(score_str):
    """Parse '2 - 1' -> (2, 1)"""
    if not score_str:
        return None, None
    parts = str(score_str).split('-')
    if len(parts) == 2:
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            return None, None
    return None, None


def _parse_json_file(filepath):
    """Parse a single team JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    team_name = data.get('team_name', '')
    league_name = data.get('league_name', '')
    matches = data.get('matches', [])
    rows = []

    for m in matches:
        md = m.get('match_data', {})
        mi = md.get('Match_Info', {})

        home = mi.get('Home', '')
        away = mi.get('Away', '')
        score = mi.get('Score', '')
        tournament = mi.get('Tournament', '')
        timestamp = mi.get('Timestamp', 0)

        if timestamp:
            date = datetime.fromtimestamp(timestamp)
        else:
            continue

        home_goals, away_goals = _parse_score(score)
        if home_goals is None:
            continue

        # Find home and away team stats
        home_team_data = md.get(home, {})
        away_team_data = md.get(away, {})

        home_stats_raw = home_team_data.get('Istatistikler', {})
        away_stats_raw = away_team_data.get('Istatistikler', {})

        # Flatten Istatistikler - it's a dict of category -> dict of stat_name -> value
        def _flatten_stats(raw):
            flat = {}
            if isinstance(raw, dict):
                for cat_name, cat_data in raw.items():
                    if isinstance(cat_data, dict):
                        for stat_name, val in cat_data.items():
                            key = STAT_MAP.get(stat_name)
                            if key:
                                flat[key] = _parse_pct(val)
            return flat

        h_flat = _flatten_stats(home_stats_raw)
        a_flat = _flatten_stats(away_stats_raw)

        row = {
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "tournament": tournament,
            "league": league_name,
        }

        # Add home stats with h_ prefix
        for k, v in h_flat.items():
            row[f"h_{k}"] = v

        # Add away stats with a_ prefix
        for k, v in a_flat.items():
            row[f"a_{k}"] = v

        rows.append(row)

    return rows


def parse_all():
    """Parse all team JSONs into a single DataFrame."""
    print("=" * 70)
    print("  PARSE ALL SOFASCORE JSONs")
    print("=" * 70)

    all_rows = []
    league_teams = defaultdict(set)
    json_files = []

    for root, dirs, files in os.walk(TAKIMLAR_DIR):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))

    print(f"Found {len(json_files)} JSON files")
    t0 = time.time()

    for i, fp in enumerate(json_files):
        try:
            rows = _parse_json_file(fp)
            all_rows.extend(rows)
            if rows:
                league_teams[rows[0].get('league', '?')].add(rows[0].get('home_team', '?'))
        except Exception as e:
            print(f"  ERROR {fp}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(json_files)}... ({time.time()-t0:.0f}s, {len(all_rows)} rows)")

    print(f"  Done: {len(all_rows)} matches from {len(json_files)} files ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(all_rows)
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicates (same match from both teams' files)
    df["match_key"] = df["date"].astype(str) + "|" + df["home_team"] + "|" + df["away_team"]
    before = len(df)
    df = df.drop_duplicates(subset="match_key").drop(columns="match_key").reset_index(drop=True)
    print(f"  After dedup: {len(df)} (removed {before - len(df)})")

    # Stats summary
    stat_cols = [c for c in df.columns if c.startswith('h_') or c.startswith('a_')]
    print(f"\n  Stat columns: {len(stat_cols)}")
    for c in sorted(stat_cols):
        non_null = df[c].notna().sum()
        print(f"    {c}: {non_null}/{len(df)} ({non_null/len(df)*100:.0f}%)")

    print(f"\n  Leagues: {len(league_teams)}")
    for lg, teams in sorted(league_teams.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {lg}: {len(teams)} teams")

    return df


def compute_elo(df):
    """Compute Elo ratings from match results."""
    print("\n" + "=" * 70)
    print("  COMPUTE ELO RATINGS")
    print("=" * 70)

    elo = defaultdict(lambda: ELO_INIT)
    atk_elo = defaultdict(lambda: ELO_INIT)
    def_elo = defaultdict(lambda: ELO_INIT)

    home_elo_list = []
    away_elo_list = []
    home_atk_list = []
    away_atk_list = []
    home_def_list = []
    away_def_list = []

    t0 = time.time()

    for idx in range(len(df)):
        row = df.iloc[idx]
        ht = row["home_team"]
        at = row["away_team"]
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        h_elo = elo[ht]
        a_elo = elo[at]

        # Expected
        h_exp = 1.0 / (1.0 + 10 ** ((a_elo + HOME_ADV - h_elo) / 400))
        a_exp = 1.0 - h_exp

        # Actual
        if hg > ag:
            h_act, a_act = 1.0, 0.0
        elif hg == ag:
            h_act, a_act = 0.5, 0.5
        else:
            h_act, a_act = 0.0, 1.0

        elo[ht] = h_elo + ELO_K * (h_act - h_exp)
        elo[at] = a_elo + ELO_K * (a_act - a_exp)

        # Attack/Defence Elo
        h_atk = atk_elo[ht]
        a_def = def_elo[at]
        h_atk_exp = 1.0 / (1.0 + 10 ** ((a_def + HOME_ADV - h_atk) / 400))
        atk_elo[ht] = h_atk + ELO_K * (h_act - h_atk_exp)
        def_elo[at] = def_elo[at] + ELO_K * (a_act - (1.0 - h_atk_exp))

        a_atk = atk_elo[at]
        h_def = def_elo[ht]
        a_atk_exp = 1.0 / (1.0 + 10 ** ((h_def + HOME_ADV - a_atk) / 400))
        atk_elo[at] = a_atk + ELO_K * (a_act - a_atk_exp)
        def_elo[ht] = def_elo[ht] + ELO_K * (h_act - (1.0 - a_atk_exp))

        home_elo_list.append(round(h_elo))
        away_elo_list.append(round(a_elo))
        home_atk_list.append(round(h_atk))
        away_atk_list.append(round(a_atk))
        home_def_list.append(round(h_def))
        away_def_list.append(round(a_def))

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_attack_elo"] = home_atk_list
    df["away_attack_elo"] = away_atk_list
    df["home_defence_elo"] = home_def_list
    df["away_defence_elo"] = away_def_list
    df["attack_elo_diff"] = df["home_attack_elo"] - df["away_attack_elo"]
    df["defence_elo_diff"] = df["home_defence_elo"] - df["away_defence_elo"]

    print(f"  Done: {time.time()-t0:.0f}s")
    print(f"  Teams tracked: {len(elo)}")

    # Show top teams
    top = sorted(elo.items(), key=lambda x: -x[1])[:15]
    print(f"\n  TOP 15 ELO:")
    for name, rating in top:
        print(f"    {rating:.0f} {name}")

    # Check if current season teams have meaningful Elo
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest = df["date"].max()
    recent = df[df["date"] >= latest - pd.Timedelta(days=30)]
    if len(recent) > 0:
        print(f"\n  Last 30 days ({len(recent)} matches):")
        print(f"    Home elo range: {recent['home_elo'].min():.0f} - {recent['home_elo'].max():.0f}")
        print(f"    Away elo range: {recent['away_elo'].min():.0f} - {recent['away_elo'].max():.0f}")
        print(f"    Home elo mean: {recent['home_elo'].mean():.0f}")

    return df


def build_league_stats(df):
    """Compute per-league rolling averages for corners, goals, draw rate, etc."""
    print("\n  Computing league stats...")
    df = df.sort_values("date").reset_index(drop=True)

    # Global league averages (from all data)
    league_avg = {}
    for lg in df["league"].unique():
        lf = df[df["league"] == lg]
        league_avg[lg] = {
            "lg_avg_goals": (lf["home_goals"] + lf["away_goals"]).mean(),
            "lg_home_goal_avg": lf["home_goals"].mean(),
            "lg_away_goal_avg": lf["away_goals"].mean(),
            "lg_draw_rate": (lf["home_goals"] == lf["away_goals"]).mean(),
            "lg_btts_rate": ((lf["home_goals"] > 0) & (lf["away_goals"] > 0)).mean(),
            "lg_over25_rate": ((lf["home_goals"] + lf["away_goals"]) > 2.5).mean(),
        }
        # Corner average from h_corners if available
        if "h_corners" in lf.columns:
            league_avg[lg]["lg_corner_avg"] = (lf["h_corners"].fillna(0) + lf["a_corners"].fillna(0)).mean()
        else:
            league_avg[lg]["lg_corner_avg"] = 9.0
        if "h_yellows" in lf.columns:
            league_avg[lg]["lg_card_avg"] = (lf["h_yellows"].fillna(0) + lf["a_yellows"].fillna(0)).mean()
        else:
            league_avg[lg]["lg_card_avg"] = 3.5

    # Add league stats to each row
    for col in ["lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
                "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg"]:
        df[col] = df["league"].map(lambda lg: league_avg.get(lg, {}).get(col, 0))

    return df


if __name__ == "__main__":
    # Step 1: Parse all JSONs
    df = parse_all()

    # Step 2: Compute Elo
    df = compute_elo(df)

    # Step 3: League stats
    df = build_league_stats(df)

    # Step 4: Derived columns
    df["result"] = df.apply(lambda r: "H" if r["home_goals"] > r["away_goals"] else ("D" if r["home_goals"] == r["away_goals"] else "A"), axis=1)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    df["over15"] = (df["total_goals"] > 1.5).astype(int)
    df["over25"] = (df["total_goals"] > 2.5).astype(int)
    df["over35"] = (df["total_goals"] > 3.5).astype(int)
    if "h_corners" in df.columns:
        df["total_corners"] = df["h_corners"].fillna(0) + df["a_corners"].fillna(0)
    else:
        df["total_corners"] = 9.0

    # H2H features (rolling)
    print("\n  Computing H2H features...")
    team_pairs = defaultdict(list)
    h2h_home_win = []
    h2h_draw = []
    h2h_away_win = []
    h2h_goals = []
    h2h_btts = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        ht, at = row["home_team"], row["away_team"]
        key = tuple(sorted([ht, at]))
        prev = team_pairs[key][-20:]

        if len(prev) > 0:
            hw = sum(1 for p in prev if (p["winner"] == "home" and p["h"] == ht) or (p["winner"] == "away" and p["h"] == at))
            dr = sum(1 for p in prev if p["result"] == "D")
            aw = sum(1 for p in prev if (p["winner"] == "away" and p["h"] == ht) or (p["winner"] == "home" and p["h"] == at))
            n = len(prev)
            h2h_home_win.append(hw / n)
            h2h_draw.append(dr / n)
            h2h_away_win.append(aw / n)
            h2h_goals.append(np.mean([p["total"] for p in prev]))
            h2h_btts.append(np.mean([p["btts"] for p in prev]))
        else:
            h2h_home_win.append(0.45)
            h2h_draw.append(0.25)
            h2h_away_win.append(0.30)
            h2h_goals.append(2.5)
            h2h_btts.append(0.5)

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        winner = "home" if hg > ag else ("away" if ag > hg else "draw")
        team_pairs[key].append({
            "h": ht, "a": at, "hg": hg, "ag": ag,
            "total": hg + ag, "result": row["result"],
            "btts": int(hg > 0 and ag > 0), "winner": winner,
        })

    df["h2h_home_win"] = h2h_home_win
    df["h2h_draw"] = h2h_draw
    df["h2h_away_win"] = h2h_away_win
    df["h2h_goals_avg"] = h2h_goals
    df["h2h_btts"] = h2h_btts

    # Save
    out_path = os.path.join(OUT_DIR, "sofascore_full.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Matches: {len(df)}")
