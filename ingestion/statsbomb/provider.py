"""StatsBomb Open Data provider (FREE, unlimited).

StatsBomb open-data: 80+ free competition-seasons, match-level data.
Free competitions: Premier League, La Liga, Bundesliga, Ligue 1, Serie A,
Champions League, World Cup, Euros, and more.

Match-level data: scores, teams, dates, referees, stadium.
No event-level download (too slow); focus on match metadata for
dedup and coverage enrichment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from configs import settings

logger = logging.getLogger(__name__)


def _safe_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _safe_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def build_statsbomb_matches() -> pd.DataFrame:
    """Build matches from StatsBomb Open Data (FREE, 80+ competitions).

    Downloads match-level data only (no event-level processing).
    Provides: scores, teams, dates, referees. Used for dedup enrichment.
    """
    from statsbombpy import sb

    cache_path = Path(settings.DATA_DIR) / "gold" / "statsbomb_matches.parquet"
    if cache_path.exists():
        logger.info("StatsBomb: cache loaded from %s", cache_path)
        return pd.read_parquet(cache_path)

    print("[SB] Loading StatsBomb Open Data (80+ competition-seasons)...")

    comps = sb.competitions()
    if comps.empty:
        logger.warning("StatsBomb: hic turnuva bulunamadi")
        return pd.DataFrame()

    all_matches = []
    team_ids = {}
    next_team_id = 7000000

    for _, comp in comps.iterrows():
        comp_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        comp_name = str(comp.get("competition_name", ""))
        season_name = str(comp.get("season_name", ""))

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches.empty:
                continue

            for _, mrow in matches.iterrows():
                home_team = str(mrow.get("home_team", ""))
                away_team = str(mrow.get("away_team", ""))
                home_score = _safe_int(mrow.get("home_score"))
                away_score = _safe_int(mrow.get("away_score"))
                match_date = pd.to_datetime(mrow.get("match_date"), errors="coerce")

                if pd.isna(match_date) or home_score is None or away_score is None:
                    continue

                # Team IDs
                h_key = (comp_name, home_team)
                a_key = (comp_name, away_team)
                if h_key not in team_ids:
                    team_ids[h_key] = next_team_id
                    next_team_id += 1
                if a_key not in team_ids:
                    team_ids[a_key] = next_team_id
                    next_team_id += 1

                if home_score > away_score:
                    result = "H"
                elif away_score > home_score:
                    result = "A"
                else:
                    result = "D"

                season_str = str(match_date.year)
                league_code = f"SB_{comp_id}"

                record = {
                    "league": league_code,
                    "league_name": comp_name,
                    "season": season_str,
                    "date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_team_id": team_ids[h_key],
                    "away_team_id": team_ids[a_key],
                    "home_goals": home_score,
                    "away_goals": away_score,
                    "ht_home_goals": None,
                    "ht_away_goals": None,
                    "result": result,
                    "home_shots": None,
                    "away_shots": None,
                    "home_sot": None,
                    "away_sot": None,
                    "home_corners": None,
                    "away_corners": None,
                    "home_fouls": None,
                    "away_fouls": None,
                    "home_yellow": None,
                    "away_yellow": None,
                    "home_red": None,
                    "away_red": None,
                    "referee": mrow.get("referee"),
                    "home_xg": None,
                    "away_xg": None,
                    "avg_home_odds": None,
                    "avg_draw_odds": None,
                    "avg_away_odds": None,
                    "b365_home_odds": None,
                    "b365_draw_odds": None,
                    "b365_away_odds": None,
                    "avg_over25_odds": None,
                    "avg_under25_odds": None,
                }
                all_matches.append(record)

            n = len(matches)
            if n > 0:
                print(f"    {comp_name} {season_name}: {n} mac")

        except Exception as e:
            logger.debug("SB fetch basarisiz: %s %s: %s", comp_name, season_name, e)
            continue

    if not all_matches:
        logger.warning("StatsBomb: hic mac indirilemedi")
        return pd.DataFrame()

    df = pd.DataFrame(all_matches)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    logger.info(
        "StatsBomb: %d mac, %d lig",
        len(df), df["league"].nunique(),
    )
    return df
