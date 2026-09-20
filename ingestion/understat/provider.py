"""Understat provider (FREE, scraping-based).

Understat provides xG data for 6 leagues from 2014/15 onwards:
- Premier League, La Liga, Ligue 1, Serie A, Bundesliga, Russian Premier League

Data per match: xG (home/away), goals, datetime, teams.
No API key required. Shots/SOT not available at league level.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

from configs import settings

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUES = {
    "epl": "England Premier League",
    "la_liga": "Spain LaLiga",
    "bundesliga": "Germany Bundesliga",
    "serie_a": "Italy Serie A",
    "ligue_1": "France Ligue 1",
    "rpl": "Russia Premier League",
}

UNDERSTAT_SEASONS = [
    "2014", "2015", "2016", "2017", "2018",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025",
]

US_PREFIX = "US_"


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v):
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


async def _fetch_all() -> list[dict]:
    """Fetch all Understat matches across 6 leagues."""
    from understat import Understat

    all_matches = []
    async with aiohttp.ClientSession() as session:
        u = Understat(session)
        for league_key, league_name in UNDERSTAT_LEAGUES.items():
            for season in UNDERSTAT_SEASONS:
                try:
                    results = await u.get_league_results(league_key, season)
                    if not results:
                        continue

                    for m in results:
                        if not m.get("isResult", True):
                            continue

                        home_team = m.get("h", {}).get("title", "")
                        away_team = m.get("a", {}).get("title", "")

                        goals = m.get("goals", {})
                        xg = m.get("xG", {})

                        if isinstance(goals, dict):
                            home_goals = _safe_int(goals.get("h"))
                            away_goals = _safe_int(goals.get("a"))
                        else:
                            continue

                        if isinstance(xg, dict):
                            home_xg = _safe_float(xg.get("h"))
                            away_xg = _safe_float(xg.get("a"))
                        else:
                            home_xg = None
                            away_xg = None

                        if home_goals is None or away_goals is None:
                            continue

                        match_date = pd.to_datetime(m.get("datetime"), errors="coerce")
                        if pd.isna(match_date):
                            continue

                        if home_goals > away_goals:
                            result = "H"
                        elif away_goals > home_goals:
                            result = "A"
                        else:
                            result = "D"

                        all_matches.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "home_xg": home_xg,
                            "away_xg": away_xg,
                            "result": result,
                            "date": match_date,
                            "season": season,
                            "league_name": league_name,
                            "league_code": f"{US_PREFIX}{league_key}",
                        })

                    if results:
                        print(f"    {league_name} {season}: {len(results)} mac")

                except Exception as e:
                    logger.debug("Understat %s %s basarisiz: %s", league_key, season, e)
                    continue

    return all_matches


def _run_async():
    """Run async fetch in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(_fetch_all())
        else:
            return loop.run_until_complete(_fetch_all())
    except RuntimeError:
        return asyncio.run(_fetch_all())


def build_understat_matches() -> pd.DataFrame:
    """Build matches from Understat (6 leagues, 2014-2025).

    Returns canonical match DataFrame with xG data.
    """
    cache_path = Path(settings.DATA_DIR) / "gold" / "understat_matches.parquet"
    if cache_path.exists():
        logger.info("Understat: cache loaded from %s", cache_path)
        return pd.read_parquet(cache_path)

    print("[US] Loading Understat data (6 leagues, 2014-2025)...")

    raw = _run_async()
    if not raw:
        logger.warning("Understat: hic mac indirilemedi")
        return pd.DataFrame()

    team_ids = {}
    next_team_id = 8000000
    rows = []

    for m in raw:
        h_key = (m["league_name"], m["home_team"])
        a_key = (m["league_name"], m["away_team"])
        if h_key not in team_ids:
            team_ids[h_key] = next_team_id
            next_team_id += 1
        if a_key not in team_ids:
            team_ids[a_key] = next_team_id
            next_team_id += 1

        rows.append({
            "league": m["league_code"],
            "league_name": m["league_name"],
            "season": m["season"],
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_team_id": team_ids[h_key],
            "away_team_id": team_ids[a_key],
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "ht_home_goals": None,
            "ht_away_goals": None,
            "result": m["result"],
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
            "referee": None,
            "home_xg": m["home_xg"],
            "away_xg": m["away_xg"],
            "avg_home_odds": None,
            "avg_draw_odds": None,
            "avg_away_odds": None,
            "b365_home_odds": None,
            "b365_draw_odds": None,
            "b365_away_odds": None,
            "avg_over25_odds": None,
            "avg_under25_odds": None,
        })

    df = pd.DataFrame(rows)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    n_xg = df["home_xg"].notna().sum()
    logger.info("Understat: %d mac, %d lig (xG: %d)", len(df), df["league"].nunique(), n_xg)
    return df
