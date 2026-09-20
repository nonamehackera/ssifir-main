"""xgabora Club Football Match Data provider - GERCEK istatistikli veri.

Kaynak: https://github.com/xgabora/Club-Football-Match-Data-2000-2025
- 27 ulke, 38 lig, 230K mac (2000-2025)
- GERCEK istatistikler: shots, SOT, corners, cards, fouls + odds + Elo + form
- API key gerektirmez, indirilebilir CSV (tek seferde, hizli)
- FD disindaki ulkeler: ARG, BRA, USA(MLS), MEX, JAP, CHN, ROM, POL, SWE,
  NOR, RUS, DEN, IRL, FIN, AUT, SUI
- Veri Football-Data.co.uk'ten uretilmis (F1, E0 gibi kodlar ayni)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
RAW_CACHE_FILE = Path("data/raw/xgabora_matches.csv.parquet")
CACHE_FILE = Path("data/gold/xgabora_matches.parquet")

# Division kodu -> (league_name, ulke)
LEAGUE_NAMES = {
    "E0": "England Premier League", "E1": "England Championship",
    "E2": "England League One", "E3": "England League Two",
    "EC": "England Conference", "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship", "SC2": "Scotland League One",
    "SC3": "Scotland League Two", "SP1": "Spain LaLiga", "SP2": "Spain Segunda Division",
    "I1": "Italy Serie A", "I2": "Italy Serie B", "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga", "F1": "France Ligue 1", "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie", "P1": "Portugal Primeira Liga",
    "B1": "Belgium Pro League", "T1": "Turkey Super Lig", "G1": "Greece Super League",
    "ARG": "Argentina Liga Profesional", "BRA": "Brazil Serie A",
    "USA": "USA Major League Soccer", "MEX": "Mexico Liga MX",
    "JAP": "Japan J1 League", "CHN": "China Super League",
    "ROM": "Romania Liga I", "POL": "Poland Ekstraklasa",
    "SWE": "Sweden Allsvenskan", "NOR": "Norway Eliteserien",
    "RUS": "Russia Premier League", "DEN": "Denmark Superliga",
    "IRL": "Ireland Premier Division", "FIN": "Finland Veikkausliiga",
    "AUT": "Austria Bundesliga", "SUI": "Switzerland Super League",
}


def _download() -> pd.DataFrame:
    """CSV'yi indir veya cache'ten oku."""
    if RAW_CACHE_FILE.exists():
        try:
            df = pd.read_parquet(RAW_CACHE_FILE)
            logger.info("xgabora raw cache okundu: %d mac", len(df))
            return df
        except Exception:
            pass

    logger.info("xgabora CSV indiriliyor: %s", DATASET_URL)
    df = pd.read_csv(DATASET_URL, low_memory=False)
    RAW_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_CACHE_FILE, index=False)
    logger.info("xgabora indirildi: %d mac", len(df))
    return df


def build_xgabora_matches() -> pd.DataFrame:
    """xgabora verisini canonical schema'ya cevirir."""
    df = _download()
    if df.empty:
        return df

    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"])

    rows = []
    for _, r in df.iterrows():
        try:
            hg, ag = int(r["FTHome"]), int(r["FTAway"])
        except (ValueError, TypeError):
            continue

        league_name = LEAGUE_NAMES.get(r["Division"], f"League {r['Division']}")
        country = league_name.split(" ")[0]

        ht_h = r.get("HTHome")
        ht_a = r.get("HTAway")
        if pd.isna(ht_h):
            ht_h, ht_a = None, None
        else:
            try:
                ht_h, ht_a = int(ht_h), int(ht_a)
            except (ValueError, TypeError):
                ht_h, ht_a = None, None

        def _int(v):
            if v is None or pd.isna(v):
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        row = {
            "date": r["MatchDate"],
            "home_team": str(r["HomeTeam"]).strip(),
            "away_team": str(r["AwayTeam"]).strip(),
            "home_goals": hg,
            "away_goals": ag,
            "ht_home_goals": ht_h,
            "ht_away_goals": ht_a,
            "result": "H" if hg > ag else ("A" if ag > hg else "D"),
            "home_shots": _int(r.get("HomeShots")),
            "away_shots": _int(r.get("AwayShots")),
            "home_sot": _int(r.get("HomeTarget")),
            "away_sot": _int(r.get("AwayTarget")),
            "home_corners": _int(r.get("HomeCorners")),
            "away_corners": _int(r.get("AwayCorners")),
            "home_fouls": _int(r.get("HomeFouls")),
            "away_fouls": _int(r.get("AwayFouls")),
            "home_yellow": _int(r.get("HomeYellow")),
            "away_yellow": _int(r.get("AwayYellow")),
            "home_red": _int(r.get("HomeRed")),
            "away_red": _int(r.get("AwayRed")),
            "b365_home_odds": r.get("OddHome") if pd.notna(r.get("OddHome")) else None,
            "b365_draw_odds": r.get("OddDraw") if pd.notna(r.get("OddDraw")) else None,
            "b365_away_odds": r.get("OddAway") if pd.notna(r.get("OddAway")) else None,
            "avg_over25_odds": r.get("MaxOver25") if pd.notna(r.get("MaxOver25")) else None,
            "avg_under25_odds": r.get("MaxUnder25") if pd.notna(r.get("MaxUnder25")) else None,
            "league": f"XG_{r['Division']}",
            "league_name": league_name,
            "season": str(r["MatchDate"].year),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Team ID'leri (lig icinde tutarli)
    next_id = 6000000
    team_map: dict[str, int] = {}
    for _, r in result.iterrows():
        for side in ["home", "away"]:
            key = (r["league"], r[f"{side}_team"].lower())
            if key not in team_map:
                team_map[key] = next_id
                next_id += 1
    result["home_team_id"] = [team_map[(lg, ht.lower())] for lg, ht in zip(result["league"], result["home_team"])]
    result["away_team_id"] = [team_map[(lg, at.lower())] for lg, at in zip(result["league"], result["away_team"])]

    for col in ["referee", "avg_home_odds", "avg_draw_odds", "avg_away_odds", "round",
                "home_xg", "away_xg"]:
        if col not in result.columns:
            result[col] = None

    # b365 odds'lardan avg odds'leri kopyala (eger avg yoksa)
    mask_no_avg = result["avg_home_odds"].isna() & result["b365_home_odds"].notna()
    result.loc[mask_no_avg, "avg_home_odds"] = result.loc[mask_no_avg, "b365_home_odds"]
    result.loc[mask_no_avg, "avg_draw_odds"] = result.loc[mask_no_avg, "b365_draw_odds"]
    result.loc[mask_no_avg, "avg_away_odds"] = result.loc[mask_no_avg, "b365_away_odds"]

    result = result.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    result = result.sort_values("date").reset_index(drop=True)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(CACHE_FILE, index=False)

    n_stats = result["home_shots"].notna().sum()
    logger.info("xgabora toplam: %d mac, %d lig, %d mac istatistikli",
                len(result), result["league"].nunique(), n_stats)
    return result