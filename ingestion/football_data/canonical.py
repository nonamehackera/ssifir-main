"""Canonical historical match builder - Football-Data.co.uk + Hugging Face + TheSportsDB.

ROADMAP bolum 2.3: Combined provider supporting 210+ countries, 2000+ leagues worldwide.

Kaynaklar:
  1. Football-Data.co.uk: 30 lig, detayli istatistik (gol, sut, sot, korner, kart, odds)
  2. Hugging Face: 195+ lig, 673K+ mac (sadece sonuc)
  3. TheSportsDB: 2000+ lig, ucretsiz (sadece sonuc)

Output: a leakage-free canonical match table saved to gold/matches.parquet.
Every row is a single completed match; the feature engine consumes these in
strict chronological order.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from configs import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FOOTBALL-DATA.CO.UK (30 leagues, 15 seasons, 1993-2025)
# ---------------------------------------------------------------------------
FD_BASE = "https://www.football-data.co.uk/mmz4281"

FD_LEAGUES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "EC": "England Conference",
    "SP1": "Spain LaLiga",
    "SP2": "Spain LaLiga 2",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "T1": "Turkey Super Lig",
    "P1": "Portugal Primeira Liga",
    "P2": "Portugal Liga 2",
    "B1": "Belgium Pro League",
    "G1": "Greece Super League",
    "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship",
    "SC2": "Scotland League One",
    "SC3": "Scotland League Two",
    "DK1": "Denmark Superligaen",
    "DK2": "Denmark 1st Division",
    "SE1": "Sweden Allsvenskan",
    "NO1": "Norway Eliteserien",
    "PL1": "Poland Ekstraklasa",
    "IR1": "Ireland Premier Division",
    "IS1": "Iceland Úrvalsdeild",
}

FD_SEASONS = [
    "1011", "1112", "1213", "1314", "1415",
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]

# ---------------------------------------------------------------------------
# HUGGING FACE DATASET (195+ leagues, 673K+ matches)
# ---------------------------------------------------------------------------
HF_LEAGUE_NAMES = {
    1: "England Premier League",
    2: "England Championship",
    3: "England League One",
    4: "England League Two",
    5: "Spain LaLiga",
    6: "Spain Segunda Division",
    7: "Italy Serie A",
    8: "Italy Serie B",
    9: "France Ligue 1",
    10: "France Ligue 2",
    11: "Germany Bundesliga",
    12: "Germany 2. Bundesliga",
    13: "Netherlands Eredivisie",
    14: "Portugal Primeira Liga",
    15: "Belgium Jupiler Pro League",
    16: "Turkey Super Lig",
    17: "Scotland Premiership",
    18: "Scotland Championship",
    19: "Brazil Serie A",
    20: "USA MLS",
    21: "Japan J1 League",
    22: "Mexico Liga MX",
    23: "Norway Eliteserien",
    24: "Russia Premier League",
    25: "Austria Bundesliga",
    26: "Switzerland Super League",
    27: "Poland Ekstraklasa",
    28: "Argentina Liga Profesional",
    29: "Sweden Allsvenskan",
    30: "Denmark Superliga",
    31: "China Super League",
    32: "Romania SuperLiga",
    33: "Finland Veikkausliiga",
    34: "Ireland Premier Division",
    35: "Greece Super League 1",
    36: "Czech Republic 1. Liga",
    37: "Saudi Arabia Pro League",
    38: "South Korea K League 1",
    39: "Morocco Botola Pro",
    40: "Egypt Premier League",
    41: "UAE Pro League",
    42: "Qatar Stars League",
    43: "Croatia HNL",
    44: "Cyprus 1. Division",
    45: "Ecuador Liga Pro Serie A",
    46: "Colombia Primera A",
    47: "Hungary NB I",
    48: "Algeria Ligue 1",
    49: "Paraguay Division Profesional",
    50: "Israel Ligat HaAl",
    51: "Uruguay Primera Division",
    52: "Costa Rica Primera Division",
    53: "Chile Primera Division",
    54: "Slovakia Nike Liga",
    55: "Slovenia Prva Liga",
    56: "Iran Persian Gulf Pro League",
    57: "Brazil Serie B",
    58: "Bolivia Primera Division",
    59: "Azerbaijan Premier League",
    60: "South Africa PSL",
    61: "Ukraine Premier League",
    62: "Australia A-League",
    63: "Serbia Super Liga",
    64: "Peru Primera Division",
    65: "Portugal Liga Portugal 2",
    66: "Bulgaria First Professional League",
    67: "Honduras Liga Nacional",
    68: "Bosnia Premier League",
    69: "Tunisia Ligue 1",
    70: "Belgium Challenger Pro League",
    71: "Georgia Erovnuli Liga",
    72: "Venezuela Primera Division",
    73: "Netherlands Eerste Divisie",
    74: "Latvia Virsliga",
    75: "Canada Premier League",
    76: "USA USL Championship",
    77: "India Super League",
    103: "England National League",
    104: "Scotland League One",
    105: "Scotland League Two",
    106: "Germany 3. Liga",
    107: "Turkey 1. Lig",
    108: "Austria 2. Liga",
    109: "Russia First League",
    110: "Thailand Thai League 1",
    111: "Belarus Premier League",
    112: "Switzerland Challenge League",
    113: "Poland I Liga",
    114: "Croatia First NL",
    115: "Romania Liga II",
    116: "Serbia Prva Liga",
    117: "South Korea K League 2",
    118: "Japan J2 League",
    119: "France National 1",
    120: "Denmark 1. Division",
    121: "Hungary NB II",
    122: "Czech Republic FNL",
    123: "Bulgaria Second League",
    124: "Greece Super League 2",
    125: "Argentina Primera Nacional",
    126: "Chile Primera B",
    127: "Colombia Primera B",
    128: "Costa Rica Liga de Ascenso",
    129: "Ireland First Division",
    130: "Northern Ireland Premiership",
    131: "Estonia Meistriliiga",
    132: "Kazakhstan Premier League",
    133: "Indonesia Liga 1",
    134: "Lithuania A Lyga",
    135: "El Salvador Primera Division",
    136: "Panama Liga Panamena",
    137: "Guatemala Liga Nacional",
    138: "Kosovo Superliga",
    139: "Malaysia Super League",
    140: "Jamaica Premier League",
    141: "Ghana Premier League",
    142: "Nigeria NPFL",
    143: "Kenya FKF Premier League",
    144: "Armenia Premier League",
    100000075: "Brazil Serie C",
    100000104: "Norway 1. Division",
    100000110: "Wales Premier League",
    100000114: "Sweden Superettan",
    100000131: "Argentina Primera B Metropolitana",
    100000165: "Iceland 1. Deild",
    100000166: "Iceland 2. Deild",
    100000189: "Australia Capital Territory NPL",
    100000191: "Australia Brisbane Premier League",
    100000243: "Ecuador Liga Pro Serie B",
    100000245: "Finland Ykkönen",
    100000247: "Finland Kakkonen",
    100000256: "USA USL League Two",
    100000275: "Indonesia Liga 2",
    100000277: "Kenya Super League",
    100000282: "Peru Segunda Division",
    100000291: "Iran Azadegan League",
    100000297: "Thailand Thai League 2",
    100000300: "Venezuela Segunda Division",
    100000328: "Estonia Esiliiga A",
    100000330: "Kuwait Premier League",
    100000331: "Kuwait Division 1",
    100000338: "Guatemala Primera Division",
    100000343: "Armenia First League",
    100000363: "Ethiopia Premier League",
    100000364: "Latvia 1. Liga",
    100000368: "Singapore Premier League",
    100000380: "Hong Kong Premier League",
    100000386: "Ivory Coast Ligue 1",
    100000390: "Lebanon Premier League",
    100000391: "Malawi Super League",
    100000392: "Malta Challenge League",
    100000393: "Malta Premier League",
    100000394: "Moldova Super Liga",
    100000396: "Nicaragua Primera Division",
    100000398: "Bangladesh Premier League",
    100000400: "Zambia Super League",
    100000402: "Sudan Premier League",
    100000407: "Northern Ireland Championship",
    100000411: "Cameroon Elite One",
    100000412: "Botswana Premier League",
    100000417: "Bahrain Premier League",
    100000422: "Barbados Premier League",
    100000481: "Australia Northern NSW NPL",
    100000489: "USA USL League One",
    100000506: "Slovakia 2. liga",
    100000549: "Sweden Damallsvenskan",
    100000563: "Sweden Ettan Norra",
    100000585: "Uganda Premier League",
    100000588: "Myanmar National League",
    100000598: "Mali Premiere Division",
    100000604: "Brazil Catarinense 1",
    100000629: "Brazil Mineiro 1",
    100000711: "Chile Segunda Division",
    100000722: "Mexico Liga Premier Serie A",
    100000758: "Gibraltar Premier Division",
    100000833: "Australia Queensland Premier League",
    100000851: "Brazil Carioca A2",
    100000907: "Cuba Primera Division",
    100000955: "New Zealand National League",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_team(name: str) -> str:
    """Cheap canonicalization for a team name within a league."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def _safe_int(v) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _parse_date(date_str: str, time_str: str | None = None) -> pd.Timestamp | None:
    try:
        d = pd.to_datetime(date_str, format="%d/%m/%Y", errors="raise")
    except Exception:
        try:
            d = pd.to_datetime(date_str, errors="raise")
        except Exception:
            return None
    if time_str and str(time_str).strip():
        try:
            hh, mm = str(time_str).split(":")
            d = d + pd.Timedelta(hours=int(hh), minutes=int(mm))
        except Exception:
            pass
    return d


# ---------------------------------------------------------------------------
# Football-Data.co.uk provider
# ---------------------------------------------------------------------------

def _download_csv(league: str, season: str) -> str | None:
    """Download a single CSV (cached under bronze). Returns text or None."""
    raw = (
        Path(settings.DATA_DIR)
        / "bronze"
        / "football_data"
        / "canonical"
        / f"{league}_{season}.csv"
    )
    if raw.exists():
        return raw.read_text(encoding="utf-8-sig")

    url = f"{FD_BASE}/{season}/{league}.csv"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            text = r.read().decode("utf-8-sig", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.info("Yok: %s %s", league, season)
            return None
        raise
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(text, encoding="utf-8")
    return text


def _build_fd_matches() -> pd.DataFrame:
    """Build matches from Football-Data.co.uk CSVs."""
    rows: list[dict] = []
    team_ids: dict[tuple[str, str], int] = {}
    next_team_id = 1

    for lg, lg_name in FD_LEAGUES.items():
        for season in FD_SEASONS:
            text = _download_csv(lg, season)
            if not text or not text.strip():
                continue
            df = pd.read_csv(pd.io.common.StringIO(text))
            df.columns = [c.strip() for c in df.columns]
            for _, r in df.iterrows():
                hg = r.get("FTHG")
                ag = r.get("FTAG")
                if pd.isna(hg) or pd.isna(ag):
                    continue
                h_name = r.get("HomeTeam")
                a_name = r.get("AwayTeam")
                if pd.isna(h_name) or pd.isna(a_name):
                    continue
                h_key = (lg, _norm_team(h_name))
                a_key = (lg, _norm_team(a_name))
                if h_key not in team_ids:
                    team_ids[h_key] = next_team_id
                    next_team_id += 1
                if a_key not in team_ids:
                    team_ids[a_key] = next_team_id
                    next_team_id += 1

                hg = int(hg)
                ag = int(ag)
                result = "H" if hg > ag else ("A" if ag > hg else "D")

                rec = {
                    "league": lg,
                    "league_name": lg_name,
                    "season": season,
                    "date": _parse_date(str(r.get("Date")), str(r.get("Time"))),
                    "home_team": str(h_name).strip(),
                    "away_team": str(a_name).strip(),
                    "home_team_id": team_ids[h_key],
                    "away_team_id": team_ids[a_key],
                    "home_goals": hg,
                    "away_goals": ag,
                    "ht_home_goals": _safe_int(r.get("HTHG")),
                    "ht_away_goals": _safe_int(r.get("HTAG")),
                    "result": result,
                    "home_shots": _safe_int(r.get("HS")),
                    "away_shots": _safe_int(r.get("AS")),
                    "home_sot": _safe_int(r.get("HST")),
                    "away_sot": _safe_int(r.get("AST")),
                    "home_corners": _safe_int(r.get("HC")),
                    "away_corners": _safe_int(r.get("AC")),
                    "home_fouls": _safe_int(r.get("HF")),
                    "away_fouls": _safe_int(r.get("AF")),
                    "home_yellow": _safe_int(r.get("HY")),
                    "away_yellow": _safe_int(r.get("AY")),
                    "home_red": _safe_int(r.get("HR")),
                    "away_red": _safe_int(r.get("AR")),
                    "referee": (str(r.get("Referee")).strip() if pd.notna(r.get("Referee")) else None),
                    "avg_home_odds": _safe_float(r.get("AvgH")),
                    "avg_draw_odds": _safe_float(r.get("AvgD")),
                    "avg_away_odds": _safe_float(r.get("AvgA")),
                    "b365_home_odds": _safe_float(r.get("B365H")),
                    "b365_draw_odds": _safe_float(r.get("B365D")),
                    "b365_away_odds": _safe_float(r.get("B365A")),
                    "avg_over25_odds": _safe_float(r.get("Avg>2.5")),
                    "avg_under25_odds": _safe_float(r.get("Avg<2.5")),
                    "avg_close_home_odds": _safe_float(r.get("AvgCH")),
                    "avg_close_draw_odds": _safe_float(r.get("AvgCD")),
                    "avg_close_away_odds": _safe_float(r.get("AvgCA")),
                    "avg_close_over25_odds": _safe_float(r.get("AvgC>2.5")),
                    "avg_close_under25_odds": _safe_float(r.get("AvgC<2.5")),
                }
                rows.append(rec)

    logger.info("Football-Data.co.uk: %d matches from %d leagues", len(rows), len(FD_LEAGUES))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_hf_matches(hf_odds_only: bool = False) -> pd.DataFrame:
    """Build matches from Hugging Face soccer dataset.

    GERCEK istatistik icin 4 dosya birlestirilir:
    - fixtures.parquet: skor, tarih, takim ID, hakem adi
    - match_stats.parquet: GERCEK istatistik (shot, SOT, corner, kart, faul,
      HT gol, xG) - 283K mac
    - odds.parquet: GERCEK oranlar (Bet365 + API-Football) - 214K mac
    - teams.parquet: takim ID -> GERCEK takim adi

    hf_odds_only=True -> SADECE odds'i olan maclar tutulur (bahis icin).
    Boylece 668K hacim yerine ~214K ORANLI mac kalir, bahis edge'i icin
    kullanisli olur (hacim !== bahis degeri).
    """
    from datasets import load_dataset

    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    print("[HF] Loading fixtures + match_stats + odds + teams (token ile)...")
    fixtures = load_dataset("eatpizzanot/soccer-dataset", data_files="fixtures.parquet", split="train").to_pandas()
    stats = load_dataset("eatpizzanot/soccer-dataset", data_files="match_stats.parquet", split="train").to_pandas()
    odds = load_dataset("eatpizzanot/soccer-dataset", data_files="odds.parquet", split="train").to_pandas()
    teams = load_dataset("eatpizzanot/soccer-dataset", data_files="teams.parquet", split="train").to_pandas()
    xg_df = load_dataset("eatpizzanot/soccer-dataset", data_files="xg_training.parquet", split="train").to_pandas()

    # xG map'i: fixture_id + side -> xg + shots + sot
    xg_df["fixture_id"] = xg_df["fixture_id"].astype(int)
    xg_home = xg_df[xg_df["side"] == "home"].set_index("fixture_id")["xg"]
    xg_away = xg_df[xg_df["side"] == "away"].set_index("fixture_id")["xg"]
    xg_shots_home = xg_df[xg_df["side"] == "home"].set_index("fixture_id")["shots_total"]
    xg_shots_away = xg_df[xg_df["side"] == "away"].set_index("fixture_id")["shots_total"]
    xg_sot_home = xg_df[xg_df["side"] == "home"].set_index("fixture_id")["shots_on_goal"]
    xg_sot_away = xg_df[xg_df["side"] == "away"].set_index("fixture_id")["shots_on_goal"]

    def _safe_val(series, fid):
        try:
            v = series.get(int(fid))
        except (ValueError, TypeError):
            return None
        if v is None or (isinstance(v, float) and v != v):
            return None
        return float(v)

    def _xg_for(fid, side):
        return _safe_val(xg_home if side == "home" else xg_away, fid)

    def _xg_shots_for(fid, side):
        return _safe_val(xg_shots_home if side == "home" else xg_shots_away, fid)

    def _xg_sot_for(fid, side):
        return _safe_val(xg_sot_home if side == "home" else xg_sot_away, fid)

    # Takim adi haritasi
    team_name_by_id = dict(zip(teams["id"].astype(str), teams["name"]))
    team_name_by_api = dict(zip(teams["api_football_id"].astype(str), teams["name"]))
    team_name_by_fd = dict(zip(teams["id"].astype(str), teams["fd_name"]))
    for k, v in team_name_by_fd.items():
        if v and str(v) != "nan":
            team_name_by_id[k] = v

    # Oranlar: her mac icin tek satir (ortalamalar)
    odds_grp = odds.groupby("fixture_id")[["home_win", "draw", "away_win"]].mean().reset_index()
    odds_map = {
        r.fixture_id: (r.home_win, r.draw, r.away_win)
        for r in odds_grp.itertuples()
    }

    # Stats haritasi: fixture_id -> dict (vectorized join icin)
    stats_map = {r.fixture_id: r for r in stats.itertuples()}

    df = fixtures[fixtures["is_played"] == True].copy()

    # League adi
    df["league_name"] = df["league_id"].map(HF_LEAGUE_NAMES)
    df = df.dropna(subset=["league_name"])
    df["league"] = "HF_" + df["league_id"].astype(str)
    df["date"] = pd.to_datetime(df["date_utc"]).dt.tz_localize(None)
    df["season"] = df["date"].dt.year.astype(str)

    # Takim ID'leri
    team_map = {}
    next_id = 1000000

    def _get_team_id(team_api_id):
        nonlocal next_id
        if team_api_id not in team_map:
            team_map[team_api_id] = next_id
            next_id += 1
        return team_map[team_api_id]

    raw_home_id = df["home_team_id"].astype(str)
    raw_away_id = df["away_team_id"].astype(str)

    def _team_name(raw_id):
        return team_name_by_api.get(raw_id) or team_name_by_id.get(raw_id)

    df["home_team"] = raw_home_id.map(_team_name)
    df["away_team"] = raw_away_id.map(_team_name)
    df["home_team"] = df["home_team"].fillna("Team" + raw_home_id)
    df["away_team"] = df["away_team"].fillna("Team" + raw_away_id)
    df["home_team_id"] = df["home_team_id"].apply(_get_team_id)
    df["away_team_id"] = df["away_team_id"].apply(_get_team_id)

    df["result"] = df.apply(
        lambda r: "H" if r["goals_home"] > r["goals_away"]
        else ("A" if r["goals_away"] > r["goals_home"] else "D"),
        axis=1,
    )

    def _stat_for(fid, col):
        try:
            s = stats_map.get(int(fid))
        except (ValueError, TypeError):
            return None
        if s is None:
            return None
        v = getattr(s, col)
        if v is None or (isinstance(v, float) and v != v):
            return None
        return v

    def _join_odds(fid):
        try:
            return odds_map.get(int(fid), (None, None, None))
        except (ValueError, TypeError):
            return (None, None, None)

    hg_ht = df["id"].map(lambda fid: _stat_for(fid, "home_goals_ht"))
    ag_ht = df["id"].map(lambda fid: _stat_for(fid, "away_goals_ht"))
    odds_vals = df["id"].map(_join_odds)

    # Stats ve xG join: once stats'den al, yoksa xG training'den shot/sot doldur
    home_shots_raw = df["id"].map(lambda fid: _stat_for(fid, "home_shots_total"))
    away_shots_raw = df["id"].map(lambda fid: _stat_for(fid, "away_shots_total"))
    home_sot_raw = df["id"].map(lambda fid: _stat_for(fid, "home_shots_on_goal"))
    away_sot_raw = df["id"].map(lambda fid: _stat_for(fid, "away_shots_on_goal"))

    # xG training'den eksik shot/sot degerlerini doldur
    home_shots = home_shots_raw.copy()
    away_shots = away_shots_raw.copy()
    home_sot = home_sot_raw.copy()
    away_sot = away_sot_raw.copy()

    for i, fid in enumerate(df["id"]):
        if pd.isna(home_shots.iloc[i]) or home_shots.iloc[i] is None:
            v = _xg_shots_for(fid, "home")
            if v is not None:
                home_shots.iloc[i] = int(round(v))
        if pd.isna(away_shots.iloc[i]) or away_shots.iloc[i] is None:
            v = _xg_shots_for(fid, "away")
            if v is not None:
                away_shots.iloc[i] = int(round(v))
        if pd.isna(home_sot.iloc[i]) or home_sot.iloc[i] is None:
            v = _xg_sot_for(fid, "home")
            if v is not None:
                home_sot.iloc[i] = int(round(v))
        if pd.isna(away_sot.iloc[i]) or away_sot.iloc[i] is None:
            v = _xg_sot_for(fid, "away")
            if v is not None:
                away_sot.iloc[i] = int(round(v))

    result = pd.DataFrame({
        "league": df["league"],
        "league_name": df["league_name"],
        "season": df["season"],
        "date": df["date"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_team_id": df["home_team_id"],
        "away_team_id": df["away_team_id"],
        "home_goals": df["goals_home"].astype(int),
        "away_goals": df["goals_away"].astype(int),
        "ht_home_goals": hg_ht,
        "ht_away_goals": ag_ht,
        "result": df["result"],
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_sot": home_sot,
        "away_sot": away_sot,
        "home_corners": df["id"].map(lambda fid: _stat_for(fid, "home_corners")),
        "away_corners": df["id"].map(lambda fid: _stat_for(fid, "away_corners")),
        "home_fouls": df["id"].map(lambda fid: _stat_for(fid, "home_fouls")),
        "away_fouls": df["id"].map(lambda fid: _stat_for(fid, "away_fouls")),
        "home_yellow": df["id"].map(lambda fid: _stat_for(fid, "home_yellow_cards")),
        "away_yellow": df["id"].map(lambda fid: _stat_for(fid, "away_yellow_cards")),
        "home_red": df["id"].map(lambda fid: _stat_for(fid, "home_red_cards")),
        "away_red": df["id"].map(lambda fid: _stat_for(fid, "away_red_cards")),
        "referee": df["referee_name"],
        "avg_home_odds": odds_vals.map(lambda o: o[0]),
        "avg_draw_odds": odds_vals.map(lambda o: o[1]),
        "avg_away_odds": odds_vals.map(lambda o: o[2]),
        "home_xg": df["id"].map(lambda fid: _xg_for(fid, "home")),
        "away_xg": df["id"].map(lambda fid: _xg_for(fid, "away")),
        "b365_home_odds": None,
        "b365_draw_odds": None,
        "b365_away_odds": None,
        "avg_over25_odds": None,
        "avg_under25_odds": None,
    })

    n_stats = result["home_shots"].notna().sum()
    n_xg = result["home_xg"].notna().sum()
    n_odds = result["avg_home_odds"].notna().sum()
    n_ref = result["referee"].notna().sum()
    n_teams = result["home_team"].str.startswith("Team").sum() + result["away_team"].str.startswith("Team").sum()
    logger.info(
        "Hugging Face: %d matches from %d leagues (stat: %d, xG: %d, oran: %d, hakem: %d, isimsiz: %d)",
        len(result), result["league_name"].nunique(), n_stats, n_xg, n_odds, n_ref, n_teams,
    )

    if hf_odds_only:
        before = len(result)
        result = result[result["avg_home_odds"].notna() &
                        result["avg_draw_odds"].notna() &
                        result["avg_away_odds"].notna()].copy()
        logger.info("HF odds_only filtresi: %d -> %d mac (sadece oranli)", before, len(result))

    return result


def _build_statsbomb_matches() -> pd.DataFrame:
    """Build matches from StatsBomb Open Data (FREE, event-level, 36+ competitions)."""
    from ingestion.statsbomb.provider import build_statsbomb_matches

    print("[SB] Loading matches from StatsBomb Open Data (PL, LaLiga, Bundesliga, Serie A, L1, UCL...)")
    return build_statsbomb_matches()


def _build_understat_matches() -> pd.DataFrame:
    """Build matches from Understat (FREE, xG for 6 leagues, 2014-2025)."""
    from ingestion.understat.provider import build_understat_matches

    print("[US] Loading matches from Understat (EPL, LaLiga, Bundesliga, Serie A, Ligue 1, RPL...)")
    return build_understat_matches()


def _build_openfootball_matches() -> pd.DataFrame:
    """Build matches from OpenFootball (free, open source, 20+ leagues)."""
    from ingestion.openfootball.provider import build_openfootball_matches

    print("[OF] Loading matches from OpenFootball (EPL, Bundesliga, LaLiga, Serie A, Ligue 1...)")
    return build_openfootball_matches()


def _build_thesportsdb_matches() -> pd.DataFrame:
    """Build matches from TheSportsDB (2000+ leagues, free API)."""
    from ingestion.thesportsdb.adapter import fetch_all_leagues, fetch_league_matches

    print("[TSDB] Loading leagues from TheSportsDB (2000+ leagues)...")
    leagues = fetch_all_leagues()
    logger.info("TheSportsDB: %d soccer ligleri bulundu", len(leagues))

    # Hedef ulkelere ait ligleri filtrele
    target_countries = [
        "England", "Spain", "Italy", "Germany", "France", "Netherlands",
        "Portugal", "Belgium", "Turkey", "Greece", "Scotland", "Denmark",
        "Sweden", "Norway", "Poland", "Czech Republic", "Austria", "Switzerland",
        "Brazil", "Argentina", "Colombia", "Chile", "Peru", "Ecuador",
        "Mexico", "USA", "Japan", "South Korea", "China", "Australia",
        "Morocco", "Egypt", "South Africa", "Nigeria", "Ghana",
        "Croatia", "Serbia", "Romania", "Bulgaria", "Hungary", "Ukraine",
        "Russia", "Finland", "Ireland", "Israel", "Saudi Arabia", "UAE",
        "India", "Indonesia", "Thailand", "Malaysia", "Singapore",
    ]

    target_leagues = leagues[
        leagues["strCountry"].isin(target_countries) |
        leagues["strLeague"].str.contains("|".join(target_countries), case=False, na=False)
    ]

    if len(target_leagues) == 0:
        target_leagues = leagues.head(100)

    logger.info("TheSportsDB: %d hedef lig secildi", len(target_leagues))

    seasons = [2022, 2023, 2024, 2025]
    frames = []
    team_map = {}
    next_team_id = 2000000

    for _, lg in target_leagues.iterrows():
        lid = int(lg["idLeague"])
        league_name = lg.get("strLeague", f"League_{lid}")

        for s in seasons:
            try:
                df = fetch_league_matches(lid, s)
                if df.empty:
                    continue

                for col in ["home_team", "away_team"]:
                    for team_name in df[col].dropna().unique():
                        key = (league_name, str(team_name).strip().lower())
                        if key not in team_map:
                            team_map[key] = next_team_id
                            next_team_id += 1

                df["league"] = f"TS_{lid}"
                df["league_name"] = league_name
                df["season"] = str(s)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["home_team_id"] = df["home_team"].apply(
                    lambda x: team_map.get((league_name, str(x).strip().lower()), 0)
                )
                df["away_team_id"] = df["away_team"].apply(
                    lambda x: team_map.get((league_name, str(x).strip().lower()), 0)
                )
                df["result"] = df.apply(
                    lambda r: "H" if r.get("home_goals", 0) > r.get("away_goals", 0)
                    else ("A" if r.get("away_goals", 0) > r.get("home_goals", 0) else "D"),
                    axis=1,
                )

                for col in ["ht_home_goals", "ht_away_goals", "home_shots", "away_shots",
                           "home_sot", "away_sot", "home_corners", "away_corners",
                           "home_fouls", "away_fouls", "home_yellow", "away_yellow",
                           "home_red", "away_red", "referee",
                           "avg_home_odds", "avg_draw_odds", "avg_away_odds",
                           "b365_home_odds", "b365_draw_odds", "b365_away_odds",
                           "avg_over25_odds", "avg_under25_odds"]:
                    if col not in df.columns:
                        df[col] = None

                frames.append(df[["league", "league_name", "season", "date",
                                  "home_team", "away_team", "home_team_id", "away_team_id",
                                  "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
                                  "result", "home_shots", "away_shots", "home_sot", "away_sot",
                                  "home_corners", "away_corners", "home_fouls", "away_fouls",
                                  "home_yellow", "away_yellow", "home_red", "away_red", "referee",
                                  "avg_home_odds", "avg_draw_odds", "avg_away_odds",
                                  "b365_home_odds", "b365_draw_odds", "b365_away_odds",
                                  "avg_over25_odds", "avg_under25_odds"]])

            except Exception as exc:
                logger.debug("TSDB fetch basarisiz: lig=%s sezon=%s: %s", lid, s, exc)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info("TheSportsDB: %d mac, %d lig", len(result), result["league"].nunique())

    cache_path = Path(settings.DATA_DIR) / "gold" / "tsdb_matches.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)

    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_matches(
    leagues: dict[str, str] | None = None,
    seasons: list[str] | None = None,
    out_path: str | None = None,
    use_huggingface: bool = True,
    use_thesportsdb: bool = False,
    use_openfootball: bool = True,
    use_fbref: bool = False,
    hf_odds_only: bool = False,
    min_year: int | None = 2015,
) -> pd.DataFrame:
    """Build the canonical match table and write gold/matches.parquet.

    Args:
        leagues: Override league dict (FD format). None = use FD_LEAGUES.
        seasons: Override seasons list. None = use FD_SEASONS.
        out_path: Output parquet path. None = gold/matches.parquet.
        use_huggingface: If True, also load Hugging Face dataset (195+ leagues).
            Default True: 673K+ mac ile dunya capinda kapsama.
        use_thesportsdb: If True, also load TheSportsDB (2000+ leagues).
            Default False: ucretsiz API cok sinirli (5 lig).
        use_openfootball: If True, also load OpenFootball (20+ leagues, ucretsiz).
            Default True: EPL, Bundesliga, LaLiga, Serie A, Ligue 1 + daha fazlasi.
        min_year: Minimum yil filtresi. None = tum yillar. Default 2015.
    """
    frames = []

    # 1) Football-Data.co.uk (detailed stats + odds) - ANA KAYNAK
    fd_df = _build_fd_matches()
    if not fd_df.empty:
        frames.append(fd_df)
        logger.info("Football-Data.co.uk: %d mac", len(fd_df))

    # 2) Hugging Face dataset (volume + global coverage)
    if use_huggingface:
        try:
            hf_df = _build_hf_matches(hf_odds_only=hf_odds_only)
            if not hf_df.empty:
                frames.append(hf_df)
                logger.info("Hugging Face: %d mac", len(hf_df))
        except Exception as e:
            logger.warning("Hugging Face load failed: %s", e)

    # 3) OpenFootball (free, open source, 20+ leagues)
    if use_openfootball:
        try:
            of_df = _build_openfootball_matches()
            if not of_df.empty:
                frames.append(of_df)
                logger.info("OpenFootball: %d mac", len(of_df))
        except Exception as e:
            logger.warning("OpenFootball load failed: %s", e)

    # 4) xgabora (GERCEK istatistikli 38 lig: ARG, BRA, USA, MEX, JAP...)
    try:
        from ingestion.xgabora.provider import build_xgabora_matches
        xg_df = build_xgabora_matches()
        if not xg_df.empty:
            frames.append(xg_df)
            logger.info("xgabora: %d mac, %d lig", len(xg_df), xg_df["league"].nunique())
    except Exception as e:
        logger.warning("xgabora load failed: %s", e)

    # 5) StatsBomb Open Data (event-level, xG, corners, free, 36+ competitions)
    try:
        sb_df = _build_statsbomb_matches()
        if not sb_df.empty:
            frames.append(sb_df)
            logger.info("StatsBomb: %d mac, %d lig", len(sb_df), sb_df["league"].nunique())
    except Exception as e:
        logger.warning("StatsBomb load failed: %s", e)

    # 6) Understat (xG for 6 leagues, 2014-2025, free)
    try:
        us_df = _build_understat_matches()
        if not us_df.empty:
            frames.append(us_df)
            logger.info("Understat: %d mac, %d lig", len(us_df), us_df["league"].nunique())
    except Exception as e:
        logger.warning("Understat load failed: %s", e)

    # 7) TheSportsDB (sinirli ucretsiz: 5 lig)
    if use_thesportsdb:
        try:
            tsdb_df = _build_thesportsdb_matches()
            if not tsdb_df.empty:
                frames.append(tsdb_df)
                logger.info("TheSportsDB: %d mac", len(tsdb_df))
        except Exception as e:
            logger.warning("TheSportsDB load failed: %s", e)

    # 5) FBref (opsiyonel: yavas scraping, xgabora tercih edilir)
    #    SADECE cache'ten okunur (scrape_new=False) - ag islemi YOK
    if use_fbref:
        try:
            from ingestion.fbref.provider import build_fbref_matches
            fb_df = build_fbref_matches(resume=True, scrape_new=False)
            if not fb_df.empty:
                frames.append(fb_df)
                logger.info("FBref: %d mac, %d lig", len(fb_df), fb_df["league"].nunique())
        except Exception as e:
            logger.warning("FBref load failed: %s", e)

    if not frames:
        raise RuntimeError("No matches parsed from any source.")

    m = pd.concat(frames, ignore_index=True)

    # min_year filtresi (hizli pipeline icin: sadece yakin donem)
    if min_year is not None:
        m = m[m["date"] >= pd.Timestamp(f"{min_year}-01-01")]
        logger.info("min_year=%d filtresi sonrasi: %d mac", min_year, len(m))

    # Duplicate temizleme: ayni tarih + ayni takimlar = ayni mac
    # once Football-Data.co.uk'i tercih et (daha fazla feature)
    m = _deduplicate_matches(m)

    # Drop rows with unparseable dates, sort chronologically
    m = m.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    m["match_id"] = np.arange(len(m), dtype=int)

    # Final istatistikler
    n_leagues = m["league"].nunique()
    n_countries = m["league_name"].nunique()
    date_range = f"{m['date'].min()} -> {m['date'].max()}"

    out_path = out_path or (Path(settings.DATA_DIR) / "gold" / "matches.parquet")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out_path, index=False)

    logger.info(
        "Canonical matches: %d rows, %d leagues, %s -> %s",
        len(m), n_leagues, date_range, out_path,
    )
    return m


def _deduplicate_matches(m: pd.DataFrame) -> pd.DataFrame:
    """Ayni maci birden fazla kaynaktan gelenlerden sadece en iyi kaynagi sec.

    Oncelik: Football-Data.co.uk > xgabora/FBref > Hugging Face > OpenFootball > TheSportsDB
    Kaynak esitliginde: istatistik dolulugu yuksek olan secilir.
    """
    m = m.copy()

    # Kaynak oncelik numarasi ata (kucuk = daha iyi)
    m["_source_priority"] = 4  # varsayilan: yuksek numara = dusuk oncelik
    m.loc[m["league"].str.startswith("XG_"), "_source_priority"] = 1  # xgabora gercek istatistikli
    m.loc[m["league"].str.startswith("FB_"), "_source_priority"] = 1  # FBref gercek istatistikli
    m.loc[m["league"].str.startswith("SB_"), "_source_priority"] = 1  # StatsBomb event-level xG
    m.loc[m["league"].str.startswith("US_"), "_source_priority"] = 2  # Understat xG
    m.loc[m["league"].str.startswith("HF_"), "_source_priority"] = 3
    m.loc[m["league"].str.startswith("OF_"), "_source_priority"] = 4
    m.loc[m["league"].str.startswith("TS_"), "_source_priority"] = 4
    # FD en oncelikli (alt cizgi icermez: E0, SP1, T1...)
    m.loc[~m["league"].str.startswith(("HF_", "OF_", "TS_", "FB_", "XG_", "SB_", "US_")), "_source_priority"] = 0

    # Istatistik dolulugu hesapla
    stat_cols = ["home_shots", "away_shots", "home_sot", "away_sot",
                 "home_corners", "away_corners", "home_yellow", "away_yellow"]
    existing_stat_cols = [c for c in stat_cols if c in m.columns]
    if existing_stat_cols:
        m["_completeness"] = m[existing_stat_cols].notna().sum(axis=1)
    else:
        m["_completeness"] = 0

    # xG dolulugu da dikkate al (StatsBomb + Understat + HF hepsi xG icerir)
    if "home_xg" in m.columns:
        m["_xg_bonus"] = m["home_xg"].notna().astype(int) * 2
    else:
        m["_xg_bonus"] = 0
    m["_score"] = m["_completeness"] + m["_xg_bonus"]

    # Tekrar edenlari bul: ayni tarih + ayni takim eslesmesi
    m["_match_key"] = (
        m["date"].astype(str).str[:10] + "_" +
        m["home_team"].str.lower().str.strip() + "_" +
        m["away_team"].str.lower().str.strip()
    )

    # Her eslesmede en iyi kaynagi sec
    m = m.sort_values(
        ["_match_key", "_source_priority", "_score"],
        ascending=[True, True, False]
    )
    m = m.drop_duplicates(subset=["_match_key"], keep="first")

    # Geici sutunlari kaldir
    m = m.drop(columns=["_source_priority", "_completeness", "_xg_bonus", "_score", "_match_key"], errors="ignore")

    logger.info("Dedup: %d mac (coklu kaynak temizlendi)", len(m))
    return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_matches()
    print(df.head())
    print("\nLeague counts:\n", df["league_name"].value_counts().head(30))
    print("\nDate range:", df["date"].min(), "->", df["date"].max())
