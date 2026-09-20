"""Hugging Face Soccer Dataset provider - 195+ leagues worldwide."""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from typing import Optional

from ingestion.base import FootballDataProvider


# League ID to readable name mapping
LEAGUE_NAMES = {
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
    31: "China CSL",
    32: "Romania SuperLiga",
    33: "Finland Veikkausliiga",
    34: "Ireland Premier Division",
    35: "Greece Super League 1",
    36: "Czech Republic 1. Liga",
    37: "Saudi Arabia Saudi Pro League",
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
    100000402: "Sudan Sudani Premier League",
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
    100000668: "Czech Republic 1. Liga U19",
    100000711: "Chile Segunda Division",
    100000722: "Mexico Liga Premier Serie A",
    100000758: "Gibraltar Premier Division",
    100000833: "Australia Queensland Premier League",
    100000851: "Brazil Carioca A2",
    100000907: "Cuba Primera Division",
    100000955: "New Zealand National League",
    100001144: "Unknown League",
    100001193: "Unknown League",
    100001194: "Unknown League",
}


class HuggingFaceProvider(FootballDataProvider):
    """Provider for Hugging Face soccer dataset (271 leagues, 673K+ matches)."""

    def __init__(self):
        self._fixtures_df = None
        self._leagues_df = None

    def _load_data(self):
        if self._fixtures_df is not None:
            return
        print("[HF] Loading fixtures from Hugging Face...")
        ds = load_dataset(
            "eatpizzanot/soccer-dataset",
            data_files="fixtures.parquet",
            split="train",
        )
        self._fixtures_df = ds.to_pandas()
        print(f"[HF] Loaded {len(self._fixtures_df)} fixtures")

    def fetch(
        self,
        since: Optional[str] = None,
        league: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        self._load_data()
        df = self._fixtures_df.copy()

        # Filter to played matches only
        df = df[df["is_played"] == True].copy()

        # Map league_id to name
        df["league_name"] = df["league_id"].map(LEAGUE_NAMES)

        # Drop unknown leagues
        df = df.dropna(subset=["league_name"])

        # Filter by date if specified
        if since:
            df = df[df["date_utc"] >= since]

        # Normalize to common schema
        result = pd.DataFrame({
            "match_id": df["api_football_id"].astype(str),
            "date": pd.to_datetime(df["date_utc"]).dt.date,
            "home_team_id": df["home_team_id"].astype(str),
            "away_team_id": df["away_team_id"].astype(str),
            "home_goals": df["goals_home"].astype(int),
            "away_goals": df["goals_away"].astype(int),
            "league": df["league_name"],
            "season": pd.to_datetime(df["date_utc"]).dt.year.astype(str),
        })

        # Derive result
        result["result"] = result.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
            else ("A" if r["away_goals"] > r["home_goals"] else "D"),
            axis=1,
        )

        result = result.sort_values("date").reset_index(drop=True)
        return result.tail(limit) if limit else result

    def list_leagues(self) -> list[str]:
        self._load_data()
        leagues = self._fixtures_df["league_id"].map(LEAGUE_NAMES).dropna().unique()
        return sorted(leagues)
