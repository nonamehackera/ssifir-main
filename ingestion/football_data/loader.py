"""Football-Data.co.uk tarihsel CSV yükleyici (ROADMAP bölüm 2.3).

Kaynak: https://www.football-data.co.uk/data.php
- Dosyalar: https://www.football-data.co.uk/mmz4281/{sezon}/{lig}.csv
  sezon örn. "2425", lig örn. "E0" (Premier League)
- Yalnızca indirme ve ham yükleme yapar; takım mapping'i Phase 2'dedir.
"""

import io
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import requests
from sqlalchemy.orm import Session

from configs import settings
from db.models import FootballDataMatch
from ingestion.storage import RawStorage

logger = logging.getLogger(__name__)

FD_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Division -> açıklama (kapsama zamanla değişebilir)
KNOWN_LEAGUES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "SP1": "Spain LaLiga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "T1": "Turkey Super Lig",
}

ODDS_COLUMNS = [
    "B365H", "B365D", "B365A",
    "PSCH", "PSCD", "PSCA",
    "AvgH", "AvgD", "AvgA",
    "MaxH", "MaxD", "MaxA",
]

CORE_COLUMNS = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
]

OPTIONAL_COLUMNS = ["Time", "FTR", "HTR", "Referee"] + ODDS_COLUMNS


def _season_codes(current: int) -> list[str]:
    years = [current - 1 - i for i in range(30)]
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in years if y >= 1993]


def download_season(league_code: str, season_code: str, timeout: int | None = None) -> str | None:
    """CSV'yi indirir, bronze'a ham olarak saklar. Döner: dosya yolu ya da None."""
    if league_code not in KNOWN_LEAGUES:
        raise ValueError(f"Bilinmeyen lig kodu: {league_code}")
    url = f"{FD_BASE_URL}/{season_code}/{league_code}.csv"
    resp = requests.get(url, timeout=timeout or settings.REQUEST_TIMEOUT)
    if resp.status_code == 404:
        logger.info("CSV yok: %s %s", league_code, season_code)
        return None
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        return None
    path = RawStorage().save(
        "football_data", "csv", {"league": league_code, "season": season_code}, text
    )
    logger.info("İndirildi: %s -> %s", url, path)
    return str(path)


def download_all(league_codes: list[str] | None = None) -> dict[str, list[str]]:
    """Bilinen tüm sezonları indirir (mevcut dosyaları atlar)."""
    codes = league_codes or list(KNOWN_LEAGUES)
    current_year = datetime.now(timezone.utc).year
    seasons = _season_codes(current_year)
    result: dict[str, list[str]] = {}
    for league in codes:
        got = []
        for season in seasons:
            path = download_season(league, season)
            if path:
                got.append(path)
        result[league] = got
    return result


def _load_csv_text(text: str) -> pd.DataFrame:
    if text.startswith("\ufeff"):
        text = text[1:]
    elif text.startswith("ï»¿"):
        text = text[3:]
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen kolonlar eksik: {missing}")
    return df


def _get(row, col: str):
    if col not in row._fields:
        return None
    value = getattr(row, col)
    return value if pd.notna(value) else None


def load_season(
    session: Session,
    league_code: str,
    season_code: str,
    bronze_path: str | None = None,
) -> int:
    """Bronze CSV'yi football_data_matches tablosuna yükler. Döner: satır sayısı."""
    if bronze_path is None:
        storage = RawStorage()
        found = storage.find("football_data", "csv", {"league": league_code, "season": season_code})
        if found is None:
            raise FileNotFoundError(
                f"Bronze CSV bulunamadı: {league_code}_{season_code}. Önce download_season() çağırın."
            )
        bronze_path = str(found)
    with open(bronze_path, encoding="utf-8") as fh:
        text = json.load(fh)["response"]
    df = _load_csv_text(text)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])

    existing = {
        (r.source_file, r.row_index)
        for r in session.query(FootballDataMatch.source_file, FootballDataMatch.row_index).all()
    }

    source_file = f"{league_code}_{season_code}"
    count = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if (source_file, i) in existing:
            continue
        session.add(
            FootballDataMatch(
                source_file=source_file,
                row_index=i,
                division=_get(row, "Div"),
                match_date=datetime.strptime(str(_get(row, "Date")), "%d/%m/%Y").replace(
                    tzinfo=timezone.utc
                ),
                home_team_name=_get(row, "HomeTeam"),
                away_team_name=_get(row, "AwayTeam"),
                home_goals=_get(row, "FTHG"),
                away_goals=_get(row, "FTAG"),
                ht_home_goals=_get(row, "HTHG"),
                ht_away_goals=_get(row, "HTAG"),
                referee=_get(row, "Referee"),
                extra=str(
                    {c: _get(row, c) for c in OPTIONAL_COLUMNS if c != "Referee"}
                ),
            )
        )
        count += 1
    session.commit()
    logger.info("%s: %d yeni satır yüklendi", source_file, count)
    return count