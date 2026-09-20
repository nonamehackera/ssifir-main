"""OpenFootball provider - ucretsiz, acik kaynak futbol verisi.

Kaynak: https://github.com/openfootball/football.json
- EPL, Championship, Bundesliga, 2. Bundesliga, 3. Liga
- LaLiga, Segunda Division
- Serie A, Serie B
- Ligue 1, Ligue 2
- Ve daha fazlasi (10+ ulke, 20+ lig)
- API key gerektirmez
- JSON formatinda, GitHub raw content ile cekilir
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

# Lig dosyalari: (dosya yolu, lig adi, ulke)
LEAGUE_FILES = [
    # Iniltere
    ("2024-25/en.1.json", "England Premier League", "England"),
    ("2024-25/en.2.json", "England Championship", "England"),
    ("2024-25/en.3.json", "England League One", "England"),
    ("2024-25/en.4.json", "England League Two", "England"),
    ("2023-24/en.1.json", "England Premier League", "England"),
    ("2023-24/en.2.json", "England Championship", "England"),
    ("2023-24/en.3.json", "England League One", "England"),
    ("2023-24/en.4.json", "England League Two", "England"),
    # Almanya
    ("2024-25/de.1.json", "Germany Bundesliga", "Germany"),
    ("2024-25/de.2.json", "Germany 2. Bundesliga", "Germany"),
    ("2024-25/de.3.json", "Germany 3. Liga", "Germany"),
    ("2023-24/de.1.json", "Germany Bundesliga", "Germany"),
    ("2023-24/de.2.json", "Germany 2. Bundesliga", "Germany"),
    ("2023-24/de.3.json", "Germany 3. Liga", "Germany"),
    # Ispanya
    ("2024-25/es.1.json", "Spain LaLiga", "Spain"),
    ("2024-25/es.2.json", "Spain Segunda Division", "Spain"),
    ("2023-24/es.1.json", "Spain LaLiga", "Spain"),
    ("2023-24/es.2.json", "Spain Segunda Division", "Spain"),
    # Italya
    ("2024-25/it.1.json", "Italy Serie A", "Italy"),
    ("2024-25/it.2.json", "Italy Serie B", "Italy"),
    ("2023-24/it.1.json", "Italy Serie A", "Italy"),
    ("2023-24/it.2.json", "Italy Serie B", "Italy"),
    # Fransa
    ("2024-25/fr.1.json", "France Ligue 1", "France"),
    ("2024-25/fr.2.json", "France Ligue 2", "France"),
    ("2023-24/fr.1.json", "France Ligue 1", "France"),
    ("2023-24/fr.2.json", "France Ligue 2", "France"),
    # Turkiye
    ("2024-25/tr.1.json", "Turkey Super Lig", "Turkey"),
    ("2023-24/tr.1.json", "Turkey Super Lig", "Turkey"),
    # Hollanda
    ("2024-25/nl.1.json", "Netherlands Eredivisie", "Netherlands"),
    ("2023-24/nl.1.json", "Netherlands Eredivisie", "Netherlands"),
    # Portekiz
    ("2024-25/pt.1.json", "Portugal Primeira Liga", "Portugal"),
    ("2023-24/pt.1.json", "Portugal Primeira Liga", "Portugal"),
    # Belcika
    ("2024-25/be.1.json", "Belgium Pro League", "Belgium"),
    ("2023-24/be.1.json", "Belgium Pro League", "Belgium"),
    # Yunanistan
    ("2024-25/gr.1.json", "Greece Super League", "Greece"),
    ("2023-24/gr.1.json", "Greece Super League", "Greece"),
    # Iskocya
    ("2024-25/sco.1.json", "Scotland Premiership", "Scotland"),
    ("2023-24/sco.1.json", "Scotland Premiership", "Scotland"),
    # Avusturya
    ("2024-25/at.1.json", "Austria Bundesliga", "Austria"),
    ("2023-24/at.1.json", "Austria Bundesliga", "Austria"),
    # Isvicre
    ("2024-25/ch.1.json", "Switzerland Super League", "Switzerland"),
    ("2023-24/ch.1.json", "Switzerland Super League", "Switzerland"),
]


def _fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        logger.debug("Fetch basarisiz: %s - %s", url, e)
        return None


def _parse_matches(data: dict) -> list[dict]:
    """OpenFootball JSON formatini match listesine donustur."""
    rows = []
    for match in data.get("matches", []):
        home = match.get("team1", "")
        away = match.get("team2", "")
        score = match.get("score", {})

        if not home or not away:
            continue

        # Skor parse - ft (full time)
        ft = score.get("ft", [])
        if len(ft) != 2:
            continue

        hg, ag = ft[0], ft[1]
        if hg is None or ag is None:
            continue

        try:
            hg, ag = int(hg), int(ag)
        except (ValueError, TypeError):
            continue

        # Tarih parse
        date_str = match.get("date", "")
        if not date_str:
            continue

        try:
            date = pd.to_datetime(date_str)
        except Exception:
            continue

        # HT skor
        ht = score.get("ht", [])
        ht_hg = ht[0] if len(ht) == 2 and ht[0] is not None else None
        ht_ag = ht[1] if len(ht) == 2 and ht[1] is not None else None

        result = "H" if hg > ag else ("A" if ag > hg else "D")

        rows.append({
            "date": date,
            "home_team": home.strip(),
            "away_team": away.strip(),
            "home_goals": hg,
            "away_goals": ag,
            "ht_home_goals": ht_hg,
            "ht_away_goals": ht_ag,
            "result": result,
            "round": match.get("round", ""),
        })

    return rows


def build_openfootball_matches(seasons: list[str] | None = None) -> pd.DataFrame:
    """OpenFootball'dan ucretsiz mac verisi ceker.

    seasons: ornek ["2024-25", "2023-24"]. None ise tum mevcut sezonlar.
    """
    if seasons is None:
        seasons = ["2024-25", "2023-24"]

    frames = []
    team_map = {}
    next_team_id = 3000000  # OF team IDs start above TSDB

    for file_path, league_name, country in LEAGUE_FILES:
        # Sezon filtresi
        season_part = file_path.split("/")[0]
        if season_part not in seasons:
            continue

        url = f"{BASE_URL}/{file_path}"
        data = _fetch_json(url)
        if not data:
            continue

        matches = _parse_matches(data)
        if not matches:
            continue

        for m in matches:
            # Team ID'leri olustur
            for team_name in [m["home_team"], m["away_team"]]:
                key = (league_name, team_name.lower())
                if key not in team_map:
                    team_map[key] = next_team_id
                    next_team_id += 1

        league_code = file_path.split("/")[1].replace(".json", "").replace(".", "_")

        for m in matches:
            m["league"] = f"OF_{league_code}"
            m["league_name"] = league_name
            m["season"] = season_part
            m["home_team_id"] = team_map.get((league_name, m["home_team"].lower()), 0)
            m["away_team_id"] = team_map.get((league_name, m["away_team"].lower()), 0)

            # Eksik sutunlari None ile doldur
            for col in ["ht_home_goals", "ht_away_goals", "home_shots", "away_shots",
                       "home_sot", "away_sot", "home_corners", "away_corners",
                       "home_fouls", "away_fouls", "home_yellow", "away_yellow",
                       "home_red", "away_red", "referee",
                       "avg_home_odds", "avg_draw_odds", "avg_away_odds",
                       "b365_home_odds", "b365_draw_odds", "b365_away_odds",
                       "avg_over25_odds", "avg_under25_odds"]:
                if col not in m:
                    m[col] = None

        frames.append(pd.DataFrame(matches))
        logger.info("OpenFootball: %s - %d mac", league_name, len(matches))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info("OpenFootball toplam: %d mac, %d lig", len(result), result["league"].nunique())

    # Cache'e kaydet
    cache_path = Path("data/gold/openfootball_matches.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)

    return result
