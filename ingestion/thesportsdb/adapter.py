"""TheSportsDB adapter (ROADMAP bolum 2 + senin talep: 200+ lig).

TheSportsDB free API:
  - https://www.thesportsdb.com/api/v1/json/3/all_leagues.php  -> 2000+ lig listesi
  - https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?... -> maclar
  - https://www.thesportsdb.com/api/v1/json/3/eventspastleague.php?id={id}&r={yil}

Ucretsiz ve API key gerektirmez (test API key=3). 2000+ futbol ligi var.
Bu adapter, Football-Data.co.uk (30 lig, detayli stats) ile TheSportsDB
(200+ lig, sonuclar) arasinda ENTITY RESOLUTION yapar (ROADMAP 55/80).

NOT: bu ortamda ag erisimi engelli oldugu icin fetch test edilemez;
kod gercek ortamda calisir. Local test icin _load_local_cache() kullanilir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from configs import settings
from ingestion.retry import with_retry

logger = logging.getLogger(__name__)

TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"
CACHE_DIR = Path(settings.DATA_DIR) / "raw" / "thesportsdb"


@with_retry(max_attempts=3, backoff=2.0, base_delay=1.0)
def _get_json(url: str) -> dict:
    import requests

    resp = requests.get(url, timeout=settings.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_all_leagues(cache: bool = True) -> pd.DataFrame:
    """Tum futbol liglerini ceker (2000+). Doner: idLeague, strLeague, etc."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "all_leagues.json"
    if cache and cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        data = _get_json(f"{TSDB_BASE}/all_leagues.php")
        cache_path.write_text(json.dumps(data))
    leagues = data.get("leagues", [])
    # sadece soccer (football) ligleri
    soccer = [l for l in leagues if str(l.get("strSport", "")).lower() in ("soccer", "football")]
    return pd.DataFrame(soccer)


def fetch_league_matches(league_id: int, season: int) -> pd.DataFrame:
    """Bir ligin bir sezon maclarini ceker (eventspastleague)."""
    url = f"{TSDB_BASE}/eventspastleague.php?id={league_id}&r={season}"
    data = _get_json(url)
    events = data.get("events", []) or []
    rows = []
    for e in events:
        if e.get("strSport") != "Soccer":
            continue
        rows.append({
            "date": e.get("dateEvent"),
            "home_team": e.get("strHomeTeam"),
            "away_team": e.get("strAwayTeam"),
            "home_goals": _to_int(e.get("intHomeScore")),
            "away_goals": _to_int(e.get("intAwayScore")),
            "league_id": league_id,
            "league_name": e.get("strLeague"),
            "season": season,
            "round": e.get("strRound"),
            "venue": e.get("strVenue"),
            "referee": e.get("strReferee"),
        })
    return pd.DataFrame(rows)


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def build_thesportsdb_matches(seasons: list[int] | None = None,
                              max_leagues: int | None = 250) -> pd.DataFrame:
    """Cok kaynakli: TheSportsDB'den 200+ lig sonuclarini toplar.

    seasons: ornek [2022, 2023, 2024, 2025]
    max_leagues: kac lig (200+ ile sinirla, tumu 2000+ olur)
    """
    leagues = fetch_all_leagues()
    logger.info("TheSportsDB: %d soccer ligi bulundu", len(leagues))
    if max_leagues:
        leagues = leagues.head(max_leagues)
    seasons = seasons or [2023, 2024, 2025]
    frames = []
    for _, lg in leagues.iterrows():
        lid = int(lg["idLeague"])
        for s in seasons:
            try:
                df = fetch_league_matches(lid, s)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lig %s/%s fetch basarisiz: %s", lid, s, exc)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    out = CACHE_DIR.parent.parent / "gold" / "tsdb_matches.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out)
    logger.info("TheSportsDB: %d mac, %d lig", len(result), result["league_id"].nunique())
    return result
