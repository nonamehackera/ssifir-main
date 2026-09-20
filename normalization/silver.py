import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import League, Referee, Season, Team, Venue
from normalization.validate import validate_fixture, validate_team

logger = logging.getLogger(__name__)

PARSE_STATUS = {
    "NS": "NS", "TBD": "NS", "PST": "NS",
    "1H": "LIVE", "HT": "LIVE", "2H": "LIVE", "ET": "LIVE", "BT": "LIVE", "P": "LIVE",
    "FT": "FT", "AET": "FT", "PEN": "FT",
    "SUSP": "SUSP", "INT": "SUSP", "ABD": "ABD", "CANC": "ABD", "POST": "POST",
}


def upsert_league(session: Session, raw: dict[str, Any]) -> League:
    league = raw.get("league", raw)
    league_id = league["id"]
    obj = session.get(League, league_id)
    if obj is None:
        obj = League(id=league_id)
        session.add(obj)
    obj.name = league.get("name") or obj.name
    obj.type = league.get("type") or obj.type
    obj.country = league.get("country") or obj.country
    obj.logo = league.get("logo") or obj.logo
    session.flush()
    return obj


def upsert_season(session: Session, league_id: int, season_raw: dict[str, Any]) -> Season:
    year = int(season_raw["year"])
    obj = session.scalar(
        select(Season).where(Season.league_id == league_id, Season.year == year)
    )
    if obj is None:
        obj = Season(league_id=league_id, year=year)
        session.add(obj)
    if season_raw.get("start"):
        obj.start_date = datetime.fromisoformat(season_raw["start"])
    if season_raw.get("end"):
        obj.end_date = datetime.fromisoformat(season_raw["end"])
    session.flush()
    return obj


def _upsert_venue(session: Session, raw: dict[str, Any]) -> int | None:
    venue_id = raw.get("id")
    if venue_id is None:
        return None
    obj = session.get(Venue, venue_id)
    if obj is None:
        obj = Venue(id=venue_id)
        session.add(obj)
    obj.name = raw.get("name") or obj.name
    obj.city = raw.get("city") or obj.city
    obj.capacity = raw.get("capacity") or obj.capacity
    obj.surface = raw.get("surface") or obj.surface
    session.flush()
    return venue_id


def upsert_team(session: Session, raw: dict[str, Any]) -> Team:
    team_id = int(raw["id"])
    team_raw = {"team_id": team_id, "team_name": raw.get("name")}
    validate_team(session, team_raw, source="api_football")

    venue_id = _upsert_venue(session, raw.get("venue") or {})

    obj = session.get(Team, team_id)
    if obj is None:
        obj = Team(id=team_id)
        session.add(obj)
    obj.name = raw.get("name") or obj.name
    obj.code = raw.get("code") or obj.code
    obj.country = raw.get("country") or obj.country
    obj.founded = raw.get("founded") or obj.founded
    obj.logo = raw.get("logo") or obj.logo
    if venue_id is not None:
        obj.venue_id = venue_id
    session.flush()
    return obj


def _upsert_referee(session: Session, name: str | None) -> int | None:
    if not name:
        return None
    obj = session.scalar(select(Referee).where(Referee.name == name))
    if obj is None:
        obj = Referee(name=name)
        session.add(obj)
        session.flush()
    return obj.id


def normalize_fixture(session: Session, raw: dict[str, Any]) -> dict[str, Any]:
    """API-Football fixture JSON -> silver fixture satırı (dict)."""
    fixture = raw["fixture"]
    league = raw["league"]
    teams = raw["teams"]
    goals = raw.get("goals") or {}
    halftime = (raw.get("score") or {}).get("halftime") or {}

    league_obj = upsert_league(session, raw)
    season_obj = upsert_season(
        session, league_obj.id, {"year": league.get("season")}
    )

    home_id = int(teams["home"]["id"])
    away_id = int(teams["away"]["id"])
    home_team = upsert_team(session, teams["home"])
    away_team = upsert_team(session, teams["away"])

    venue_raw = fixture.get("venue") or {}
    venue_id = _upsert_venue(session, venue_raw)
    referee_id = _upsert_referee(session, fixture.get("referee"))

    kickoff = datetime.fromisoformat(str(fixture["date"]).replace("Z", "+00:00"))
    status_raw = (fixture.get("status") or {}).get("short", "NS")
    status = PARSE_STATUS.get(status_raw, "UNKNOWN")

    return {
        "fixture_id": int(fixture["id"]),
        "league_id": league_obj.id,
        "season_id": season_obj.id,
        "home_team_id": home_team.id,
        "away_team_id": away_team.id,
        "kickoff_at": kickoff,
        "status": status,
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "ht_home_goals": halftime.get("home"),
        "ht_away_goals": halftime.get("away"),
        "round": league.get("round"),
        "venue_id": venue_id,
        "referee_id": referee_id,
    }


def upsert_fixture(session: Session, raw: dict[str, Any]) -> bool:
    """API-Football fixture JSON'u doğrular ve fixtures tablosuna upsert eder.

    Döner: veri kalite kontrollerinden geçti mi?
    """
    data = normalize_fixture(session, raw)
    ok = validate_fixture(session, data, source="api_football")
    if not ok:
        return False

    from db.models import Fixture

    obj = session.get(Fixture, data["fixture_id"])
    if obj is None:
        obj = Fixture(id=data["fixture_id"])
        session.add(obj)
    for field, value in data.items():
        setattr(obj, field, value)
    session.flush()
    return True