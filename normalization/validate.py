import logging
from datetime import datetime

from sqlalchemy.orm import Session

from db.models import DataQualityLog

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


def log_issue(
    session: Session,
    check_name: str,
    severity: str,
    message: str,
    fixture_id: int | None = None,
    source: str | None = None,
    flush: bool = True,
) -> None:
    session.add(
        DataQualityLog(
            fixture_id=fixture_id,
            check_name=check_name,
            severity=severity,
            message=message[:2000],
            source=source,
        )
    )
    if flush:
        session.flush()


def validate_fixture(session: Session, data: dict, source: str | None = None) -> bool:
    """ROADMAP bölüm 81: fixture satırı temel doğruluk kontrolleri."""
    fixture_id = data.get("fixture_id")
    home = data.get("home_team_id")
    away = data.get("away_team_id")
    kickoff = data.get("kickoff_at")
    ok = True

    if home is not None and away is not None and home == away:
        log_issue(
            session, "invalid_fixture", SEVERITY_ERROR,
            f"Ev ve deplasman takımı aynı: {home}", fixture_id, source,
        )
        ok = False

    if kickoff is None:
        log_issue(
            session, "missing_kickoff", SEVERITY_ERROR,
            "kickoff_at eksik", fixture_id, source,
        )
        ok = False
    elif not isinstance(kickoff, datetime):
        log_issue(
            session, "bad_timestamp", SEVERITY_ERROR,
            f"kickoff_at geçersiz tip: {type(kickoff)}", fixture_id, source,
        )
        ok = False

    for field in ("home_goals", "away_goals", "ht_home_goals", "ht_away_goals"):
        value = data.get(field)
        if value is not None and (not isinstance(value, int) or value < 0):
            log_issue(
                session, "negative_goals", SEVERITY_ERROR,
                f"{field} negatif veya geçersiz: {value}", fixture_id, source,
            )
            ok = False

    hg, ag = data.get("home_goals"), data.get("away_goals")
    hthg, htag = data.get("ht_home_goals"), data.get("ht_away_goals")
    if (
        hg is not None and hthg is not None and hthg > hg
    ) or (
        ag is not None and htag is not None and htag > ag
    ):
        log_issue(
            session, "impossible_score", SEVERITY_ERROR,
            f"İY skoru MSA'dan büyük: {hthg}-{htag} vs {hg}-{ag}", fixture_id, source,
        )
        ok = False

    return ok


def validate_team(session: Session, data: dict, source: str | None = None) -> bool:
    team_id = data.get("team_id")
    ok = True
    if not data.get("team_name"):
        log_issue(
            session, "missing_team_name", SEVERITY_WARNING,
            f"Takım adı eksik (id={team_id})", None, source,
        )
        ok = False
    return ok