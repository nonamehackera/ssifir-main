"""Normalizasyon + validasyon testleri (gerçek API gerektirmez)."""

import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from db.engine import SessionLocal  # noqa: E402
from db.models import DataQualityLog, Fixture  # noqa: E402
from normalization.silver import upsert_fixture  # noqa: E402

SAMPLE_FIXTURE = {
    "fixture": {
        "id": 11804817,
        "date": "2024-08-16T19:00:00+00:00",
        "status": {"short": "FT", "elapsed": 90},
        "venue": {"id": 556, "name": "Old Trafford", "city": "Manchester"},
        "referee": "Michael Oliver",
    },
    "league": {"id": 39, "name": "Premier League", "type": "League", "country": "England", "season": 2024, "round": "Matchweek 1"},
    "teams": {
        "home": {"id": 33, "name": "Manchester United", "code": "MUN", "country": "England", "founded": 1878},
        "away": {"id": 40, "name": "Liverpool", "code": "LIV", "country": "England", "founded": 1892},
    },
    "goals": {"home": 0, "away": 3},
    "score": {"halftime": {"home": 0, "away": 2}, "fulltime": {"home": 0, "away": 3}},
}

INVALID_FIXTURE = {
    **SAMPLE_FIXTURE,
    "fixture": {**SAMPLE_FIXTURE["fixture"], "id": 11804818},
    "goals": {"home": -1, "away": 3},
    "score": {"halftime": {"home": 5, "away": 0}},
}


def test_valid_fixture():
    with SessionLocal() as session:
        session.query(Fixture).filter(Fixture.id == 11804817).delete()
        session.commit()
        ok = upsert_fixture(session, SAMPLE_FIXTURE)
        session.commit()
        assert ok, "geçerli fixture kabul edilmeli"
        f = session.get(Fixture, 11804817)
        assert f.home_team_id == 33 and f.away_team_id == 40
        assert f.home_goals == 0 and f.away_goals == 3
        assert f.ht_home_goals == 0 and f.ht_away_goals == 2
        assert f.status == "FT"
        assert f.kickoff_at == datetime(2024, 8, 16, 19, 0, tzinfo=timezone.utc)
        assert f.league_id == 39
        assert f.round == "Matchweek 1"
    print("test_valid_fixture OK")


def test_invalid_fixture_rejected():
    with SessionLocal() as session:
        session.query(Fixture).filter(Fixture.id == 11804818).delete()
        session.commit()
        ok = upsert_fixture(session, INVALID_FIXTURE)
        session.commit()
        assert not ok, "negatif gol ve İY>MSA senaryosu reddedilmeli"
        issues = (
            session.query(DataQualityLog)
            .filter(DataQualityLog.fixture_id == 11804818)
            .all()
        )
        names = {i.check_name for i in issues}
        assert "negative_goals" in names
        assert "impossible_score" in names
        assert session.get(Fixture, 11804818) is None, "hatalı fixture kaydedilmemeli"
    print("test_invalid_fixture_rejected OK")


def test_upsert_idempotent():
    with SessionLocal() as session:
        upsert_fixture(session, SAMPLE_FIXTURE)
        session.commit()
        upsert_fixture(session, SAMPLE_FIXTURE)
        session.commit()
        count = session.query(Fixture).filter(Fixture.id == 11804817).count()
        assert count == 1, "upsert tek satır bırakmalı"
    print("test_upsert_idempotent OK")


if __name__ == "__main__":
    test_valid_fixture()
    test_invalid_fixture_rejected()
    test_upsert_idempotent()
    print("Tüm testler geçti.")