from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from db.engine import Base


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    type: Mapped[str | None] = mapped_column(String(30))
    country: Mapped[str | None] = mapped_column(String(80))
    logo: Mapped[str | None] = mapped_column(String(300))
    current_season: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("league_id", "year", name="uq_season_league_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False, index=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(160))
    city: Mapped[str | None] = mapped_column(String(80))
    capacity: Mapped[int | None] = mapped_column(Integer)
    surface: Mapped[str | None] = mapped_column(String(40))


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    code: Mapped[str | None] = mapped_column(String(10))
    country: Mapped[str | None] = mapped_column(String(80))
    founded: Mapped[int | None] = mapped_column(Integer)
    logo: Mapped[str | None] = mapped_column(String(300))
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TeamMapping(Base):
    __tablename__ = "team_mappings"
    __table_args__ = (
        UniqueConstraint("provider", "provider_id", name="uq_team_mapping_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(60), nullable=False)
    canonical_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Referee(Base):
    __tablename__ = "referees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120))
    nationality: Mapped[str | None] = mapped_column(String(80))


class Coach(Base):
    __tablename__ = "coaches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120))
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    nationality: Mapped[str | None] = mapped_column(String(80))


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False, index=True)
    season_id: Mapped[int | None] = mapped_column(ForeignKey("seasons.id"))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    status: Mapped[str | None] = mapped_column(String(30))

    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)

    ht_home_goals: Mapped[int | None] = mapped_column(Integer)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer)

    round: Mapped[str | None] = mapped_column(String(60))
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    referee_id: Mapped[int | None] = mapped_column(ForeignKey("referees.id"))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), nullable=False, index=True)
    bookmaker_id: Mapped[int | None] = mapped_column(Integer)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    market: Mapped[str | None] = mapped_column(String(50))
    selection: Mapped[str | None] = mapped_column(String(100))
    line: Mapped[float | None] = mapped_column(Numeric)
    price: Mapped[float | None] = mapped_column(Numeric)
    source: Mapped[str | None] = mapped_column(String(50))


class TeamRating(Base):
    __tablename__ = "team_ratings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    rating_type: Mapped[str] = mapped_column(String(30), nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class DataQualityLog(Base):
    __tablename__ = "data_quality_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    check_name: Mapped[str] = mapped_column(String(80), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    dataset_version: Mapped[str | None] = mapped_column(String(40))
    feature_version: Mapped[str | None] = mapped_column(String(40))
    source_versions: Mapped[str | None] = mapped_column(String(200))  # ROADMAP 88
    home_win: Mapped[float | None] = mapped_column(Numeric)
    draw: Mapped[float | None] = mapped_column(Numeric)
    away_win: Mapped[float | None] = mapped_column(Numeric)
    home_lambda: Mapped[float | None] = mapped_column(Numeric)
    away_lambda: Mapped[float | None] = mapped_column(Numeric)
    btts_yes: Mapped[float | None] = mapped_column(Numeric)
    over25: Mapped[float | None] = mapped_column(Numeric)
    payload: Mapped[str | None] = mapped_column(Text)  # ROADMAP 52 full JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("model_predictions.prediction_id"), nullable=False, unique=True
    )
    actual_home_goals: Mapped[int | None] = mapped_column(Integer)
    actual_away_goals: Mapped[int | None] = mapped_column(Integer)
    actual_result: Mapped[str | None] = mapped_column(String(10))
    log_loss: Mapped[float | None] = mapped_column(Numeric)
    brier_score: Mapped[float | None] = mapped_column(Numeric)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(60), nullable=False)
    version: Mapped[str] = mapped_column(String(30), nullable=False)
    training_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    features_hash: Mapped[str | None] = mapped_column(String(64))
    dataset_version: Mapped[str | None] = mapped_column(String(40))
    algorithm: Mapped[str | None] = mapped_column(String(40))
    hyperparameters: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[str | None] = mapped_column(Text)
    calibration_method: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class LiveSnapshot(Base):
    """Canli mac state snapshot'lari (ROADMAP bolum 22, 23).

    Her dakika / her event'te bir snapshot. Kritik kural (ROADMAP 23):
    t anindaki tahmin SADECE 0-t arasi verilerle yapilir; gelecek bilgi
    (final score, sonraki dakika) KULLANILAMAZ. Bu tablo o noktadaki
    gozlemlenen state'i saklar.
    """

    __tablename__ = "live_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    minute: Mapped[int | None] = mapped_column(Integer)

    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)

    xg_home: Mapped[float | None] = mapped_column(Float)
    xg_away: Mapped[float | None] = mapped_column(Float)
    shots_home: Mapped[int | None] = mapped_column(Integer)
    shots_away: Mapped[int | None] = mapped_column(Integer)
    sot_home: Mapped[int | None] = mapped_column(Integer)
    sot_away: Mapped[int | None] = mapped_column(Integer)
    corners_home: Mapped[int | None] = mapped_column(Integer)
    corners_away: Mapped[int | None] = mapped_column(Integer)
    red_cards_home: Mapped[int | None] = mapped_column(Integer)
    red_cards_away: Mapped[int | None] = mapped_column(Integer)
    yellow_cards_home: Mapped[int | None] = mapped_column(Integer)
    yellow_cards_away: Mapped[int | None] = mapped_column(Integer)
    possession_home: Mapped[float | None] = mapped_column(Float)

    live_odds_home: Mapped[float | None] = mapped_column(Numeric)
    live_odds_draw: Mapped[float | None] = mapped_column(Numeric)
    live_odds_away: Mapped[float | None] = mapped_column(Numeric)

    source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class FootballDataMatch(Base):
    """Football-Data.co.uk CSV arşivinden ham satırlar (staging).

    Takım adları API-Football canonical ID'lerine Phase 2'deki
    entity resolution ile bağlanır; bu tablo ham kayıt olarak saklanır.
    """

    __tablename__ = "football_data_matches"
    __table_args__ = (
        UniqueConstraint("source_file", "row_index", name="uq_fd_source_row"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(String(40), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    division: Mapped[str | None] = mapped_column(String(10))
    match_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    home_team_name: Mapped[str | None] = mapped_column(String(80))
    away_team_name: Mapped[str | None] = mapped_column(String(80))
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    ht_home_goals: Mapped[int | None] = mapped_column(Integer)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer)
    referee: Mapped[str | None] = mapped_column(String(80))
    extra: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class StandingsSnapshot(Base):
    __tablename__ = "standings_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False, index=True)
    season_id: Mapped[int | None] = mapped_column(ForeignKey("seasons.id"))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    position: Mapped[int | None] = mapped_column(Integer)
    points: Mapped[int | None] = mapped_column(Integer)
    played: Mapped[int | None] = mapped_column(Integer)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)