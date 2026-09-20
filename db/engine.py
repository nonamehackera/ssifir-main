from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from configs import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    import db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def _ensure_columns() -> None:
    """Mevcut tablolara yeni sütunlari idempotent olarak ekler.

    SQLAlchemy create_all YENI tablo kurar ama var olan tabloya sütun
    eklemez. Bu yuzden modelde yeni alan ekledigimizde (orn. prediction
    source_versions) migrasyon yerine ALTER ile tamamlariz.
    """
    alters = [
        ("model_predictions", "source_versions", "VARCHAR(200)"),
    ]
    with engine.begin() as conn:
        for table, col, typ in alters:
            exists = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name=:t AND column_name=:c"
                ),
                {"t": table, "c": col},
            ).scalar()
            if not exists:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {typ}"))


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
