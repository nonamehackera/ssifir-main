"""Odds snapshot arsivi + CLV hesabi (ROADMAP bolum 51, 62).

ROADMAP 62: canli odds saglayici tarafindan tarihsel saklanmazsa, sistem
kendi snapshot arsivini olusturmalidir. Bu modul:
  - bir mac icin odds snapshot'i PostgreSQL'e yazar (bookmaker, market,
    selection, line, price, captured_at)
  - CLV (Closing Line Value) hesaplar: tahmin anindaki odds ile kapanis
    odds'i karsilastirir (ROADMAP 51)

Free veride (Football-Data.co.uk) kapanis benzeri odds = AvgH/D/A (ortalama
kapanis). Canli sistemde ise her dakika snapshot alinir ve kapanis = son
snapshot kabul edilir.

CLV prensibi: model SÜREKLI kapanisa gore iyi fiyat yakaliyor mu?
  CLV = (odds_at_prediction / closing_odds) - 1   (selection'i kazanirsa)
  pozitif CLV => model kapanisdan once daha yuksek oran bulmus (iyi sinyal)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select

from db.engine import SessionLocal
from db.models import OddsSnapshot

logger = logging.getLogger(__name__)


def save_snapshot(
    fixture_id: int,
    market: str,
    selection: str,
    price: float,
    captured_at: datetime | None = None,
    bookmaker_id: int | None = None,
    line: float | None = None,
    source: str = "football_data",
    session=None,
) -> OddsSnapshot:
    """Tek bir odds snapshot'i yazar (ROADMAP 21 tablo yapisi)."""
    own = session is None
    session = session or SessionLocal()
    try:
        snap = OddsSnapshot(
            fixture_id=fixture_id,
            bookmaker_id=bookmaker_id,
            captured_at=captured_at or datetime.now(timezone.utc),
            market=market,
            selection=selection,
            line=line,
            price=price,
            source=source,
        )
        session.add(snap)
        session.commit()
        session.refresh(snap)
        return snap
    finally:
        if own:
            session.close()


def get_snapshots(fixture_id: int, market: str | None = None, session=None):
    own = session is None
    session = session or SessionLocal()
    try:
        q = select(OddsSnapshot).where(OddsSnapshot.fixture_id == fixture_id)
        if market:
            q = q.where(OddsSnapshot.market == market)
        return session.execute(q.order_by(OddsSnapshot.captured_at)).scalars().all()
    finally:
        if own:
            session.close()


def compute_clv(
    fixture_id: int,
    prediction_odds: dict,
    closing_odds: dict,
    won: dict,
) -> dict:
    """CLV hesabi (ROADMAP 51).

    prediction_odds: {'H': 2.10, 'D': 3.40, 'A': 3.50}  (tahmin anindaki)
    closing_odds:    {'H': 2.00, 'D': 3.50, 'A': 3.60}  (kapanis)
    won:             {'H': True, 'D': False, 'A': False} (gercek sonuc)

    CLV = (prediction_odds[won_sel] / closing_odds[won_sel]) - 1
    Pozitif => model kapanisa gore daha iyi fiyat yakaladi.
    """
    won_sel = "H" if won.get("H") else ("D" if won.get("D") else "A")
    pred = prediction_odds.get(won_sel)
    close = closing_odds.get(won_sel)
    if pred is None or close is None or close <= 0:
        return {"clv": None, "won_selection": won_sel, "note": "odds eksik"}
    clv = (pred / close) - 1.0
    return {
        "clv": round(float(clv), 4),
        "won_selection": won_sel,
        "prediction_odds": pred,
        "closing_odds": close,
        "edge": round(float(clv), 4),
    }


def batch_clv_report(rows: list[dict]) -> dict:
    """Birden cok macin CLV ortalamasi (ROADMAP 50: CLV metrigi)."""
    clvs = [r["clv"] for r in rows if r.get("clv") is not None]
    if not clvs:
        return {"n": 0, "mean_clv": None}
    return {
        "n": len(clvs),
        "mean_clv": round(float(np.mean(clvs)), 4),
        "positive_rate": round(float(np.mean([c > 0 for c in clvs])), 3),
    }
