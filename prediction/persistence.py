"""Tahmin kaliciligi ve sonuc degerlendirmesi (ROADMAP bolum 88).

Her tahmin DB'ye yazilir ki:
  - model neden dogru/yanlis tahmin etti sonradan anlasilabilsin
  - walk-forward disi gercek performans olculsin (ROADMAP 50)
  - drift monitoring (monitoring/drift.py) buradan beslensin

prediction_id formati (ROADMAP 40): {fixture_id}_{SNAPSHOT}
  ornek: 12345_T-24H, 12345_T-15M, 12345_LIVE65
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from db.engine import SessionLocal
from db.models import ModelPrediction, PredictionResult
from evaluation.metrics import (
    accuracy_1x2,
    brier_1x2,
    log_loss_1x2,
)

logger = logging.getLogger(__name__)


def save_prediction(
    prediction_id: str,
    fixture_id: int,
    prediction_time,
    payload: dict,
    model_version: str = "unknown",
    dataset_version: str | None = None,
    feature_version: str | None = None,
    source_versions: dict | None = None,
    session=None,
) -> ModelPrediction:
    """Pipeline ciktisini model_predictions tablosuna yazar (ROADMAP 88)."""
    own = session is None
    session = session or SessionLocal()
    try:
        # idempotent: ayni prediction_id varsa guncelleme
        existing = (session.execute(
            select(ModelPrediction).where(ModelPrediction.prediction_id == prediction_id)
        ).scalars().first())
        if existing is None:
            existing = ModelPrediction(prediction_id=prediction_id)
            session.add(existing)
        res = payload.get("result", {})
        goals = payload.get("goals", {})
        btts = payload.get("btts", {})
        ou = payload.get("over_under_2_5", {})
        existing.fixture_id = fixture_id
        existing.prediction_time = prediction_time or datetime.now(timezone.utc)
        existing.model_version = model_version
        existing.dataset_version = dataset_version
        existing.feature_version = feature_version
        existing.source_versions = json.dumps(source_versions or {}, ensure_ascii=False)
        existing.home_win = res.get("home")
        existing.draw = res.get("draw")
        existing.away_win = res.get("away")
        existing.home_lambda = goals.get("home_lambda")
        existing.away_lambda = goals.get("away_lambda")
        existing.btts_yes = btts.get("yes")
        existing.over25 = ou.get("over")
        existing.payload = json.dumps(payload, ensure_ascii=False, default=str)
        session.commit()
        session.refresh(existing)
        return existing
    finally:
        if own:
            session.close()


def record_result(
    prediction_id: str,
    actual_home_goals: int,
    actual_away_goals: int,
    session=None,
) -> PredictionResult | None:
    """Gercek sonucu tahminle eslestirip log-loss / brier hesaplar (ROADMAP 50)."""
    own = session is None
    session = session or SessionLocal()
    try:
        pred = (session.execute(
            select(ModelPrediction).where(ModelPrediction.prediction_id == prediction_id)
        ).scalars().first())
        if pred is None:
            logger.warning("Sonuc kaydedilemedi, tahmin yok: %s", prediction_id)
            return None
        actual_result = ("H" if actual_home_goals > actual_away_goals
                         else "A" if actual_away_goals > actual_home_goals else "D")
        y = [_np_select(actual_result)]
        ph, pd_, pa = float(pred.home_win), float(pred.draw), float(pred.away_win)
        ll = log_loss_1x2([actual_result], [ph], [pd_], [pa])
        br = brier_1x2([actual_result], [ph], [pd_], [pa])
        # idempotent
        pr = (session.execute(
            select(PredictionResult).where(PredictionResult.prediction_id == prediction_id)
        ).scalars().first())
        if pr is None:
            pr = PredictionResult(prediction_id=prediction_id)
            session.add(pr)
        pr.actual_home_goals = actual_home_goals
        pr.actual_away_goals = actual_away_goals
        pr.actual_result = actual_result
        pr.log_loss = ll
        pr.brier_score = br
        session.commit()
        session.refresh(pr)
        return pr
    finally:
        if own:
            session.close()


def _np_select(r: str) -> int:
    return 0 if r == "H" else 1 if r == "D" else 2


def get_prediction(prediction_id: str, session=None) -> ModelPrediction | None:
    own = session is None
    session = session or SessionLocal()
    try:
        return (session.execute(
            select(ModelPrediction).where(ModelPrediction.prediction_id == prediction_id)
        ).scalars().first())
    finally:
        if own:
            session.close()


def predictions_for_fixture(fixture_id: int, session=None):
    own = session is None
    session = session or SessionLocal()
    try:
        return session.execute(
            select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
            .order_by(ModelPrediction.prediction_time)
        ).scalars().all()
    finally:
        if own:
            session.close()
