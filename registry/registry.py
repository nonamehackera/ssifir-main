"""Model registry (ROADMAP bolum 67, 68, 88).

Her egitilen model bir ModelVersion kaydi ile PostgreSQL'e yazilir:
    model_id, version, status (candidate -> staging -> production),
    metrics (JSON), features_hash, dataset_version, calibration_method, ...

Akis (ROADMAP 68):
    candidate  : yeni egitildi, henuz backtest edilmedi
    staging    : walk-forward basarili, canli olmayan degerlendirmede
    production : onaylandi, sadece bu status'teki model uretime cikar

Bu modul hem kayit (register) hem de "production modeli getir" islemlerini
yapar. Boylece prediction pipeline hangi modeli kullanacagini DB'den bilir
(ROADMAP 88: her tahmin model_version tasir).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from db.engine import SessionLocal
from db.models import ModelVersion

logger = logging.getLogger(__name__)


def _features_hash(feature_cols) -> str:
    if not feature_cols:
        return "none"
    h = hashlib.sha256()
    for c in sorted(feature_cols):
        h.update(str(c).encode())
    return h.hexdigest()[:16]


def register_model(
    model_id: str,
    version: str,
    metrics: dict,
    algorithm: str,
    dataset_version: str,
    features_hash: str | None = None,
    feature_cols=None,
    calibration_method: str | None = None,
    hyperparameters: dict | None = None,
    status: str = "candidate",
    session=None,
) -> ModelVersion:
    """Yeni bir model versiyonunu DB'ye kaydeder (candidate olarak)."""
    own = session is None
    session = session or SessionLocal()
    try:
        mv = ModelVersion(
            model_id=model_id,
            version=version,
            training_start=datetime.now(timezone.utc),
            training_end=datetime.now(timezone.utc),
            features_hash=features_hash or _features_hash(feature_cols),
            dataset_version=dataset_version,
            algorithm=algorithm,
            hyperparameters=json.dumps(hyperparameters or {}, ensure_ascii=False),
            metrics=json.dumps(metrics, ensure_ascii=False, default=str),
            calibration_method=calibration_method,
            status=status,
        )
        session.add(mv)
        session.commit()
        session.refresh(mv)
        logger.info("Model kaydedildi: %s@%s status=%s", model_id, version, status)
        return mv
    finally:
        if own:
            session.close()


def promote(model_id: str, version: str, to_status: str, session=None) -> None:
    """Modeli candidate -> staging -> production asamalarina gecirir.

    Kurallar (ROADMAP 68):
      - sadece 'candidate' -> 'staging' veya 'staging' -> 'production' gecislerine izin ver
      - production'a gecerken, ayni model_id'nin diger production kaydi
        otomatik 'archived' yapilir (tek production prensibi)
    """
    allowed = {"staging": "candidate", "production": "staging"}
    if to_status not in allowed:
        raise ValueError(f"Gecis hedefi gecersiz: {to_status}")

    own = session is None
    session = session or SessionLocal()
    try:
        mv = session.query(ModelVersion).filter_by(model_id=model_id, version=version).first()
        if mv is None:
            raise ValueError(f"Model bulunamadi: {model_id}@{version}")
        if mv.status != allowed[to_status]:
            raise ValueError(
                f"{model_id}@{version} status='{mv.status}' -> '{to_status}' gecisine izin yok "
                f"(beklenen: '{allowed[to_status]}')"
            )
        mv.status = to_status
        if to_status == "production":
            # diger production kayitlarini archive et
            for other in session.query(ModelVersion).filter_by(model_id=model_id, status="production").all():
                if other.version != version:
                    other.status = "archived"
                    logger.info("Eski production arsivlendi: %s@%s", model_id, other.version)
        session.commit()
        logger.info("Model terfi: %s@%s -> %s", model_id, version, to_status)
    finally:
        if own:
            session.close()


def get_production_model(model_id: str, session=None) -> ModelVersion | None:
    """Uretimdeki (status='production') modeli dondurur."""
    own = session is None
    session = session or SessionLocal()
    try:
        return session.query(ModelVersion).filter_by(model_id=model_id, status="production").first()
    finally:
        if own:
            session.close()


def list_models(model_id: str | None = None, session=None) -> list[ModelVersion]:
    q = SessionLocal().query(ModelVersion) if session is None else session.query(ModelVersion)
    if model_id:
        q = q.filter_by(model_id=model_id)
    own = session is None
    try:
        return q.order_by(ModelVersion.id.desc()).all()
    finally:
        if own:
            SessionLocal().close()
