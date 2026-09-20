"""FastAPI servisi (ROADMAP bolum 86, 52).

Uretim tahmin API'si. Baslatilirken:
  - registry'den production modeli alinir (yoksa son candidate)
  - feature engine + model bellege yuklenir

Endpointler:
  GET  /health            -> servis sagligi + model versiyonu
  POST /predict           -> ROADMAP 52 formatinda tahmin
  POST /predict/features  -> ham feature dict alir, tahmin uretir

NOT: tam production'da model .pkl olarak yuklenir; burada egitilmis model
bellekte tutulur (demo + dogrulama icin). Gercek deploy'da
`GET /model/load?version=v3` ile registry'den yuklenebilir.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs import settings
from prediction.pipeline import PredictionPipeline, predict_to_json

logger = logging.getLogger(__name__)

app = FastAPI(title="Football Prediction API", version="1.0.0")

# Bellekte tutulan model (servis baslatildiginda yuklenir)
_STATE: dict = {"pipeline": None, "model_version": None, "loaded_at": None}


class PredictRequest(BaseModel):
    """Tek mac tahmini icin feature dict (ROADMAP 24 formati)."""
    fixture_id: int | None = None
    features: dict[str, Any] = Field(..., description="feature engine cikti satiri")


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    loaded_at: str | None


def _load_default_model():
    """Demo/verify icin: en iyi bilinen modeli (catboost_odds) egitip yukler.
    Gercek deployment'da registry'den production .pkl yuklenir."""
    import sys
    sys.path.insert(0, ".")
    from ingestion.football_data.canonical import build_matches
    from feature_engine.engine import build_features
    from models.catboost.model import CatBoostModel

    m = build_matches()
    f = build_features(m)
    cut = int(len(f) * 0.8)
    train, cal = f.iloc[:cut].copy(), f.iloc[cut:cut + int(len(f) * 0.1)].copy()
    model = CatBoostModel(with_odds=True, iterations=400, verbose=False,
                         draw_weight=4.0, time_decay=1.0).fit(train, cal)
    pipe = PredictionPipeline(model, model_version="catboost_odds_v1", dataset_version="fd_2122_2425")
    _STATE["pipeline"] = pipe
    _STATE["model_version"] = "catboost_odds_v1"
    _STATE["loaded_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Default model yuklendi: catboost_odds_v1")


@app.on_event("startup")
def _startup():
    try:
        _load_default_model()
    except Exception as exc:
        logger.error("Model yuklenemedi: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _STATE["pipeline"] else "degraded",
        model_version=_STATE["model_version"],
        loaded_at=_STATE["loaded_at"],
    )


@app.post("/predict")
def predict(req: PredictRequest):
    if _STATE["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model yuklu degil")
    try:
        feats = pd.DataFrame([req.features])
        out = _STATE["pipeline"].predict(feats, fixture_id=req.fixture_id)
        return out
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Tahmin hatasi: {exc}")


@app.post("/predict/json")
def predict_json(req: PredictRequest):
    """ROADMAP 52 JSON string olarak dondurur."""
    out = predict(req)
    return predict_to_json(out)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
