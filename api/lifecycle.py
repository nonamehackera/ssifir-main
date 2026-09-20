"""Mac yasam dongusu orkestratoru (ROADMAP 40 + 66 + 85 + 88).

Bir macin tum omru boyunca:
  1. PRE-MATCH: T-48h..KO snapshot'lari uret + DB'ye yaz (persistence)
  2. LIVE: canli state geldikce snapshot + canli tahmin (live/model)
  3. RESULT: mac biter bitmez sonucu kaydet + evaluate (brier/logloss)
  4. monitoring/drift beslemesi icin donem metrigi kaydet

Kullanim:
  orch = MatchLifecycle(predict_fn, live_source=None)
  orch.run_pre_match(fixture_id, kickoff_at, features_row)
  orch.ingest_live(fixture_id, snapshot)   # canli veri her geldiginde
  orch.finalize(fixture_id, home_goals, away_goals)  # mac sonu
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from prediction.persistence import save_prediction, record_result
from prediction.snapshot_lifecycle import (
    emit_snapshots,
    prediction_id,
    stability,
)
from live.model import live_predict

logger = logging.getLogger(__name__)


class MatchLifecycle:
    def __init__(self, predict_fn, source_versions: dict | None = None,
                 model_version: str = "unknown", feature_version: str = "v1"):
        self.predict_fn = predict_fn
        self.source_versions = source_versions or {}
        self.model_version = model_version
        self.feature_version = feature_version
        self._live_trail: dict[int, list[dict]] = {}
        self._pre_lambda: dict[int, tuple[float, float]] = {}

    # ---- 1. PRE-MATCH ----
    def run_pre_match(self, fixture_id: int, kickoff_at: datetime, features_row) -> list[dict]:
        """Tum pre-match snapshot'larini uretir ve DB'ye yazar (ROADMAP 40)."""
        def _pf(label, pt):
            payload = self.predict_fn(features_row, snapshot_label=label, prediction_time=pt)
            save_prediction(
                prediction_id(fixture_id, label), fixture_id, pt, payload,
                model_version=self.model_version, feature_version=self.feature_version,
                source_versions=self.source_versions,
            )
            return payload

        emitted = emit_snapshots(fixture_id, kickoff_at, _pf)
        # stability (reliability icin)
        stab = stability(emitted)
        logger.info("Pre-match %d snapshot kaydedildi (stability=%.4f)", fixture_id, stab)
        return emitted

    # ---- 2. LIVE ----
    def ingest_live(self, fixture_id: int, snapshot: dict) -> dict:
        """Canli state snapshot'i geldikce canli tahmin uretir (ROADMAP 41-43).

        snapshot: {minute, xg_home, xg_away, score_home, score_away,
                   live_odds_home, live_odds_draw, live_odds_away}
        pre_lambda: pre-match beklenen goller (run_pre_match sonucundan veya ayrica)
        """
        pre = self._pre_lambda.get(fixture_id, (1.4, 1.1))
        pred = live_predict(
            pre[0], pre[1],
            minute=snapshot.get("minute", 0),
            xg_home=snapshot.get("xg_home"), xg_away=snapshot.get("xg_away"),
            score_home=snapshot.get("score_home", 0), score_away=snapshot.get("score_away", 0),
            live_odds_home=snapshot.get("live_odds_home"),
            live_odds_draw=snapshot.get("live_odds_draw"),
            live_odds_away=snapshot.get("live_odds_away"),
        )
        pid = prediction_id(fixture_id, f"LIVE{snapshot.get('minute', 0)}")
        save_prediction(
            pid, fixture_id, datetime.now(timezone.utc),
            {"result": pred["result"], "goals": pred["expected_goals"], "live": True,
             "score": pred["score"]},
            model_version=self.model_version, feature_version=self.feature_version,
            source_versions=self.source_versions,
        )
        self._live_trail.setdefault(fixture_id, []).append(pred)
        return pred

    def set_pre_lambda(self, fixture_id: int, home_lambda: float, away_lambda: float) -> None:
        self._pre_lambda[fixture_id] = (home_lambda, away_lambda)

    # ---- 3. RESULT ----
    def finalize(self, fixture_id: int, home_goals: int, away_goals: int) -> dict:
        """Mac biter bitmez tum snapshot'larin sonucunu kaydeder + evaluate."""
        # pre-match son snapshot'i (KO) ve tum live snapshot'lari sonuclandir
        from db.engine import SessionLocal
        from db.models import ModelPrediction
        from sqlalchemy import select

        session = SessionLocal()
        try:
            preds = session.execute(
                select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
            ).scalars().all()
            results = []
            for p in preds:
                # payload DB'de JSON string olarak saklanir
                try:
                    is_live = isinstance(p.payload, str) or (p.payload or {}).get("live", False)
                    if isinstance(p.payload, str):
                        import json as _json
                        _pl = _json.loads(p.payload)
                        is_live = _pl.get("live", False)
                    else:
                        is_live = (p.payload or {}).get("live", False)
                except Exception:
                    is_live = False
                # hem pre-match hem live sonuclandir (ROADMAP 88: her tahmin)
                rec = record_result(p.prediction_id, home_goals, away_goals, session=session)
                if rec:
                    results.append(rec)
            session.commit()
            return {"fixture_id": fixture_id, "evaluated": len(results)}
        finally:
            session.close()
