"""Drift monitoring + otomatik retrain hook (ROADMAP bolum 69, 70).

Model zamanla bozulur (feature drift, prediction drift, calibration drift).
Bu modul:

1. evaluate_period(): belirli bir donemdeki tahminlerin gercek sonuclarla
   karsilastirip Brier / log-loss / accuracy uretir (ROADMAP 70 ornek).
2. detect_drift(): onceki donem metrigi ile karsilastirir, esik asilirsa
   ALARM dondurur.
3. maybe_retrain(): drift alarmi varsa retrain tetikler (hook). Gercek
   retrain isi disaridan bir callable ile verilir (jobs retrain fonksiyonu).

ROADMAP 70 ornek:
  2024 Brier = 0.192
  2025 Brier = 0.209
  2026 Brier = 0.234  -> alarm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from evaluation.metrics import (
    accuracy_1x2,
    brier_1x2,
    log_loss_1x2,
)

logger = logging.getLogger(__name__)

# Alarm esikleri (ROADMAP 70 prensibi: kucuk dalgalanma degil,
# surdurulebilir kotulesme alarmi)
BRIER_DRIFT_THRESHOLD = 0.02      # onceki doneme gore +0.02 artis
LOGLOSS_DRIFT_THRESHOLD = 0.03    # onceki doneme gore +0.03 artis
ACCURACY_DROP_THRESHOLD = 0.03    # onceki doneme gore -0.03 dusus


@dataclass
class PeriodMetrics:
    period: str
    n: int
    log_loss: float
    brier: float
    accuracy: float


def evaluate_period(y_true, p_home, p_draw, p_away, period: str = "current") -> PeriodMetrics:
    """Bir donemin tahmin metriklerini hesaplar."""
    return PeriodMetrics(
        period=period,
        n=len(y_true),
        log_loss=log_loss_1x2(y_true, p_home, p_draw, p_away),
        brier=brier_1x2(y_true, p_home, p_draw, p_away),
        accuracy=accuracy_1x2(y_true, p_home, p_draw, p_away),
    )


@dataclass
class DriftReport:
    drift_detected: bool
    reasons: list[str] = field(default_factory=list)
    prev: PeriodMetrics | None = None
    curr: PeriodMetrics | None = None


def detect_drift(prev: PeriodMetrics, curr: PeriodMetrics) -> DriftReport:
    """Iki donem metrigini karsilastirir, esik asilirsa drift alarmi."""
    reasons = []
    if curr.brier - prev.brier >= BRIER_DRIFT_THRESHOLD:
        reasons.append(f"Brier drift: {prev.brier:.4f} -> {curr.brier:.4f} (Δ+{curr.brier-prev.brier:.4f})")
    if curr.log_loss - prev.log_loss >= LOGLOSS_DRIFT_THRESHOLD:
        reasons.append(f"LogLoss drift: {prev.log_loss:.4f} -> {curr.log_loss:.4f}")
    if prev.accuracy - curr.accuracy >= ACCURACY_DROP_THRESHOLD:
        reasons.append(f"Accuracy drop: {prev.accuracy:.3f} -> {curr.accuracy:.3f}")
    return DriftReport(
        drift_detected=len(reasons) > 0,
        reasons=reasons,
        prev=prev,
        curr=curr,
    )


def maybe_retrain(drift: DriftReport, retrain_fn=None) -> dict:
    """Drift varsa retrain tetikler. retrain_fn disaridan verilir.

    Ornek retrain_fn (jobs/retrain.py):
        def do_retrain():
            from jobs.retrain import run
            return run(with_odds=True, register=True)

    Doner: {'action': 'retrain'|'skip', 'reasons': [...]}
    """
    if not drift.drift_detected:
        return {"action": "skip", "reasons": ["drift yok"]}
    logger.warning("DRIFT ALARM: %s", drift.reasons)
    if retrain_fn is not None:
        try:
            result = retrain_fn()
            return {"action": "retrain", "reasons": drift.reasons, "status": "triggered",
                    "result": result}
        except Exception as exc:
            logger.error("Retrain basarisiz: %s", exc)
            return {"action": "retrain", "reasons": drift.reasons, "status": "failed", "error": str(exc)}
    return {"action": "retrain", "reasons": drift.reasons, "status": "no_hook"}


# Basit InMemory metrik gecmisi (gercek sistemde DB'ye yazilir)
_HISTORY: list[PeriodMetrics] = []


def record_period(m: PeriodMetrics) -> None:
    _HISTORY.append(m)


def check_latest_drift(retrain_fn=None) -> DriftReport | None:
    """Son iki donemi karsilastirir (ROADMAP 70 benzeri)."""
    if len(_HISTORY) < 2:
        return None
    return detect_drift(_HISTORY[-2], _HISTORY[-1])
