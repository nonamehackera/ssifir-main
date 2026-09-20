"""Probability ensemble (ROADMAP bolum 33) + temperature calibration (34).

fit(train, cal): trains every sub-model on `train`, then:
  1. optimizes sub-model weights on `cal` to minimize 1X2 log loss
     (bounded simplex, scipy SLSQP)
  2. fits temperature T on `cal` to minimize NLL of the weighted ensemble

predict(feats): weighted average of sub-model probabilities, temperature
scaled. All probabilities are re-normalized afterwards.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.base import PredictionResult

logger = logging.getLogger(__name__)


class EnsembleModel:
    name = "ensemble"

    def __init__(self, models: list) -> None:
        if not models:
            raise ValueError("Ensemble en az bir model ister.")
        self.models = models
        self.weights: np.ndarray | None = None
        self.temperature: float = 1.0

    def fit(self, train, cal=None) -> "EnsembleModel":
        for m in self.models:
            m.fit(train, cal)
        self.weights = np.full(len(self.models), 1.0 / len(self.models))
        if cal is not None and len(cal) > 50:
            self._fit_weights_and_temp(cal)
        else:
            logger.warning("Calibration dilimi yok; eşit ağırlıklar kullanılıyor.")
        return self

    def _sub_predictions(self, feats: pd.DataFrame) -> list[PredictionResult]:
        return [m.predict(feats) for m in self.models]

    def _prob_matrix(self, preds: list[PredictionResult]) -> np.ndarray:
        """(n, 3, k) per-sub-model 1X2 probability tensor."""
        cols = []
        for p in preds:
            cols.append(
                np.column_stack(
                    [np.asarray(p.home_win, float),
                     np.asarray(p.draw, float),
                     np.asarray(p.away_win, float)]
                )
            )
        return np.stack(cols, axis=-1)

    def _fit_weights_and_temp(self, cal: pd.DataFrame) -> None:
        y = cal["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        P3 = self._prob_matrix(self._sub_predictions(cal))  # (n, 3, k)
        n = len(self.models)

        def nll_from_weights(w: np.ndarray) -> float:
            probs = np.einsum("ijk,k->ij", P3, w)  # (n, 3)
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            return float(-np.mean(np.log(probs[np.arange(len(y)), y])))

        w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        res = minimize(nll_from_weights, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-6})
        if res.success:
            self.weights = np.clip(res.x, 0, 1)
            self.weights = self.weights / self.weights.sum()

        # temperature on the weighted ensemble (softmax on log-probs)
        Pw = np.clip(np.einsum("ijk,k->ij", P3, self.weights), 1e-9, 1 - 1e-9)

        def nll_temp(t: float) -> float:
            logp = np.log(Pw) / t
            logp = logp - logp.max(axis=1, keepdims=True)
            p = np.exp(logp)
            p = p / p.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))

        res_t = minimize(nll_temp, 1.0, method="Nelder-Mead", options={"xatol": 1e-4})
        if res_t.success:
            self.temperature = float(res_t.x[0])
        logger.info(
            "Ensemble weights=%s T=%.3f (cal=%d)",
            np.round(self.weights, 3), self.temperature, len(cal),
        )

    def _agg(self, preds: list[PredictionResult], field: str) -> np.ndarray | None:
        """Field bazında None çıktıları filtreleyerek ağırlıklı ortalama."""
        vals, ws = [], []
        for p, w in zip(preds, self.weights):
            v = getattr(p, field)
            if v is None:
                continue
            vals.append(np.asarray(v, float))
            ws.append(w)
        if not vals:
            return None
        ws = np.array(ws)
        ws = ws / ws.sum()
        return np.average(np.column_stack(vals), axis=1, weights=ws)

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        preds = self._sub_predictions(feats)
        P3 = self._prob_matrix(preds)
        probs = np.clip(np.einsum("ijk,k->ij", P3, self.weights), 1e-12, 1)

        # temperature scaling on 1X2 (softmax on log-probs)
        logp = np.log(probs) / self.temperature
        logp = logp - logp.max(axis=1, keepdims=True)
        scaled = np.exp(logp)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)

        # aggregate the rest (BTTS / over25 / lambdas) by weighted average
        btts = self._agg(preds, "btts_yes")
        over25 = self._agg(preds, "over25")
        lam_h = self._agg(preds, "home_lambda")
        lam_a = self._agg(preds, "away_lambda")
        return PredictionResult(
            home_win=scaled[:, 0],
            draw=scaled[:, 1],
            away_win=scaled[:, 2],
            home_lambda=lam_h,
            away_lambda=lam_a,
            btts_yes=np.clip(btts, 0, 1) if btts is not None else None,
            over25=np.clip(over25, 0, 1) if over25 is not None else None,
        )