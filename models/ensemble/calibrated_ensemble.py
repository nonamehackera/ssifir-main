"""Ensemble with class-weight aware training + temperature calibration.

The key insight: use weighted average of sub-model probabilities + temperature
scaling. Isotonic regression and meta-learners break when ensemble variance is low.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.base import PredictionResult

logger = logging.getLogger(__name__)


class CalibratedEnsemble:
    name = "calibrated_ensemble"

    def __init__(self, models: list) -> None:
        if not models:
            raise ValueError("Ensemble en az bir model ister.")
        self.models = models
        self.weights: np.ndarray | None = None
        self.temperature: float = 1.0
        self._error_memory: list[dict] = []  # mispredicted rows from past folds

    def fit(self, train, cal=None, sample_weight=None) -> "CalibratedEnsemble":
        # If we have error memory from previous folds, boost those rows
        if self._error_memory and sample_weight is None:
            mem_ids = {r["match_id"] for r in self._error_memory}
            sample_weight = train["match_id"].isin(mem_ids).astype(float) + 1.0

        for m in self.models:
            if hasattr(m, "fit"):
                m.fit(train, cal, sample_weight=sample_weight)
            else:
                m.fit(train, cal)

        n = len(self.models)
        self.weights = np.full(n, 1.0 / n)

        if cal is not None and len(cal) > 200:
            self._fit_weights_and_temp(cal)
        else:
            logger.warning("Calibration verisi yetersiz.")

        return self

    def remember_errors(self, valid: pd.DataFrame, pred) -> None:
        """Store mispredicted matches so the NEXT fit can up-weight them."""
        y = valid["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        ph, pd_, pa = np.asarray(pred.home_win), np.asarray(pred.draw), np.asarray(pred.away_win)
        preds = np.select([ph >= pd_, pa >= ph], [0, 2], default=1)
        wrong = valid.iloc[(preds != y).nonzero()[0]]
        for _, row in wrong.iterrows():
            self._error_memory.append({"match_id": row["match_id"]})
        # Cap memory to avoid runaway growth
        if len(self._error_memory) > 50000:
            self._error_memory = self._error_memory[-50000:]

    def _get_prob_matrix(self, feats: pd.DataFrame) -> np.ndarray:
        """(n, 3, k) per-sub-model probabilities."""
        cols = []
        for m in self.models:
            p = m.predict(feats)
            cols.append(np.column_stack([
                np.asarray(p.home_win, float),
                np.asarray(p.draw, float),
                np.asarray(p.away_win, float),
            ]))
        return np.stack(cols, axis=-1)

    def _fit_weights_and_temp(self, cal: pd.DataFrame) -> None:
        y = cal["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        P3 = self._get_prob_matrix(cal)
        n = len(self.models)

        def nll(w):
            probs = np.einsum("ijk,k->ij", P3, w)
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            return -np.mean(np.log(probs[np.arange(len(y)), y]))

        res = minimize(nll, self.weights, method="L-BFGS-B",
                       bounds=[(0, 1)] * n,
                       options={"maxiter": 200})
        if res.success:
            self.weights = np.clip(res.x, 0, 1)
            self.weights /= self.weights.sum()

        Pw = np.einsum("ijk,k->ij", P3, self.weights)

        def nll_temp(t):
            logp = np.log(np.clip(Pw, 1e-9, 1)) / t
            logp -= logp.max(axis=1, keepdims=True)
            p = np.exp(logp)
            p /= p.sum(axis=1, keepdims=True)
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))

        res_t = minimize(nll_temp, 1.0, method="Nelder-Mead",
                         options={"xatol": 1e-4})
        if res_t.success:
            self.temperature = float(res_t.x[0])

        logger.info("Ensemble: weights=%s T=%.3f", np.round(self.weights, 3), self.temperature)

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        P3 = self._get_prob_matrix(feats)
        probs = np.einsum("ijk,k->ij", P3, self.weights)

        # Temperature scaling
        logp = np.log(np.clip(probs, 1e-12, 1)) / self.temperature
        logp -= logp.max(axis=1, keepdims=True)
        scaled = np.exp(logp)
        scaled /= scaled.sum(axis=1, keepdims=True)

        # Aggregate other outputs
        lam_h = self._agg(feats, "home_lambda")
        lam_a = self._agg(feats, "away_lambda")
        btts = self._agg(feats, "btts_yes")
        over25 = self._agg(feats, "over25")

        return PredictionResult(
            home_win=scaled[:, 0],
            draw=scaled[:, 1],
            away_win=scaled[:, 2],
            home_lambda=lam_h,
            away_lambda=lam_a,
            btts_yes=np.clip(btts, 0, 1) if btts is not None else None,
            over25=np.clip(over25, 0, 1) if over25 is not None else None,
        )

    def _agg(self, feats: pd.DataFrame, field: str) -> np.ndarray | None:
        vals, ws = [], []
        for m, w in zip(self.models, self.weights):
            p = m.predict(feats)
            v = getattr(p, field)
            if v is not None:
                vals.append(np.asarray(v, float))
                ws.append(w)
        if not vals:
            return None
        ws = np.array(ws)
        ws /= ws.sum()
        return np.average(np.column_stack(vals), axis=1, weights=ws)
