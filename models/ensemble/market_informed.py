"""Market-informed ensemble (ROADMAP bolum 17+).

Sorun: saf model (catboost no_odds) acc ~47%, bookmaker favori ~46%.
Saf model bookmaker'dan BELLI BIR ORANDA daha iyi ama tutarsiz.

Cozum: bookmaker implied olasiligini PRIOR olarak al, modeli de "korreksiyon"
olarak kullan. Bayes vari:
  p_final = softmax( log(p_bookmaker) + alpha * log(p_model / p_bookmaker) )
  alpha=0 -> saf bookmaker (no edge)
  alpha=1 -> model bookmaker'i tam override eder (kopyalama)
  alpha∈(0,1) -> dengeli: model bookmaker'dan sapabilir ama bookmaker
                         guvenliği korunur.

Bu, "with_odds=True" modellerinin bookmaker'i KOPYALAMASINDAN farklidir:
orda bookmaker input olarak verilir ve model ona bagimli kalir.
Burada bookmaker PRIOR'dir, model onu "duzeltmeye" calisir.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.base import PredictionResult

logger = logging.getLogger(__name__)


class MarketInformedEnsemble:
    name = "market_informed"

    def __init__(self, model, alpha: float = 0.35) -> None:
        """
        model: tek bir alt model (olasilik ureten, with_odds=False onerilir)
        alpha: model agirligi [0,1]. 0.35 = bookmaker %65 agirlikli prior.
        """
        self.model = model
        self.alpha = alpha

    def fit(self, train, cal=None) -> "MarketInformedEnsemble":
        self.model.fit(train, cal)
        return self

    @staticmethod
    def _combine(p_book, p_model, alpha):
        """Bayes vari combine: log-odds karisimi + renormalize."""
        pb = np.clip(p_book, 1e-6, 1 - 1e-6)
        pm = np.clip(p_model, 1e-6, 1 - 1e-6)
        # log odds of model relative to bookmaker
        log_rel = np.log(pm) - np.log(pb)
        # final = bookmaker * model^alpha  (logaritmik)
        log_final = np.log(pb) + alpha * log_rel
        # stabilize
        log_final -= log_final.max(axis=1, keepdims=True)
        out = np.exp(log_final)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        p = self.model.predict(feats)
        ph = np.asarray(p.home_win, float)
        pd_ = np.asarray(p.draw, float)
        pa = np.asarray(p.away_win, float)

        pb_h = feats.get("mkt_home_prob")
        pb_d = feats.get("mkt_draw_prob")
        pb_a = feats.get("mkt_away_prob")
        if pb_h is None or pb_d is None or pb_a is None:
            # oran yoksa saf modele dus
            logger.warning("MarketInformed: mkt prob yok, saf modele dusuldu")
            return p
        pb_h = np.asarray(pb_h, float)
        pb_d = np.asarray(pb_d, float)
        pb_a = np.asarray(pb_a, float)

        Pm = np.column_stack([ph, pd_, pa])
        Pb = np.column_stack([pb_h, pb_d, pb_a])
        # her satir normalize (bookmaker overround var)
        Pb = Pb / Pb.sum(axis=1, keepdims=True)
        Pm = Pm / Pm.sum(axis=1, keepdims=True)

        out = self._combine(Pb, Pm, self.alpha)
        return PredictionResult(
            home_win=out[:, 0], draw=out[:, 1], away_win=out[:, 2],
            home_lambda=p.home_lambda, away_lambda=p.away_lambda,
            btts_yes=p.btts_yes, over25=p.over25,
        )
