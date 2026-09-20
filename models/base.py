"""Shared model interface.

Every model exposes:
  fit(train_feats, val_feats=None) -> self
  predict(feats) -> PredictionResult  (probabilities + expected goals)

PredictionResult carries, per match (numpy arrays, length = len(feats)):
  home_win, draw, away_win     # 1X2 probabilities (each model normalizes its own)
  home_lambda, away_lambda     # expected goals (Dixon-Coles / regressors)
  btts_yes, over25             # market probabilities
This lets the ensemble average whatever each model produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class PredictionResult:
    home_win: object = None
    draw: object = None
    away_win: object = None
    home_lambda: object = None
    away_lambda: object = None
    btts_yes: object = None
    over15: object = None
    over25: object = None
    over35: object = None
    score_matrix: object = None  # optional list of (h, a, p)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
            "home_lambda": self.home_lambda,
            "away_lambda": self.away_lambda,
            "btts_yes": self.btts_yes,
            "over15": self.over15,
            "over25": self.over25,
            "over35": self.over35,
        }


class MatchPredictor(Protocol):
    name: str

    def fit(self, train, val=None) -> "MatchPredictor":
        ...

    def predict(self, feats) -> PredictionResult:
        ...


def normalize_1x2(home, draw, away):
    import numpy as np

    s = home + draw + away
    s = np.where(s <= 0, 1.0, s)
    return home / s, draw / s, away / s
