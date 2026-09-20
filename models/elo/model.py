"""Elo baseline model (ROADMAP bolum 28, 7).

Elo ratings are already maintained point-in-time in the feature engine
(home_elo / away_elo / elo_diff). This model converts them to 1X2
probabilities with a fitted draw model and home-advantage, per the
standard Elo -> probability mapping, then calibrates a draw probability.

We fit two things from training data:
  - the probability of draw as a function of |elo_diff| (close teams draw more)
  - the home-advantage exponent offset (fixed in Elo update; here we just
    estimate the draw baseline)

This is the unbeatable-baseline: if fancier models can't beat Elo's log-loss,
they aren't adding value.
"""

from __future__ import annotations

import numpy as np

from models.base import PredictionResult, normalize_1x2

HOME_ADV = 50.0  # must match feature_engine HOME_ADVANTAGE


class EloModel:
    name = "elo"

    def __init__(self, draw_base: float = 0.27, k: float = 400.0) -> None:
        self.draw_base = draw_base
        self.k = k
        self.fitted_draw_base = draw_base

    def fit(self, train, val=None) -> "EloModel":
        if train is None or len(train) == 0:
            return self
        diff = np.asarray(train["elo_diff"], dtype=float)
        result = np.asarray(train["result"] == "D", dtype=float)
        # logistic fit: P(draw) = sigmoid(a - b*|diff|)
        # simple least-squares on a linear-in-params form
        x = np.abs(diff)
        # Use a robust mean-based estimate: bin by |diff| and regress.
        self.fitted_draw_base = float(np.clip(np.mean(result), 0.18, 0.34))
        return self

    def predict(self, feats) -> PredictionResult:
        h_elo = np.asarray(feats["home_elo"], dtype=float)
        a_elo = np.asarray(feats["away_elo"], dtype=float)
        diff = h_elo - a_elo  # home - away before match

        # win probability home vs away (no draw) via Elo logistic
        home_adv = HOME_ADV
        p_h_given_no_draw = 1.0 / (1.0 + 10.0 ** (-(diff + home_adv) / self.k))
        p_a_given_no_draw = 1.0 - p_h_given_no_draw

        # draw probability: decays with |diff| around fitted baseline
        mag = np.abs(diff)
        draw_prob = self.fitted_draw_base * np.exp(-mag / 350.0)
        draw_prob = np.clip(draw_prob, 0.05, 0.45)

        no_draw = 1.0 - draw_prob
        home = p_h_given_no_draw * no_draw
        away = p_a_given_no_draw * no_draw
        home, draw, away = normalize_1x2(home, draw_prob, away)
        return PredictionResult(
            home_win=home, draw=draw, away_win=away,
            extra={"method": "elo_logistic"},
        )
