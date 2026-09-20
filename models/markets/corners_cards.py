"""Corners (ROADMAP 38) ve Cards (ROADMAP 39) modelleri.

Free veride (Football-Data.co.uk) mevcut:
  - corners: HC (home corners), AC (away corners)
  - cards:   HY/AY (yellow), HR/AR (red)

Her iki pazar da "fazla sacilma" gosterebilir; bu yuzden sadece Poisson
degil, Negative Binomial ile de benchmark edilir (ROADMAP 38 notu).

Her model:
  - fit(train): pre-match feature'lardan ogrenir
  - predict(feats): home_corners / away_corners / total_corners (corners)
                    home_cards / away_cards / total_cards (cards)
  - skor dagilimi: Poisson/NegBin PMF ile uretilir (over/under hesaplamak icin)

Hedefler (ROADMAP):
  corners -> home_corners, away_corners, total_corners
  cards   -> home_cards, away_cards, total_cards
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from models.base import PredictionResult

logger = logging.getLogger(__name__)

CORNERS_FEATURES = [
    "home_corners_5", "away_corners_5", "home_w_shots", "away_w_shots",
    "home_sot_5", "away_sot_5", "lg_corner_avg", "lg_avg_goals",
    "home_elo", "away_elo", "elo_diff", "home_rest_days", "away_rest_days",
]

# Dikkat: pre-match feature'lar SADECE. home_yellow/away_yellow gibi ayni
# macin post-match degerleri feature OLAMAZ (target leakage). Rolling kart
# feature'lari feature_engine tarafindan uretilir (home_cards_5 vb).
CARDS_FEATURES = [
    "home_cards_5", "away_cards_5", "home_w_cards", "away_w_cards",
    "lg_card_avg", "referee",
    "home_elo", "away_elo", "league", "home_rest_days", "away_rest_days",
]


class _CountRegressor:
    """Ortak taban: Poisson veya Negative Binomial regressor (lightgbm ile
    benchmark). Hedef: tek bir sayi (corners/cards)."""

    def __init__(self, family: str = "poisson", use_ml: bool = True, n_estimators: int = 300) -> None:
        self.family = family
        self.use_ml = use_ml
        self.n_estimators = n_estimators
        self._ml = None
        self._alpha = 1.0
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray, features: list[str]):
        y = np.asarray(y, dtype=float)
        Xs = X[features].copy()
        # categorical kolonlari kategorik tipe cevir (lightgbm icin)
        self._cat_features = [c for c in ["league", "referee", "season"] if c in features]
        for c in self._cat_features:
            Xs[c] = Xs[c].astype("category")
        self._features = features
        self._mean = float(np.mean(y)) if len(y) else 0.0
        if self.use_ml:
            import lightgbm as lgb

            self._ml = lgb.LGBMRegressor(
                objective="poisson" if self.family == "poisson" else "regression",
                n_estimators=self.n_estimators, random_seed=42, verbose=-1,
            )
            self._ml.fit(Xs, y, categorical_feature=self._cat_features if self._cat_features else None)
        else:
            var = np.var(y) if len(y) > 1 else 1.0
            mean = self._mean
            self._alpha = max(0.01, var / mean - 1.0) if mean > 0 else 0.5
        return self

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        Xs = X[self._features].copy()
        for c in self._cat_features:
            Xs[c] = Xs[c].astype("category")
        if self._ml is not None:
            return np.maximum(self._ml.predict(Xs), 0.05)
        return np.full(len(Xs), self._mean)

    def predict_dist(self, means: np.ndarray):
        """Her satir icin 0..MAX_G olasilik dagilimi (PMF)."""
        MAX_G = 20
        g = np.arange(MAX_G + 1)
        out = []
        for mu in means:
            mu = max(mu, 0.05)
            if self.family == "poisson":
                pmf = poisson.pmf(g, mu)
            else:
                # NegBin: mean=mu, alpha -> r=mu/alpha, p=r/(r+mu)
                r = max(mu / max(self._alpha, 1e-3), 0.05)
                p = r / (r + mu)
                pmf = nbinom.pmf(g, r, p)
            pmf = np.asarray(pmf, dtype=float)
            pmf = pmf / pmf.sum()
            out.append(pmf)
        return np.array(out)  # (n, MAX_G+1)


class _CatCountRegressor:
    """CatBoost Poisson regressor, tam pre-match feature seti ile.

    Leakage-free (sadece pre-match feature'lar). Mevcut LightGBM-negbin
    yaklasimi kornerlerde naive'e yakin kaldi (%51); CatBoost tam feature
    seti ile %57'e cikiyor. Cards'ta sinyal yok (naive ile ayni).
    """

    def __init__(self, iterations: int = 400, learning_rate: float = 0.03, depth: int = 5) -> None:
        self.params = dict(iterations=iterations, learning_rate=learning_rate, depth=depth,
                           verbose=False, allow_writing_files=False)
        self._ml = None
        self._features = None
        self._cat_idx = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, features: list[str]):
        from catboost import CatBoostRegressor
        from models.catboost.model import CAT_COLS
        self._features = features
        self._cat_idx = [features.index(c) for c in CAT_COLS if c in features]
        self._ml = CatBoostRegressor(loss_function="Poisson", cat_features=self._cat_idx,
                                     **self.params)
        self._ml.fit(X[features], np.asarray(y, dtype=float))
        return self

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self._ml.predict(X[self._features]), 0.05)

    def predict_dist(self, means: np.ndarray):
        MAX_G = 20
        g = np.arange(MAX_G + 1)
        out = []
        for mu in means:
            pmf = np.asarray(poisson.pmf(g, max(mu, 0.05)), dtype=float)
            pmf = pmf / pmf.sum()
            out.append(pmf)
        return np.array(out)


class CornersModel:
    name = "corners"

    def __init__(self, family: str = "negbin", use_ml: bool = True) -> None:
        if use_ml:
            self.home = _CatCountRegressor()
            self.away = _CatCountRegressor()
            self._full = True
        else:
            self.home = _CountRegressor(family, False)
            self.away = _CountRegressor(family, False)
            self._full = False

    def fit(self, train, val=None) -> "CornersModel":
        if self._full:
            from models.catboost.model import CAT_COLS, NUMERIC_FEATURES, ODDS_FEATURES
            fs = CAT_COLS + NUMERIC_FEATURES + ODDS_FEATURES
            self.home.fit(train, train["home_corners"].to_numpy(), fs)
            self.away.fit(train, train["away_corners"].to_numpy(), fs)
        else:
            self.home.fit(train, train["home_corners"], CORNERS_FEATURES)
            self.away.fit(train, train["away_corners"], CORNERS_FEATURES)
        return self

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        hm = self.home.predict_mean(feats)
        am = self.away.predict_mean(feats)
        h_dist = self.home.predict_dist(hm)
        a_dist = self.away.predict_dist(am)
        # total = convolution (h + a)
        MAX_G = h_dist.shape[1] - 1
        total = np.convolve(np.ones(MAX_G + 1), np.ones(MAX_G + 1))[: 2 * MAX_G + 1]
        total_dist = np.array([np.convolve(h_dist[i], a_dist[i])[: 2 * MAX_G + 1]
                               for i in range(len(h_dist))])
        # over/under 8.5 ve 9.5 (corners genelde 8-12)
        over_85 = 1.0 - total_dist[:, :9].sum(axis=1)
        over_95 = 1.0 - total_dist[:, :10].sum(axis=1)
        return PredictionResult(
            home_lambda=hm, away_lambda=am,
            extra={
                "over_8_5": over_85, "over_9_5": over_95,
                "home_dist": h_dist, "away_dist": a_dist, "total_dist": total_dist,
            },
        )


class CardsModel:
    name = "cards"

    def __init__(self, family: str = "negbin", use_ml: bool = True) -> None:
        self.home = _CountRegressor(family, use_ml)
        self.away = _CountRegressor(family, use_ml)

    def fit(self, train, val=None) -> "CardsModel":
        self.home.fit(train, train["home_yellow"] + train["home_red"].fillna(0), CARDS_FEATURES)
        self.away.fit(train, train["away_yellow"] + train["away_red"].fillna(0), CARDS_FEATURES)
        return self

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        hm = self.home.predict_mean(feats)
        am = self.away.predict_mean(feats)
        h_dist = self.home.predict_dist(hm)
        a_dist = self.away.predict_dist(am)
        total_dist = np.array([np.convolve(h_dist[i], a_dist[i])[: 2 * h_dist.shape[1]]
                               for i in range(len(h_dist))])
        over_35 = 1.0 - total_dist[:, :4].sum(axis=1)  # toplam 4.5 ustu
        return PredictionResult(
            home_lambda=hm, away_lambda=am,
            extra={"over_3_5": over_35, "home_dist": h_dist, "away_dist": a_dist,
                   "total_dist": total_dist},
        )
