"""LightGBM benchmark model (ROADMAP bolum 31).

Numeric-feature benchmark alongside CatBoost. Same leakage-safe feature
pipeline; league/season/team IDs are passed as categorical features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.base import PredictionResult
from models.catboost.model import feature_columns

logger = logging.getLogger(__name__)


CAT_FEATURES = ["league", "season"]


class LightGBMModel:
    """Numeric-feature benchmark (ROADMAP bolum 31).

    Takım kimliği bilgisi kategorik olarak eklenmez: kategorik takım
    ID'leri LightGBM'de aşırı güvene yol açıyor; takım gücü elo/opp_elo
    ile taşınır.
    """

    name = "lightgbm"

    def __init__(
        self,
        with_odds: bool = False,
        num_leaves: int = 31,
        learning_rate: float = 0.03,
        n_estimators: int = 1000,
        max_depth: int = 5,
        min_child_samples: int = 80,
        subsample: float = 0.7,
        colsample_bytree: float = 0.6,
        reg_alpha: float = 0.5,
        reg_lambda: float = 5.0,
        max_bin: int = 63,
        random_seed: int = 42,
        verbose: int = -1,
        draw_weight: float = 4.0,
        time_decay: float = 1.0,
    ) -> None:
        self.with_odds = with_odds
        self.draw_weight = draw_weight
        self.time_decay = time_decay
        self.feature_cols = [
            c for c in feature_columns(with_odds) if c not in ("home_team_id", "away_team_id")
        ]
        self.cat_features = [c for c in CAT_FEATURES if c in self.feature_cols]
        self.params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "max_bin": max_bin,
            "random_seed": random_seed,
            "verbose": verbose,
        }
        self._clf = None
        self._reg_home = None
        self._reg_away = None
        self._clf_btts = None
        self._clf_over25 = None

    def _X(self, feats: pd.DataFrame) -> pd.DataFrame:
        df = feats[self.feature_cols].copy()
        for c in self.cat_features:
            df[c] = df[c].astype("category")
        return df

    def fit(self, train, val=None) -> "LightGBMModel":
        import lightgbm as lgb

        X = self._X(train)
        y = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

        # TIME DECAY: yeni maclar daha onemli
        if "date" in train.columns and self.time_decay > 0:
            t_days = (train["date"].max() - train["date"]).dt.days.clip(lower=0)
            decay = np.exp(-self.time_decay * t_days / 365.0)
            sample_weight = decay.to_numpy()
        else:
            sample_weight = None

        def _callbacks():
            if val is not None:
                return [lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=0)]
            return None

        self._clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=3,
            class_weight={0: 1, 1: self.draw_weight, 2: 1},
            max_cat_to_onehot=32, **self.params
        )
        if val is not None:
            Xv = self._X(val)
            yv = val["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
            self._clf.fit(X, y, eval_set=[(Xv, yv)], callbacks=_callbacks(),
                          categorical_feature=self.cat_features,
                          sample_weight=sample_weight)
        else:
            self._clf.fit(X, y, categorical_feature=self.cat_features,
                         sample_weight=sample_weight)

        y_btts = train["btts"].to_numpy()
        self._clf_btts = lgb.LGBMClassifier(objective="binary", **self.params)
        if val is not None:
            yv_btts = val["btts"].to_numpy()
            self._clf_btts.fit(X, y_btts, eval_set=[(Xv, yv_btts)], callbacks=_callbacks(),
                               categorical_feature=self.cat_features,
                               sample_weight=sample_weight)
        else:
            self._clf_btts.fit(X, y_btts, categorical_feature=self.cat_features,
                              sample_weight=sample_weight)

        y_over25 = train["over25"].to_numpy()
        self._clf_over25 = lgb.LGBMClassifier(objective="binary", **self.params)
        if val is not None:
            yv_over25 = val["over25"].to_numpy()
            self._clf_over25.fit(X, y_over25, eval_set=[(Xv, yv_over25)], callbacks=_callbacks(),
                                 categorical_feature=self.cat_features,
                                 sample_weight=sample_weight)
        else:
            self._clf_over25.fit(X, y_over25, categorical_feature=self.cat_features,
                                sample_weight=sample_weight)

        y_hg = train["home_goals"].to_numpy()
        self._reg_home = lgb.LGBMRegressor(objective="poisson", **self.params)
        if val is not None:
            yv_hg = val["home_goals"].to_numpy()
            self._reg_home.fit(X, y_hg, eval_set=[(Xv, yv_hg)], callbacks=_callbacks(),
                               categorical_feature=self.cat_features,
                               sample_weight=sample_weight)
        else:
            self._reg_home.fit(X, y_hg, categorical_feature=self.cat_features,
                              sample_weight=sample_weight)

        y_ag = train["away_goals"].to_numpy()
        self._reg_away = lgb.LGBMRegressor(objective="poisson", **self.params)
        if val is not None:
            yv_ag = val["away_goals"].to_numpy()
            self._reg_away.fit(X, y_ag, eval_set=[(Xv, yv_ag)], callbacks=_callbacks(),
                               categorical_feature=self.cat_features,
                               sample_weight=sample_weight)
        else:
            self._reg_away.fit(X, y_ag, categorical_feature=self.cat_features,
                              sample_weight=sample_weight)
        return self

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        if self._clf is None:
            raise RuntimeError("Model henuz fit edilmedi.")
        X = self._X(feats)
        p1x2 = self._clf.predict_proba(X)
        return PredictionResult(
            home_win=p1x2[:, 0],
            draw=p1x2[:, 1],
            away_win=p1x2[:, 2],
            home_lambda=np.maximum(self._reg_home.predict(X), 0.05),
            away_lambda=np.maximum(self._reg_away.predict(X), 0.05),
            btts_yes=self._clf_btts.predict_proba(X)[:, 1],
            over25=self._clf_over25.predict_proba(X)[:, 1],
        )