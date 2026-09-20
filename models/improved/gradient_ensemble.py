"""GUCLU GRADIENT ENSEMBLE (LightGBM + XGBoost + CatBoost).

Tek CatBoost %55-57 veriyordu. 3 gradient boosting'i birlestirip
voting ile accuracy'yi %60+'a cikarmayi hedefliyoruz.

Her model:
  - Ayni SELECTED_FEATURES kullanir
  - Time-decay + draw_weight tune (ImprovedMatchModel'dan alinir)
  - Farkli algoritma oldugu icin ensemble'da complementarity olur
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.improved.model import SELECTED_FEATURES, ImprovedMatchModel

logger = logging.getLogger(__name__)

CAT_COLS = ["league", "season"]


class GradientEnsemble:
    name = "gradient_ensemble"

    def __init__(
        self,
        iterations: int = 1500,
        n_models: int = 2,  # her algoritma icin voting
        time_decay: float = 1.0,
        draw_weight_grid: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
        verbose: bool = False,
    ) -> None:
        self.iterations = iterations
        self.n_models = n_models
        self.time_decay = time_decay
        self.draw_weight_grid = draw_weight_grid
        self.verbose = verbose
        self._models: list = []
        self.feature_cols: list = []
        self.best_draw_weight = 2.0

    def _X(self, feats):
        cols = [c for c in self.feature_cols if c in feats.columns]
        return feats[cols].copy()

    def _sample_weight(self, train, y, base_w):
        draw_mask = (y == 1)
        sw = base_w.copy()
        sw[draw_mask] = base_w[draw_mask] * self.best_draw_weight
        return sw

    def _time_decay_w(self, train):
        if "date" in train.columns and self.time_decay > 0:
            t_days = (train["date"].max() - train["date"]).dt.days.clip(lower=0)
            return np.exp(-self.time_decay * t_days / 365.0).to_numpy()
        return np.ones(len(train))

    def fit(self, train, val=None) -> "GradientEnsemble":
        from evaluation.metrics import log_loss_1x2

        self.feature_cols = [c for c in SELECTED_FEATURES if c in train.columns]
        self._cat_indices = [self.feature_cols.index(c) for c in CAT_COLS if c in self.feature_cols]

        X = self._X(train)
        y = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        val_X = self._X(val) if val is not None else None
        val_y = val["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy() if val is not None else None
        base_w = self._time_decay_w(train)

        # draw_weight tune
        best_w, best_ll = self.draw_weight_grid[0], float("inf")
        for w in self.draw_weight_grid:
            sw = base_w.copy()
            sw[y == 1] = base_w[y == 1] * w
            clf = self._train_one(X, y, val_X, val_y, sw, seed=42, algo="lgbm")
            p = clf.predict_proba(X if val_X is None else val_X)
            ll = log_loss_1x2(y if val_X is None else val_y, p[:, 0], p[:, 1], p[:, 2])
            if ll < best_ll:
                best_ll, best_w = ll, w
        self.best_draw_weight = best_w
        logger.info("GradientEnsemble draw_weight tune: %s (ll=%.4f)", best_w, best_ll)

        # final: her algoritma n_models kez farkli seed
        self._models = []
        for algo in ["lgbm", "xgb", "cat"]:
            for i in range(self.n_models):
                sw = base_w.copy()
                sw[y == 1] = base_w[y == 1] * self.best_draw_weight
                clf = self._train_one(X, y, val_X, val_y, sw, seed=100 + i * 7, algo=algo)
                self._models.append((algo, clf))
        return self

    def _train_one(self, X, y, val_X, val_y, sw, seed, algo):
        if algo == "lgbm":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=self.iterations,
                learning_rate=0.03, max_depth=6, num_leaves=31,
                min_child_samples=20, reg_lambda=8.0, subsample=0.8,
                colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
            )
            if val_X is not None:
                clf.fit(X, y, sample_weight=sw, eval_set=[(val_X, val_y)],
                        eval_metric="multi_logloss", callbacks=[])
            else:
                clf.fit(X, y, sample_weight=sw)
            return clf
        elif algo == "xgb":
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                objective="multi:softprob", num_class=3, n_estimators=self.iterations,
                learning_rate=0.03, max_depth=6, reg_lambda=8.0,
                subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                early_stopping_rounds=80, verbosity=0,
            )
            if val_X is not None:
                clf.fit(X, y, sample_weight=sw, eval_set=[(val_X, val_y)],
                        verbose=False)
            else:
                clf.fit(X, y, sample_weight=sw, verbose=False)
            return clf
        else:  # cat
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier(
                loss_function="MultiClass", cat_features=self._cat_indices,
                iterations=self.iterations, learning_rate=0.03, depth=6,
                l2_leaf_reg=8.0, random_strength=1.5, bagging_temperature=1.0,
                random_seed=seed, verbose=False, allow_writing_files=False,
            )
            if val_X is not None:
                clf.fit(X, y, eval_set=(val_X, val_y), use_best_model=True,
                        early_stopping_rounds=80, plot=False, sample_weight=sw)
            else:
                clf.fit(X, y, plot=False, sample_weight=sw)
            return clf

    def predict(self, feats):
        from models.base import PredictionResult
        X = self._X(feats)
        probas = []
        for algo, clf in self._models:
            probas.append(clf.predict_proba(X))
        p = np.mean(probas, axis=0)
        return PredictionResult(home_win=p[:, 0], draw=p[:, 1], away_win=p[:, 2])
