"""Iyilestirilmis mac modeli (ROADMAP bolum 30+).

Model neden 'aptal saptal' calisiyordu:
  1. 89 feature'da boguluyordu (cogu gurultu: lg_card_avg, w_corners...)
  2. draw_weight=1.2 rastgele sabitti (draw %27 ama model yaklasmiyordu)
  3. Ensemble esit agirlikla basliyordu

Cozum:
  - LEAKAGE-FREE guclu feature alt kumesi (elo_diff, form, xG, H2H, opp)
  - draw_weight validation grid ile optimize (class imbalance cozumu)
  - CatBoost MultiClass + erken durdurma (overfit önleme)
  - (opsiyonel) farkli seed'lerde birkac model -> voting

NOT: with_odds=False (bookmaker kopyalamaz, saf sinyal).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.base import PredictionResult

logger = logging.getLogger(__name__)

# Sadece anlamli + leakage-free feature'lar. Gurultuyu attik.
SELECTED_FEATURES = [
    # Elo (en guc lu sinyal)
    "elo_diff", "home_elo", "away_elo",
    "attack_elo_diff", "defence_elo_diff",
    "home_attack_elo", "home_defence_elo",
    "away_attack_elo", "away_defence_elo",
    # Form (recency-weighted)
    "home_w_gf", "home_w_ga", "home_w_shots", "home_w_sot",
    "away_w_gf", "away_w_ga", "away_w_shots", "away_w_sot",
    "home_gf_5", "home_ga_5", "home_pts_5",
    "away_gf_5", "away_ga_5", "away_pts_5",
    "home_gf_3", "away_gf_3",
    "home_gf_8", "away_gf_8",
    "home_gf_20", "away_gf_20",
    # Home/Away split form
    "home_hgf_5", "home_hga_5", "home_hpts_5",
    "away_agf_5", "away_aga_5", "away_apts_5",
    # xG (GERCEK sinyal: sot/shots'tan turetilmis)
    "home_xg_real", "away_xg_real", "total_xg_real", "xg_diff_real",
    "shots_diff", "sot_diff",
    # H2H
    "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts",
    # Opponent strength
    "home_opp_elo", "away_opp_elo",
    # Rest
    "home_rest_days", "away_rest_days",
    # League baselines
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
]

CAT_COLS = ["league", "season"]


class ImprovedMatchModel:
    name = "improved_match"

    def __init__(
        self,
        iterations: int = 1500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 8.0,
        draw_weight_grid: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
        n_models: int = 3,  # voting icin farkli seed
        time_decay: float = 1.0,  # time-decay lambda (0=esit agirlik, 1=yeni mac 2.7x)
        verbose: bool = False,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.draw_weight_grid = draw_weight_grid
        self.n_models = n_models
        self.time_decay = time_decay
        self.verbose = verbose
        self.best_draw_weight = 2.0
        self._models: list = []
        self.feature_cols: list = []

    def _X(self, feats: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.feature_cols if c in feats.columns]
        return feats[cols].copy()

    def _train_one(self, X, y, val_X, val_y, draw_weight, seed, sample_weight=None):
        from catboost import CatBoostClassifier
        # auto_class_weights='Balanced' -> CatBoost sinif frekansina gore agirlik verir
        # (draw %27 oldugu icin draw'a otomatik agirlik). class_weights ile BIRLESTIRILEMEZ.
        clf = CatBoostClassifier(
            loss_function="MultiClass",
            cat_features=self._cat_indices,
            class_weights={0: 1.0, 1: self.best_draw_weight, 2: 1.0},
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=64,
            random_strength=1.5,
            bagging_temperature=1.0,
            random_seed=seed,
            verbose=self.verbose,
            allow_writing_files=False,
        )
        if val_X is not None:
            clf.fit(X, y, eval_set=(val_X, val_y),
                    use_best_model=True,
                    early_stopping_rounds=80, plot=False,
                    sample_weight=sample_weight)
        else:
            clf.fit(X, y, plot=False, sample_weight=sample_weight)
        return clf

    def fit(self, train, val=None, sample_weight=None) -> "ImprovedMatchModel":
        from evaluation.metrics import log_loss_1x2

        # feature kolonlarini sadece var olanlarla sinirla
        self.feature_cols = [c for c in SELECTED_FEATURES if c in train.columns]
        self._cat_indices = [self.feature_cols.index(c) for c in CAT_COLS if c in self.feature_cols]

        X = self._X(train)
        y = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        val_X = self._X(val) if val is not None else None
        val_y = val["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy() if val is not None else None

        # --- TIME DECAY: yeni maclar daha onemli ---
        # train tarihine gore exp(-lambda * gun_farki) agirligi.
        # Boylece 3 ay onceki mac, dunekinden dusuk agirlikta.
        if "date" in train.columns and self.time_decay > 0:
            t_days = (train["date"].max() - train["date"]).dt.days.clip(lower=0)
            decay = np.exp(-self.time_decay * t_days / 365.0)
            base_w = decay.to_numpy()
        else:
            base_w = np.ones(len(train))
        # draw'a ek agirlik (class imbalance)
        draw_mask = (y == 1)
        sample_weight = base_w.copy()
        sample_weight[draw_mask] *= self.draw_weight_grid[0]  # tune'da guncellenecek

        # --- draw_weight grid tune (class imbalance cozumu) ---
        # tune asamasinda sadece draw agirligini degistir, time_decay sabit kalir
        sw_base = sample_weight.copy()
        best_w, best_ll = self.draw_weight_grid[0], float("inf")
        for w in self.draw_weight_grid:
            sw = sw_base.copy()
            sw[draw_mask] = base_w[draw_mask] * w
            clf = self._train_one(X, y, val_X, val_y, w, seed=42, sample_weight=sw)
            p = clf.predict_proba(X if val_X is None else val_X)
            if val_X is None:
                ll = log_loss_1x2(y, p[:, 0], p[:, 1], p[:, 2])
            else:
                ll = log_loss_1x2(val_y, p[:, 0], p[:, 1], p[:, 2])
            if ll < best_ll:
                best_ll, best_w = ll, w
        self.best_draw_weight = best_w
        logger.info("ImprovedMatchModel draw_weight tune: en iyi=%s (ll=%.4f)", best_w, best_ll)

        # --- final: n_models farkli seed ile voting ---
        self._models = []
        for i in range(self.n_models):
            sw = base_w.copy()
            sw[draw_mask] = base_w[draw_mask] * self.best_draw_weight
            clf = self._train_one(X, y, val_X, val_y, self.best_draw_weight, seed=100 + i * 7, sample_weight=sw)
            self._models.append(clf)
        return self

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        if not self._models:
            raise RuntimeError("Model henuz fit edilmedi.")
        X = self._X(feats)
        # voting: tum modellerin ortalamasi
        probas = [m.predict_proba(X) for m in self._models]
        p = np.mean(probas, axis=0)
        return PredictionResult(
            home_win=p[:, 0], draw=p[:, 1], away_win=p[:, 2],
        )
