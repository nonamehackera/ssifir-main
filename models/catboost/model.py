"""CatBoost model family (ROADMAP bolum 30, 35-36).

Leakage-safe tabular models:
  1X2 classifier      (target: result H/D/A)
  BTTS classifier     (target: both teams scored)
  Over 2.5 classifier (target: total >= 3)

Feature selection is explicit: only pre-match information. Post-match
columns (goals, shots, HT goals, results, market *closing* implied probs
outside with_odds mode) are excluded, per ROADMAP bolum 45.

with_odds=True trains Model B (odds-aware); with_odds=False is Model A
(pure football signal, ROADMAP bolum 17).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from models.base import PredictionResult

logger = logging.getLogger(__name__)

CAT_COLS = ["league", "home_team_id", "away_team_id", "season"]
ODDS_FEATURES = [
    "mkt_home_prob", "mkt_draw_prob", "mkt_away_prob", "mkt_over25_prob",
    "avg_home_odds", "avg_draw_odds", "avg_away_odds", "avg_over25_odds",
    "mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob", "mkt_close_over25_prob",
    "avg_close_home_odds", "avg_close_draw_odds", "avg_close_away_odds", "avg_close_over25_odds",
]

# Load unified team ID mapping for proper team identification
try:
    _TEAM_ID_MAP = json.load(open("data/gold/team_id_unified.json", encoding="utf-8"))
except Exception:
    _TEAM_ID_MAP = {}


def feature_columns(with_odds: bool = False) -> list[str]:
    cols = CAT_COLS + NUMERIC_FEATURES
    if with_odds:
        cols = cols + ODDS_FEATURES
    return cols


NUMERIC_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_gf_5", "home_ga_5", "home_pts_5", "home_shots_5", "home_sot_5",
    "home_corners_5", "home_w_gf", "home_w_ga", "home_w_shots", "home_w_sot",
    "home_w_corners",
    "home_hgf_5", "home_hga_5", "home_hpts_5",
    "home_gf_3", "home_ga_3", "home_pts_3",
    "home_gf_8", "home_ga_8",
    "home_gf_20", "home_ga_20",
    "home_gf_std", "home_ga_std", "home_pts_std",
    "away_gf_5", "away_ga_5", "away_pts_5", "away_shots_5", "away_sot_5",
    "away_corners_5", "away_w_gf", "away_w_ga", "away_w_shots", "away_w_sot",
    "away_w_corners",
    "away_agf_5", "away_aga_5", "away_apts_5",
    "away_gf_3", "away_ga_3", "away_pts_3",
    "away_gf_8", "away_ga_8",
    "away_gf_20", "away_ga_20",
    "away_gf_std", "away_ga_std", "away_pts_std",
    "home_attack_elo", "home_defence_elo", "away_attack_elo", "away_defence_elo",
    "attack_elo_diff", "defence_elo_diff",
    "home_rest_days", "away_rest_days",
    "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts",
    "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg", "lg_draw_rate",
    "lg_btts_rate", "lg_over25_rate", "lg_corner_avg", "lg_card_avg",
    "data_completeness",
    "home_xg", "away_xg", "total_xg_real", "xg_diff_real",
    "home_score_prob", "away_score_prob", "btts_xprob", "over25_xprob",
    "xg_diff_abs", "draw_xprob",
    # NEW: momentum, streaks, form consistency
    "home_momentum", "away_momentum",
    "home_wins_last5", "home_draws_last5",
    "away_wins_last5", "away_draws_last5",
    "home_form_std", "away_form_std",
    "home_gdiff5", "away_gdiff5",
    # NEW: home/away split form (real)
    "home_gf_5_real", "home_ga_5_real",
    "away_gf_5_real", "away_ga_5_real",
]


class CatBoostModel:
    name = "catboost"

    def __init__(
            self,
            with_odds: bool = False,
            iterations: int = 1000,
            learning_rate: float = 0.03,
            depth: int = 7,
            l2_leaf_reg: float = 8.0,
            border_count: int = 64,
            random_strength: float = 2.0,
            bagging_temperature: float = 1.0,
            random_seed: int = 42,
            verbose: bool = False,
            draw_weight: float = 1.0,
            btts_scale_pos_weight: float | None = None,
            time_decay: float = 1.0,
    ) -> None:
        self.with_odds = with_odds
        self.draw_weight = draw_weight
        self.time_decay = time_decay
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "border_count": border_count,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "random_seed": random_seed,
            "verbose": verbose,
            "allow_writing_files": False,
        }
        self.btts_scale_pos_weight = btts_scale_pos_weight
        self.feature_cols = feature_columns(with_odds)
        self.cat_indices = [self.feature_cols.index(c) for c in CAT_COLS if c in self.feature_cols]
        self._clf_1x2 = None
        self._clf_btts = None
        self._clf_over15 = None
        self._clf_over25 = None
        self._clf_over35 = None
        self._reg_home = None
        self._reg_away = None

    def _X(self, feats: pd.DataFrame) -> pd.DataFrame:
        return feats[self.feature_cols].copy()

    def fit(self, train, val=None, sample_weight=None) -> "CatBoostModel":
        from catboost import CatBoostClassifier, CatBoostRegressor

        # TIME DECAY: yeni maclar daha onemli
        if "date" in train.columns and self.time_decay > 0:
            t_days = (train["date"].max() - train["date"]).dt.days.clip(lower=0)
            decay = np.exp(-self.time_decay * t_days / 365.0)
            base_w = decay.to_numpy()
        else:
            base_w = np.ones(len(train))

        X = self._X(train)
        y_result = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        y_btts = train["btts"].to_numpy()
        y_over15 = train["over15"].to_numpy() if "over15" in train.columns else (train["home_goals"] + train["away_goals"] > 1.5).astype(int).to_numpy()
        y_over25 = train["over25"].to_numpy()
        y_over35 = train["over35"].to_numpy() if "over35" in train.columns else (train["home_goals"] + train["away_goals"] > 3.5).astype(int).to_numpy()
        y_hg = train["home_goals"].to_numpy()
        y_ag = train["away_goals"].to_numpy()

        # Draw'a ek agirlik (class imbalance)
        draw_mask = (y_result == 1)
        sw_1x2 = base_w.copy()
        sw_1x2[draw_mask] *= self.draw_weight

        # Diger modeller icin de time decay uygula
        sw_bin = base_w.copy()

        # BTTS pozitif sınıf ağırlığı (class balance)
        if self.btts_scale_pos_weight is None:
            n_pos = int(np.sum(y_btts))
            n_neg = len(y_btts) - n_pos
            if n_pos > 0 and n_neg > 0:
                self.btts_scale_pos_weight = n_neg / n_pos
            else:
                self.btts_scale_pos_weight = 1.0

        Xv = self._X(val) if val is not None else None
        eval_1x2 = (
            (Xv, val["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy())
            if val is not None
            else None
        )
        eval_btts = (Xv, val["btts"].to_numpy()) if val is not None else None
        eval_over = (Xv, val["over25"].to_numpy()) if val is not None else None
        eval_hg = (Xv, val["home_goals"].to_numpy()) if val is not None else None
        eval_ag = (Xv, val["away_goals"].to_numpy()) if val is not None else None

        early_stop_rounds = 80 if val is not None else None

        # Auto-retrain: weight mispredicted rows higher (sample_weight)
        if sample_weight is not None:
            sw_1x2 = sw_1x2 * sample_weight
            sw_bin = sw_bin * sample_weight

        self._clf_1x2 = CatBoostClassifier(
            loss_function="MultiClass", cat_features=self.cat_indices,
            class_weights={0: 1.0, 1: 4.0, 2: 1.3},
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._clf_1x2.fit(
            X, y_result, eval_set=eval_1x2, use_best_model=val is not None, plot=False,
            sample_weight=sw_1x2,
        )

        self._clf_btts = CatBoostClassifier(
            loss_function="Logloss", cat_features=self.cat_indices,
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._clf_btts.fit(X, y_btts, eval_set=eval_btts, use_best_model=val is not None, plot=False,
                           sample_weight=sw_bin, scale_pos_weight=self.btts_scale_pos_weight)

        self._clf_over25 = CatBoostClassifier(
            loss_function="Logloss", cat_features=self.cat_indices,
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._clf_over25.fit(X, y_over25, eval_set=eval_over, use_best_model=val is not None, plot=False,
                             sample_weight=sw_bin)

        self._clf_over15 = CatBoostClassifier(
            loss_function="Logloss", cat_features=self.cat_indices,
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._clf_over15.fit(X, y_over15, eval_set=eval_over, use_best_model=val is not None, plot=False,
                             sample_weight=sw_bin)

        self._clf_over35 = CatBoostClassifier(
            loss_function="Logloss", cat_features=self.cat_indices,
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._clf_over35.fit(X, y_over35, eval_set=eval_over, use_best_model=val is not None, plot=False,
                             sample_weight=sw_bin)

        self._reg_home = CatBoostRegressor(
            loss_function="Poisson", cat_features=self.cat_indices,
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._reg_home.fit(X, y_hg, eval_set=eval_hg, use_best_model=val is not None, plot=False,
                           sample_weight=sw_bin)

        self._reg_away = CatBoostRegressor(
            loss_function="Poisson", cat_features=self.cat_indices,
            early_stopping_rounds=early_stop_rounds, **self.params
        )
        self._reg_away.fit(X, y_ag, eval_set=eval_ag, use_best_model=val is not None, plot=False,
                           sample_weight=sw_bin)

        logger.info(
            "CatBoost(%s) fit: %d satir, %d feature, odds=%s",
            self.name, len(X), len(self.feature_cols), self.with_odds,
        )
        return self

    def predict(self, feats: pd.DataFrame) -> PredictionResult:
        if self._clf_1x2 is None:
            raise RuntimeError("Model henuz fit edilmedi.")
        X = self._X(feats)
        p1x2 = self._clf_1x2.predict_proba(X)
        btts = self._clf_btts.predict_proba(X)[:, 1]
        over15 = self._clf_over15.predict_proba(X)[:, 1]
        over25 = self._clf_over25.predict_proba(X)[:, 1]
        over35 = self._clf_over35.predict_proba(X)[:, 1]
        lam_h = self._reg_home.predict(X)
        lam_a = self._reg_away.predict(X)
        return PredictionResult(
            home_win=p1x2[:, 0],
            draw=p1x2[:, 1],
            away_win=p1x2[:, 2],
            home_lambda=np.maximum(lam_h, 0.05),
            away_lambda=np.maximum(lam_a, 0.05),
            btts_yes=btts,
            over15=over15,
            over25=over25,
            over35=over35,
        )