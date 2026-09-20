"""Eksik feature tahmin modelleri.

Football-Data.co.uk'de: shot, sot, corner, kart, foul verileri mevcut.
Hugging Face / TheSportsDB: sadece sonuc (gol) var.

Bu modeller FD verisinde egitilir, sonra eksik olan HF/TSDB maclarina
uygulanarak feature seti zenginlestirilir. Point-in-time uyumlu:
sadece mac oncesi bilgi kullanir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StatsPrediction:
    home_shots: np.ndarray | None = None
    away_shots: np.ndarray | None = None
    home_sot: np.ndarray | None = None
    away_sot: np.ndarray | None = None
    home_corners: np.ndarray | None = None
    away_corners: np.ndarray | None = None
    home_yellow: np.ndarray | None = None
    away_yellow: np.ndarray | None = None
    home_red: np.ndarray | None = None
    away_red: np.ndarray | None = None
    home_fouls: np.ndarray | None = None
    away_fouls: np.ndarray | None = None


class MissingStatsPredictor:
    """Eksik istatistiksel feature'lari tahmin eden model.

    FD verisinde egitilen Poisson regresorlar kullanir.
    Input: elo, form, gol beklentisi gibi match-oncesi feature'lar.
    Output: shot, sot, corner, kart tahminleri.
    """

    FEATURE_COLS = [
        "home_elo", "away_elo", "elo_diff",
        "home_gf_5", "home_ga_5", "home_pts_5",
        "away_gf_5", "away_ga_5", "away_pts_5",
        "home_shots_5", "away_shots_5",
        "home_sot_5", "away_sot_5",
        "home_corners_5", "away_corners_5",
        "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
        "lg_corner_avg", "lg_card_avg",
        "data_completeness",
        "home_attack_elo", "home_defence_elo",
        "away_attack_elo", "away_defence_elo",
        "attack_elo_diff", "defence_elo_diff",
        "home_rest_days", "away_rest_days",
        "home_gf_3", "home_ga_3",
        "away_gf_3", "away_ga_3",
        "home_momentum", "away_momentum",
        "home_wins_last5", "away_wins_last5",
        "mkt_home_prob", "mkt_away_prob",
    ]

    TARGET_COLS = [
        "home_shots", "away_shots",
        "home_sot", "away_sot",
        "home_corners", "away_corners",
        "home_yellow", "away_yellow",
        "home_red", "away_red",
        "home_fouls", "away_fouls",
    ]

    def __init__(self):
        self._models: dict[str, object] = {}
        self._available_targets: list[str] = []

    def fit(self, feats: pd.DataFrame) -> "MissingStatsPredictor":
        """FD verisinde egit (sadece istatistigi olan maclar)."""
        from lightgbm import LGBMRegressor

        available_feats = [c for c in self.FEATURE_COLS if c in feats.columns]
        available_targets = [c for c in self.TARGET_COLS if c in feats.columns]

        if not available_targets:
            logger.warning("Stats predictor: hicbir target colonu bulunamadi")
            return self

        # Istatistigi olmayan maclari atla
        any_stats = feats[available_targets].notna().any(axis=1)
        train_df = feats[any_stats].copy()

        if len(train_df) < 1000:
            logger.warning("Istatistik egitim verisi cok az: %d", len(train_df))
            return self

        logger.info("Stats predictor egitimi: %d ornek ile", len(train_df))

        X = train_df[available_feats].apply(pd.to_numeric, errors="coerce").fillna(0)

        for target in available_targets:
            if target not in feats.columns:
                continue
            y_raw = train_df[target]
            if y_raw.notna().sum() < 500:
                continue

            # Eksik degerleri ortalama ile doldur
            y = y_raw.fillna(y_raw.mean())

            model = LGBMRegressor(
                objective="poisson" if "red" not in target else "regression",
                n_estimators=400,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=2.0,
                verbose=-1,
                random_seed=42,
            )
            model.fit(X, y)
            self._models[target] = model
            self._available_targets.append(target)

        logger.info("Stats predictor: %d model egitildi: %s",
                     len(self._models), list(self._models.keys()))
        return self

    def predict(self, feats: pd.DataFrame) -> StatsPrediction:
        """Eksik feature'lari tahmin et."""
        if not self._models:
            return StatsPrediction()

        available_feats = [c for c in self.FEATURE_COLS if c in feats.columns]
        X = feats[available_feats].apply(pd.to_numeric, errors="coerce").fillna(0)

        result = StatsPrediction()
        for target, model in self._models.items():
            pred = np.maximum(model.predict(X), 0)
            # Tam sayiya yuvarla ( Poisson )
            if "red" not in target:
                pred = np.round(pred).astype(int)
            else:
                pred = np.clip(np.round(pred).astype(int), 0, 3)

            setattr(result, target, pred)

        return result

    def fill_missing(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Eksik istatistikleri tahminle ve doldur.

        Tum kaynaklarda (FD, HF, XG, OF) eksik istatistikleri doldurur.
        Model egitiminde gercek istatistik kullanan kaynaklar tercih edilir.
        """
        result = self.predict(feats)
        df = feats.copy()

        filled_count = 0
        available_targets = [c for c in self.TARGET_COLS if c in df.columns]
        for target in available_targets:
            pred_values = getattr(result, target, None)
            if pred_values is None or target not in df.columns:
                continue

            missing_mask = df[target].isna()
            if missing_mask.sum() == 0:
                continue

            df.loc[missing_mask, target] = pred_values[missing_mask.values]
            filled_count += missing_mask.sum()

        if filled_count > 0:
            logger.info("Stats predictor: %d eksik deger dolduruldu (tum kaynaklar)", filled_count)

        return df


def build_enriched_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Eksik istatistikleri tahmin ederek mac verisini zenginlestir.

    Istatistik verisi olan tum kaynaklarda (FD, XG, HF stats'li maclar) egitilir.
    Sonra tum kaynaklardaki eksik istatistikleri tahmin ederek doldurur.
    """
    predictor = MissingStatsPredictor()

    # Istatistik verisi olan tum kaynaklarda egit (FD, XG, HF stats'li)
    any_stats_cols = ["home_shots", "away_shots", "home_sot", "away_sot"]
    existing_stats_cols = [c for c in any_stats_cols if c in matches.columns]
    if existing_stats_cols:
        has_stats = matches[existing_stats_cols].notna().any(axis=1)
        train_data = matches[has_stats].copy()
    else:
        train_data = matches.copy()

    if len(train_data) > 1000:
        predictor.fit(train_data)

    enriched = predictor.fill_missing(matches)

    # data_completeness guncelle
    stat_cols = ["home_shots", "away_shots", "home_sot", "away_sot",
                 "home_corners", "away_corners", "home_yellow", "away_yellow"]
    if "data_completeness" in enriched.columns:
        stat_completeness = enriched[stat_cols].notna().mean(axis=1)
        enriched["data_completeness"] = enriched[["data_completeness"]].join(
            stat_completeness.rename("_stat_comp")
        ).mean(axis=1)

    return enriched
