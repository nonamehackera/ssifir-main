"""Feedback Loop Integration - Mevcut tahmin sistemine geri besleme döngüsü entegre eder.

Bu modül, mevcut predict.py ve API ile kullanılmak üzere tasarlanmıştır.
Tahminleri kaydeder, hata analizi yapar ve düzeltme önerileri sunar.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from feedback.orchestrator import FeedbackOrchestrator

logger = logging.getLogger(__name__)


class FeedbackIntegration:
    """Mevcut tahmin sistemine geri besleme entegrasyonu."""

    def __init__(self):
        self.orchestrator = FeedbackOrchestrator(
            data_dir="data/feedback",
            auto_correct=True,
            min_predictions_for_correction=50,
        )
        self._initialized = True

    def record_prediction(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        actual_result: str,
        home_goals: int = 0,
        away_goals: int = 0,
        model_name: str = "LGB+XGB v6",
    ):
        """Tahmin sonucunu kaydet."""
        self.orchestrator.record_prediction(
            match_id=match_id,
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual_result,
            league=league,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            model_name=model_name,
        )

    def get_team_correction(self, team: str) -> dict:
        """Takım için düzeltme önerileri al."""
        features = self.orchestrator.feature_generator.generate_team_features(team)
        return features

    def get_league_correction(self, league: str) -> dict:
        """Lig için düzeltme önerileri al."""
        features = self.orchestrator.feature_generator.generate_league_features(league)
        return features

    def run_feedback_cycle(
        self,
        current_weights: Optional[np.ndarray] = None,
        current_temperature: float = 1.0,
    ) -> dict:
        """Geri besleme döngüsü çalıştır."""
        result = self.orchestrator.run_cycle(current_weights, current_temperature)
        return {
            "cycle_id": result.cycle_id,
            "patterns_detected": result.patterns_detected,
            "correction_applied": result.correction_applied,
            "improvement": result.improvement,
        }

    def get_report(self) -> str:
        """Tam rapor al."""
        return self.orchestrator.generate_report()

    def get_adjusted_prediction(
        self,
        home_team: str,
        away_team: str,
        league: str,
        raw_prediction: dict,
    ) -> dict:
        """Ham tahmini geri besleme ile ayarla."""
        # Takım hata özelliklerini al
        home_features = self.orchestrator.feature_generator.generate_team_features(home_team)
        away_features = self.orchestrator.feature_generator.generate_team_features(away_team)

        # Hata oranlarını hesapla
        home_error = home_features.get(f"{home_team}_error_rate", 0.5)
        away_error = away_features.get(f"{away_team}_error_rate", 0.5)

        # Güven skorunu hesapla
        confidence_factor = 1.0 - (home_error + away_error) / 2

        # Tahmini ayarla
        adjusted = raw_prediction.copy()

        # Düşük güvenli tahminlerde daha temkinli ol
        if confidence_factor < 0.6:
            # Olasılıkları daha dengeli hale getir
            probs = [raw_prediction["home_win"], raw_prediction["draw"], raw_prediction["away_win"]]
            max_prob = max(probs)
            if max_prob > 0.7:
                # Yüksek olasılıkları biraz azalt
                adjustment = (max_prob - 0.7) * 0.3
                probs[probs.index(max_prob)] -= adjustment
                # Diğerlerini orantılı olarak artır
                other_probs = [p for p in probs if p != max_prob]
                if other_probs:
                    increase = adjustment / len(other_probs)
                    for i in range(len(probs)):
                        if probs[i] != max_prob:
                            probs[i] += increase

                # Normalize et
                total = sum(probs)
                probs = [p / total for p in probs]

                adjusted["home_win"] = probs[0]
                adjusted["draw"] = probs[1]
                adjusted["away_win"] = probs[2]

        # Güven bilgisini ekle
        adjusted["confidence_factor"] = confidence_factor
        adjusted["error_rates"] = {
            "home": home_error,
            "away": away_error,
        }

        return adjusted


# Global örnek
_feedback = None


def get_feedback() -> FeedbackIntegration:
    """Global geri besleme örneğini al."""
    global _feedback
    if _feedback is None:
        _feedback = FeedbackIntegration()
    return _feedback
