"""Feedback v4 - Takim Bazli Hata Ogrenme.

GERCEK MANTIK:
  Mac bitti, sonuc belli. Model ne dedi, gercek ne oldu?
  "Galatasaray deplasmanda favori gosterildi ama kaybetti"
  -> Bir daha Galatasaray deplasman tahmininde dikkatli ol

  HER TAKIM ICIN:
  - Ev sahibi iken kac kez kazandi/kaybetti/berabere kaldi?
  - Deplasman iken kac kez kazandi/kaybetti/berabere kaldi?
  - Model ne sIKLIKLA tahmin etti vs gercek sonuc ne oldu?

  NE ZAMAN DUZELTME YAPAR:
  - Takim icin yeterli ornek varsa (min 5 mac)
  - Modelin tahmini ile gercek arasinda buyuk fark varsa
  - Duzeltme KUCUK olmali (max %5)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FeedbackAdjustment:
    original_home: float
    original_draw: float
    original_away: float
    adjusted_home: float
    adjusted_draw: float
    adjusted_away: float
    adjustments_applied: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    confidence_factor: float = 1.0


class FeedbackPredictorV4:
    """Takim bazli hata ogrenen feedback predictor."""

    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_file = self.data_dir / "team_profiles_v4.json"
        self.profiles = self._load()

        # Parametreler
        self.min_samples = 5        # Minimum 5 mac
        self.max_adjustment = 0.05  # Maks %5 duzeltme
        self.learning_rate = 0.3    # Yavas ogrenme

    def _load(self) -> dict:
        if self.profile_file.exists():
            try:
                with open(self.profile_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save(self):
        try:
            with open(self.profile_file, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, indent=2, ensure_ascii=False)
        except:
            pass

    def _ensure_team(self, team_id: str):
        if team_id not in self.profiles:
            self.profiles[team_id] = {
                "home": {"H": 0, "D": 0, "A": 0, "total": 0},
                "away": {"H": 0, "D": 0, "A": 0, "total": 0},
                "predictions": {"H": 0, "D": 0, "A": 0},
                "correct_predictions": {"H": 0, "D": 0, "A": 0},
            }

    def record_match(
        self,
        home_team_id: int,
        away_team_id: int,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        actual_result: str,
        elo_diff: float = 0.0,
    ):
        """Mac sonucunu kaydet - her iki takim icin de guncelle."""
        home_key = str(home_team_id)
        away_key = str(away_team_id)

        # Ev sahibi takim
        self._ensure_team(home_key)
        h = self.profiles[home_key]
        h["home"][actual_result] += 1
        h["home"]["total"] += 1

        # Tahmin bilgisi
        pred_result = ["H", "D", "A"][np.argmax([pred_home, pred_draw, pred_away])]
        h["predictions"][pred_result] += 1
        if pred_result == actual_result:
            h["correct_predictions"][pred_result] += 1

        # Deplasman takimi
        self._ensure_team(away_key)
        a = self.profiles[away_key]
        a["away"][actual_result] += 1
        a["away"]["total"] += 1

        a_pred_result = ["H", "D", "A"][np.argmax([pred_home, pred_draw, pred_away])]
        # Deplasman takimi icin sonucu tersine cevir
        reverse_map = {"H": "A", "D": "D", "A": "H"}
        a_pred_for_away = reverse_map[pred_result]
        a_actual_for_away = reverse_map[actual_result]
        a["predictions"][a_pred_for_away] += 1
        if a_pred_for_away == a_actual_for_away:
            a["correct_predictions"][a_pred_for_away] += 1

        self._save()

    def adjust_prediction(
        self,
        home_team_id: int,
        away_team_id: int,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        elo_diff: float = 0.0,
        **kwargs,
    ) -> FeedbackAdjustment:
        """Tahmini ayarla - takim bazli hata profillerini kullanarak."""
        home_key = str(home_team_id)
        away_key = str(away_team_id)

        adj_home = pred_home
        adj_draw = pred_draw
        adj_away = pred_away
        reasons = []
        adjustments = []

        # Ev sahibi takim profilini kontrol et
        if home_key in self.profiles:
            h = self.profiles[home_key]
            home_stats = h["home"]

            if home_stats["total"] >= self.min_samples:
                # Ev sahibi takimin ev sahibi performansi
                actual_h_rate = home_stats["H"] / home_stats["total"]
                actual_d_rate = home_stats["D"] / home_stats["total"]
                actual_a_rate = home_stats["A"] / home_stats["total"]

                # Model "H" diyor ama takim surekli mi kaybediyor?
                if actual_h_rate < 0.30 and pred_home > 0.45:
                    # Bu takim evde bile kaybediyor, H fazla yuksek
                    correction = (pred_home - actual_h_rate) * self.learning_rate
                    correction = min(correction, self.max_adjustment)
                    adj_home -= correction
                    adj_draw += correction * 0.5
                    adj_away += correction * 0.5
                    reasons.append(f"Ev sahibi {home_key}: evde kazanma={actual_h_rate:.2f} ama model {pred_home:.2f} diyor")
                    adjustments.append("home_team_home_bias")

                # Beraberlik orani yuksek mi?
                if actual_d_rate > 0.40 and pred_draw < 0.25:
                    # Bu takim cok berabere kaliyor, D dusuk
                    correction = (actual_d_rate - pred_draw) * self.learning_rate
                    correction = min(correction, self.max_adjustment)
                    adj_draw += correction
                    adj_home -= correction * 0.5
                    adj_away -= correction * 0.5
                    reasons.append(f"Ev sahibi {home_key}: beraberlik={actual_d_rate:.2f} ama model {pred_draw:.2f} diyor")
                    adjustments.append("home_team_draw_bias")

        # Deplasman takim profilini kontrol et
        if away_key in self.profiles:
            a = self.profiles[away_key]
            away_stats = a["away"]

            if away_stats["total"] >= self.min_samples:
                actual_h_rate = away_stats["H"] / away_stats["total"]
                actual_a_rate = away_stats["A"] / away_stats["total"]

                # Model "A" diyor ama deplasman takimi hic kazanamıyor mu?
                if actual_a_rate < 0.20 and pred_away > 0.40:
                    correction = (pred_away - actual_a_rate) * self.learning_rate
                    correction = min(correction, self.max_adjustment)
                    adj_away -= correction
                    adj_home += correction * 0.5
                    adj_draw += correction * 0.5
                    reasons.append(f"Deplasman {away_key}: deplasmanda kazanma={actual_a_rate:.2f} ama model {pred_away:.2f} diyor")
                    adjustments.append("away_team_away_bias")

                # Deplasman takimi cok beraberlik mi yapiyor?
                actual_d_rate = away_stats["D"] / away_stats["total"]
                if actual_d_rate > 0.40 and pred_draw < 0.25:
                    correction = (actual_d_rate - pred_draw) * self.learning_rate
                    correction = min(correction, self.max_adjustment)
                    adj_draw += correction
                    adj_home -= correction * 0.5
                    adj_away -= correction * 0.5
                    reasons.append(f"Deplasman {away_key}: beraberlik={actual_d_rate:.2f} ama model {pred_draw:.2f} diyor")
                    adjustments.append("away_team_draw_bias")

        # Normalize et
        total = adj_home + adj_draw + adj_away
        if total > 0:
            adj_home /= total
            adj_draw /= total
            adj_away /= total

        return FeedbackAdjustment(
            original_home=pred_home,
            original_draw=pred_draw,
            original_away=pred_away,
            adjusted_home=adj_home,
            adjusted_draw=adj_draw,
            adjusted_away=adj_away,
            adjustments_applied=adjustments,
            reasons=reasons,
        )

    def get_team_summary(self, team_id: int) -> dict:
        """Takim ozeti."""
        key = str(team_id)
        if key not in self.profiles:
            return {"status": "no_data"}
        return self.profiles[key]

    def get_stats(self) -> dict:
        """Genel istatistikler."""
        total_teams = len(self.profiles)
        total_matches = sum(
            p["home"]["total"] + p["away"]["total"]
            for p in self.profiles.values()
        )
        return {
            "total_teams": total_teams,
            "total_matches": total_matches,
        }
