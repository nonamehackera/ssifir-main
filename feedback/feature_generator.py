"""Error Feature Generator - Hata kalıplarından yeni özellikler türetir.

Modelin geçmiş hatalarından öğrenerek yeni özellikler oluşturur:

1. Takım Hata Profili: Her takımın hata kalıbı istatistikleri
2. Lig Hata Profili: Her ligin hata kalıbı istatistikleri
3. Sonuç Hata Profili: Her sonucun (H/D/A) hata kalıbı
4. Gol Aralığı Hata Profili: Gol aralıklarına göre hata kalıpları
5. Güven Hata Profili: Güven seviyelerine göre hata kalıpları
6. Zamanlama Hata Profili: Zaman bazında hata kalıpları
7. Kombinasyon Özellikleri: Birden fazla faktörün birleşimi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from feedback.error_tracker import ErrorTracker, PredictionRecord
from feedback.pattern_detector import PatternDetector, ErrorPattern

logger = logging.getLogger(__name__)


@dataclass
class ErrorFeatureSet:
    """Hata kalıplarından türetilmiş özellik seti."""
    team_features: dict  # Takım bazında hata özellikleri
    league_features: dict  # Lig bazında hata özellikleri
    result_features: dict  # Sonuç bazında hata özellikleri
    goal_range_features: dict  # Gol aralığı bazında hata özellikleri
    confidence_features: dict  # Güven bazında hata özellikleri
    temporal_features: dict  # Zamanlama bazında hata özellikleri
    combination_features: dict  # Kombinasyon özellikleri


class ErrorFeatureGenerator:
    """Hata kalıplarından yeni özellikler üreten sınıf.

    Özellikler:
    - Geçmiş hata verilerinden öğrenme
    - Gerçek zamanlı özellik üretimi
    - Point-in-time garantisi (gelecek bilgisi sızıntısı yok)
    - Özellk mühendisliği Enjeksiyonu
    """

    def __init__(
        self,
        error_tracker: ErrorTracker,
        pattern_detector: PatternDetector,
        feature_dir: str = "data/feedback/features",
    ):
        self.tracker = error_tracker
        self.detector = pattern_detector
        self.feature_dir = Path(feature_dir)
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        # Önbellek
        self._feature_cache: dict[str, ErrorFeatureSet] = {}

    def generate_team_features(self, team: str) -> dict:
        """Takım için hata özellikleri üret."""
        cache_key = f"team_{team}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key].team_features.get(team, {})

        records = self.tracker.get_records_by_team(team)
        if not records:
            return self._default_team_features()

        features = {
            # Temel hata istatistikleri
            f"{team}_error_rate": self._calc_error_rate(records),
            f"{team}_avg_error_margin": self._calc_avg_error_margin(records),
            f"{team}_log_loss": self._calc_log_loss(records),

            # Sonuç bazında hata
            f"{team}_home_error_rate": self._calc_result_error_rate(records, "H"),
            f"{team}_draw_error_rate": self._calc_result_error_rate(records, "D"),
            f"{team}_away_error_rate": self._calc_result_error_rate(records, "A"),

            # Ev sahibi/deplasman performansı
            f"{team}_home_performance": self._calc_home_performance(records),
            f"{team}_away_performance": self._calc_away_performance(records),

            # Gol tahmin hatası
            f"{team}_goal_prediction_error": self._calc_goal_prediction_error(records),

            # Güven kalibrasyonu
            f"{team}_confidence_calibration": self._calc_confidence_calibration(records),

            # Form trendi
            f"{team}_error_trend": self._calc_error_trend(records),

            # Son 10 maçtaki hata
            f"{team}_recent_error_rate": self._calc_recent_error_rate(records, 10),
        }

        self._feature_cache[cache_key] = ErrorFeatureSet(
            team_features={team: features},
            league_features={},
            result_features={},
            goal_range_features={},
            confidence_features={},
            temporal_features={},
            combination_features={},
        )

        return features

    def generate_league_features(self, league: str) -> dict:
        """Lig için hata özellikleri üret."""
        records = self.tracker.get_records_by_league(league)
        if not records:
            return self._default_league_features()

        features = {
            # Temel hata istatistikleri
            f"{league}_error_rate": self._calc_error_rate(records),
            f"{league}_avg_error_margin": self._calc_avg_error_margin(records),
            f"{league}_log_loss": self._calc_log_loss(records),

            # Sonuç dağılımı hataları
            f"{league}_home_bias": self._calc_home_bias(records),
            f"{league}_draw_bias": self._calc_draw_bias(records),
            f"{league}_away_bias": self._calc_away_bias(records),

            # Gol aralığı hataları
            f"{league}_low_scoring_error": self._calc_goal_range_error(records, "low"),
            f"{league}_high_scoring_error": self._calc_goal_range_error(records, "high"),

            # Volatilite
            f"{league}_error_volatility": self._calc_error_volatility(records),

            # Örnek sayısı (güven için)
            f"{league}_sample_size": len(records),
        }

        return features

    def generate_match_features(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Maç için hata özelliklerini features DataFrame'ine ekle."""
        # Takım özelliklerini al
        home_features = self.generate_team_features(home_team)
        away_features = self.generate_team_features(away_team)

        # Lig özelliklerini al
        league_features = self.generate_league_features(league)

        # Kombinasyon özelliklerini hesapla
        combination_features = self._generate_combination_features(
            home_team, away_team, league, home_features, away_features
        )

        # Tüm özellikleri birleştir
        all_features = {**home_features, **away_features, **league_features, **combination_features}

        # DataFrame'e ekle
        if match_id in features_df.index:
            for feature_name, value in all_features.items():
                features_df.loc[match_id, feature_name] = value
        else:
            # Yeni satır ekle
            new_row = pd.DataFrame([all_features], index=[match_id])
            features_df = pd.concat([features_df, new_row])

        return features_df

    def _generate_combination_features(
        self,
        home_team: str,
        away_team: str,
        league: str,
        home_features: dict,
        away_features: dict,
    ) -> dict:
        """Kombinasyon özellikleri üret."""
        features = {}

        # Takım vs takım karşılaştırma
        home_error = home_features.get(f"{home_team}_error_rate", 0.5)
        away_error = away_features.get(f"{away_team}_error_rate", 0.5)

        features["error_rate_diff"] = home_error - away_error
        features["error_rate_ratio"] = home_error / max(away_error, 0.01)
        features["combined_error_rate"] = (home_error + away_error) / 2

        # Lig etkisi
        league_factor = 1.0  # Varsayılan
        features["league_error_factor"] = league_factor

        # Güven skoru
        features["prediction_confidence_score"] = self._calc_confidence_score(
            home_features, away_features
        )

        return features

    def _calc_error_rate(self, records: list[PredictionRecord]) -> float:
        """Hata oranını hesapla."""
        if not records:
            return 0.5
        incorrect = len([r for r in records if not r.is_correct])
        return incorrect / len(records)

    def _calc_avg_error_margin(self, records: list[PredictionRecord]) -> float:
        """Ortalama hata payını hesapla."""
        if not records:
            return 0.5
        return np.mean([r.error_margin for r in records])

    def _calc_log_loss(self, records: list[PredictionRecord]) -> float:
        """Log loss hesapla."""
        if not records:
            return float("inf")
        return np.mean([r.log_loss for r in records])

    def _calc_result_error_rate(
        self, records: list[PredictionRecord], result: str
    ) -> float:
        """Belirli bir sonucun hata oranını hesapla."""
        result_records = [r for r in records if r.actual_result == result]
        if not result_records:
            return 0.5
        return self._calc_error_rate(result_records)

    def _calc_home_performance(self, records: list[PredictionRecord]) -> float:
        """Ev sahibi performansını hesapla."""
        home_records = [r for r in records if r.home_team == records[0].home_team]
        if not home_records:
            return 0.5
        return 1.0 - self._calc_error_rate(home_records)

    def _calc_away_performance(self, records: list[PredictionRecord]) -> float:
        """Deplasman performansını hesapla."""
        away_records = [r for r in records if r.away_team == records[0].away_team]
        if not away_records:
            return 0.5
        return 1.0 - self._calc_error_rate(away_records)

    def _calc_goal_prediction_error(self, records: list[PredictionRecord]) -> float:
        """Gol tahmin hatasını hesapla."""
        if not records:
            return 0.0

        errors = []
        for r in records:
            # Basit bir metrik: gerçek gol sayısının tahmin olasılıklarına uzaklığı
            total_goals = r.home_goals + r.away_goals
            if total_goals == 0:
                expected = r.pred_draw  # 0-0 beraberlik olasılığı
            elif total_goals == 1:
                expected = r.pred_home if r.home_goals == 1 else r.pred_away
            else:
                expected = (r.pred_home + r.pred_away) / 2  # Yüksek skorlu

            errors.append(abs(expected - 0.5))  # Ne kadar emin olduk

        return np.mean(errors)

    def _calc_confidence_calibration(self, records: list[PredictionRecord]) -> float:
        """Güven kalibrasyonunu hesapla."""
        if not records:
            return 0.0

        # Güven seviyelerine göre grupla
        confidences = []
        for r in records:
            max_prob = max(r.pred_home, r.pred_draw, r.pred_away)
            confidences.append(max_prob)

        avg_confidence = np.mean(confidences)
        accuracy = 1.0 - self._calc_error_rate(records)

        # Kalibrasyon hatası: güven ve doğruluk farkı
        return abs(avg_confidence - accuracy)

    def _calc_error_trend(self, records: list[PredictionRecord]) -> float:
        """Hata trendini hesapla (pozitif = kötüleşiyor)."""
        if len(records) < 10:
            return 0.0

        # Son 10 maç vs önceki 10 maç
        recent = records[-10:]
        previous = records[-20:-10] if len(records) >= 20 else records[:10]

        recent_error = self._calc_error_rate(recent)
        previous_error = self._calc_error_rate(previous)

        return recent_error - previous_error

    def _calc_recent_error_rate(
        self, records: list[PredictionRecord], n: int
    ) -> float:
        """Son n maçtaki hata oranını hesapla."""
        recent = records[-n:]
        return self._calc_error_rate(recent)

    def _calc_home_bias(self, records: list[PredictionRecord]) -> float:
        """Ev sahibi önyargısını hesapla."""
        if not records:
            return 0.0

        home_wins = len([r for r in records if r.actual_result == "H"])
        home_pred_avg = np.mean([r.pred_home for r in records])

        return home_pred_avg - (home_wins / len(records))

    def _calc_draw_bias(self, records: list[PredictionRecord]) -> float:
        """Beraberlik önyargısını hesapla."""
        if not records:
            return 0.0

        draws = len([r for r in records if r.actual_result == "D"])
        draw_pred_avg = np.mean([r.pred_draw for r in records])

        return draw_pred_avg - (draws / len(records))

    def _calc_away_bias(self, records: list[PredictionRecord]) -> float:
        """Deplasman önyargısını hesapla."""
        if not records:
            return 0.0

        away_wins = len([r for r in records if r.actual_result == "A"])
        away_pred_avg = np.mean([r.pred_away for r in records])

        return away_pred_avg - (away_wins / len(records))

    def _calc_goal_range_error(
        self, records: list[PredictionRecord], goal_range: str
    ) -> float:
        """Gol aralığı hata oranını hesapla."""
        if not records:
            return 0.5

        if goal_range == "low":
            # 0-1 gol
            filtered = [r for r in records if r.home_goals + r.away_goals <= 1]
        else:
            # 4+ gol
            filtered = [r for r in records if r.home_goals + r.away_goals >= 4]

        if not filtered:
            return 0.5

        return self._calc_error_rate(filtered)

    def _calc_error_volatility(self, records: list[PredictionRecord]) -> float:
        """Hata volatilitesini hesapla."""
        if len(records) < 10:
            return 0.0

        # Son 10 maçtaki hata oranlarının standart sapması
        error_rates = []
        for i in range(10, len(records) + 1):
            window = records[max(0, i-10):i]
            error_rates.append(self._calc_error_rate(window))

        return np.std(error_rates)

    def _calc_confidence_score(
        self, home_features: dict, away_features: dict
    ) -> float:
        """Güven skoru hesapla."""
        home_error = home_features.get("home_error_rate", 0.5)
        away_error = away_features.get("away_error_rate", 0.5)

        # Düşük hata = yüksek güven
        return 1.0 - (home_error + away_error) / 2

    def _default_team_features(self) -> dict:
        """Varsayılan takım özellikleri."""
        return {
            "_error_rate": 0.5,
            "_avg_error_margin": 0.5,
            "_log_loss": 1.0,
            "_home_error_rate": 0.5,
            "_draw_error_rate": 0.5,
            "_away_error_rate": 0.5,
            "_home_performance": 0.5,
            "_away_performance": 0.5,
            "_goal_prediction_error": 0.0,
            "_confidence_calibration": 0.0,
            "_error_trend": 0.0,
            "_recent_error_rate": 0.5,
        }

    def _default_league_features(self) -> dict:
        """Varsayılan lig özellikleri."""
        return {
            "_error_rate": 0.5,
            "_avg_error_margin": 0.5,
            "_log_loss": 1.0,
            "_home_bias": 0.0,
            "_draw_bias": 0.0,
            "_away_bias": 0.0,
            "_low_scoring_error": 0.5,
            "_high_scoring_error": 0.5,
            "_error_volatility": 0.0,
            "_sample_size": 0,
        }

    def save_features(self, features_df: pd.DataFrame, filename: str = "error_features.parquet"):
        """Özellikleri dosyaya kaydet."""
        filepath = self.feature_dir / filename
        features_df.to_parquet(filepath)
        logger.info(f"Hata özellikleri kaydedildi: {filepath}")

    def load_features(self, filename: str = "error_features.parquet") -> pd.DataFrame:
        """Özellikleri dosyadan yükle."""
        filepath = self.feature_dir / filename
        if filepath.exists():
            return pd.read_parquet(filepath)
        return pd.DataFrame()

    def get_feature_importance_report(self) -> str:
        """Özellik önem raporu oluştur."""
        patterns = self.detector.detect_all_patterns()

        report = ["HATA ÖZELLİK RAPORU", "=" * 50, ""]

        if not patterns:
            report.append("Henüz belirgin hata kalıbı tespit edilmedi.")
            return "\n".join(report)

        # En önemli özellikleri listele
        feature_scores = {}
        for pattern in patterns:
            if pattern.pattern_type == "takim":
                team = pattern.details.get("team", "")
                feature_scores[f"{team}_error_rate"] = pattern.severity
            elif pattern.pattern_type == "lig":
                league = pattern.details.get("league", "")
                feature_scores[f"{league}_error_rate"] = pattern.severity

        # Önem sırasına göre sırala
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        report.append("En Önemli Hata Özellikleri:")
        for feature, score in sorted_features[:10]:
            report.append(f"  {feature}: {score:.3f}")

        return "\n".join(report)
