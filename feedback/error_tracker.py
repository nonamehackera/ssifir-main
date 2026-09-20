"""Error Tracker - Tahminleri ve gerçek sonuçları kaydedip analiz eder.

Her tahmin kaydedilir:
- Tahmin olasılıkları (home_win, draw, away_win)
- Gerçek sonuç (H/D/A)
- Maç meta verileri (lig, takım, tarih, vb.)
- Hata metrikleri (log loss, Brier, error margin)

Analiz fonksiyonları:
- Periyodik hata analizi (lig, takım, zaman dilimi bazında)
- Büyük hatalı tahminleri tespit etme
- Hata dağılımı istatistikleri
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.metrics import log_loss_1x2, brier_1x2, accuracy_1x2

logger = logging.getLogger(__name__)

EPS = 1e-12


@dataclass
class PredictionRecord:
    """Tek bir tahmin kaydı."""
    match_id: str
    prediction_time: str
    actual_result: str  # H/D/A

    # Tahmin olasılıkları
    pred_home: float
    pred_draw: float
    pred_away: float

    # Gerçek olasılıklar (one-hot)
    actual_home: float = 0.0
    actual_draw: float = 0.0
    actual_away: float = 0.0

    # Meta veriler
    league: str = ""
    season: str = ""
    home_team: str = ""
    away_team: str = ""
    home_goals: int = 0
    away_goals: int = 0

    # Ek bilgiler
    model_name: str = ""
    confidence: float = 0.0

    def __post_init__(self):
        # Gerçek olasılıkları one-hot olarak ayarla
        if self.actual_result == "H":
            self.actual_home = 1.0
        elif self.actual_result == "D":
            self.actual_draw = 1.0
        elif self.actual_result == "A":
            self.actual_away = 1.0

    @property
    def predicted_result(self) -> str:
        """Tahmin edilen sonucu döndür."""
        probs = [self.pred_home, self.pred_draw, self.pred_away]
        idx = np.argmax(probs)
        return ["H", "D", "A"][idx]

    @property
    def is_correct(self) -> bool:
        """Tahmin doğru mu?"""
        return self.predicted_result == self.actual_result

    @property
    def error_margin(self) -> float:
        """Tahmin hata payı (doğru sonuca verilen olasılık)."""
        if self.actual_result == "H":
            return 1.0 - self.pred_home
        elif self.actual_result == "D":
            return 1.0 - self.pred_draw
        else:
            return 1.0 - self.pred_away

    @property
    def log_loss(self) -> float:
        """Tek kayıt için log loss."""
        if self.actual_result == "H":
            return -np.log(max(self.pred_home, EPS))
        elif self.actual_result == "D":
            return -np.log(max(self.pred_draw, EPS))
        else:
            return -np.log(max(self.pred_away, EPS))

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorTracker:
    """Tahmin hatalarını takip eden ve analiz eden sınıf.

    Özellikler:
    - Tahminleri kalıcı olarak kaydeder (JSON/Parquet)
    - Periyodik analiz yapar
    - Büyük hatalı tahminleri tespit eder
    - Hata dağılımı istatistikleri üretir
    """

    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.records_file = self.data_dir / "prediction_records.parquet"
        self.analysis_dir = self.data_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        # Mevcut kayıtları yükle
        self.records: list[PredictionRecord] = []
        self._load_records()

    def _load_records(self):
        """Mevcut kayıtları dosyadan yükle."""
        if self.records_file.exists():
            try:
                df = pd.read_parquet(self.records_file)
                for _, row in df.iterrows():
                    record = PredictionRecord(**row.to_dict())
                    self.records.append(record)
                logger.info(f"{len(self.records)} kayıt yüklendi.")
            except Exception as e:
                logger.warning(f"Kayıtlar yüklenemedi: {e}")

    def record_prediction(
        self,
        match_id: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        actual_result: str,
        league: str = "",
        season: str = "",
        home_team: str = "",
        away_team: str = "",
        home_goals: int = 0,
        away_goals: int = 0,
        model_name: str = "",
    ) -> PredictionRecord:
        """Yeni bir tahmin kaydı ekle."""
        record = PredictionRecord(
            match_id=match_id,
            prediction_time=datetime.now().isoformat(),
            actual_result=actual_result,
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            league=league,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            model_name=model_name,
        )
        self.records.append(record)
        self._save_records()
        return record

    def _save_records(self):
        """Kayıtları dosyaya kaydet."""
        if not self.records:
            return
        df = pd.DataFrame([r.to_dict() for r in self.records])
        df.to_parquet(self.records_file, index=False)

    def get_recent_records(self, n: int = 100) -> list[PredictionRecord]:
        """Son n kaydı döndür."""
        return self.records[-n:]

    def get_records_by_league(self, league: str) -> list[PredictionRecord]:
        """Belirli bir lige ait kayıtları döndür."""
        return [r for r in self.records if r.league == league]

    def get_records_by_team(self, team: str) -> list[PredictionRecord]:
        """Belirli bir takıma ait kayıtları döndür."""
        return [
            r for r in self.records
            if r.home_team == team or r.away_team == team
        ]

    def get_incorrect_predictions(self) -> list[PredictionRecord]:
        """Yanlış tahminleri döndür."""
        return [r for r in self.records if not r.is_correct]

    def get_worst_predictions(self, n: int = 20) -> list[PredictionRecord]:
        """En kötü tahminleri döndür (hata payına göre)."""
        return sorted(self.records, key=lambda r: r.error_margin, reverse=True)[:n]

    def analyze_by_category(self) -> dict:
        """Kategori bazında hata analizi."""
        if not self.records:
            return {}

        analysis = {
            "total_matches": len(self.records),
            "accuracy": accuracy_1x2(
                [r.actual_result for r in self.records],
                [r.pred_home for r in self.records],
                [r.pred_draw for r in self.records],
                [r.pred_away for r in self.records],
            ),
            "log_loss": log_loss_1x2(
                [r.actual_result for r in self.records],
                [r.pred_home for r in self.records],
                [r.pred_draw for r in self.records],
                [r.pred_away for r in self.records],
            ),
            "brier": brier_1x2(
                [r.actual_result for r in self.records],
                [r.pred_home for r in self.records],
                [r.pred_draw for r in self.records],
                [r.pred_away for r in self.records],
            ),
        }

        # Lig bazında analiz
        leagues = {}
        for r in self.records:
            if r.league not in leagues:
                leagues[r.league] = []
            leagues[r.league].append(r)

        analysis["by_league"] = {}
        for league, records in leagues.items():
            if len(records) >= 10:  # Minimum örnek sayısı
                analysis["by_league"][league] = {
                    "n": len(records),
                    "accuracy": accuracy_1x2(
                        [r.actual_result for r in records],
                        [r.pred_home for r in records],
                        [r.pred_draw for r in records],
                        [r.pred_away for r in records],
                    ),
                    "avg_error_margin": np.mean([r.error_margin for r in records]),
                }

        # Sonuç bazında analiz (H/D/A)
        result_groups = {"H": [], "D": [], "A": []}
        for r in self.records:
            result_groups[r.actual_result].append(r)

        analysis["by_result"] = {}
        for result, records in result_groups.items():
            if records:
                analysis["by_result"][result] = {
                    "n": len(records),
                    "avg_pred_for": np.mean([
                        r.pred_home if result == "H" else
                        r.pred_draw if result == "D" else
                        r.pred_away
                        for r in records
                    ]),
                    "avg_error_margin": np.mean([r.error_margin for r in records]),
                }

        # Gol aralığı bazında analiz
        goal_bins = [
            (0, 0, "0-0"),
            (1, 0, "1-0 veya 0-1"),
            (1, 1, "1-1"),
            (2, 0, "2-0 veya 0-2"),
            (2, 1, "2-1 veya 1-2"),
            (2, 2, "2-2"),
            (3, None, "3+"),
        ]

        analysis["by_goals"] = {}
        for r in self.records:
            total_goals = r.home_goals + r.away_goals
            if total_goals >= 3:
                bucket = "3+"
            elif total_goals == 0:
                bucket = "0-0"
            elif total_goals == 1:
                bucket = "1-0 veya 0-1"
            elif total_goals == 2:
                if r.home_goals == 2 or r.away_goals == 2:
                    bucket = "2-0 veya 0-2"
                else:
                    bucket = "2-1 veya 1-2"
            else:
                bucket = "2-2"

            if bucket not in analysis["by_goals"]:
                analysis["by_goals"][bucket] = []
            analysis["by_goals"][bucket].append(r)

        for bucket, records in analysis["by_goals"].items():
            if records:
                analysis["by_goals"][bucket] = {
                    "n": len(records),
                    "accuracy": accuracy_1x2(
                        [r.actual_result for r in records],
                        [r.pred_home for r in records],
                        [r.pred_draw for r in records],
                        [r.pred_away for r in records],
                    ),
                    "avg_error_margin": np.mean([r.error_margin for r in records]),
                }

        return analysis

    def get_confidence_distribution(self) -> dict:
        """Tahmin güven dağılımını analiz et."""
        if not self.records:
            return {}

        confidences = []
        for r in self.records:
            max_prob = max(r.pred_home, r.pred_draw, r.pred_away)
            confidences.append(max_prob)

        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(confidences, bins=bins)

        return {
            "bins": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
            "counts": hist.tolist(),
            "avg_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
        }

    def save_analysis(self, analysis: dict, filename: str = "error_analysis.json"):
        """Analiz sonuçlarını dosyaya kaydet."""
        filepath = self.analysis_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"Analiz kaydedildi: {filepath}")

    def export_to_dataframe(self) -> pd.DataFrame:
        """Kayıtları DataFrame olarak döndür."""
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.records])
