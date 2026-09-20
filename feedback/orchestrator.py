"""Feedback Loop Orchestrator - Tüm bileşenleri koordine eder.

Geri besleme döngüsünün ana koordinatörü:

1. Tahmin kaydı → 2. Hata analizi → 3. Kalıp tespiti → 4. Düzeltme → 5. Yeni özellikler

Bu döngü sürekli çalışır ve modelin zamanla kendini geliştirmesini sağlar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from feedback.error_tracker import ErrorTracker, PredictionRecord
from feedback.pattern_detector import PatternDetector, ErrorPattern
from feedback.self_correction import SelfCorrectionEngine, CorrectionResult
from feedback.feature_generator import ErrorFeatureGenerator, ErrorFeatureSet

logger = logging.getLogger(__name__)


@dataclass
class FeedbackCycleResult:
    """Her geri besleme döngüsünün sonucu."""
    cycle_id: int
    start_time: str
    end_time: str
    duration_seconds: float

    # Tahmin kayıtları
    predictions_recorded: int
    total_predictions: int

    # Kalıp tespiti
    patterns_detected: int
    top_patterns: list[dict]

    # Düzeltme
    correction_applied: bool
    correction_result: Optional[dict]

    # Yeni özellikler
    features_generated: int

    # Metrikler
    metrics_before: dict
    metrics_after: dict
    improvement: float


class FeedbackOrchestrator:
    """Geri besleme döngüsünün ana koordinatörü.

    Özellikler:
    - Otomatik döngü çalıştırma
    - Periyodik analiz ve düzeltme
    - Geçmiş takibi ve raporlama
    - Güvenli düzeltme mekanizmaları
    """

    def __init__(
        self,
        data_dir: str = "data/feedback",
        auto_correct: bool = True,
        min_predictions_for_correction: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Bileşenleri başlat
        self.error_tracker = ErrorTracker(str(self.data_dir))
        self.pattern_detector = PatternDetector(self.error_tracker)
        self.self_correction = SelfCorrectionEngine(
            self.error_tracker, self.pattern_detector, str(self.data_dir / "corrections")
        )
        self.feature_generator = ErrorFeatureGenerator(
            self.error_tracker, self.pattern_detector, str(self.data_dir / "features")
        )

        # Ayarlar
        self.auto_correct = auto_correct
        self.min_predictions_for_correction = min_predictions_for_correction

        # Döngü geçmişi
        self.cycle_history: list[FeedbackCycleResult] = []
        self._load_history()

    def _load_history(self):
        """Döngü geçmişini yükle."""
        history_file = self.data_dir / "cycle_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.cycle_history = [FeedbackCycleResult(**item) for item in data]
            except Exception as e:
                logger.warning(f"Döngü geçmişi yüklenemedi: {e}")

    def _save_history(self):
        """Döngü geçmişini kaydet."""
        history_file = self.data_dir / "cycle_history.json"
        data = [
            {
                "cycle_id": c.cycle_id,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "duration_seconds": c.duration_seconds,
                "predictions_recorded": c.predictions_recorded,
                "total_predictions": c.total_predictions,
                "patterns_detected": c.patterns_detected,
                "top_patterns": c.top_patterns,
                "correction_applied": c.correction_applied,
                "correction_result": c.correction_result,
                "features_generated": c.features_generated,
                "metrics_before": c.metrics_before,
                "metrics_after": c.metrics_after,
                "improvement": c.improvement,
            }
            for c in self.cycle_history
        ]

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

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
        return self.error_tracker.record_prediction(
            match_id=match_id,
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual_result,
            league=league,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            model_name=model_name,
        )

    def run_cycle(
        self,
        current_weights: Optional[np.ndarray] = None,
        current_temperature: float = 1.0,
    ) -> FeedbackCycleResult:
        """Tek bir geri besleme döngüsü çalıştır."""
        start_time = datetime.now()
        cycle_id = len(self.cycle_history) + 1

        logger.info(f"Geri besleme döngüsü #{cycle_id} başlıyor...")

        # Mevcut metrikleri hesapla
        metrics_before = self.self_correction._calculate_metrics()

        # Tahmin sayısını kontrol et
        total_predictions = len(self.error_tracker.records)
        predictions_recorded = 0

        # Kalıpları tespit et
        patterns = self.pattern_detector.detect_all_patterns()
        top_patterns = [
            {
                "type": p.pattern_type,
                "description": p.description,
                "severity": p.severity,
                "confidence": p.confidence,
            }
            for p in patterns[:5]
        ]

        # Düzeltme uygula
        correction_applied = False
        correction_result = None

        if self.auto_correct and total_predictions >= self.min_predictions_for_correction:
            result = self.self_correction.analyze_and_correct(
                current_weights, current_temperature
            )
            correction_applied = True
            correction_result = {
                "success": result.success,
                "improvement": result.improvement,
                "actions_count": len(result.actions_taken),
            }

        # Yeni özellikler üret
        features_generated = 0
        if patterns:
            features_generated = self._generate_features_from_patterns(patterns)

        # Düzeltilmiş metrikleri hesapla
        metrics_after = self.self_correction._calculate_metrics()
        improvement = self.self_correction._calculate_improvement(metrics_before, metrics_after)

        end_time = datetime.now()

        result = FeedbackCycleResult(
            cycle_id=cycle_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            predictions_recorded=predictions_recorded,
            total_predictions=total_predictions,
            patterns_detected=len(patterns),
            top_patterns=top_patterns,
            correction_applied=correction_applied,
            correction_result=correction_result,
            features_generated=features_generated,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=improvement,
        )

        self.cycle_history.append(result)
        self._save_history()

        logger.info(
            f"Geri besleme döngüsü #{cycle_id} tamamlandı: "
            f"{len(patterns)} kalıp, {correction_applied} düzeltme, "
            f"{improvement:.4f} iyileştirme"
        )

        return result

    def _generate_features_from_patterns(self, patterns: list[ErrorPattern]) -> int:
        """Kalıplardan yeni özellikler üret."""
        features_count = 0

        for pattern in patterns:
            if pattern.pattern_type == "takim":
                team = pattern.details.get("team", "")
                if team:
                    self.feature_generator.generate_team_features(team)
                    features_count += 1

            elif pattern.pattern_type == "lig":
                league = pattern.details.get("league", "")
                if league:
                    self.feature_generator.generate_league_features(league)
                    features_count += 1

        return features_count

    def get_latest_cycle(self) -> Optional[FeedbackCycleResult]:
        """Son döngü sonucunu döndür."""
        if self.cycle_history:
            return self.cycle_history[-1]
        return None

    def get_cycle_summary(self) -> dict:
        """Döngü özetini döndür."""
        if not self.cycle_history:
            return {"total_cycles": 0}

        return {
            "total_cycles": len(self.cycle_history),
            "average_improvement": np.mean([c.improvement for c in self.cycle_history]),
            "total_improvement": sum(c.improvement for c in self.cycle_history),
            "average_patterns": np.mean([c.patterns_detected for c in self.cycle_history]),
            "correction_rate": sum(1 for c in self.cycle_history if c.correction_applied) / len(self.cycle_history),
        }

    def generate_report(self) -> str:
        """Tam bir geri besleme raporu oluştur."""
        report = [
            "GERİ BESLEME DÖNGÜSÜ RAPORU",
            "=" * 60,
            "",
        ]

        # Genel özet
        summary = self.get_cycle_summary()
        report.extend([
            "GENEL ÖZET",
            f"  Toplam döngü: {summary.get('total_cycles', 0)}",
            f"  Ortalama iyileştirme: {summary.get('average_improvement', 0):.4f}",
            f"  Toplam iyileştirme: {summary.get('total_improvement', 0):.4f}",
            f"  Düzeltme oranı: %{summary.get('correction_rate', 0) * 100:.1f}",
            "",
        ])

        # Son döngü
        latest = self.get_latest_cycle()
        if latest:
            report.extend([
                "SON DÖNGÜ",
                f"  Döngü #{latest.cycle_id}",
                f"  Süre: {latest.duration_seconds:.1f} saniye",
                f"  Tespit edilen kalıp: {latest.patterns_detected}",
                f"  Düzeltme uygulandı: {latest.correction_applied}",
                f"  İyileştirme: {latest.improvement:.4f}",
                "",
            ])

            if latest.top_patterns:
                report.append("  EN ÖNEMLİ KALIPLAR:")
                for i, pattern in enumerate(latest.top_patterns, 1):
                    report.append(
                        f"    {i}. [{pattern['type']}] {pattern['description']}"
                    )
                report.append("")

        # Mevcut metrikler
        metrics = self.self_correction._calculate_metrics()
        report.extend([
            "MEVCUT METRİKLER",
            f"  Doğruluk: %{metrics.get('accuracy', 0) * 100:.1f}",
            f"  Log Loss: {metrics.get('log_loss', 0):.4f}",
            f"  Brier Score: {metrics.get('brier', 0):.4f}",
            "",
        ])

        # Kalıp analizi
        patterns = self.pattern_detector.detect_all_patterns()
        if patterns:
            report.extend([
                "TESPIT EDİLEN KALIPLAR",
                f"  Toplam kalıp: {len(patterns)}",
                "",
            ])

            for i, pattern in enumerate(patterns[:5], 1):
                report.extend([
                    f"  {i}. {pattern.pattern_type.upper()}",
                    f"     Açıklama: {pattern.description}",
                    f"     Ciddiyet: {pattern.severity:.2f}",
                    f"     Güven: {pattern.confidence:.2f}",
                    "",
                ])

        # Öneriler
        report.extend([
            "ÖNERİLER",
            self._generate_recommendations(patterns),
        ])

        return "\n".join(report)

    def _generate_recommendations(self, patterns: list[ErrorPattern]) -> str:
        """Kalıplara göre öneriler oluştur."""
        if not patterns:
            return "  Henüz belirgin bir sorun tespit edilmedi."

        recommendations = []

        for pattern in patterns[:3]:
            if pattern.pattern_type == "lig":
                recommendations.append(
                    f"  - {pattern.details.get('league', '')} ligi için özellik mühendisliği iyileştirilmeli"
                )
            elif pattern.pattern_type == "takim":
                recommendations.append(
                    f"  - {pattern.details.get('team', '')} takımı için özel model eğitilmeli"
                )
            elif pattern.pattern_type == "sonuc":
                recommendations.append(
                    f"  - '{pattern.details.get('result', '')}' sonucu için kalibrasyon ayarlanmalı"
                )

        return "\n".join(recommendations) if recommendations else "  Genel iyileştirme önerileri mevcut değil."
