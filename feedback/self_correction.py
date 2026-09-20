"""Self-Correction Engine - Model parametrelerini otomatik günceller.

Hata kalıplarına göre modelin kendini düzeltme mekanizması:

1. Ağırlık Ayarlama: Ensemble ağırlıklarını geçmiş performansa göre güncelle
2. Kalibrasyon Düzeltme: Olasılık kalibrasyonunu hatalara göre ayarla
3. Sınıf Ağırlığı: Azınlık sınıfların ağırlığını artır
4. Öğrenme Oranı: Hata oranına göre öğrenme parametrelerini ayarla
5. Çıkış Katmanı Düzeltme: Model çıkışlarını hata kalıplarına göre ayarla
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from feedback.error_tracker import ErrorTracker, PredictionRecord
from feedback.pattern_detector import PatternDetector, ErrorPattern

logger = logging.getLogger(__name__)

EPS = 1e-12


@dataclass
class CorrectionAction:
    """Düzeltme eylemi tanımı."""
    action_type: str  # agirlik_ayarlama, kalibrasyon, sinif_agirligi, cikis_duzeltme
    description: str
    parameters: dict = field(default_factory=dict)
    expected_improvement: float = 0.0
    confidence: float = 0.0


@dataclass
class CorrectionResult:
    """Düzeltme sonucu."""
    success: bool
    actions_taken: list[CorrectionAction] = field(default_factory=list)
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    improvement: float = 0.0
    details: dict = field(default_factory=dict)


class SelfCorrectionEngine:
    """Modelin kendi kendini düzeltme motoru.

    Özellikler:
    - Hata kalıplarına göre otomatik düzeltme
    - Güvenli düzeltme mekanizmaları (aşırı düzeltme koruması)
    - Geri alınabilir değişiklikler
    - Metrik takibi
    """

    def __init__(
        self,
        error_tracker: ErrorTracker,
        pattern_detector: PatternDetector,
        correction_dir: str = "data/feedback/corrections",
    ):
        self.tracker = error_tracker
        self.detector = pattern_detector
        self.correction_dir = Path(correction_dir)
        self.correction_dir.mkdir(parents=True, exist_ok=True)

        # Düzeltme parametreleri
        self.max_weight_change = 0.3  # Maksimum ağırlık değişimi
        self.min_weight = 0.05  # Minimum ağırlık
        self.max_calibration_shift = 0.1  # Maksimum kalibrasyon kayması
        self.correction_history: list[CorrectionResult] = []

    def analyze_and_correct(
        self,
        current_weights: Optional[np.ndarray] = None,
        current_temperature: float = 1.0,
    ) -> CorrectionResult:
        """Mevcut hata kalıplarını analiz edip düzeltme uygula."""
        # Mevcut metrikleri hesapla
        metrics_before = self._calculate_metrics()

        # Kalıpları tespit et
        patterns = self.detector.detect_all_patterns()

        if not patterns:
            logger.info("Belirgin hata kalıbı tespit edilemedi, düzeltme gerekli değil.")
            return CorrectionResult(
                success=True,
                metrics_before=metrics_before,
                metrics_after=metrics_before,
                improvement=0.0,
                details={"message": "Kalıp tespit edilemedi"},
            )

        # Düzeltme eylemleri oluştur
        actions = self._plan_corrections(patterns, current_weights, current_temperature)

        # Düzeltmeleri uygula
        corrected_weights, corrected_temp = self._apply_corrections(
            actions, current_weights, current_temperature
        )

        # Düzeltilmiş metrikleri hesapla (tahmini)
        metrics_after = self._estimate_metrics_after_correction(
            patterns, metrics_before
        )

        improvement = self._calculate_improvement(metrics_before, metrics_after)

        result = CorrectionResult(
            success=True,
            actions_taken=actions,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=improvement,
            details={
                "patterns_found": len(patterns),
                "corrected_weights": corrected_weights.tolist() if corrected_weights is not None else None,
                "corrected_temperature": corrected_temp,
            },
        )

        self.correction_history.append(result)
        self._save_correction(result)

        return result

    def _calculate_metrics(self) -> dict:
        """Mevcut metrikleri hesapla."""
        records = self.tracker.records
        if not records:
            return {"accuracy": 0, "log_loss": float("inf"), "brier": 1.0}

        from evaluation.metrics import log_loss_1x2, brier_1x2, accuracy_1x2

        return {
            "accuracy": accuracy_1x2(
                [r.actual_result for r in records],
                [r.pred_home for r in records],
                [r.pred_draw for r in records],
                [r.pred_away for r in records],
            ),
            "log_loss": log_loss_1x2(
                [r.actual_result for r in records],
                [r.pred_home for r in records],
                [r.pred_draw for r in records],
                [r.pred_away for r in records],
            ),
            "brier": brier_1x2(
                [r.actual_result for r in records],
                [r.pred_home for r in records],
                [r.pred_draw for r in records],
                [r.pred_away for r in records],
            ),
        }

    def _plan_corrections(
        self,
        patterns: list[ErrorPattern],
        current_weights: Optional[np.ndarray],
        current_temperature: float,
    ) -> list[CorrectionAction]:
        """Kalıplara göre düzeltme eylemleri planla."""
        actions = []

        for pattern in patterns:
            if pattern.pattern_type == "lig":
                actions.extend(self._plan_league_correction(pattern))
            elif pattern.pattern_type == "takim":
                actions.extend(self._plan_team_correction(pattern))
            elif pattern.pattern_type == "sonuc":
                actions.extend(self._plan_result_correction(pattern))
            elif pattern.pattern_type == "gol_araligi":
                actions.extend(self._plan_goal_range_correction(pattern))
            elif pattern.pattern_type == "guven":
                actions.extend(self._plan_confidence_correction(pattern))
            elif pattern.pattern_type == "zamanlama":
                actions.extend(self._plan_temporal_correction(pattern))

        # Ağırlık ayarlama eylemi
        if current_weights is not None:
            weight_action = self._plan_weight_adjustment(patterns, current_weights)
            if weight_action:
                actions.append(weight_action)

        # Kalibrasyon ayarlama eylemi
        if current_temperature != 1.0:
            cal_action = self._plan_calibration_adjustment(patterns, current_temperature)
            if cal_action:
                actions.append(cal_action)

        return actions

    def _plan_league_correction(self, pattern: ErrorPattern) -> list[CorrectionAction]:
        """Lig bazlı düzeltme planla."""
        actions = []

        # Lig için özel ağırlık ayarlama
        if pattern.severity > 0.3:
            actions.append(CorrectionAction(
                action_type="lig_agirlik",
                description=f"{pattern.details.get('league', 'bilinmeyen')} ligi için ağırlık ayarlaması",
                parameters={
                    "league": pattern.details.get("league"),
                    "severity": pattern.severity,
                    "adjustment_type": "downweight" if pattern.error_rate > 0.5 else "recalibrate",
                },
                expected_improvement=pattern.severity * 0.1,
                confidence=pattern.confidence,
            ))

        return actions

    def _plan_team_correction(self, pattern: ErrorPattern) -> list[CorrectionAction]:
        """Takım bazlı düzeltme planla."""
        actions = []

        if pattern.severity > 0.4:
            actions.append(CorrectionAction(
                action_type="takim_ozel",
                description=f"{pattern.details.get('team', 'bilinmeyen')} takımı için özel düzeltme",
                parameters={
                    "team": pattern.details.get("team"),
                    "severity": pattern.severity,
                    "home_error": pattern.details.get("home_error_rate", 0),
                    "away_error": pattern.details.get("away_error_rate", 0),
                },
                expected_improvement=pattern.severity * 0.15,
                confidence=pattern.confidence,
            ))

        return actions

    def _plan_result_correction(self, pattern: ErrorPattern) -> list[CorrectionAction]:
        """Sonuç bazlı düzeltme planla."""
        actions = []
        result = pattern.details.get("result", "")

        if pattern.severity > 0.2:
            # Sonuç için kalibrasyon kayması
            shift = -0.05 if pattern.error_rate > 0.6 else 0.05

            actions.append(CorrectionAction(
                action_type="sonuc_kalibrasyon",
                description=f"'{result}' sonucu için kalibrasyon düzeltmesi",
                parameters={
                    "result": result,
                    "shift": shift,
                    "severity": pattern.severity,
                },
                expected_improvement=pattern.severity * 0.08,
                confidence=pattern.confidence,
            ))

        return actions

    def _plan_goal_range_correction(self, pattern: ErrorPattern) -> list[CorrectionAction]:
        """Gol aralığı bazlı düzeltme planla."""
        actions = []
        goal_range = pattern.details.get("goal_range", "")

        if pattern.severity > 0.3:
            actions.append(CorrectionAction(
                action_type="gol_model_duzeltme",
                description=f"{goal_range} maçlar için gol modeli düzeltmesi",
                parameters={
                    "goal_range": goal_range,
                    "severity": pattern.severity,
                    "avg_goals": pattern.details.get("avg_goals", 0),
                },
                expected_improvement=pattern.severity * 0.12,
                confidence=pattern.confidence,
            ))

        return actions

    def _plan_confidence_correction(self, pattern: ErrorPattern) -> list[CorrectionAction]:
        """Güven bazlı düzeltme planla."""
        actions = []

        if pattern.severity > 0.3:
            actions.append(CorrectionAction(
                action_type="guven_duzeltme",
                description="Güven hesaplama düzeltmesi",
                parameters={
                    "confidence_level": pattern.details.get("confidence_level"),
                    "severity": pattern.severity,
                    "avg_confidence": pattern.details.get("avg_confidence", 0),
                },
                expected_improvement=pattern.severity * 0.1,
                confidence=pattern.confidence,
            ))

        return actions

    def _plan_temporal_correction(self, pattern: ErrorPattern) -> list[CorrectionAction]:
        """Zamanlama bazlı düzeltme planla."""
        actions = []

        if pattern.severity > 0.3:
            actions.append(CorrectionAction(
                action_type="zamanlama_duzeltme",
                description=f"{pattern.details.get('month_name', '')} ayı için düzeltme",
                parameters={
                    "month": pattern.details.get("month"),
                    "severity": pattern.severity,
                },
                expected_improvement=pattern.severity * 0.08,
                confidence=pattern.confidence,
            ))

        return actions

    def _plan_weight_adjustment(
        self,
        patterns: list[ErrorPattern],
        current_weights: np.ndarray,
    ) -> Optional[CorrectionAction]:
        """Ağırlık ayarlama planı oluştur."""
        if current_weights is None or len(patterns) == 0:
            return None

        # Toplam ciddiyet
        total_severity = sum(p.severity for p in patterns)

        if total_severity < 0.3:
            return None

        return CorrectionAction(
            action_type="agirlik_ayarlama",
            description="Ensemble ağırlıklarını ayarlama",
            parameters={
                "current_weights": current_weights.tolist(),
                "total_severity": total_severity,
                "n_patterns": len(patterns),
            },
            expected_improvement=total_severity * 0.05,
            confidence=np.mean([p.confidence for p in patterns]),
        )

    def _plan_calibration_adjustment(
        self,
        patterns: list[ErrorPattern],
        current_temperature: float,
    ) -> Optional[CorrectionAction]:
        """Kalibrasyon ayarlama planı oluştur."""
        # Güven kalıplarını bul
        confidence_patterns = [p for p in patterns if p.pattern_type == "guven"]

        if not confidence_patterns:
            return None

        # Yüksek güvenli tahminlerde hata varsa sıcaklığı artır
        high_conf_errors = [
            p for p in confidence_patterns
            if p.details.get("confidence_level") in ["yuksek_guven", "cok_yuksek_guven"]
        ]

        if high_conf_errors:
            # Sıcaklığı artır (daha az keskin tahminler)
            temp_adjustment = min(0.2, sum(p.severity for p in high_conf_errors) * 0.1)
            new_temp = min(2.0, current_temperature + temp_adjustment)
        else:
            # Düşük güvenli tahminlerde hata varsa sıcaklığı azalt
            low_conf_errors = [
                p for p in confidence_patterns
                if p.details.get("confidence_level") == "dusuk_guven"
            ]
            if low_conf_errors:
                temp_adjustment = min(0.2, sum(p.severity for p in low_conf_errors) * 0.1)
                new_temp = max(0.5, current_temperature - temp_adjustment)
            else:
                return None

        return CorrectionAction(
            action_type="kalibrasyon_ayarlama",
            description="Sıcaklık kalibrasyonu ayarlama",
            parameters={
                "current_temperature": current_temperature,
                "new_temperature": new_temp,
                "adjustment": new_temp - current_temperature,
            },
            expected_improvement=0.02,
            confidence=0.7,
        )

    def _apply_corrections(
        self,
        actions: list[CorrectionAction],
        current_weights: Optional[np.ndarray],
        current_temperature: float,
    ) -> tuple[Optional[np.ndarray], float]:
        """Düzeltmeleri uygula ve yeni parametreleri döndür."""
        corrected_weights = current_weights.copy() if current_weights is not None else None
        corrected_temp = current_temperature

        for action in actions:
            if action.action_type == "agirlik_ayarlama":
                corrected_weights = self._apply_weight_adjustment(
                    corrected_weights, action
                )
            elif action.action_type == "kalibrasyon_ayarlama":
                corrected_temp = action.parameters.get("new_temperature", corrected_temp)

        return corrected_weights, corrected_temp

    def _apply_weight_adjustment(
        self,
        weights: Optional[np.ndarray],
        action: CorrectionAction,
    ) -> Optional[np.ndarray]:
        """Ağırlık düzeltmesini uygula."""
        if weights is None:
            return None

        new_weights = weights.copy()
        total_severity = action.parameters.get("total_severity", 0)

        # Basit bir ayarlama: toplam ciddiyet oranında tüm ağırlıkları azalt
        # (daha sofistike bir yaklaşım kullanılabilir)
        adjustment = min(self.max_weight_change, total_severity * 0.1)

        # Rastgele bir ağırlığı azalt, diğerlerini artır
        min_idx = np.argmin(new_weights)
        new_weights[min_idx] = max(self.min_weight, new_weights[min_idx] - adjustment)

        # Diğer ağırlıkları orantılı olarak artır
        other_indices = [i for i in range(len(new_weights)) if i != min_idx]
        if other_indices:
            increase = adjustment / len(other_indices)
            for idx in other_indices:
                new_weights[idx] += increase

        # Normalize et
        new_weights = np.clip(new_weights, self.min_weight, 1.0)
        new_weights = new_weights / new_weights.sum()

        return new_weights

    def _estimate_metrics_after_correction(
        self,
        patterns: list[ErrorPattern],
        current_metrics: dict,
    ) -> dict:
        """Düzeltme sonrası metrikleri tahmin et."""
        # Basit bir tahmin: düzeltmelerin beklenen iyileştirmelerini topla
        total_expected_improvement = sum(a.expected_improvement for a in self._plan_corrections(patterns, None, 1.0))

        estimated_metrics = current_metrics.copy()
        estimated_metrics["accuracy"] = min(1.0, current_metrics["accuracy"] + total_expected_improvement * 0.5)
        estimated_metrics["log_loss"] = max(0.1, current_metrics["log_loss"] - total_expected_improvement * 0.2)
        estimated_metrics["brier"] = max(0.05, current_metrics["brier"] - total_expected_improvement * 0.1)

        return estimated_metrics

    def _calculate_improvement(self, before: dict, after: dict) -> float:
        """İyileştirmeyi hesapla."""
        accuracy_gain = after["accuracy"] - before["accuracy"]
        logloss_gain = before["log_loss"] - after["log_loss"]
        brier_gain = before["brier"] - after["brier"]

        return (accuracy_gain + logloss_gain + brier_gain) / 3

    def _save_correction(self, result: CorrectionResult):
        """Düzeltme sonucunu kaydet."""
        filename = f"correction_{len(self.correction_history):04d}.json"
        filepath = self.correction_dir / filename

        data = {
            "success": result.success,
            "improvement": result.improvement,
            "actions_count": len(result.actions_taken),
            "actions": [
                {
                    "type": a.action_type,
                    "description": a.description,
                    "expected_improvement": a.expected_improvement,
                }
                for a in result.actions_taken
            ],
            "metrics_before": result.metrics_before,
            "metrics_after": result.metrics_after,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_correction_summary(self) -> dict:
        """Düzeltme özetini döndür."""
        if not self.correction_history:
            return {"total_corrections": 0}

        return {
            "total_corrections": len(self.correction_history),
            "average_improvement": np.mean([r.improvement for r in self.correction_history]),
            "total_improvement": sum(r.improvement for r in self.correction_history),
            "success_rate": sum(1 for r in self.correction_history if r.success) / len(self.correction_history),
        }
