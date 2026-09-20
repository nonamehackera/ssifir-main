"""Pattern Detector - Sistemli hata kalıplarını tespit eder.

Bu modül, modelin sürekli olarak yanlış tahmin ettiği durumları
tespit ederek kalıplarını ortaya çıkarır.

Tespit edilen kalıp türleri:
1. Lig bazında sistemli önyargı (bir ligde hep yanlış)
2. Takım bazında önyargı (bir takımı hep yanlış tahmin etme)
3. Sonuç bazında önyargı (örn: beraberlikleri hep kaçırma)
4. Gol aralığı bazında önyargı (yüksek skorlu maçları tahmin edememe)
5. Zamanlamalı önyargı (sezon sonu gibi dönemsel sapmalar)
6. Güven bazında önyargı (yüksek güvenli tahminlerde hata)
7. Kombinasyon kalıpları (birden fazla faktörün birleşimi)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from feedback.error_tracker import ErrorTracker, PredictionRecord

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Tespit edilen bir hata kalıbı."""
    pattern_type: str  # lig, takim, sonuc, gol_araligi, zamanlama, guven, kombinasyon
    description: str
    severity: float  # 0-1 arası (1 = çok ciddi)
    confidence: float  # 0-1 arası (kalıbın güvenilirliği)
    affected_matches: int
    affected_ratio: float  # Toplam maç oranı
    error_rate: float  # Bu kalıpta hata oranı
    expected_error_rate: float  # Beklenen hata oranı
    details: dict = field(default_factory=dict)
    recommendation: str = ""


class PatternDetector:
    """Hata kalıplarını tespit eden sınıf.

    Özellikler:
    - Çoklu kalıp tespiti
    - İstatistiksel anlamlılık testi
    - Kalıpları önceliklendirme
    - Düzeltme önerileri
    """

    def __init__(self, error_tracker: ErrorTracker):
        self.tracker = error_tracker
        self.min_sample_size = 20  # Minimum örnek sayısı
        self.significance_level = 0.05  # İstatistiksel anlamlılık eşiği

    def detect_all_patterns(self) -> list[ErrorPattern]:
        """Tüm kalıp türlerini tespit et."""
        patterns = []

        patterns.extend(self._detect_league_bias())
        patterns.extend(self._detect_team_bias())
        patterns.extend(self._detect_result_bias())
        patterns.extend(self._detect_goal_range_bias())
        patterns.extend(self._detect_confidence_bias())
        patterns.extend(self._detect_temporal_bias())

        # Önceliklendirme (ciddiyet * güven)
        patterns.sort(key=lambda p: p.severity * p.confidence, reverse=True)

        return patterns

    def _detect_league_bias(self) -> list[ErrorPattern]:
        """Lig bazında sistemli önyargı tespiti."""
        patterns = []
        records = self.tracker.records

        if not records:
            return patterns

        # Tüm ligleri analiz et
        leagues = {}
        for r in records:
            if r.league:
                if r.league not in leagues:
                    leagues[r.league] = []
                leagues[r.league].append(r)

        # Genel hata oranı
        total_error_rate = len([r for r in records if not r.is_correct]) / len(records)

        for league, league_records in leagues.items():
            if len(league_records) < self.min_sample_size:
                continue

            incorrect = len([r for r in league_records if not r.is_correct])
            error_rate = incorrect / len(league_records)

            # Hata oranı genel ortalamadan anlamlı derecede farklı mı?
            if error_rate > total_error_rate * 1.2:  # %20 daha fazla hata
                # Binomial test
                result = stats.binomtest(
                    incorrect,
                    len(league_records),
                    total_error_rate,
                    alternative='greater'
                )
                p_value = result.pvalue

                if p_value < self.significance_level:
                    severity = min(1.0, (error_rate - total_error_rate) / total_error_rate)
                    confidence = 1 - p_value

                    # Ligdeki spesifik sorunları analiz et
                    details = self._analyze_league_details(league_records)

                    patterns.append(ErrorPattern(
                        pattern_type="lig",
                        description=f"{league} liginde sistemli hata: %{error_rate*100:.1f} hata oranı (genel: %{total_error_rate*100:.1f})",
                        severity=severity,
                        confidence=confidence,
                        affected_matches=len(league_records),
                        affected_ratio=len(league_records) / len(records),
                        error_rate=error_rate,
                        expected_error_rate=total_error_rate,
                        details=details,
                        recommendation=f"{league} ligi için model ayarları gözden geçirilmeli"
                    ))

        return patterns

    def _analyze_league_details(self, records: list[PredictionRecord]) -> dict:
        """Ligdeki spesifik hata detaylarını analiz et."""
        details = {
            "total_matches": len(records),
            "incorrect_matches": len([r for r in records if not r.is_correct]),
        }

        # Sonuç bazında analiz
        result_errors = {"H": 0, "D": 0, "A": 0}
        result_counts = {"H": 0, "D": 0, "A": 0}

        for r in records:
            result_counts[r.actual_result] += 1
            if not r.is_correct:
                result_errors[r.actual_result] += 1

        details["result_error_rates"] = {
            result: result_errors[result] / max(result_counts[result], 1)
            for result in ["H", "D", "A"]
        }

        # En çok hangi sonucu kaçırıyor
        worst_result = max(result_errors, key=lambda x: result_errors[x] / max(result_counts[x], 1))
        details["worst_predicted_result"] = worst_result

        return details

    def _detect_team_bias(self) -> list[ErrorPattern]:
        """Takım bazında sistemli önyargı tespiti."""
        patterns = []
        records = self.tracker.records

        if not records:
            return patterns

        # Takımları topla
        teams = {}
        for r in records:
            if r.home_team:
                if r.home_team not in teams:
                    teams[r.home_team] = {"home": [], "away": []}
                teams[r.home_team]["home"].append(r)
            if r.away_team:
                if r.away_team not in teams:
                    teams[r.away_team] = {"home": [], "away": []}
                teams[r.away_team]["away"].append(r)

        total_error_rate = len([r for r in records if not r.is_correct]) / len(records)

        for team, team_records in teams.items():
            all_records = team_records["home"] + team_records["away"]
            if len(all_records) < self.min_sample_size:
                continue

            incorrect = len([r for r in all_records if not r.is_correct])
            error_rate = incorrect / len(all_records)

            if error_rate > total_error_rate * 1.3:  # %30 daha fazla hata
                p_value = stats.binomtest(
                    incorrect,
                    len(all_records),
                    total_error_rate,
                    alternative='greater'
                ).pvalue

                if p_value < self.significance_level:
                    severity = min(1.0, (error_rate - total_error_rate) / total_error_rate)
                    confidence = 1 - p_value

                    # Takım için spesifik sorunları analiz et
                    details = self._analyze_team_details(all_records)

                    patterns.append(ErrorPattern(
                        pattern_type="takim",
                        description=f"{team} takımı için sistemli hata: %{error_rate*100:.1f} hata oranı",
                        severity=severity,
                        confidence=confidence,
                        affected_matches=len(all_records),
                        affected_ratio=len(all_records) / len(records),
                        error_rate=error_rate,
                        expected_error_rate=total_error_rate,
                        details=details,
                        recommendation=f"{team} takımı için özellik mühendisliği iyileştirilmeli"
                    ))

        return patterns

    def _analyze_team_details(self, records: list[PredictionRecord]) -> dict:
        """Takım için spesifik hata detaylarını analiz et."""
        details = {
            "total_matches": len(records),
            "home_matches": len([r for r in records if r.home_team == records[0].home_team]),
            "away_matches": len([r for r in records if r.away_team == records[0].home_team]),
        }

        # Ev sahibi/deplasman performansı
        home_records = [r for r in records if r.home_team == records[0].home_team]
        away_records = [r for r in records if r.away_team == records[0].home_team]

        if home_records:
            home_error = len([r for r in home_records if not r.is_correct]) / len(home_records)
            details["home_error_rate"] = home_error

        if away_records:
            away_error = len([r for r in away_records if not r.is_correct]) / len(away_records)
            details["away_error_rate"] = away_error

        return details

    def _detect_result_bias(self) -> list[ErrorPattern]:
        """Sonuç bazında sistemli önyargı tespiti (H/D/A)."""
        patterns = []
        records = self.tracker.records

        if not records:
            return patterns

        # Sonuç gruplarını analiz et
        result_groups = {"H": [], "D": [], "A": []}
        for r in records:
            result_groups[r.actual_result].append(r)

        total_error_rate = len([r for r in records if not r.is_correct]) / len(records)

        for result, result_records in result_groups.items():
            if len(result_records) < self.min_sample_size:
                continue

            incorrect = len([r for r in result_records if not r.is_correct])
            error_rate = incorrect / len(result_records)

            # Belirli bir sonucu tahmin etmede sistemli sorun var mı?
            if error_rate > total_error_rate * 1.25:
                p_value = stats.binomtest(
                    incorrect,
                    len(result_records),
                    total_error_rate,
                    alternative='greater'
                ).pvalue

                if p_value < self.significance_level:
                    severity = min(1.0, (error_rate - total_error_rate) / total_error_rate)
                    confidence = 1 - p_value

                    # Ortalama tahmin olasılıklarını analiz et
                    avg_preds = {
                        "pred_home": np.mean([r.pred_home for r in result_records]),
                        "pred_draw": np.mean([r.pred_draw for r in result_records]),
                        "pred_away": np.mean([r.pred_away for r in result_records]),
                    }

                    patterns.append(ErrorPattern(
                        pattern_type="sonuc",
                        description=f"'{result}' sonuçlarını tahmin etmede sistemli hata: %{error_rate*100:.1f}",
                        severity=severity,
                        confidence=confidence,
                        affected_matches=len(result_records),
                        affected_ratio=len(result_records) / len(records),
                        error_rate=error_rate,
                        expected_error_rate=total_error_rate,
                        details={
                            "result": result,
                            "avg_predictions": avg_preds,
                            "sample_size": len(result_records),
                        },
                        recommendation=f"'{result}' sonucu için kalibrasyon ayarlanmalı"
                    ))

        return patterns

    def _detect_goal_range_bias(self) -> list[ErrorPattern]:
        """Gol aralığı bazında sistemli önyargı tespiti."""
        patterns = []
        records = self.tracker.records

        if not records:
            return patterns

        # Gol aralıklarına göre grupla
        goal_ranges = {
            "dusuk_skorlu": [],  # 0-1 gol
            "orta_skorlu": [],   # 2-3 gol
            "yuksek_skorlu": [], # 4+ gol
        }

        for r in records:
            total_goals = r.home_goals + r.away_goals
            if total_goals <= 1:
                goal_ranges["dusuk_skorlu"].append(r)
            elif total_goals <= 3:
                goal_ranges["orta_skorlu"].append(r)
            else:
                goal_ranges["yuksek_skorlu"].append(r)

        total_error_rate = len([r for r in records if not r.is_correct]) / len(records)

        for goal_range, range_records in goal_ranges.items():
            if len(range_records) < self.min_sample_size:
                continue

            incorrect = len([r for r in range_records if not r.is_correct])
            error_rate = incorrect / len(range_records)

            if error_rate > total_error_rate * 1.3:
                p_value = stats.binomtest(
                    incorrect,
                    len(range_records),
                    total_error_rate,
                    alternative='greater'
                ).pvalue

                if p_value < self.significance_level:
                    severity = min(1.0, (error_rate - total_error_rate) / total_error_rate)
                    confidence = 1 - p_value

                    patterns.append(ErrorPattern(
                        pattern_type="gol_araligi",
                        description=f"{goal_range} maçlarda sistemli hata: %{error_rate*100:.1f}",
                        severity=severity,
                        confidence=confidence,
                        affected_matches=len(range_records),
                        affected_ratio=len(range_records) / len(records),
                        error_rate=error_rate,
                        expected_error_rate=total_error_rate,
                        details={
                            "goal_range": goal_range,
                            "avg_goals": np.mean([r.home_goals + r.away_goals for r in range_records]),
                        },
                        recommendation=f"{goal_range} maçlar için gol modeli iyileştirilmeli"
                    ))

        return patterns

    def _detect_confidence_bias(self) -> list[ErrorPattern]:
        """Güven bazında sistemli önyargı tespiti."""
        patterns = []
        records = self.tracker.records

        if not records:
            return patterns

        # Güven seviyelerine göre grupla
        confidence_groups = {
            "dusuk_guven": [],   # <0.5
            "orta_guven": [],    # 0.5-0.7
            "yuksek_guven": [],  # 0.7-0.9
            "cok_yuksek_guven": [],  # >0.9
        }

        for r in records:
            max_prob = max(r.pred_home, r.pred_draw, r.pred_away)
            if max_prob < 0.5:
                confidence_groups["dusuk_guven"].append(r)
            elif max_prob < 0.7:
                confidence_groups["orta_guven"].append(r)
            elif max_prob < 0.9:
                confidence_groups["yuksek_guven"].append(r)
            else:
                confidence_groups["cok_yuksek_guven"].append(r)

        total_error_rate = len([r for r in records if not r.is_correct]) / len(records)

        for conf_group, group_records in confidence_groups.items():
            if len(group_records) < self.min_sample_size:
                continue

            incorrect = len([r for r in group_records if not r.is_correct])
            error_rate = incorrect / len(group_records)

            # Yüksek güvenli tahminlerde hata oranı yüksekse bu ciddi bir sorun
            if conf_group in ["yuksek_guven", "cok_yuksek_guven"] and error_rate > 0.15:
                p_value = stats.binomtest(
                    incorrect,
                    len(group_records),
                    total_error_rate,
                    alternative='greater'
                ).pvalue

                if p_value < self.significance_level:
                    severity = min(1.0, error_rate / 0.3)  # %30 hata = maksimum ciddiyet
                    confidence = 1 - p_value

                    patterns.append(ErrorPattern(
                        pattern_type="guven",
                        description=f"{conf_group} tahminlerde güvenilirlik sorunu: %{error_rate*100:.1f} hata",
                        severity=severity,
                        confidence=confidence,
                        affected_matches=len(group_records),
                        affected_ratio=len(group_records) / len(records),
                        error_rate=error_rate,
                        expected_error_rate=total_error_rate,
                        details={
                            "confidence_level": conf_group,
                            "avg_confidence": np.mean([
                                max(r.pred_home, r.pred_draw, r.pred_away)
                                for r in group_records
                            ]),
                        },
                        recommendation=f"Güven hesaplama mekanizması gözden geçirilmeli"
                    ))

        return patterns

    def _detect_temporal_bias(self) -> list[ErrorPattern]:
        """Zamanlamalı önyargı tespiti (sezon, ay bazında)."""
        patterns = []
        records = self.tracker.records

        if not records:
            return patterns

        # Tarih bilgisi olanları filtrele
        dated_records = []
        for r in records:
            try:
                dt = pd.to_datetime(r.prediction_time)
                dated_records.append((r, dt))
            except:
                continue

        if len(dated_records) < self.min_sample_size * 2:
            return patterns

        # Aylara göre grupla
        monthly_groups = {}
        for r, dt in dated_records:
            month = dt.month
            if month not in monthly_groups:
                monthly_groups[month] = []
            monthly_groups[month].append(r)

        total_error_rate = len([r for r in records if not r.is_correct]) / len(records)

        for month, month_records in monthly_groups.items():
            if len(month_records) < self.min_sample_size:
                continue

            incorrect = len([r for r in month_records if not r.is_correct])
            error_rate = incorrect / len(month_records)

            if error_rate > total_error_rate * 1.3:
                p_value = stats.binomtest(
                    incorrect,
                    len(month_records),
                    total_error_rate,
                    alternative='greater'
                ).pvalue

                if p_value < self.significance_level:
                    severity = min(1.0, (error_rate - total_error_rate) / total_error_rate)
                    confidence = 1 - p_value

                    month_names = [
                        "", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                        "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"
                    ]

                    patterns.append(ErrorPattern(
                        pattern_type="zamanlama",
                        description=f"{month_names[month]} ayında sistemli hata: %{error_rate*100:.1f}",
                        severity=severity,
                        confidence=confidence,
                        affected_matches=len(month_records),
                        affected_ratio=len(month_records) / len(records),
                        error_rate=error_rate,
                        expected_error_rate=total_error_rate,
                        details={
                            "month": month,
                            "month_name": month_names[month],
                            "sample_size": len(month_records),
                        },
                        recommendation=f"Sezon dinamikleri için zamanlama özellikleri eklenmeli"
                    ))

        return patterns

    def get_top_patterns(self, n: int = 5) -> list[ErrorPattern]:
        """En önemli n kalıbı döndür."""
        all_patterns = self.detect_all_patterns()
        return all_patterns[:n]

    def generate_report(self) -> str:
        """Kalıp tespiti raporu oluştur."""
        patterns = self.detect_all_patterns()

        if not patterns:
            return "Belirgin hata kalıbı tespit edilemedi."

        report = ["HATA KALIBI RAPORU", "=" * 50, ""]

        for i, pattern in enumerate(patterns, 1):
            report.extend([
                f"{i}. {pattern.pattern_type.upper()} KALIBI",
                f"   Açıklama: {pattern.description}",
                f"   Ciddiyet: {pattern.severity:.2f}",
                f"   Güven: {pattern.confidence:.2f}",
                f"   Etkilenen maç: {pattern.affected_matches} (%{pattern.affected_ratio*100:.1f})",
                f"   Hata oranı: %{pattern.error_rate*100:.1f} (beklenen: %{pattern.expected_error_rate*100:.1f})",
                f"   Öneri: {pattern.recommendation}",
                "",
            ])

        return "\n".join(report)
