"""Feedback-Aware Predictor - Tahminleri otomatik iyilestiren modul.

Tahmin yapilirken:
1. Takimlarin gecmis hata profillerini kontrol eder
2. Ligdeki sistematik onyargilari tespit eder
3. Benzer mac tiplerini (derbi, yuksek ELO) analiz eder
4. Tahmini buna gore ayarlar
5. Tahmini kaydeder (gelecekte ogrenmek icin)

Sonuc bilindiginde:
6. Gercek sonucu kaydeder
7. Hata kaliblarini gunceller
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-10


@dataclass
class FeedbackAdjustment:
    """Geri besleme ayarlamasi sonucu."""
    original_home: float
    original_draw: float
    original_away: float
    adjusted_home: float
    adjusted_draw: float
    adjusted_away: float
    home_adjustment: float = 0.0
    draw_adjustment: float = 0.0
    away_adjustment: float = 0.0
    confidence_factor: float = 1.0
    adjustments_applied: list = field(default_factory=list)
    reasons: list = field(default_factory=list)


class FeedbackAwarePredictor:
    """Geri besleme ile iyilestirilmis tahminci.

    Ozellikleri:
    - Takim bazli hata profilleri
    - Lig bazli hata profilleri
    - Mac tipi bazli ayarlama
    - Dinamik kalibrasyon
    - Otomatik ogrenme
    """

    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Hata profilleri dosyalari
        self.team_profiles_file = self.data_dir / "team_error_profiles.json"
        self.league_profiles_file = self.data_dir / "league_error_profiles.json"
        self.match_type_profiles_file = self.data_dir / "match_type_profiles.json"
        self.prediction_log_file = self.data_dir / "prediction_log.json"

        # Profilleri yukle
        self.team_profiles = self._load_json(self.team_profiles_file, {})
        self.league_profiles = self._load_json(self.league_profiles_file, {})
        self.match_type_profiles = self._load_json(self.match_type_profiles_file, {})
        self.prediction_log = self._load_json(self.prediction_log_file, [])

        # Ayarlam parametreleri
        self.max_adjustment_per_team = 0.15  # Takim basina maks ayarlama
        self.max_adjustment_per_league = 0.10  # Lig basina maks ayarlama
        self.min_samples_for_adjustment = 10  # Minimum ornek sayisi
        self.recent_weight = 0.7  # Son tahminlerin agirligi
        self.combination_max = 0.25  # Kombinasyon ayarlamas maximum

    def _load_json(self, filepath: Path, default):
        """JSON dosyasini yukle."""
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"JSON yuklenemedi {filepath}: {e}")
        return default

    def _save_json(self, filepath: Path, data):
        """JSON dosyasina kaydet."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON kaydedilemedi {filepath}: {e}")

    def adjust_prediction(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        elo_diff: float = 0.0,
        home_elo: float = 1500.0,
        away_elo: float = 1500.0,
    ) -> FeedbackAdjustment:
        """Tahmini geri besleme ile ayarla.

        1. Takim profillerinden ayarlama
        2. Lig profilinden ayarlama
        3. Mac tipi profilinden ayarlama
        4. Kombinasyon etkisi
        5. Normalize et
        """
        home_key = str(home_team_id)
        away_key = str(away_team_id)

        # Baslangic degerleri
        adj_home, adj_draw, adj_away = pred_home, pred_draw, pred_away
        reasons = []
        adjustments_applied = []

        # 1. TAKIM BAZLI AYARLAMA
        home_adj = self._get_team_adjustment(home_key, "home")
        away_adj = self._get_team_adjustment(away_key, "away")

        if home_adj is not None:
            # Ev sahibi takimin hata kalibi
            h_bias = home_adj.get("home_bias", 0)  # Ev sahibi lehine onyargi
            d_bias = home_adj.get("draw_bias", 0)  # Beraberlik onyargisi
            a_bias = home_adj.get("away_bias", 0)  # Deplasman onyargisi

            if abs(h_bias) > 0.02:
                adj_home += h_bias * self.max_adjustment_per_team
                reasons.append(f"Ev sahibi takim hatasi: {h_bias:.3f}")
                adjustments_applied.append("home_team")

            if abs(d_bias) > 0.02:
                adj_draw += d_bias * self.max_adjustment_per_team
                reasons.append(f"Ev sahibi beraberlik hatasi: {d_bias:.3f}")

            if abs(a_bias) > 0.02:
                adj_away += a_bias * self.max_adjustment_per_team
                reasons.append(f"Ev sahibi deplasman hatasi: {a_bias:.3f}")

        if away_adj is not None:
            h_bias = away_adj.get("home_bias", 0)
            d_bias = away_adj.get("draw_bias", 0)
            a_bias = away_adj.get("away_bias", 0)

            if abs(h_bias) > 0.02:
                adj_home += h_bias * self.max_adjustment_per_team * 0.5
                reasons.append(f"Dep. takim ev sahibi hatasi: {h_bias:.3f}")
                adjustments_applied.append("away_team")

            if abs(d_bias) > 0.02:
                adj_draw += d_bias * self.max_adjustment_per_team * 0.5

            if abs(a_bias) > 0.02:
                adj_away += a_bias * self.max_adjustment_per_team * 0.5

        # 2. LIG BAZLI AYARLAMA
        league_adj = self._get_league_adjustment(league)
        if league_adj is not None:
            lg_home_bias = league_adj.get("home_bias", 0)
            lg_draw_bias = league_adj.get("draw_bias", 0)
            lg_away_bias = league_adj.get("away_bias", 0)

            if abs(lg_home_bias) > 0.02:
                adj_home += lg_home_bias * self.max_adjustment_per_league
                reasons.append(f"Lig ev sahibi hatasi: {lg_home_bias:.3f}")
                adjustments_applied.append("league")

            if abs(lg_draw_bias) > 0.02:
                adj_draw += lg_draw_bias * self.max_adjustment_per_league
                reasons.append(f"Lig beraberlik hatasi: {lg_draw_bias:.3f}")

            if abs(lg_away_bias) > 0.02:
                adj_away += lg_away_bias * self.max_adjustment_per_league
                reasons.append(f"Lig deplasman hatasi: {lg_away_bias:.3f}")

        # 3. MAC TIPI BAZLI AYARLAMA
        match_type = self._classify_match_type(elo_diff, home_elo, away_elo)
        type_adj = self._get_match_type_adjustment(match_type)
        if type_adj is not None:
            t_home = type_adj.get("home_bias", 0)
            t_draw = type_adj.get("draw_bias", 0)
            t_away = type_adj.get("away_bias", 0)

            if abs(t_home) > 0.02:
                adj_home += t_home * 0.08
                reasons.append(f"Mac tipi ({match_type}) hatasi: {t_home:.3f}")
                adjustments_applied.append("match_type")

            if abs(t_draw) > 0.02:
                adj_draw += t_draw * 0.08
                reasons.append(f"Mac tipi ({match_type}) beraberlik hatasi: {t_draw:.3f}")

            if abs(t_away) > 0.02:
                adj_away += t_away * 0.08

        # 4. KOMBINASYON ETKISI
        if len(adjustments_applied) >= 2:
            # Birden fazla ayarlama varsa etkileşim kontrolü
            combo_effect = self._calculate_combination_effect(
                home_adj, away_adj, league_adj, type_adj
            )
            if abs(combo_effect) > 0.01:
                adj_draw += combo_effect * 0.5
                reasons.append(f"Kombinasyon etkisi: {combo_effect:.3f}")
                adjustments_applied.append("combination")

        # 5. GUVEN FAKTORU
        confidence_factor = self._calculate_confidence_factor(
            home_key, away_key, league
        )

        # 6. NORMALIZE ET
        total = adj_home + adj_draw + adj_away
        if total > 0:
            adj_home /= total
            adj_draw /= total
            adj_away /= total
        else:
            adj_home, adj_draw, adj_away = pred_home, pred_draw, pred_away

        # Ciddi sapmalar icin sinirliyic
        max_change = 0.20
        adj_home = np.clip(adj_home, pred_home - max_change, pred_home + max_change)
        adj_draw = np.clip(adj_draw, pred_draw - max_change, pred_draw + max_change)
        adj_away = np.clip(adj_away, pred_away - max_change, pred_away + max_change)

        # Tekrar normalize et
        total = adj_home + adj_draw + adj_away
        if total > 0:
            adj_home /= total
            adj_draw /= total
            adj_away /= total

        return FeedbackAdjustment(
            original_home=pred_home,
            original_draw=pred_draw,
            original_away=pred_away,
            adjusted_home=float(adj_home),
            adjusted_draw=float(adj_draw),
            adjusted_away=float(adj_away),
            home_adjustment=float(adj_home - pred_home),
            draw_adjustment=float(adj_draw - pred_draw),
            away_adjustment=float(adj_away - pred_away),
            confidence_factor=confidence_factor,
            adjustments_applied=adjustments_applied,
            reasons=reasons,
        )

    def _get_team_adjustment(self, team_key: str, side: str) -> Optional[dict]:
        """Takim icin ayarlama degeri al."""
        if team_key not in self.team_profiles:
            return None

        profile = self.team_profiles[team_key]
        n = profile.get("total_predictions", 0)

        if n < self.min_samples_for_adjustment:
            return None

        # Son tahminlere daha fazla agirlik ver
        recent = profile.get("recent_predictions", [])
        if len(recent) > 5:
            recent = recent[-10:]

        home_errors = []
        draw_errors = []
        away_errors = []

        for pred in recent:
            if pred.get("was_correct"):
                continue

            actual = pred.get("actual_result", "")
            pred_h = pred.get("pred_home", 0.33)
            pred_d = pred.get("pred_draw", 0.33)
            pred_a = pred.get("pred_away", 0.33)

            # Hata yonunu hesapla
            if actual == "H":
                home_errors.append(1 - pred_h)
                draw_errors.append(-pred_d)
                away_errors.append(-pred_a)
            elif actual == "D":
                home_errors.append(-pred_h)
                draw_errors.append(1 - pred_d)
                away_errors.append(-pred_a)
            elif actual == "A":
                home_errors.append(-pred_h)
                draw_errors.append(-pred_d)
                away_errors.append(1 - pred_a)

        if not home_errors:
            return None

        # Agirlikli ortalama (son hatalara daha fazla agirlik)
        weights = np.exp(np.linspace(-1, 0, len(home_errors)))
        weights /= weights.sum()

        return {
            "home_bias": float(np.average(home_errors, weights=weights)),
            "draw_bias": float(np.average(draw_errors, weights=weights)),
            "away_bias": float(np.average(away_errors, weights=weights)),
            "sample_size": n,
        }

    def _get_league_adjustment(self, league: str) -> Optional[dict]:
        """Lig icin ayarlama degeri al."""
        if league not in self.league_profiles:
            return None

        profile = self.league_profiles[league]
        n = profile.get("total_predictions", 0)

        if n < self.min_samples_for_adjustment:
            return None

        return {
            "home_bias": profile.get("home_bias", 0),
            "draw_bias": profile.get("draw_bias", 0),
            "away_bias": profile.get("away_bias", 0),
            "sample_size": n,
        }

    def _classify_match_type(self, elo_diff: float, home_elo: float, away_elo: float) -> str:
        """Mac tipini siniflandir."""
        abs_diff = abs(elo_diff)

        if abs_diff < 30:
            return "derbi"
        elif abs_diff < 100:
            return "yaklasik_esit"
        elif abs_diff < 200:
            return "hafif_favori"
        elif home_elo > 1700 or away_elo > 1700:
            return " buyuk_mac"
        else:
            return "net_favori"

    def _get_match_type_adjustment(self, match_type: str) -> Optional[dict]:
        """Mac tipi icin ayarlama degeri al."""
        if match_type not in self.match_type_profiles:
            return None

        profile = self.match_type_profiles[match_type]
        n = profile.get("total_predictions", 0)

        if n < 5:
            return None

        return {
            "home_bias": profile.get("home_bias", 0),
            "draw_bias": profile.get("draw_bias", 0),
            "away_bias": profile.get("away_bias", 0),
        }

    def _calculate_combination_effect(
        self, home_adj, away_adj, league_adj, type_adj
    ) -> float:
        """Kombinasyon etkisini hesapla."""
        effects = []
        if home_adj is not None:
            effects.append(home_adj.get("home_bias", 0))
        if away_adj is not None:
            effects.append(away_adj.get("away_bias", 0))
        if league_adj is not None:
            effects.append(league_adj.get("draw_bias", 0))
        if type_adj is not None:
            effects.append(type_adj.get("draw_bias", 0))

        if not effects:
            return 0.0

        # Etkilerin ayni yonde olup olmadigini kontrol et
        positive = sum(1 for e in effects if e > 0)
        negative = sum(1 for e in effects if e < 0)

        if positive > len(effects) / 2:
            return np.mean([e for e in effects if e > 0]) * 0.3
        elif negative > len(effects) / 2:
            return np.mean([e for e in effects if e < 0]) * 0.3

        return 0.0

    def _calculate_confidence_factor(
        self, home_key: str, away_key: str, league: str
    ) -> float:
        """Guven faktorunu hesapla."""
        factors = []

        # Takim guvenilirligi
        if home_key in self.team_profiles:
            n = self.team_profiles[home_key].get("total_predictions", 0)
            factors.append(min(1.0, n / 50))

        if away_key in self.team_profiles:
            n = self.team_profiles[away_key].get("total_predictions", 0)
            factors.append(min(1.0, n / 50))

        # Lig guvenilirligi
        if league in self.league_profiles:
            n = self.league_profiles[league].get("total_predictions", 0)
            factors.append(min(1.0, n / 100))

        if not factors:
            return 0.5

        return float(np.mean(factors))

    def record_prediction(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        elo_diff: float = 0.0,
        match_id: str = "",
    ):
        """Tahmini kaydet (gelecekte ogrenmek icin)."""
        home_key = str(home_team_id)
        away_key = str(away_team_id)

        prediction_record = {
            "match_id": match_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "league": league,
            "pred_home": pred_home,
            "pred_draw": pred_draw,
            "pred_away": pred_away,
            "elo_diff": elo_diff,
            "timestamp": pd.Timestamp.now().isoformat(),
            "actual_result": None,
            "was_correct": None,
        }

        # Tahmin kaydini ekle
        self.prediction_log.append(prediction_record)

        # Takim profillerini guncelle
        self._update_team_profile(home_key, prediction_record)
        self._update_team_profile(away_key, prediction_record)

        # Lig profilini guncelle
        self._update_league_profile(league, prediction_record)

        # Mac tipi profilini guncelle
        match_type = self._classify_match_type(elo_diff, 1500, 1500)
        self._update_match_type_profile(match_type, prediction_record)

        # Kaydet
        self._save_json(self.prediction_log_file, self.prediction_log)
        self._save_json(self.team_profiles_file, self.team_profiles)
        self._save_json(self.league_profiles_file, self.league_profiles)
        self._save_json(self.match_type_profiles_file, self.match_type_profiles)

    def record_result(self, match_id: str, actual_result: str):
        """Gercek sonucu kaydet ve profilleri guncelle."""
        for pred in self.prediction_log:
            if pred.get("match_id") == match_id:
                pred["actual_result"] = actual_result
                pred["was_correct"] = self._check_correct(pred, actual_result)

                # Takim profillerini guncelle
                home_key = str(pred["home_team_id"])
                away_key = str(pred["away_team_id"])
                self._update_team_profile_with_result(home_key, pred, actual_result)
                self._update_team_profile_with_result(away_key, pred, actual_result)

                # Lig profilini guncelle
                self._update_league_profile_with_result(pred["league"], pred, actual_result)

                # Mac tipi profilini guncelle
                match_type = self._classify_match_type(pred.get("elo_diff", 0), 1500, 1500)
                self._update_match_type_profile_with_result(match_type, pred, actual_result)

                break

        # Kaydet
        self._save_json(self.prediction_log_file, self.prediction_log)
        self._save_json(self.team_profiles_file, self.team_profiles)
        self._save_json(self.league_profiles_file, self.league_profiles)
        self._save_json(self.match_type_profiles_file, self.match_type_profiles)

    def _check_correct(self, pred: dict, actual_result: str) -> bool:
        """Tahminin dogru olup olmadigini kontrol et."""
        probs = [pred["pred_home"], pred["pred_draw"], pred["pred_away"]]
        predicted = ["H", "D", "A"][np.argmax(probs)]
        return predicted == actual_result

    def _update_team_profile(self, team_key: str, pred: dict):
        """Takim profilini guncelle."""
        if team_key not in self.team_profiles:
            self.team_profiles[team_key] = {
                "total_predictions": 0,
                "recent_predictions": [],
                "home_bias": 0,
                "draw_bias": 0,
                "away_bias": 0,
            }

        profile = self.team_profiles[team_key]
        profile["total_predictions"] = profile.get("total_predictions", 0) + 1

        # Son 20 tahmini sakla
        recent = profile.get("recent_predictions", [])
        recent.append({
            "pred_home": pred["pred_home"],
            "pred_draw": pred["pred_draw"],
            "pred_away": pred["pred_away"],
            "league": pred["league"],
            "actual_result": None,
            "was_correct": None,
        })
        profile["recent_predictions"] = recent[-20:]

    def _update_team_profile_with_result(self, team_key: str, pred: dict, actual_result: str):
        """Takim profilini sonuc ile guncelle."""
        if team_key not in self.team_profiles:
            return

        profile = self.team_profiles[team_key]

        # Son tahmini guncelle
        recent = profile.get("recent_predictions", [])
        for r in reversed(recent):
            if r.get("pred_home") == pred.get("pred_home"):
                r["actual_result"] = actual_result
                r["was_correct"] = self._check_correct(pred, actual_result)
                break

        # Bias hesapla
        errors_h, errors_d, errors_a = [], [], []
        for r in recent:
            if r.get("actual_result") is None:
                continue
            if r.get("was_correct"):
                continue

            actual = r["actual_result"]
            if actual == "H":
                errors_h.append(1 - r["pred_home"])
                errors_d.append(-r["pred_draw"])
                errors_a.append(-r["pred_away"])
            elif actual == "D":
                errors_h.append(-r["pred_home"])
                errors_d.append(1 - r["pred_draw"])
                errors_a.append(-r["pred_away"])
            elif actual == "A":
                errors_h.append(-r["pred_home"])
                errors_d.append(-r["pred_draw"])
                errors_a.append(1 - r["pred_away"])

        if errors_h:
            weights = np.exp(np.linspace(-1, 0, len(errors_h)))
            weights /= weights.sum()
            profile["home_bias"] = float(np.average(errors_h, weights=weights))
            profile["draw_bias"] = float(np.average(errors_d, weights=weights))
            profile["away_bias"] = float(np.average(errors_a, weights=weights))

    def _update_league_profile(self, league: str, pred: dict):
        """Lig profilini guncelle."""
        if league not in self.league_profiles:
            self.league_profiles[league] = {
                "total_predictions": 0,
                "home_bias": 0,
                "draw_bias": 0,
                "away_bias": 0,
                "recent_errors": [],
            }

        profile = self.league_profiles[league]
        profile["total_predictions"] = profile.get("total_predictions", 0) + 1

        # Son hatalari sakla
        recent_errors = profile.get("recent_errors", [])
        recent_errors.append({
            "pred_home": pred["pred_home"],
            "pred_draw": pred["pred_draw"],
            "pred_away": pred["pred_away"],
            "actual_result": None,
        })
        profile["recent_errors"] = recent_errors[-50:]

    def _update_league_profile_with_result(self, league: str, pred: dict, actual_result: str):
        """Lig profilini sonuc ile guncelle."""
        if league not in self.league_profiles:
            return

        profile = self.league_profiles[league]

        # Son hatalari guncelle
        recent_errors = profile.get("recent_errors", [])
        for r in reversed(recent_errors):
            if r.get("pred_home") == pred.get("pred_home") and r.get("actual_result") is None:
                r["actual_result"] = actual_result
                break

        # Bias hesapla
        errors_h, errors_d, errors_a = [], [], []
        for r in recent_errors:
            if r.get("actual_result") is None:
                continue

            actual = r["actual_result"]
            if actual == "H":
                errors_h.append(1 - r["pred_home"])
                errors_d.append(-r["pred_draw"])
                errors_a.append(-r["pred_away"])
            elif actual == "D":
                errors_h.append(-r["pred_home"])
                errors_d.append(1 - r["pred_draw"])
                errors_a.append(-r["pred_away"])
            elif actual == "A":
                errors_h.append(-r["pred_home"])
                errors_d.append(-r["pred_draw"])
                errors_a.append(1 - r["pred_away"])

        if errors_h:
            weights = np.exp(np.linspace(-1, 0, len(errors_h)))
            weights /= weights.sum()
            profile["home_bias"] = float(np.average(errors_h, weights=weights))
            profile["draw_bias"] = float(np.average(errors_d, weights=weights))
            profile["away_bias"] = float(np.average(errors_a, weights=weights))

    def _update_match_type_profile(self, match_type: str, pred: dict):
        """Mac tipi profilini guncelle."""
        if match_type not in self.match_type_profiles:
            self.match_type_profiles[match_type] = {
                "total_predictions": 0,
                "home_bias": 0,
                "draw_bias": 0,
                "away_bias": 0,
                "recent_errors": [],
            }

        profile = self.match_type_profiles[match_type]
        profile["total_predictions"] = profile.get("total_predictions", 0) + 1

        recent_errors = profile.get("recent_errors", [])
        recent_errors.append({
            "pred_home": pred["pred_home"],
            "pred_draw": pred["pred_draw"],
            "pred_away": pred["pred_away"],
            "actual_result": None,
        })
        profile["recent_errors"] = recent_errors[-30:]

    def _update_match_type_profile_with_result(self, match_type: str, pred: dict, actual_result: str):
        """Mac tipi profilini sonuc ile guncelle."""
        if match_type not in self.match_type_profiles:
            return

        profile = self.match_type_profiles[match_type]

        recent_errors = profile.get("recent_errors", [])
        for r in reversed(recent_errors):
            if r.get("pred_home") == pred.get("pred_home") and r.get("actual_result") is None:
                r["actual_result"] = actual_result
                break

        errors_h, errors_d, errors_a = [], [], []
        for r in recent_errors:
            if r.get("actual_result") is None:
                continue

            actual = r["actual_result"]
            if actual == "H":
                errors_h.append(1 - r["pred_home"])
                errors_d.append(-r["pred_draw"])
                errors_a.append(-r["pred_away"])
            elif actual == "D":
                errors_h.append(-r["pred_home"])
                errors_d.append(1 - r["pred_draw"])
                errors_a.append(-r["pred_away"])
            elif actual == "A":
                errors_h.append(-r["pred_home"])
                errors_d.append(-r["pred_draw"])
                errors_a.append(1 - r["pred_away"])

        if errors_h:
            weights = np.exp(np.linspace(-1, 0, len(errors_h)))
            weights /= weights.sum()
            profile["home_bias"] = float(np.average(errors_h, weights=weights))
            profile["draw_bias"] = float(np.average(errors_d, weights=weights))
            profile["away_bias"] = float(np.average(errors_a, weights=weights))

    def get_stats(self) -> dict:
        """Istatistikleri dondur."""
        return {
            "total_predictions": len(self.prediction_log),
            "teams_tracked": len(self.team_profiles),
            "leagues_tracked": len(self.league_profiles),
            "match_types_tracked": len(self.match_type_profiles),
            "recent_predictions": len([
                p for p in self.prediction_log
                if p.get("actual_result") is not None
            ]),
        }

    def get_team_report(self, team_id: int) -> dict:
        """Takim icin rapor olustur."""
        team_key = str(team_id)
        if team_key not in self.team_profiles:
            return {"error": "Takim bulunamadi"}

        profile = self.team_profiles[team_key]
        return {
            "team_id": team_id,
            "total_predictions": profile.get("total_predictions", 0),
            "home_bias": profile.get("home_bias", 0),
            "draw_bias": profile.get("draw_bias", 0),
            "away_bias": profile.get("away_bias", 0),
            "recent_predictions": len(profile.get("recent_predictions", [])),
        }

    def get_league_report(self, league: str) -> dict:
        """Lig icin rapor olustur."""
        if league not in self.league_profiles:
            return {"error": "Lig bulunamadi"}

        profile = self.league_profiles[league]
        return {
            "league": league,
            "total_predictions": profile.get("total_predictions", 0),
            "home_bias": profile.get("home_bias", 0),
            "draw_bias": profile.get("draw_bias", 0),
            "away_bias": profile.get("away_bias", 0),
        }
