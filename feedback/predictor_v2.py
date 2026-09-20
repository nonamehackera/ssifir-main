"""
Football Prediction Feedback System V2

Uses match-type patterns instead of team-specific patterns.
Groups matches by ELO difference ranges and home team ELO ranges
to create 12 match types, each with its own calibration profile.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class FeedbackAdjustment:
    home_adj: float
    draw_adj: float
    away_adj: float
    match_type: str
    confidence: str  # "applied", "insufficient_data", "no_bias"
    samples: int
    z_score: float
    p_value: float


class FeedbackPredictorV2:
    """Match-type based prediction feedback system."""

    ELO_DIFF_BINS = [
        (-9999, 50, "close"),
        (50, 150, "slight"),
        (150, 300, "clear"),
        (300, 9999, "dominant"),
    ]

    HOME_ELO_BINS = [
        (-9999, 1400, "weak"),
        (1400, 1600, "mid"),
        (1600, 9999, "strong"),
    ]

    MIN_SAMPLES = 30
    MAX_ADJUSTMENT = 0.04
    LEARNING_RATE = 0.3
    Z_THRESHOLD = 1.96  # p < 0.05

    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.data_dir / "match_type_profile.json"
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> dict:
        if self.profile_path.exists():
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_profiles(self):
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2)

    def _ensure_match_type(self, match_type: str):
        if match_type not in self.profiles:
            self.profiles[match_type] = {
                "pred_H": {"correct": 0, "total": 0, "sum_pred": 0.0},
                "pred_D": {"correct": 0, "total": 0, "sum_pred": 0.0},
                "pred_A": {"correct": 0, "total": 0, "sum_pred": 0.0},
            }

    def _classify_match_type(self, elo_diff: float, home_elo: float) -> str:
        elo_label = None
        for low, high, label in self.ELO_DIFF_BINS:
            if low <= elo_diff < high:
                elo_label = label
                break
        if elo_label is None:
            elo_label = "dominant"

        home_label = None
        for low, high, label in self.HOME_ELO_BINS:
            if low <= home_elo < high:
                home_label = label
                break
        if home_label is None:
            home_label = "mid"

        return f"{elo_label}_{home_label}"

    def record_prediction(
        self,
        home_team_id: str,
        away_team_id: str,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        elo_diff: float,
        match_id: str,
    ):
        match_type = self._classify_match_type(elo_diff, 0)
        self._ensure_match_type(match_type)

        profile = self.profiles[match_type]
        predicted_outcome = max(
            [("pred_H", pred_home), ("pred_D", pred_draw), ("pred_A", pred_away)],
            key=lambda x: x[1],
        )

        for key, prob in [("pred_H", pred_home), ("pred_D", pred_draw), ("pred_A", pred_away)]:
            if key == predicted_outcome[0]:
                profile[key]["total"] += 1
                profile[key]["sum_pred"] += prob

        self._save_profiles()

    def record_result(
        self,
        home_team_id: str,
        away_team_id: str,
        elo_diff: float,
        home_elo: float,
        actual_result: str,
    ):
        match_type = self._classify_match_type(elo_diff, home_elo)
        self._ensure_match_type(match_type)

        profile = self.profiles[match_type]

        result_map = {"H": "pred_H", "D": "pred_D", "A": "pred_A"}
        actual_key = result_map.get(actual_result)
        if actual_key is None:
            return

        # Find which outcome was the model's top prediction for this match
        # We record which outcome the model predicted most strongly
        # and whether that outcome was correct
        best_pred_key = None
        best_prob = -1.0
        for key in ["pred_H", "pred_D", "pred_A"]:
            total = profile[key]["total"]
            if total > 0:
                avg_pred = profile[key]["sum_pred"] / total
                if avg_pred > best_prob:
                    best_prob = avg_pred
                    best_pred_key = key

        # If the model's top prediction was this actual outcome, mark correct
        if best_pred_key is not None and best_pred_key == actual_key:
            profile[best_pred_key]["correct"] += 1

        # Also update the actual outcome's stats regardless
        # This tracks how often each outcome occurs when the model predicts it
        # We need to match predictions to results by match, so we do a simpler approach:
        # Update the actual outcome's count for the match type
        profile[actual_key]["total"] += 1

        self._save_profiles()

    def record_result_v2(
        self,
        match_type: str,
        predicted_outcome: str,
        actual_result: str,
    ):
        """Direct recording with known match type and predicted outcome."""
        self._ensure_match_type(match_type)
        profile = self.profiles[match_type]

        result_map = {"H": "pred_H", "D": "pred_D", "A": "pred_A"}
        pred_key = result_map.get(predicted_outcome)
        actual_key = result_map.get(actual_result)

        if pred_key is None or actual_key is None:
            return

        profile[pred_key]["total"] += 1
        if pred_key == actual_key:
            profile[pred_key]["correct"] += 1

        self._save_profiles()

    def adjust_prediction(
        self,
        home_team_id: str,
        away_team_id: str,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        elo_diff: float,
        home_elo: float,
        away_elo: float,
    ) -> FeedbackAdjustment:
        match_type = self._classify_match_type(elo_diff, home_elo)
        self._ensure_match_type(match_type)

        profile = self.profiles[match_type]

        # Find which outcome the model predicts most strongly
        probs = {"H": pred_home, "D": pred_draw, "A": pred_away}
        predicted_outcome = max(probs, key=probs.get)
        pred_key = f"pred_{predicted_outcome}"
        pred_prob = probs[predicted_outcome]

        stats = profile[pred_key]
        total = stats["total"]

        if total < self.MIN_SAMPLES:
            return FeedbackAdjustment(
                home_adj=0.0,
                draw_adj=0.0,
                away_adj=0.0,
                match_type=match_type,
                confidence="insufficient_data",
                samples=total,
                z_score=0.0,
                p_value=1.0,
            )

        # Calculate actual success rate for this prediction type
        actual_rate = stats["correct"] / total if total > 0 else 0.0
        avg_pred = stats["sum_pred"] / total if total > 0 else 0.0

        # Z-test: is actual_rate significantly different from avg_pred?
        # H0: actual_rate == avg_pred
        if avg_pred <= 0 or avg_pred >= 1 or total < 2:
            return FeedbackAdjustment(
                home_adj=0.0,
                draw_adj=0.0,
                away_adj=0.0,
                match_type=match_type,
                confidence="no_bias",
                samples=total,
                z_score=0.0,
                p_value=1.0,
            )

        se = np.sqrt(avg_pred * (1 - avg_pred) / total)
        if se == 0:
            return FeedbackAdjustment(
                home_adj=0.0,
                draw_adj=0.0,
                away_adj=0.0,
                match_type=match_type,
                confidence="no_bias",
                samples=total,
                z_score=0.0,
                p_value=1.0,
            )

        z_score = (actual_rate - avg_pred) / se
        p_value = 2.0 * (1.0 - self._normal_cdf(abs(z_score)))

        home_adj = 0.0
        draw_adj = 0.0
        away_adj = 0.0
        confidence = "no_bias"

        if abs(z_score) > self.Z_THRESHOLD:
            # Model is biased for this match type and prediction
            bias = actual_rate - avg_pred
            raw_adjustment = -bias * self.LEARNING_RATE
            clamped = np.clip(raw_adjustment, -self.MAX_ADJUSTMENT, self.MAX_ADJUSTMENT)

            # Apply adjustment only to the predicted outcome
            result_map = {"H": "home_adj", "D": "draw_adj", "A": "away_adj"}
            setattr(self, result_map[predicted_outcome], float(clamped))
            home_adj = float(clamped) if predicted_outcome == "H" else 0.0
            draw_adj = float(clamped) if predicted_outcome == "D" else 0.0
            away_adj = float(clamped) if predicted_outcome == "A" else 0.0

            # Redistribute the adjustment to other outcomes proportionally
            other_outcomes = [o for o in ["H", "D", "A"] if o != predicted_outcome]
            other_probs = [probs[o] for o in other_outcomes]
            other_total = sum(other_probs)

            if other_total > 0:
                for adj_key, outcome in zip(
                    ["home_adj", "draw_adj", "away_adj"], other_outcomes
                ):
                    current = locals()[adj_key]
                    redistribution = -clamped * (probs[outcome] / other_total)
                    new_val = current + redistribution
                    if adj_key == "home_adj":
                        home_adj = float(new_val)
                    elif adj_key == "draw_adj":
                        draw_adj = float(new_val)
                    else:
                        away_adj = float(new_val)

            confidence = "applied"

        return FeedbackAdjustment(
            home_adj=home_adj,
            draw_adj=draw_adj,
            away_adj=away_adj,
            match_type=match_type,
            confidence=confidence,
            samples=total,
            z_score=float(z_score),
            p_value=float(p_value),
        )

    def get_stats(self) -> dict:
        stats = {}
        for match_type, profile in self.profiles.items():
            stats[match_type] = {}
            for pred_key in ["pred_H", "pred_D", "pred_A"]:
                data = profile[pred_key]
                total = data["total"]
                correct = data["correct"]
                sum_pred = data["sum_pred"]
                avg_pred = sum_pred / total if total > 0 else 0.0
                success_rate = correct / total if total > 0 else 0.0
                bias = success_rate - avg_pred if total > 0 else 0.0
                stats[match_type][pred_key] = {
                    "total": total,
                    "correct": correct,
                    "avg_pred": round(avg_pred, 4),
                    "success_rate": round(success_rate, 4),
                    "bias": round(bias, 4),
                }
        return stats

    @staticmethod
    def _normal_cdf(x: float) -> float:
        import math as _math
        return 0.5 * (1.0 + _math.erf(x / _math.sqrt(2.0)))

    def reset(self):
        self.profiles = {}
        self._save_profiles()

    def print_summary(self):
        stats = self.get_stats()
        print("=" * 70)
        print("FEEDBACK PREDICTOR V2 - Match Type Calibration Summary")
        print("=" * 70)
        for match_type in sorted(stats.keys()):
            print(f"\n  Match Type: {match_type}")
            for pred_key in ["pred_H", "pred_D", "pred_A"]:
                s = stats[match_type][pred_key]
                if s["total"] > 0:
                    print(
                        f"    {pred_key}: {s['total']:>4} samples, "
                        f"avg_pred={s['avg_pred']:.3f}, "
                        f"actual={s['success_rate']:.3f}, "
                        f"bias={s['bias']:+.3f}"
                    )
        print("=" * 70)
