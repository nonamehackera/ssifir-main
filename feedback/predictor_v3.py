"""Feedback Predictor v3 - Prediction Flipper.

APPROACH:
  Instead of adjusting probabilities (which rarely flips predictions),
  learn WHEN to flip the model's top prediction.

  When model predicts H but for this match type H is historically
  correct only 30% of the time, flip to the next best prediction.

RULES:
  1. Track per match-type: when model predicts X, how often is X correct?
  2. Only flip when: >= 50 samples AND success rate < 40% (strong evidence)
  3. Flip to the outcome with highest historical rate for this match type
  4. Never flip when model is confident (>65% for any outcome)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


class FeedbackPredictorV3:
    """Simple prediction flipper based on match-type success rates."""

    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_file = self.data_dir / "flipper_v3.json"
        self.profiles = self._load_json(self.profile_file, {})

        # Parameters
        self.min_samples = 50
        self.flip_threshold = 0.40  # Flip if success rate < 40%
        self.min_gap = 0.10  # Gap between best and 2nd best outcome

    def _load_json(self, filepath: Path, default):
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return default

    def _save_json(self, filepath: Path, data):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except:
            pass

    def _classify(self, elo_diff, home_elo):
        ed = abs(elo_diff)
        if ed < 50:
            ed_bin = "close"
        elif ed < 150:
            ed_bin = "slight"
        elif ed < 300:
            ed_bin = "clear"
        else:
            ed_bin = "dominant"

        if home_elo < 1400:
            h_bin = "weak"
        elif home_elo < 1600:
            h_bin = "mid"
        else:
            h_bin = "strong"

        return f"{ed_bin}_{h_bin}"

    def record_prediction(
        self, home_team_id, away_team_id, league,
        pred_home, pred_draw, pred_away, elo_diff, match_id,
    ):
        """Record what the model predicted."""
        home_elo = 1500  # default; not used for classification
        mt = self._classify(elo_diff, home_elo)
        if mt not in self.profiles:
            self.profiles[mt] = {}
        # Store prediction for later matching with result
        if match_id not in self.profiles[mt]:
            self.profiles[mt][match_id] = {
                "pred": ["H", "D", "A"][np.argmax([pred_home, pred_draw, pred_away])],
                "result": None,
            }
        self._save_json(self.profile_file, self.profiles)

    def record_result(
        self, home_team_id, away_team_id, elo_diff, home_elo, actual_result,
    ):
        """Record actual result and update success rates."""
        mt = self._classify(elo_diff, home_elo)
        # Update aggregate stats (not per-match, just aggregate)
        if mt not in self.profiles:
            self.profiles[mt] = {}

        # Use aggregate tracking instead of per-match
        agg_key = "__aggregate__"
        if agg_key not in self.profiles[mt]:
            self.profiles[mt][agg_key] = {
                "pred_H": {"correct": 0, "total": 0},
                "pred_D": {"correct": 0, "total": 0},
                "pred_A": {"correct": 0, "total": 0},
            }
        # We need to know what was predicted - store externally
        self._save_json(self.profile_file, self.profiles)

    def record_match(self, match_type, predicted, actual):
        """Direct match recording with known prediction and result."""
        if match_type not in self.profiles:
            self.profiles[match_type] = {}

        agg_key = "__aggregate__"
        if agg_key not in self.profiles[match_type]:
            self.profiles[match_type][agg_key] = {
                "pred_H": {"correct": 0, "total": 0},
                "pred_D": {"correct": 0, "total": 0},
                "pred_A": {"correct": 0, "total": 0},
            }

        agg = self.profiles[match_type][agg_key]
        pred_key = f"pred_{predicted}"
        agg[pred_key]["total"] += 1
        if predicted == actual:
            agg[pred_key]["correct"] += 1

        self._save_json(self.profile_file, self.profiles)

    def adjust_prediction(
        self, home_team_id, away_team_id, league,
        pred_home, pred_draw, pred_away, elo_diff, home_elo, away_elo,
    ) -> FeedbackAdjustment:
        """Adjust prediction - only flip when strong evidence exists."""
        mt = self._classify(elo_diff, home_elo)

        if mt not in self.profiles or "__aggregate__" not in self.profiles[mt]:
            return self._no_adjust(pred_home, pred_draw, pred_away)

        agg = self.profiles[mt]["__aggregate__"]
        probs = {"H": pred_home, "D": pred_draw, "A": pred_away}
        predicted = max(probs, key=probs.get)
        pred_key = f"pred_{predicted}"
        stats = agg[pred_key]

        n = stats["total"]
        if n < self.min_samples:
            return self._no_adjust(pred_home, pred_draw, pred_away)

        success_rate = stats["correct"] / n
        avg_pred_prob = probs[predicted]

        # Check if model is confident - don't flip confident predictions
        if avg_pred_prob > 0.65:
            return self._no_adjust(pred_home, pred_draw, pred_away)

        # Check if model is bad at this prediction
        if success_rate < self.flip_threshold:
            # Find best alternative outcome
            alt_rates = {}
            for outcome in ["H", "D", "A"]:
                if outcome != predicted:
                    alt_key = f"pred_{outcome}"
                    alt_n = agg[alt_key]["total"]
                    if alt_n >= 10:
                        alt_rates[outcome] = agg[alt_key]["correct"] / alt_n
                    else:
                        alt_rates[outcome] = probs[outcome]  # fallback to model prob

            # Pick best alternative
            best_alt = max(alt_rates, key=alt_rates.get)
            best_alt_rate = alt_rates[best_alt]
            gap = best_alt_rate - success_rate

            if gap >= self.min_gap:
                # FLIP!
                adj_home = pred_home
                adj_draw = pred_draw
                adj_away = pred_away

                flip_factor = 0.15  # How much to shift probability
                if best_alt == "H":
                    adj_home += flip_factor
                    adj_draw -= flip_factor * 0.5
                    adj_away -= flip_factor * 0.5
                elif best_alt == "D":
                    adj_draw += flip_factor
                    adj_home -= flip_factor * 0.5
                    adj_away -= flip_factor * 0.5
                else:
                    adj_away += flip_factor
                    adj_home -= flip_factor * 0.5
                    adj_draw -= flip_factor * 0.5

                # Normalize
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
                    adjustments_applied=[f"flip_{predicted}_to_{best_alt}"],
                    reasons=[f"Match type {mt}: {predicted} success rate={success_rate:.2f} (n={n}), flipping to {best_alt} (rate={best_alt_rate:.2f})"],
                )

        return self._no_adjust(pred_home, pred_draw, pred_away)

    def _no_adjust(self, ph, pd, pa):
        return FeedbackAdjustment(
            original_home=ph, original_draw=pd, original_away=pa,
            adjusted_home=ph, adjusted_draw=pd, adjusted_away=pa,
        )

    def get_stats(self):
        stats = {}
        for mt, data in self.profiles.items():
            if "__aggregate__" in data:
                agg = data["__aggregate__"]
                stats[mt] = {}
                for pk in ["pred_H", "pred_D", "pred_A"]:
                    s = agg[pk]
                    n = s["total"]
                    rate = s["correct"] / n if n > 0 else 0
                    stats[mt][pk] = {"total": n, "correct": s["correct"], "success_rate": round(rate, 3)}
        return stats
