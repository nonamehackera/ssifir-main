"""ErrorMemory - Yanlis tahminleri ogrenen hafiza.

GERCEK MANTIK:
  1. Mac bitti, model "H" dedi ama "A" oldu
  2. Bu takim icin not al: "Bu takim hakkinda yanlis tahmin ettin"
  3. Bir daha bu takimi gordonce: "Dikkat et, son seferde yanlis yaptin"
  4. Daha temkinli tahmin yap

HER MAC ICIN KAYDET:
  - hangi takim
  - model ne dedi
  - gercek ne oldu
  - yanlis miydi

SORGULAMA:
  - bu takim icin son N macte kac kez yanlis tahmin yaptim?
  - bu lig icin yanlis tahmin orani kac?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ErrorMemory:
    """Yanlis tahminleri ogrenen hafiza sistemi."""

    def __init__(self, data_dir: str = "data/feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.data_dir / "error_memory.json"
        self.memory = self._load()

        # Parametreler
        self.max_history = 50       # Her takim icin son 50 mac
        self.warning_threshold = 3  # 3+ hata varsa uyar
        self.adjustment_factor = 0.15  # Hata orani kadar duzeltme

    def _load(self) -> dict:
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"predictions": [], "team_errors": {}, "league_errors": {}}

    def _save(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except:
            pass

    def record_prediction(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        actual_result: str,
    ):
        """Tahmini ve sonucu kaydet."""
        predicted = ["H", "D", "A"][np.argmax([pred_home, pred_draw, pred_away])]
        was_correct = predicted == actual_result

        # Tahmini kaydet
        entry = {
            "home_id": int(home_team_id),
            "away_id": int(away_team_id),
            "league": league,
            "pred_home": round(pred_home, 3),
            "pred_draw": round(pred_draw, 3),
            "pred_away": round(pred_away, 3),
            "predicted": predicted,
            "actual": actual_result,
            "correct": was_correct,
        }
        self.memory["predictions"].append(entry)

        # Son 500 tane tut
        if len(self.memory["predictions"]) > 500:
            self.memory["predictions"] = self.memory["predictions"][-500:]

        # Takim hatalarini guncelle
        for tid in [home_team_id, away_team_id]:
            tid_str = str(int(tid))
            if tid_str not in self.memory["team_errors"]:
                self.memory["team_errors"][tid_str] = {"errors": 0, "total": 0, "wrong_predictions": []}

            te = self.memory["team_errors"][tid_str]
            te["total"] += 1

            if not was_correct:
                te["errors"] += 1
                te["wrong_predictions"].append({
                    "predicted": predicted,
                    "actual": actual_result,
                    "league": league,
                    "pred_home": round(pred_home, 3),
                    "pred_draw": round(pred_draw, 3),
                    "pred_away": round(pred_away, 3),
                })
                # Son 10 hatayi tut
                if len(te["wrong_predictions"]) > 10:
                    te["wrong_predictions"] = te["wrong_predictions"][-10:]

        # Lig hatalarini guncelle
        if league not in self.memory["league_errors"]:
            self.memory["league_errors"][league] = {"errors": 0, "total": 0}
        self.memory["league_errors"][league]["total"] += 1
        if not was_correct:
            self.memory["league_errors"][league]["errors"] += 1

        self._save()

    def get_team_error_rate(self, team_id: int) -> float:
        """Takimin hata orani (0-1)."""
        tid_str = str(int(team_id))
        if tid_str not in self.memory["team_errors"]:
            return 0.0
        te = self.memory["team_errors"][tid_str]
        if te["total"] < 3:
            return 0.0
        return te["errors"] / te["total"]

    def get_team_warning(self, team_id: int) -> str:
        """Takim icin uyari mesaji."""
        tid_str = str(int(team_id))
        if tid_str not in self.memory["team_errors"]:
            return ""
        te = self.memory["team_errors"][tid_str]
        if te["total"] < 3:
            return ""
        error_rate = te["errors"] / te["total"]
        if error_rate > 0.4:
            return f"DIKKAT: Bu takim hakkinda %{error_rate*100:.0f} hata yaptin ({te['errors']}/{te['total']})"
        return ""

    def get_adjustment(self, team_id: int) -> float:
        """Takim icin duzeltme faktoru (negatif = daha temkinli)."""
        error_rate = self.get_team_error_rate(team_id)
        if error_rate > 0.4:
            return -self.adjustment_factor
        return 0.0

    def get_league_error_rate(self, league: str) -> float:
        """Ligdeki hata orani."""
        if league not in self.memory["league_errors"]:
            return 0.0
        le = self.memory["league_errors"][league]
        if le["total"] < 5:
            return 0.0
        return le["errors"] / le["total"]

    def get_stats(self) -> dict:
        """Genel istatistikler."""
        total = len(self.memory["predictions"])
        errors = sum(1 for p in self.memory["predictions"] if not p["correct"])
        return {
            "total_predictions": total,
            "total_errors": errors,
            "error_rate": errors / total if total > 0 else 0,
            "teams_tracked": len(self.memory["team_errors"]),
            "leagues_tracked": len(self.memory["league_errors"]),
        }
