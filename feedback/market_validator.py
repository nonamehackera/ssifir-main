"""Market Validator v2 - Tahminleri GERCEKTEN duzelt.

HER PAZAR ICIN:
  1. Takim istatistikleri (son 20 mac ortalamasi)
  2. Lig ortalamalari
  3. Model tahmini ile karsilastir
  4. Fark buyukse -> TAHMINI DEGISDIR
  5. Degisiklik made -> ayni birak

PAZARLAR:
  - 1X2 (ev kazanir / beraberlik / deplasman kazanir)
  - Toplam gol (ust 1.5, ust 2.5, ust 3.5)
  - BTTS (var/yok)
  - Korner (ust/alt 7.5, 8.5, 9.5)
  - Ilk yari sonucu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ValidatedPrediction:
    """Dogrulanmis ve duzeltilmis tahmin."""
    # 1X2
    home_win: float
    draw: float
    away_win: float
    # Gol pazarlari
    btts_yes: float
    over15: float
    over25: float
    over35: float
    # Korner
    corner_total: float
    corner_over75: float
    corner_over85: float
    corner_over95: float
    # Duzeltmeler
    adjustments: list = field(default_factory=list)


class MarketValidator:
    """Tum pazarlari dogrulayan ve duzelten sistem."""

    def __init__(self, features_df: pd.DataFrame):
        df = features_df[features_df["date"] >= "2023-01-01"].copy()
        self._compute_team_stats(df)
        self._compute_league_stats(df)

    def _compute_team_stats(self, df):
        self.team_stats = {}
        for _, row in df.iterrows():
            tid = int(row["home_team_id"])
            if tid not in self.team_stats:
                self.team_stats[tid] = {
                    "home_gf": [], "home_ga": [], "home_btts": [], "home_o25": [],
                    "away_gf": [], "away_ga": [], "away_btts": [], "away_o25": [],
                    "home_corners": [], "away_corners": [],
                    "home_hg": [], "away_ag": [],
                }
            ts = self.team_stats[tid]
            hg = row.get("home_goals", 0)
            ag = row.get("away_goals", 0)
            hc = row.get("home_corners", 0)
            ac = row.get("away_corners", 0)
            ht_hg = row.get("ht_home_goals", np.nan)
            ht_ag = row.get("ht_away_goals", np.nan)

            if pd.notna(hg):
                ts["home_gf"].append(float(hg))
            if pd.notna(ag):
                ts["home_ga"].append(float(ag))
            if pd.notna(hg) and pd.notna(ag):
                ts["home_btts"].append(1 if hg > 0 and ag > 0 else 0)
                ts["home_o25"].append(1 if hg + ag > 2.5 else 0)
            if pd.notna(hc):
                ts["home_corners"].append(float(hc))
            if pd.notna(ht_hg):
                ts["home_hg"].append(float(ht_hg))

            atid = int(row["away_team_id"])
            if atid not in self.team_stats:
                self.team_stats[atid] = {
                    "home_gf": [], "home_ga": [], "home_btts": [], "home_o25": [],
                    "away_gf": [], "away_ga": [], "away_btts": [], "away_o25": [],
                    "home_corners": [], "away_corners": [],
                    "home_hg": [], "away_ag": [],
                }
            ats = self.team_stats[atid]
            if pd.notna(ag):
                ats["away_gf"].append(float(ag))
            if pd.notna(hg):
                ats["away_ga"].append(float(hg))
            if pd.notna(hg) and pd.notna(ag):
                ats["away_btts"].append(1 if hg > 0 and ag > 0 else 0)
                ats["away_o25"].append(1 if hg + ag > 2.5 else 0)
            if pd.notna(ac):
                ats["away_corners"].append(float(ac))
            if pd.notna(ht_ag):
                ats["away_ag"].append(float(ht_ag))

    def _compute_league_stats(self, df):
        self.league_stats = {}
        for _, row in df.iterrows():
            lg = row.get("league", "Unknown")
            if lg not in self.league_stats:
                self.league_stats[lg] = {
                    "home_gf": [], "away_gf": [], "total_goals": [],
                    "btts": [], "o25": [], "corners": [],
                }
            ls = self.league_stats[lg]
            hg = row.get("home_goals", 0)
            ag = row.get("away_goals", 0)
            tc = row.get("home_corners", 0) + row.get("away_corners", 0)
            if pd.notna(hg) and pd.notna(ag):
                ls["home_gf"].append(float(hg))
                ls["away_gf"].append(float(ag))
                ls["total_goals"].append(float(hg + ag))
                ls["btts"].append(1 if hg > 0 and ag > 0 else 0)
                ls["o25"].append(1 if hg + ag > 2.5 else 0)
            if pd.notna(tc):
                ls["corners"].append(float(tc))

    def _avg(self, vals, window=20):
        if not vals:
            return 0.0
        return float(np.mean(vals[-window:]))

    def _team(self, tid, stat):
        if tid not in self.team_stats:
            return 0.0
        return self._avg(self.team_stats[tid].get(stat, []))

    def _league(self, lg, stat):
        if lg not in self.league_stats:
            return 0.0
        return self._avg(self.league_stats[lg].get(stat, []))

    def _blend(self, team_val, league_val, team_weight=0.7):
        if team_val > 0 and league_val > 0:
            return team_val * team_weight + league_val * (1 - team_weight)
        return team_val if team_val > 0 else league_val

    def validate(
        self,
        home_id: int,
        away_id: int,
        league: str,
        pred_home: float,
        pred_draw: float,
        pred_away: float,
        btts_yes: float,
        over15: float,
        over25: float,
        over35: float,
        corner_total: float,
        corner_over75: float,
        corner_over85: float,
        corner_over95: float,
    ) -> ValidatedPrediction:
        """Tum pazarlari dogrula ve duzelt."""
        adj = []

        # ── TAKIM ISTATISTIKLERI ──────────────────────────────────────
        h_gf = self._team(home_id, "home_gf")
        h_ga = self._team(home_id, "home_ga")
        a_gf = self._team(away_id, "away_gf")
        a_ga = self._team(away_id, "away_ga")
        h_btts = self._team(home_id, "home_btts")
        a_btts = self._team(away_id, "away_btts")
        h_o25 = self._team(home_id, "home_o25")
        a_o25 = self._team(away_id, "away_o25")
        h_corners = self._team(home_id, "home_corners")
        a_corners = self._team(away_id, "away_corners")

        # Lig ortalamalari
        lg_hg = self._league(league, "home_gf")
        lg_ag = self._league(league, "away_gf")
        lg_btts = self._league(league, "btts")
        lg_o25 = self._league(league, "o25")
        lg_corners = self._league(league, "corners")

        # ── 1. GOL TAHMINLERI ─────────────────────────────────────────
        exp_hg = self._blend(h_gf, lg_hg)
        exp_ag = self._blend(a_gf, lg_ag)
        exp_total = exp_hg + exp_ag

        # BTTS
        exp_btts = self._blend((h_btts + a_btts) / 2, lg_btts)
        if exp_btts > 0 and abs(btts_yes - exp_btts) > 0.15:
            new_btts = btts_yes * 0.5 + exp_btts * 0.5
            adj.append(f"BTTS: {btts_yes:.0%} -> {new_btts:.0%} (takim ort={exp_btts:.0%})")
            btts_yes = new_btts

        # Over 2.5
        exp_o25 = self._blend((h_o25 + a_o25) / 2, lg_o25)
        if exp_o25 > 0 and abs(over25 - exp_o25) > 0.15:
            new_o25 = over25 * 0.5 + exp_o25 * 0.5
            adj.append(f"Over2.5: {over25:.0%} -> {new_o25:.0%} (takim ort={exp_o25:.0%})")
            over25 = new_o25

        # Over 1.5 - Over 2.5'ten daha yuksek olmali
        if over15 < over25:
            over15 = over25 * 1.2
            over15 = min(over15, 0.98)

        # Over 3.5 - Over 2.5'ten daha dusuk olmali
        if over35 > over25:
            over35 = over25 * 0.6
            over35 = max(over35, 0.02)

        # ── 2. KORNER TAHMINLERI ──────────────────────────────────────
        exp_corners = self._blend(h_corners + a_corners, lg_corners)
        if exp_corners > 0 and corner_total > 0:
            corner_diff = abs(corner_total - exp_corners) / exp_corners
            if corner_diff > 0.20:
                new_total = corner_total * 0.4 + exp_corners * 0.6
                adj.append(f"Korner: {corner_total:.1f} -> {new_total:.1f} (takim ort={exp_corners:.1f})")
                corner_total = new_total

        # Korner ust/alt olasiliklari - toplama gore guncelle
        if corner_total < 7.5:
            corner_over75 = min(0.30, corner_over75 * 0.6)
            corner_over85 = min(0.20, corner_over85 * 0.5)
            corner_over95 = min(0.10, corner_over95 * 0.4)
        elif corner_total < 9.5:
            corner_over75 = min(0.65, corner_over75)
            corner_over85 = min(0.45, corner_over85)
            corner_over95 = min(0.25, corner_over95 * 0.7)
        else:
            corner_over75 = max(0.60, corner_over75)
            corner_over85 = max(0.40, corner_over85)
            corner_over95 = max(0.25, corner_over95)

        # ── 3. 1X2 TAHMINLERI ─────────────────────────────────────────
        # Eger gol beklentileri cok farkliysa 1X2'yi de ayarla
        if exp_hg > 0 and exp_ag > 0:
            goal_ratio = exp_hg / exp_ag

            # Ev sahibi cok guclu ise ve model az verdiyse
            if goal_ratio > 1.8 and pred_home < 0.50:
                boost = min(0.08, (goal_ratio - 1.8) * 0.05)
                pred_home += boost
                pred_away -= boost * 0.5
                pred_draw -= boost * 0.5
                adj.append(f"1X2: ev gucu arttirildi (gol ratio={goal_ratio:.2f})")

            # Deplasman cok guclu ise
            if goal_ratio < 0.55 and pred_away < 0.45:
                boost = min(0.08, (0.55 - goal_ratio) * 0.05)
                pred_away += boost
                pred_home -= boost * 0.5
                pred_draw -= boost * 0.5
                adj.append(f"1X2: dep gucu arttirildi (gol ratio={goal_ratio:.2f})")

        # ── NORMALIZASYON ─────────────────────────────────────────────
        total = pred_home + pred_draw + pred_away
        if total > 0:
            pred_home /= total
            pred_draw /= total
            pred_away /= total

        pred_home = max(0.02, min(0.95, pred_home))
        pred_draw = max(0.02, min(0.50, pred_draw))
        pred_away = max(0.02, min(0.95, pred_away))

        total = pred_home + pred_draw + pred_away
        if total > 0:
            pred_home /= total
            pred_draw /= total
            pred_away /= total

        return ValidatedPrediction(
            home_win=pred_home,
            draw=pred_draw,
            away_win=pred_away,
            btts_yes=btts_yes,
            over15=over15,
            over25=over25,
            over35=over35,
            corner_total=corner_total,
            corner_over75=corner_over75,
            corner_over85=corner_over85,
            corner_over95=corner_over95,
            adjustments=adj,
        )
