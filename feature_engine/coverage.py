"""Feature coverage matrisi (ROADMAP bolum 79).

Her (league, season, feature) kombinasyonu icin coverage %'sini uretir.
Boylece hangi feature'in hangi lig/sezonda eksik oldugu gorulur ve
ROADMAP 79'daki ornek gibi ikinci kaynak / fallback karari verilir.

Ornek cikti:
  league season   feature        coverage
  E0     2425     xg             0.98
  E1     2425     home_corners   0.34
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Free Football-Data.co.uk'un CSV'lerinde guvenilir olan sutunlar.
# (xG yok -> coverage 0; corners/cards kismen var -> dusuk coverage)
CORE_FEATURES = [
    "home_goals", "away_goals", "home_shots", "away_shots",
    "home_sot", "away_sot", "home_corners", "away_corners",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "avg_home_odds", "avg_draw_odds", "avg_away_odds",
]

# Free veride KESINLIKLE olmayanlar (uzak kaynak gerekir) -> coverage 0
UNAVAILABLE_IN_FREE = [
    "xg", "xga", "npxg", "xg_open_play", "xg_set_play",
    "player_xg", "expected_starting_xi", "lineup_strength",
    "missing_xg", "coach_tenure", "referee_cards",
    "weather_temp", "weather_rain", "travel_distance",
]


def coverage_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """features: build_features() ciktisi (league, season, + feature sutunlari)."""
    rows = []
    grp = features.groupby(["league", "season"])
    n = len(features)
    for (lg, se), sub in grp:
        for col in CORE_FEATURES:
            if col in sub.columns:
                cov = 1.0 - sub[col].isna().mean()
                rows.append({"league": lg, "season": se, "feature": col, "coverage": round(float(cov), 3)})
        for col in UNAVAILABLE_IN_FREE:
            rows.append({"league": lg, "season": se, "feature": col, "coverage": 0.0})
    return pd.DataFrame(rows)


def low_coverage_report(matrix: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """threshold altindaki (league, season, feature) kombinasyonlari."""
    return matrix[matrix["coverage"] < threshold].sort_values(["coverage", "league"])


def print_report(matrix: pd.DataFrame) -> None:
    low = low_coverage_report(matrix)
    print(f"Coverage matrisi: {len(matrix)} hucresi, dusuk (<0.5) = {len(low)}")
    if len(low):
        print("Dusuk coverage ornekleri:")
        print(low.head(10).to_string(index=False))
    # ozet: free veride hic olmayanlar
    missing = sorted(set(UNAVAILABLE_IN_FREE))
    print("\nFree veride KESINLIKLE yok (uzak kaynak gerekir):")
    print(", ".join(missing))
