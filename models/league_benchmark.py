"""Lig ozel model benchmark (ROADMAP bolum 71).

Global model her ligi tek modelde ogrenir. Ancak buyuk liglerde (E0, E1,
D1, I1, SP1) ayri model daha iyi olabilir mi? Bu modul walk-forward ile
global vs league-specific karsilastirmasini yapar ve sadece destekleniyorsa
ayri model onerir.

NOT: free veride her lig icin ayri model egitmek icin yeterli veri (6+ sezon)
sadece birkac buyuk ligde var. Bu yuzden benchmark sonucu layera gore
KARAR verir, magick olarak lig modeli kurmaz.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from evaluation.metrics import log_loss_1x2
from models.catboost.model import CatBoostModel

logger = logging.getLogger(__name__)

# Yeterli veri olan buyuk ligler (free veride 4 sezon x ~300-600 mac)
BIG_LEAGUES = ["E0", "E1", "D1", "I1", "SP1", "F1", "N1", "P1", "B1", "T1"]


def benchmark(features: pd.DataFrame, target_leagues=None, with_odds: bool = True) -> pd.DataFrame:
    """Global vs league-specific 1X2 log-loss karsilastirmasi."""
    target_leagues = target_leagues or BIG_LEAGUES
    cut = int(len(features) * 0.8)
    train, test = features.iloc[:cut], features.iloc[cut:].reset_index(drop=True)

    # global model
    gmod = CatBoostModel(with_odds=with_odds, iterations=300, verbose=False)
    try:
        gmod = gmod.fit(train)
        gp = gmod.predict(test)
        global_ll = log_loss_1x2(test["result"], gp.home_win, gp.draw, gp.away_win)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Global model basarisiz: %s", exc)
        global_ll = float("nan")

    rows = []
    for lg in target_leagues:
        tr = train[train["league"] == lg]
        te = test[test["league"] == lg].reset_index(drop=True)
        if len(tr) < 200 or len(te) < 30:
            rows.append({"league": lg, "n_train": len(tr), "n_test": len(te),
                         "global_ll": global_ll, "league_ll": None, "better": None})
            continue
        try:
            m = CatBoostModel(with_odds=with_odds, iterations=200, verbose=False).fit(tr)
            p = m.predict(te)
            ll = log_loss_1x2(te["result"], p.home_win, p.draw, p.away_win)
            rows.append({"league": lg, "n_train": len(tr), "n_test": len(te),
                         "global_ll": global_ll, "league_ll": ll,
                         "better": ll < global_ll})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lig %s modeli basarisiz: %s", lg, exc)
            rows.append({"league": lg, "n_train": len(tr), "n_test": len(te),
                         "global_ll": global_ll, "league_ll": None, "better": None})
    return pd.DataFrame(rows)


def recommend_league_models(report: pd.DataFrame) -> list[str]:
    """Sadece league-specific modelin GLOBALL'dan anlamli olcude daha iyi
    oldugu ligleri dondurur (ROADMAP 71: sadece validation destekliyorsa)."""
    out = []
    for _, r in report.iterrows():
        if r["league_ll"] is not None and r["league_ll"] < r["global_ll"] - 0.01:
            out.append(r["league"])
    return out
