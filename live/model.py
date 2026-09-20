"""Canli model (ROADMAP bolum 41-43, 23).

Pre-match modelinin ciktiini (beklenen goller, 1X2) canli state ile
birlestirir. Kritik kural (ROADMAP 23): t dakikadaki tahmin SADECE 0-t
arasi gozlemlenen verilerle yapilir.

Yaklasim:
  - Pre-match: home_lambda_0, away_lambda_0 (mac oncesi beklenen gol)
  - Canli: t dakikada xG_home_t, xG_away_t (o ana kadar uretilen)
  - Kalan dakika orani: (90 - t) / 90
  - Kalan gol beklenen: pre_lambda * kalan_oran  (basit, ancak guclu)
  - Toplam tahmini gol: canli_xg + kalan_beklenen
  - 1X2: Poisson score matrix (canli_xg + kalan_beklenen) uzerinden

Bu, "pre-match model ve canli xG iki ayri bilgi kaynagi" prensibini
(ROADMAP 43) uygular: ikisi de kullanilir, gelecek bilgisi kullanilmaz.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)

MATCH_LENGTH = 90.0


def _poisson_1x2(home_lam: float, away_lam: float, max_g: int = 10) -> tuple[float, float, float]:
    gh = np.arange(max_g + 1)
    ph = poisson.pmf(gh, home_lam)
    pa = poisson.pmf(gh, away_lam)
    mat = ph[:, None] * pa[None, :]
    s = mat.sum()
    if s <= 0:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3
    mat = mat / s
    g1, g2 = np.meshgrid(gh, gh, indexing="ij")
    home = mat[g1 > g2].sum()
    away = mat[g1 < g2].sum()
    draw = mat[g1 == g2].sum()
    return float(home), float(draw), float(away)


def live_predict(
    pre_home_lambda: float,
    pre_away_lambda: float,
    minute: float,
    xg_home: float | None = None,
    xg_away: float | None = None,
    score_home: int = 0,
    score_away: int = 0,
    live_odds_home: float | None = None,
    live_odds_draw: float | None = None,
    live_odds_away: float | None = None,
) -> dict:
    """Canli tahmin uretir.

    pre_home_lambda: mac oncesi beklenen ev sahibi gol (pre-match modelinden)
    minute:           su anki dakika (0-90+)
    xg_home/xg_away:  o ana kadar uretilen canli xG (yoksa None)
    score_*:         su anki skor (gelecek degil, mevcut durum)

    Donecek: ROADMAP 52 benzeri canli tahmin JSON.
    """
    minute = max(0.0, min(float(minute), 120.0))
    remaining = max(0.0, (MATCH_LENGTH - minute) / MATCH_LENGTH)

    # canli xG yoksa, pre-match beklenenin kalan orani kadarini kullan
    if xg_home is None:
        xg_home = pre_home_lambda * (1.0 - remaining)
    if xg_away is None:
        xg_away = pre_away_lambda * (1.0 - remaining)

    # kalan beklenen gol: pre-match lambda * kalan sure orani
    remaining_home = pre_home_lambda * remaining
    remaining_away = pre_away_lambda * remaining

    # toplam tahmini (mevcut canli xG + kalan beklenen)
    total_home = float(xg_home) + remaining_home
    total_away = float(xg_away) + remaining_away

    home_p, draw_p, away_p = _poisson_1x2(total_home, total_away)

    # canli odds varsa implied probability (sadece referans)
    market = None
    if live_odds_home and live_odds_draw and live_odds_away:
        inv = np.array([live_odds_home, live_odds_draw, live_odds_away])
        implied = 1.0 / inv
        implied = implied / implied.sum()
        market = {"home": round(float(implied[0]), 3),
                  "draw": round(float(implied[1]), 3),
                  "away": round(float(implied[2]), 3)}

    return {
        "minute": int(minute),
        "score": f"{score_home}-{score_away}",
        "expected_goals": {
            "home_total": round(total_home, 3),
            "away_total": round(total_away, 3),
            "remaining_home": round(remaining_home, 3),
            "remaining_away": round(remaining_away, 3),
        },
        "result": {
            "home": round(home_p, 3),
            "draw": round(draw_p, 3),
            "away": round(away_p, 3),
        },
        "market_implied": market,
        "note": "ROADMAP 23: sadece 0-%d arasi veri kullanildi." % int(minute),
    }


def backtest_live(fixture_snapshots: list[dict], pre_lambda: tuple) -> dict:
    """Verilen canli snapshot serisini kullanarak her dakika tahmin uretir.

    fixture_snapshots: [{minute, xg_home, xg_away, score_home, score_away,
                         live_odds_home, live_odds_draw, live_odds_away}, ...]
    pre_lambda: (pre_home_lambda, pre_away_lambda)

    Donecek: her snapshot icin tahmin + son gercek sonuc.
    """
    out = []
    for snap in fixture_snapshots:
        pred = live_predict(
            pre_lambda[0], pre_lambda[1],
            minute=snap.get("minute", 0),
            xg_home=snap.get("xg_home"), xg_away=snap.get("xg_away"),
            score_home=snap.get("score_home", 0), score_away=snap.get("score_away", 0),
            live_odds_home=snap.get("live_odds_home"),
            live_odds_draw=snap.get("live_odds_draw"),
            live_odds_away=snap.get("live_odds_away"),
        )
        out.append(pred)
    return {"n_snapshots": len(out), "predictions": out}
