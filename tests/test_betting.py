"""Betting katmani birim testleri (gercek kanit: betting/strategy.py calisiyor)."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betting.strategy import (
    evaluate_betting, kelly_fraction, market_margin, value_bets,
)
from models.base import PredictionResult


def test_kelly():
    # f* = (p*o - 1)/(o - 1)
    assert abs(kelly_fraction(0.5, 2.1, 1.0) - (0.5 * 2.1 - 1) / (2.1 - 1)) < 1e-9
    # negatif edge -> 0 stake
    assert kelly_fraction(0.3, 1.5, 1.0) == 0.0
    # half-kelly yariya duser
    full = kelly_fraction(0.5, 2.1, 1.0)
    half = kelly_fraction(0.5, 2.1, 0.5)
    assert abs(half - full * 0.5) < 1e-9
    print("OK test_kelly")


def test_market_margin():
    m = market_margin(2.0, 3.5, 4.0)
    assert m > 0  # bookmaker komisyonu
    assert abs(m - 0.0357) < 1e-3
    print("OK test_market_margin")


def test_value_bets_positive_edge():
    ph = np.array([0.50, 0.40]); pd_ = np.array([0.25, 0.30]); pa = np.array([0.25, 0.30])
    oh = np.array([1.90, 3.0]); od = np.array([3.50, 3.2]); oa = np.array([4.50, 2.6])
    vb = value_bets(ph, pd_, pa, oh, od, oa, edge_threshold=0.03)
    # mac0: away edge = 0.25*4.5-1 = +0.125 -> bahis (2)
    assert vb["selections"][0] == 2 and abs(vb["edges"][0] - 0.125) < 1e-9
    # mac1: home edge = 0.40*3.0-1 = +0.20 -> bahis (0)
    assert vb["selections"][1] == 0 and abs(vb["edges"][1] - 0.20) < 1e-9
    print("OK test_value_bets_positive_edge")


def test_value_bets_no_bet_when_negative():
    ph = np.array([0.30]); pd_ = np.array([0.30]); pa = np.array([0.40])
    oh = np.array([3.0]); od = np.array([3.0]); oa = np.array([2.0])
    # tum edge'ler negatif -> bahis yok
    vb = value_bets(ph, pd_, pa, oh, od, oa, 0.03)
    assert vb["selections"][0] == -1
    assert vb["n_bets"] == 0
    print("OK test_value_bets_no_bet_when_negative")


def test_evaluate_betting_end_to_end():
    n = 200
    rng = np.random.default_rng(42)
    valid = pd.DataFrame({
        "result": rng.choice(["H", "D", "A"], n, p=[0.45, 0.27, 0.28]),
        "avg_home_odds": np.round(rng.uniform(1.5, 3.0, n), 2),
        "avg_draw_odds": np.round(rng.uniform(3.0, 4.0, n), 2),
        "avg_away_odds": np.round(rng.uniform(2.5, 5.0, n), 2),
    })
    ohv = valid["avg_home_odds"].values; odv = valid["avg_draw_odds"].values; oav = valid["avg_away_odds"].values
    # model = bookmaker + kucuk gurultu (gercekci: model bookie'yi kopyalar)
    ph_ = 1 / ohv / 1.08; pd__ = 1 / odv / 1.08; pa_ = 1 / oav / 1.08
    s = ph_ + pd__ + pa_; ph_ /= s; pd__ /= s; pa_ /= s
    pred = PredictionResult(home_win=ph_, draw=pd__, away_win=pa_)
    rep = evaluate_betting(valid, pred, edge_threshold=0.03)
    assert all(k in rep for k in ("value_bet", "always_best", "clv"))
    assert "roi" in rep["value_bet"] and "roi" in rep["always_best"]
    print("OK test_evaluate_betting_end_to_end")


if __name__ == "__main__":
    test_kelly()
    test_market_margin()
    test_value_bets_positive_edge()
    test_value_bets_no_bet_when_negative()
    test_evaluate_betting_end_to_end()
    print("\nALL BETTING UNIT TESTS PASSED")
