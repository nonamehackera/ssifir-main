"""Bahis EDGE (deger bahsi) katmani (ROADMAP bolum 51+).

Sistem neden 'bahsi oynayamiyor'?
  Model olasilik uretiyor ama HICBIR YERDE 'bu maca bahis oynayalim mi,
  edgeimiz var mi?' hesabi yok. Sadece argmax (en yuksek olasilikli
  sonucu sec) ile bahis oynarsan bookmaker vig'i (komisyon) yuzunden
  hep KAYBEDERSIN. Cunku favori orani zaten favori olma ihtimaline gore
  ayarli + bookmaker payi.

Dogru yaklasim: VALUE BETTING.
  edge = model_prob * odds - 1
  edge > threshold (orn. 0.03 = %%3 deger) olan maclara bahis oyna.
  Boylece modelin bookmaker'dan daha iyi bildigi (edge'li) maclari oynarsin.

Bu modul:
  - value_bets(): hangi (H/D/A) secimlerinde pozitif edge var
  - kelly_fraction(): optimal stake (tum parayi yakmamak icin)
  - evaluate_betting(): gercek OOS ROI / yield / hit_rate / CLV raporu
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from evaluation.metrics import betting_roi

logger = logging.getLogger(__name__)

# Bookmaker margin (vig) hesabi: 1/p_h + 1/p_d + 1/p_a - 1
# FD avg odds'u icin bu genelde %0.05-0.10 arasinda.
DEFAULT_EDGE_THRESHOLD = 0.03  # %%3 deger: bahis oynamak icin min edge


def implied_prob(odds: float) -> float:
    """Orani olasiliga cevir (1/odds)."""
    return 1.0 / odds if odds and odds > 0 else 0.0


def market_margin(odds_h: float, odds_d: float, odds_a: float) -> float:
    """Bookmaker komisyonu (overround). 0 ise mukemmel piyasa."""
    s = 0.0
    for o in (odds_h, odds_d, odds_a):
        if o and o > 0:
            s += 1.0 / o
    return s - 1.0


def value_bets(
    p_home: np.ndarray,
    p_draw: np.ndarray,
    p_away: np.ndarray,
    odds_h: np.ndarray,
    odds_d: np.ndarray,
    odds_a: np.ndarray,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> dict:
    """Her mac icin pozitif edge'li (H/D/A) secimleri dondur.

    Doner:
      selections: (n,) secilen sinif (0/1/2) veya -1 (bahis yok)
      edges:      (n,) o secimdeki edge (model_prob*odds - 1)
      probs:      (n,) secilen sinifin model olasiligi
      bet_odds:   (n,) secilen sinifin orani
    """
    p_home = np.asarray(p_home, float)
    p_draw = np.asarray(p_draw, float)
    p_away = np.asarray(p_away, float)
    odds_h = np.asarray(odds_h, float)
    odds_d = np.asarray(odds_d, float)
    odds_a = np.asarray(odds_a, float)

    # her sinif icin edge
    e_h = np.where((odds_h > 0) & np.isfinite(odds_h), p_home * odds_h - 1, -1e9)
    e_d = np.where((odds_d > 0) & np.isfinite(odds_d), p_draw * odds_d - 1, -1e9)
    e_a = np.where((odds_a > 0) & np.isfinite(odds_a), p_away * odds_a - 1, -1e9)

    E = np.column_stack([e_h, e_d, e_a])
    sel = np.argmax(E, axis=1)
    best_edge = E[np.arange(len(sel)), sel]

    # edge threshold altindaysa bahis yok
    no_bet = best_edge < edge_threshold
    sel = np.where(no_bet, -1, sel)
    best_edge = np.where(no_bet, 0.0, best_edge)

    bet_odds = np.select(
        [sel == 0, sel == 1, sel == 2],
        [odds_h, odds_d, odds_a],
        default=0.0,
    )
    sel_prob = np.select(
        [sel == 0, sel == 1, sel == 2],
        [p_home, p_draw, p_away],
        default=0.0,
    )

    return {
        "selections": sel,
        "edges": best_edge,
        "probs": sel_prob,
        "bet_odds": bet_odds,
        "n_bets": int((sel >= 0).sum()),
    }


def kelly_fraction(prob: float, odds: float, frac: float = 1.0) -> float:
    """Kelly kriteri ile optimal stake yuzdesi.

    f* = (p*(o-1) - (1-p)) / (o-1) = (p*o - 1) / (o - 1)
    frac: risk azaltma (0.5 = half-Kelly, agresif degil).
    """
    if odds <= 1 or prob <= 0:
        return 0.0
    f = (prob * odds - 1.0) / (odds - 1.0)
    f = max(0.0, f) * frac
    return float(min(f, 0.25))  # tek macta max %%25 stake (guvenlik)


def evaluate_betting(
    valid: pd.DataFrame,
    pred,
    odds_cols: tuple[str, str, str] = ("avg_home_odds", "avg_draw_odds", "avg_away_odds"),
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    closing_cols: tuple[str, str, str] | None = None,
    stake: float = 1.0,
) -> dict:
    """Tek script'te gercek OOS bahis raporu.

    pred: PredictionResult (home_win/draw/away_win array'leri)
    odds_cols: modelin bahis oynadigi anki oranlar (FD avg = acilis benzeri)
    closing_cols: kapanis oranlari (CLV icin). Yoksa odds_cols kullanilir.

    Doner:
      value_bet: edge threshold'lu deger bahsi raporu (ROI/yield/hit)
      always_best: argmax sonucu her maca bahis raporu (karsilastirma)
      clv: ortalama CLV (model kapanisa gore iyi fiyat buldu mu?)
    """
    p_home = np.asarray(pred.home_win, float)
    p_draw = np.asarray(pred.draw_win if hasattr(pred, "draw_win") else pred.draw, float)
    p_away = np.asarray(pred.away_win, float)

    oh = valid[odds_cols[0]].to_numpy(dtype=float)
    od = valid[odds_cols[1]].to_numpy(dtype=float)
    oa = valid[odds_cols[2]].to_numpy(dtype=float)

    y = valid["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    # --- 1) VALUE BET (edge threshold) ---
    vb = value_bets(p_home, p_draw, p_away, oh, od, oa, edge_threshold)
    prof_vb = np.zeros(len(y))
    invest_vb = np.zeros(len(y))
    # Kelly stake ile (half-kelly)
    for i in range(len(y)):
        if vb["selections"][i] < 0:
            continue
        sel = int(vb["selections"][i])
        o = vb["bet_odds"][i]
        p = vb["probs"][i]
        kf = kelly_fraction(p, o, frac=0.5)
        if kf <= 0:
            continue
        s = stake * kf
        invest_vb[i] = s
        if y[i] == sel:
            prof_vb[i] = s * (o - 1)
    n_bets_vb = int((invest_vb > 0).sum())
    roi_vb = prof_vb.sum() / invest_vb.sum() if invest_vb.sum() > 0 else 0.0
    yield_vb = prof_vb.sum() / len(y) if len(y) > 0 else 0.0
    hit_vb = (prof_vb > 0).sum() / n_bets_vb if n_bets_vb > 0 else 0.0
    avg_edge_vb = vb["edges"][vb["selections"] >= 0].mean() if n_bets_vb > 0 else 0.0

    # --- 2) ALWAYS BEST (argmax her maca bahis) ---
    ab_sel = np.select([p_home >= p_draw, p_away >= p_home], [0, 2], default=1)
    ab_odds = np.select([ab_sel == 0, ab_sel == 1, ab_sel == 2], [oh, od, oa], default=0.0)
    prof_ab = np.zeros(len(y))
    invest_ab = np.full(len(y), stake)
    for i in range(len(y)):
        if ab_odds[i] <= 0:
            invest_ab[i] = 0
            continue
        if y[i] == ab_sel[i]:
            prof_ab[i] = stake * (ab_odds[i] - 1)
    n_bets_ab = int((invest_ab > 0).sum())
    roi_ab = prof_ab.sum() / invest_ab.sum() if invest_ab.sum() > 0 else 0.0
    yield_ab = prof_ab.sum() / len(y) if len(y) > 0 else 0.0
    hit_ab = (prof_ab > 0).sum() / n_bets_ab if n_bets_ab > 0 else 0.0

    # --- 3) CLV (kapanisa gore) ---
    clv_list = []
    close_src = closing_cols if closing_cols else odds_cols
    co_h = valid[close_src[0]].to_numpy(dtype=float)
    co_d = valid[close_src[1]].to_numpy(dtype=float)
    co_a = valid[close_src[2]].to_numpy(dtype=float)
    for i in range(len(y)):
        if vb["selections"][i] < 0:
            continue
        sel = int(vb["selections"][i])
        pred_o = vb["bet_odds"][i]
        close_o = [co_h, co_d, co_a][sel][i]
        if close_o and close_o > 0 and pred_o and pred_o > 0:
            clv_list.append(pred_o / close_o - 1.0)
    mean_clv = float(np.mean(clv_list)) if clv_list else 0.0
    clv_pos = float(np.mean([c > 0 for c in clv_list])) if clv_list else 0.0

    return {
        "n_valid": len(y),
        "value_bet": {
            "n_bets": n_bets_vb,
            "roi": float(roi_vb),
            "yield": float(yield_vb),
            "hit_rate": float(hit_vb),
            "avg_edge": float(avg_edge_vb),
            "edge_threshold": edge_threshold,
        },
        "always_best": {
            "n_bets": n_bets_ab,
            "roi": float(roi_ab),
            "yield": float(yield_ab),
            "hit_rate": float(hit_ab),
        },
        "clv": {
            "mean_clv": mean_clv,
            "positive_rate": clv_pos,
            "n": len(clv_list),
        },
    }


def summarize_betting(reports: list[dict]) -> pd.DataFrame:
    """Fold raporlarini tabloya cevir."""
    rows = []
    for r in reports:
        vb = r["value_bet"]
        ab = r["always_best"]
        rows.append({
            "fold": r.get("fold", "?"),
            "n": r["n_valid"],
            "vb_bets": vb["n_bets"],
            "vb_roi": vb["roi"],
            "vb_yield": vb["yield"],
            "vb_hit": vb["hit_rate"],
            "vb_edge": vb["avg_edge"],
            "ab_roi": ab["roi"],
            "ab_hit": ab["hit_rate"],
            "clv": r["clv"]["mean_clv"],
            "clv_pos": r["clv"]["positive_rate"],
        })
    return pd.DataFrame(rows)
