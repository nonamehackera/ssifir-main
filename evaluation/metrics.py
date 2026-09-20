"""Evaluation metrics (ROADMAP bolum 50).

Out-of-sample metrics for every model output:
  1X2:       log loss, Brier score, accuracy, macro-F1, balanced accuracy
  Goals:      MAE, RMSE, Poisson deviance
  BTTS/O-U:   log loss, Brier, AUC, calibration error
  Betting:    ROI, yield, CLV, max drawdown, hit rate (only OOS)

All functions accept either pandas Series/arrays. Predictions are clipped to
avoid log(0).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _asarray(x):
    return np.asarray(x, dtype=float)


def _clip(p):
    return np.clip(_asarray(p), EPS, 1 - EPS)


def log_loss_1x2(y_true, p_home, p_draw, p_away) -> float:
    """y_true: result string 'H'/'D'/'A' or one-hot cols."""
    ph, pd_, pa = _clip(p_home), _clip(p_draw), _clip(p_away)
    if isinstance(y_true, (list, np.ndarray, object)):
        y = np.asarray(y_true)
        if y.dtype.kind in ("U", "S", "O"):
            y = np.select([y == "H", y == "D", y == "A"], [0, 1, 2], default=0).astype(int)
    else:
        y = np.asarray(y_true, dtype=int)
    p = np.select([y == 0, y == 1, y == 2], [ph, pd_, pa], default=ph)
    return float(-np.mean(np.log(p)))


def brier_1x2(y_true, p_home, p_draw, p_away) -> float:
    ph, pd_, pa = _clip(p_home), _clip(p_draw), _clip(p_away)
    y = np.asarray(y_true)
    if y.dtype.kind in ("U", "S", "O"):
        y = np.select([y == "H", y == "D", y == "A"], [0, 1, 2], default=0).astype(int)
    else:
        y = y.astype(int)
    oh = np.zeros((len(y), 3))
    oh[np.arange(len(y)), y] = 1
    pred = np.column_stack([ph, pd_, pa])
    return float(np.mean(np.sum((pred - oh) ** 2, axis=1)))


def accuracy_1x2(y_true, p_home, p_draw, p_away) -> float:
    ph, pd_, pa = _asarray(p_home), _asarray(p_draw), _asarray(p_away)
    pred = np.select([ph >= pd_, pa >= ph, pa >= pd_], [0, 2, 1], default=0)
    y = np.asarray(y_true)
    if y.dtype.kind in ("U", "S", "O"):
        y = np.select([y == "H", y == "D", y == "A"], [0, 1, 2], default=0).astype(int)
    else:
        y = y.astype(int)
    return float(np.mean(pred == y))


def macro_f1_1x2(y_true, p_home, p_draw, p_away) -> float:
    from sklearn.metrics import f1_score
    ph, pd_, pa = _asarray(p_home), _asarray(p_draw), _asarray(p_away)
    pred = np.select([ph >= pd_, pa >= ph, pa >= pd_], [0, 2, 1], default=0)
    y = np.asarray(y_true)
    if y.dtype.kind in ("U", "S", "O"):
        y = np.select([y == "H", y == "D", y == "A"], [0, 1, 2], default=0).astype(int)
    else:
        y = y.astype(int)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def brier_binary(y_true, p) -> float:
    y = _asarray(y_true).astype(float)
    p = _clip(p)
    return float(np.mean((p - y) ** 2))


def log_loss_binary(y_true, p) -> float:
    y = _asarray(y_true).astype(float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc_binary(y_true, p) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(_asarray(y_true).astype(int), _asarray(p)))
    except Exception:
        return float("nan")


def calibration_error(y_true, p, n_bins: int = 10) -> float:
    """ECE-style calibration error for binary target."""
    y = _asarray(y_true).astype(float)
    p = _asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    total = 0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        w = mask.sum()
        err += w * abs(conf - acc)
        total += w
    return float(err / total) if total > 0 else float("nan")


def mae_goals(y_true, pred) -> float:
    return float(np.mean(np.abs(_asarray(y_true) - _asarray(pred))))


def rmse_goals(y_true, pred) -> float:
    return float(np.sqrt(np.mean((_asarray(y_true) - _asarray(pred)) ** 2)))


def poisson_deviance(y_true, lam) -> float:
    """Poisson deviance for count predictions lam (expected)."""
    y = _asarray(y_true).astype(float)
    lam = np.clip(_asarray(lam), EPS, None)
    ll = y * np.log(y / lam) - (y - lam)
    ll = np.where(y == 0, lam, ll)  # 0*log(0) -> limit
    return float(2 * np.mean(ll))


def betting_roi(y_true_win: np.ndarray, odds: np.ndarray, p=None,
                stake: float = 1.0, threshold: float | None = None) -> dict:
    """ROI / yield for backing the home/draw/away selection at given odds.

    y_true_win: integer outcome (0/1/2). odds: (n,3) columns for H/D/A.
    If threshold given, only bet when model prob >= threshold (and bet the
    argmax selection).
    """
    y = np.asarray(y_true_win).astype(int)
    odds = _asarray(odds)
    if p is not None:
        p = _asarray(p)
    n = len(y)
    profit = np.zeros(n)
    invested = np.zeros(n)
    for i in range(n):
        if threshold is not None and p is not None:
            sel = int(np.argmax(p[i]))
            if p[i][sel] < threshold:
                continue
        else:
            sel = int(np.argmax(odds[i])) if p is None else int(np.argmax(p[i]))
        invested[i] = stake
        if y[i] == sel:
            profit[i] = stake * (odds[i][sel] - 1)
    total_invested = invested.sum()
    total_profit = profit.sum()
    roi = (total_profit / total_invested) if total_invested > 0 else 0.0
    return {
        "roi": float(roi),
        "yield": float(total_profit / n) if n > 0 else 0.0,
        "bets": int((invested > 0).sum()),
        "hit_rate": float((profit > 0).sum() / (invested > 0).sum()) if (invested > 0).sum() > 0 else 0.0,
    }
