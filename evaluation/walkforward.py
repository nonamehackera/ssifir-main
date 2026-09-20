"""Walk-forward evaluation (ROADMAP bolum 49).

Chronological folds; every model is trained ONLY on matches before the
validation window. Within each fold the training set is split into
fit (last portion) and calibration slice for ensemble weights + temperature,
keeping weight fitting honest.

Aggregated report: 1X2 log loss / Brier / accuracy + BTTS & Over2.5
log loss / Brier / AUC per model family across folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evaluation.metrics import (
    accuracy_1x2,
    auc_binary,
    brier_1x2,
    calibration_error,
    log_loss_1x2,
    log_loss_binary,
    brier_binary,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    start: object
    end: object
    n_train: int
    n_cal: int
    n_valid: int
    models: dict[str, dict] = field(default_factory=dict)


def make_folds(feats: pd.DataFrame, n_folds: int = 4) -> list[tuple[object, object]]:
    """n_folds ardışık üçer aylık validation penceresi (walk-forward)."""
    start = pd.Timestamp("2023-07-01")
    months = 3
    return [
        (start + pd.DateOffset(months=i * months),
         start + pd.DateOffset(months=(i + 1) * months))
        for i in range(n_folds)
    ]


def eval_1x2(df: pd.DataFrame, pred) -> dict:
    y = df["result"]
    return {
        "log_loss": log_loss_1x2(y, pred.home_win, pred.draw, pred.away_win),
        "brier": brier_1x2(y, pred.home_win, pred.draw, pred.away_win),
        "accuracy": accuracy_1x2(y, pred.home_win, pred.draw, pred.away_win),
        "ece": calibration_error((df["result"] == "H").astype(int), pred.home_win),
    }


def eval_binary(df: pd.DataFrame, y_col: str, p) -> dict:
    y = df[y_col].to_numpy()
    return {
        "log_loss": log_loss_binary(y, p),
        "brier": brier_binary(y, p),
        "auc": auc_binary(y, p),
    }


def run_walkforward(
    feats: pd.DataFrame,
    model_factories: dict[str, callable],
    n_folds: int = 4,
    cal_frac: float = 0.15,
) -> list[FoldResult]:
    folds = make_folds(feats, n_folds)
    results: list[FoldResult] = []

    for i, (v_start, v_end) in enumerate(folds):
        valid = feats[(feats["date"] >= v_start) & (feats["date"] < v_end)].copy()
        if len(valid) < 50:
            logger.warning("Fold %d cok kucuk (%d), atlaniyor", i, len(valid))
            continue
        train_all = feats[feats["date"] < v_start].copy()
        train_all = train_all.sort_values("date").reset_index(drop=True)
        n_cal = max(50, int(len(train_all) * cal_frac))
        cal = train_all.iloc[-n_cal:]
        train = train_all.iloc[:-n_cal]

        fr = FoldResult(
            fold=i, start=v_start, end=v_end,
            n_train=len(train), n_cal=len(cal), n_valid=len(valid),
        )
        logger.info("== Fold %d: train=%d cal=%d valid=%d ==", i, len(train), len(cal), len(valid))

        for name, factory in model_factories.items():
            model = factory()
            try:
                model.fit(train, cal)
                pred = model.predict(valid)
                fr.models[name] = {
                    "1x2": eval_1x2(valid, pred),
                    "btts": eval_binary(valid, "btts", pred.btts_yes),
                    "over25": eval_binary(valid, "over25", pred.over25),
                }
                logger.info(
                    "  %-14s logloss=%.4f brier=%.4f acc=%.3f btts_auc=%.3f",
                    name,
                    fr.models[name]["1x2"]["log_loss"],
                    fr.models[name]["1x2"]["brier"],
                    fr.models[name]["1x2"]["accuracy"],
                    fr.models[name]["btts"]["auc"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Fold %d model %s hata: %s", i, name, exc)

        results.append(fr)

    return results


def summarize(results: list[FoldResult]) -> pd.DataFrame:
    """Model x metrik tablosu (fold ortalamalari)."""
    rows = []
    for fr in results:
        for name, metrics in fr.models.items():
            rows.append(
                {
                    "fold": fr.fold,
                    "model": name,
                    "n": fr.n_valid,
                    **metrics["1x2"],
                    "btts_ll": metrics["btts"]["log_loss"],
                    "btts_brier": metrics["btts"]["brier"],
                    "btts_auc": metrics["btts"]["auc"],
                    "over25_ll": metrics["over25"]["log_loss"],
                    "over25_auc": metrics["over25"]["auc"],
                }
            )
    return pd.DataFrame(rows).groupby("model", as_index=False).agg(
        n=("n", "sum"),
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        accuracy=("accuracy", "mean"),
        ece=("ece", "mean"),
        btts_ll=("btts_ll", "mean"),
        btts_auc=("btts_auc", "mean"),
        over25_ll=("over25_ll", "mean"),
        over25_auc=("over25_auc", "mean"),
    ).sort_values("log_loss")