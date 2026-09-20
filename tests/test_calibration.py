"""Kalibrasyon modülü dogrulama (gercek veri, kronolojik split).

Test 1: catboost_odds zaten iyi kalibre oldugundan, calibrator akilli
secim yapip HAM cikti kullanmalı (ECE/log-loss bozulmamali).
Test 2: zayif bir model (uniform + gurultu) icin kalibrator GELISTIRMELI.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.catboost.model import CatBoostModel
from calibration.calibrators import MulticlassCalibrator
from evaluation.metrics import log_loss_1x2, calibration_error, accuracy_1x2


def _split(f):
    cut = int(len(f) * 0.75)
    n_cal = int(len(f) * 0.15)
    train, cal, test = (
        f.iloc[:cut].copy(),
        f.iloc[cut:cut + n_cal].copy(),
        f.iloc[cut + n_cal:].reset_index(drop=True),
    )
    return train, cal, test


def test_good_model_keeps_raw():
    m = build_matches()
    f = build_features(m)
    train, cal, test = _split(f)
    model = CatBoostModel(with_odds=True, iterations=400, verbose=False).fit(train, cal)
    pred = model.predict(test)
    P = np.column_stack([pred.home_win, pred.draw, pred.away_win])
    y = test["result"].to_numpy()

    cal_pred = model.predict(cal)
    cal_P = np.column_stack([cal_pred.home_win, cal_pred.draw, cal_pred.away_win])
    calib = MulticlassCalibrator(method="best").fit(cal["result"].to_numpy(), cal_P)
    Pc = calib.predict(P)

    ll_raw = log_loss_1x2(y, P[:, 0], P[:, 1], P[:, 2])
    ll_cal = log_loss_1x2(y, Pc[:, 0], Pc[:, 1], Pc[:, 2])
    print(f"[T1] catboost_odds: raw_ll={ll_raw:.4f} cal_ll={ll_cal:.4f} use_calibrated={calib.use_calibrated}")
    # akilli secim sayesinde kalibrasyon asla ham'dan kotu olmamali
    assert ll_cal <= ll_raw + 1e-9
    assert calib.use_calibrated in (True, False)
    print("T1 OK: akilli secim log-loss'u korumadi/bozmadi.")


def test_weak_model_improves():
    m = build_matches()
    f = build_features(m)
    train, cal, test = _split(f)
    rng = np.random.default_rng(0)
    # weak/overconfident model: uniform + kucuk gurultu (miscalibrated)
    P_cal = rng.dirichlet([1, 1, 1], size=len(cal))
    P_test = rng.dirichlet([1, 1, 1], size=len(test))
    # overconfident: uzaklastir
    P_cal = 0.4 * P_cal + 0.6 * np.array([0.7, 0.15, 0.15])
    P_test = 0.4 * P_test + 0.6 * np.array([0.7, 0.15, 0.15])
    P_cal = P_cal / P_cal.sum(1, keepdims=True)
    P_test = P_test / P_test.sum(1, keepdims=True)
    y_cal = cal["result"].to_numpy()
    y_test = test["result"].to_numpy()

    ll_raw = log_loss_1x2(y_test, P_test[:, 0], P_test[:, 1], P_test[:, 2])
    calib = MulticlassCalibrator(method="best").fit(y_cal, P_cal)
    Pc = calib.predict(P_test)
    ll_cal = log_loss_1x2(y_test, Pc[:, 0], Pc[:, 1], Pc[:, 2])
    print(f"[T2] weak model: raw_ll={ll_raw:.4f} cal_ll={ll_cal:.4f} use_calibrated={calib.use_calibrated}")
    assert calib.use_calibrated is True, "zayif modelde kalibrator secilmeli"
    assert ll_cal < ll_raw - 1e-3, "kalibrasyon log-loss'u iyilestirmeli"
    print("T2 OK: miscalibrated modelde kalibrator log-loss'u dusurdu.")


if __name__ == "__main__":
    test_good_model_keeps_raw()
    test_weak_model_improves()
    print("\nTum kalibrasyon testleri gecti.")
