"""Probability calibration (ROADMAP bolum 34).

Model olasiliklari guvenilir mi? Yani model "Home %72" diyorsa, o maclarin
gercekten ~%72'sini mi kazaniyor? Bu modul bunu olcer ve duzeltir.

Uç yontem (binary hedefler icin):
  - Platt Scaling   : logit(p) -> logistic regression (smooth, az veriyle calisir)
  - Isotonic        : sklearn IsotonicRegression (daha esnek, cok veri ister)
  - Beta Calibration: p_cal = a*p^c / (a*p^c + (1-p)^c)  (Kullback 2015)

Multiclass (1X2) icin: her sinif one-vs-rest olarak ayri binary
calibrator ile kalibre edilir, sonra yeniden normalize edilir. Bu,
temperature scaling'den daha esnektir (her sinif kendi egrisine sahip olur).

Onemli: calibrator YALNIZCA calibration diliminde (train disi, ama
gelecekteki test'e sizdirilmayan) fit edilir. Walk-forward icinde
`cal` parametresiyle calistirilir.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p) - np.log(1 - p)


class BinaryCalibrator:
    """Tek bir hedefin (evet/hayir) olasiligini kalibre eder.

    fit(y_true, p_raw): y_true 0/1, p_raw modelin ham olasiligi.
    predict(p_raw): kalibre olasilik.
    method: 'platt' | 'isotonic' | 'beta' | 'best' (validation log-loss'a gore)
    """

    def __init__(self, method: str = "best") -> None:
        self.method = method
        self._fitted_method: str | None = None
        self._params: dict = {}
        self._isotonic: IsotonicRegression | None = None

    # ---- Platt ----
    def _fit_platt(self, y: np.ndarray, p: np.ndarray):
        try:
            z = _logit(p).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs")
            lr.fit(z, y)
            self._params["platt"] = (lr.intercept_[0], lr.coef_[0][0])
        except Exception:
            # fallback: sabit
            self._params["platt"] = (0.0, 1.0)

    @staticmethod
    def _apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
        z = _logit(p)
        return 1.0 / (1.0 + np.exp(-(a + b * z)))

    # ---- Beta ----
    def _fit_beta(self, y: np.ndarray, p: np.ndarray):
        p = _clip(p)

        def nll(theta):
            a, c = theta[0], theta[1]
            a = max(a, 1e-3)
            c = max(c, 1e-3)
            num = a * np.power(p, c)
            den = num + np.power(1 - p, c)
            pc = np.clip(num / den, 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))

        try:
            res = minimize(nll, [1.0, 1.0], method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-4})
            a = max(float(res.x[0]), 1e-3)
            c = max(float(res.x[1]), 1e-3)
        except Exception:
            a, c = 1.0, 1.0
        self._params["beta"] = (a, c)

    @staticmethod
    def _apply_beta(p: np.ndarray, a: float, c: float) -> np.ndarray:
        p = _clip(p)
        num = a * np.power(p, c)
        den = num + np.power(1 - p, c)
        return np.clip(num / den, 1e-9, 1 - 1e-9)

    # ---- Isotonic ----
    def _fit_isotonic(self, y: np.ndarray, p: np.ndarray):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(p, y)
        self._isotonic = iso

    # ---- fit ----
    def fit(self, y_true, p_raw) -> "BinaryCalibrator":
        y = np.asarray(y_true, dtype=float)
        p = _clip(np.asarray(p_raw, dtype=float))

        self._fit_platt(y, p)
        self._fit_beta(y, p)
        try:
            self._fit_isotonic(y, p)
        except Exception:
            self._isotonic = None

        if self.method == "best":
            from evaluation.metrics import log_loss_binary

            cands = {
                "platt": self._apply_platt(p, *self._params["platt"]),
                "beta": self._apply_beta(p, *self._params["beta"]),
            }
            if self._isotonic is not None:
                cands["isotonic"] = self._isotonic.predict(p)
            scored = {k: log_loss_binary(y, v) for k, v in cands.items()}
            best = min(scored, key=scored.get)
            self._fitted_method = best
            logger.info("Binary calibrator secildi: %s (ll=%.4f)", best, scored[best])
        else:
            self._fitted_method = self.method
        return self

    def predict(self, p_raw) -> np.ndarray:
        p = _clip(np.asarray(p_raw, dtype=float))
        m = self._fitted_method
        if m == "platt":
            return self._apply_platt(p, *self._params["platt"])
        if m == "beta":
            return self._apply_beta(p, *self._params["beta"])
        if m == "isotonic" and self._isotonic is not None:
            return np.clip(self._isotonic.predict(p), 1e-6, 1 - 1e-6)
        return p


class MulticlassCalibrator:
    """1X2 (H/D/A) olasiliklarini one-vs-rest ile kalibre eder.

    Girdi: (n,3) ham olasilik matrisi. Her sinif ayri BinaryCalibrator ile
    kalibre edilir, ardindan satir bazinda normalize edilir.

    ONEMLI: calibration her zaman iyilestirmez. Eger model zaten iyi
    kalibreyse (CatBoost gibi) ve cal dilimi kucukse, esnek bir kalibrator
    overfit olup zarar verebilir. Bu yuzden fit() hem kalibre hem ham
    ciktiinin validation log-loss'unu olcer ve DAHA IYI OLANI secer
    (self.use_calibrated bayragi). predict() otomatik olarak en iyisini dondurur.
    """

    def __init__(self, method: str = "best") -> None:
        self.method = method
        self._cal: list[BinaryCalibrator] = []
        self.use_calibrated = True
        self._cal_ll: float | None = None
        self._raw_ll: float | None = None

    def fit(self, y_true, P: np.ndarray, P_raw_eval: np.ndarray | None = None) -> "MulticlassCalibrator":
        """y_true: sonuclar; P: kalibrasyon diliminin ham olasiliklari (n,3).
        P_raw_eval: dogrulama icin ayni dilim (genelde P ile ayni)."""
        from evaluation.metrics import log_loss_1x2

        y = np.asarray(y_true)
        P = np.asarray(P, dtype=float)
        if y.dtype.kind in ("U", "S", "O"):
            y = np.select([y == "H", y == "D", y == "A"], [0, 1, 2], default=0).astype(int)
        else:
            y = y.astype(int)
        self._cal = []
        for k in range(3):
            bc = BinaryCalibrator(self.method)
            bc.fit((y == k).astype(int), P[:, k])
            self._cal.append(bc)

        # kalibre edilmis matrisi dogrudan hesapla (predict() bayragindan bagimsiz)
        Pc = np.column_stack([self._cal[k].predict(P[:, k]) for k in range(3)])
        s = Pc.sum(axis=1, keepdims=True)
        s = np.where(s <= 0, 1.0, s)
        Pc = Pc / s
        self._cal_ll = log_loss_1x2(y, Pc[:, 0], Pc[:, 1], Pc[:, 2])
        self._raw_ll = log_loss_1x2(y, P[:, 0], P[:, 1], P[:, 2])
        # gecersiz (nan/inf) kalibrasyon her zaman ham'dan kotudur
        if not np.isfinite(self._cal_ll):
            self.use_calibrated = False
        else:
            # Kalibrasyon HER ZAMAN iyilestirmez. Model zaten iyi kalibreyse
            # (dusuk ECE) ve cal dilimi kucukse, esnek kalibrator overfit olup
            # test'te ZARAR verebilir. Bu yuzden kalibrasyonu YALNIZCA
            # (a) ham'dan dusuk log-loss veriyorsa VE
            # (b) model ham'da belirgin miscalibrated ise (ece eşigi) uygula.
            from evaluation.metrics import calibration_error

            ece_raw = calibration_error((y == 0).astype(int), P[:, 0])
            improves = self._cal_ll <= self._raw_ll - 1e-3  # anlamli iyilestirme
            miscalibrated = ece_raw > 0.02
            self.use_calibrated = bool(improves and miscalibrated)
        logger.info(
            "Multiclass calibrator: raw_ll=%.4f cal_ll=%.4f -> %s",
            self._raw_ll, self._cal_ll,
            "kalibre kullan" if self.use_calibrated else "ham kullan",
        )
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        if not self.use_calibrated:
            s = P.sum(axis=1, keepdims=True)
            s = np.where(s <= 0, 1.0, s)
            return P / s
        out = np.column_stack([self._cal[k].predict(P[:, k]) for k in range(3)])
        s = out.sum(axis=1, keepdims=True)
        s = np.where(s <= 0, 1.0, s)
        return out / s


def select_best_method(y_true, p_raw, methods=("platt", "isotonic", "beta")) -> str:
    """Validation uzerinde hangi kalibrasyon yonteminin en dusuk log-loss
    verdigini dondurur (BinaryCalibrator.fit(method='best') ile ayni, ama
    disaridan cagrilabilir)."""
    from evaluation.metrics import log_loss_binary

    y = np.asarray(y_true, dtype=float)
    bc = BinaryCalibrator(method="best")
    bc.fit(y, p_raw)
    return bc._fitted_method
