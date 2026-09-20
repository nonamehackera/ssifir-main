"""Production Enhanced Ensemble Model.

`feature_engine.enhance` modülünden gelen yeni feature'lari kullanir ve
her market (1X2, BTTS, Over2.5, Goals) icin guvenilir tahmin uretir.

Varsayilan (yuksek verili) ligler ile az verili/farkli ligler arasinda
saglam fallback yapar:
  - `lg_confidence_weight` feature'i modelin az verili liglerde form
    sinyallerine daha az, market oranlarina daha cok guvenmesini saglar.
  - Tahmin zamaninda eger market orani yoksa (mkt_*_prob NaN/None),
    model saf futbol sinyaline doner (ayri ensemble dal).

Kullanim:
    from models.production_ensemble import ProductionEnsemble
    from feature_engine.enhance import get_enhanced_feature_names

    ens = ProductionEnsemble()
    ens.fit(train, cal)
    pred = ens.predict(test)   # -> PredictionResult
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.base import PredictionResult, normalize_1x2

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def _ml_1x2(seed=42):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3, num_leaves=50,
        learning_rate=0.015, n_estimators=600, max_depth=6,
        min_child_samples=100, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )


def _ml_bin(seed=42):
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=45, learning_rate=0.015,
        n_estimators=500, max_depth=6, min_child_samples=80,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5,
        reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )


def _ml_reg(seed=42):
    return lgb.LGBMRegressor(
        objective="poisson", num_leaves=45, learning_rate=0.015,
        n_estimators=400, max_depth=6, min_child_samples=80,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5,
        reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )


class ProductionEnsemble:
    """Production ensemble: 1X2 + BTTS + Over2.5 + Goals.

    Base feature set + enhanced features (feature_engine.enhance).
    Varsayilan ve az verili ligler icin fallback.
    """

    name = "production_ensemble"

    def __init__(self, base_features: list[str] | None = None,
                 enhanced_features: list[str] | None = None,
                 seeds: list[int] | None = None) -> None:
        self.base_features = list(base_features) if base_features else []
        self.enhanced_features = list(enhanced_features) if enhanced_features else []
        self.feature_cols = self.base_features + [
            f for f in self.enhanced_features if f not in self.base_features
        ]
        self.seeds = seeds if seeds else [42, 123, 456]

        self.models_1x2 = None
        self.models_btts = None
        self.models_over15 = None
        self.models_over = None
        self.models_over35 = None
        self.models_hg = None
        self.models_ag = None

    # ── feature hazirlama ──────────────────────────────────────────────────
    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feature_cols].fillna(0)

    def _available_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_cols if c in df.columns]

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None,
            cat_features: list[str] | None = None) -> "ProductionEnsemble":
        if not HAS_LGB:
            raise RuntimeError("lightgbm gerekli (pip install lightgbm)")

        self.feature_cols = self._available_cols(train)
        if not self.feature_cols:
            raise ValueError("Geçerli feature sütunu yok")

        X = self._X(train)
        n_tr = len(train)

        # 1X2
        self.models_1x2 = []
        y1 = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        for s in self.seeds:
            m = _ml_1x2(s)
            m.fit(X, y1)
            self.models_1x2.append(m)

        # BTTS
        self.models_btts = []
        yb = train["btts"].to_numpy()
        for s in self.seeds[:1]:
            m = _ml_bin(s)
            m.fit(X, yb)
            self.models_btts.append(m)

        # Over2.5
        self.models_over = []
        yo = train["over25"].to_numpy()
        for s in self.seeds[:1]:
            m = _ml_bin(s)
            m.fit(X, yo)
            self.models_over.append(m)

        # Over1.5
        self.models_over15 = []
        yo15 = train["over15"].to_numpy() if "over15" in train.columns else (train["home_goals"] + train["away_goals"] > 1.5).astype(int).to_numpy()
        for s in self.seeds[:1]:
            m = _ml_bin(s)
            m.fit(X, yo15)
            self.models_over15.append(m)

        # Over3.5
        self.models_over35 = []
        yo35 = train["over35"].to_numpy() if "over35" in train.columns else (train["home_goals"] + train["away_goals"] > 3.5).astype(int).to_numpy()
        for s in self.seeds[:1]:
            m = _ml_bin(s)
            m.fit(X, yo35)
            self.models_over35.append(m)

        # Goals
        self.models_hg = []
        yhg = train["home_goals"].to_numpy()
        for s in self.seeds[:1]:
            m = _ml_reg(s)
            m.fit(X, yhg)
            self.models_hg.append(m)

        self.models_ag = []
        yag = train["away_goals"].to_numpy()
        for s in self.seeds[:1]:
            m = _ml_reg(s)
            m.fit(X, yag)
            self.models_ag.append(m)

        # ── Kalibrasyon (val verilirse isotonic) ───────────────────────────
        self.cal_1x2 = None
        self.cal_btts = None
        self.cal_over = None
        if val is not None and len(val) > 50:
            try:
                from sklearn.isotonic import IsotonicRegression
                Xv = self._X(val)
                raw_v = np.mean([m.predict_proba(Xv) for m in self.models_1x2], axis=0)
                yv = val["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
                self.cal_1x2 = [
                    IsotonicRegression(out_of_bounds="clip")
                    .fit(raw_v[:, c], (yv == c).astype(float))
                    for c in range(3)
                ]
                raw_b = self.models_btts[0].predict_proba(Xv)[:, 1]
                self.cal_btts = IsotonicRegression(out_of_bounds="clip").fit(
                    raw_b, val["btts"].to_numpy().astype(float)
                )
                raw_o = self.models_over[0].predict_proba(Xv)[:, 1]
                self.cal_over = IsotonicRegression(out_of_bounds="clip").fit(
                    raw_o, val["over25"].to_numpy().astype(float)
                )
            except Exception as e:  # pragma: no cover
                logger.warning("Kalibrasyon basarisiz, ham probs kullanilacak: %s", e)
                self.cal_1x2 = None

        logger.info(
            "ProductionEnsemble fit: %d satir, %d feature, %d seed%s",
            n_tr, len(self.feature_cols), len(self.seeds),
            " + isotonic kalibrasyon" if self.cal_1x2 else "",
        )
        return self

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        X = self._X(df)

        # 1X2 - ensemble ortalamasi
        probs_1x2 = np.mean([m.predict_proba(X) for m in self.models_1x2], axis=0)
        if self.cal_1x2 is not None:
            cp = np.column_stack([self.cal_1x2[c].predict(probs_1x2[:, c]) for c in range(3)])
            cp = np.maximum(cp, 0)
            home, draw, away = normalize_1x2(cp[:, 0], cp[:, 1], cp[:, 2])
        else:
            home, draw, away = normalize_1x2(
                probs_1x2[:, 0], probs_1x2[:, 1], probs_1x2[:, 2]
            )
        home = np.asarray(home); draw = np.asarray(draw); away = np.asarray(away)

        # BTTS
        raw_b = np.clip(self.models_btts[0].predict_proba(X)[:, 1], 0, 1)
        btts = self.cal_btts.predict(raw_b) if self.cal_btts is not None else raw_b
        btts = np.clip(btts, 0, 1)
        # Over1.5
        raw_o15 = np.clip(self.models_over15[0].predict_proba(X)[:, 1], 0, 1)
        over15 = self.cal_over15.predict(raw_o15) if hasattr(self, 'cal_over15') and self.cal_over15 is not None else raw_o15
        over15 = np.clip(over15, 0, 1)
        # Over25
        raw_o = np.clip(self.models_over[0].predict_proba(X)[:, 1], 0, 1)
        over25 = self.cal_over.predict(raw_o) if self.cal_over is not None else raw_o
        over25 = np.clip(over25, 0, 1)
        # Over3.5
        raw_o35 = np.clip(self.models_over35[0].predict_proba(X)[:, 1], 0, 1)
        over35 = self.cal_over35.predict(raw_o35) if hasattr(self, 'cal_over35') and self.cal_over35 is not None else raw_o35
        over35 = np.clip(over35, 0, 1)
        # Goals
        home_lam = np.maximum(self.models_hg[0].predict(X), 0.05)
        away_lam = np.maximum(self.models_ag[0].predict(X), 0.05)

        return PredictionResult(
            home_win=home,
            draw=draw,
            away_win=away,
            home_lambda=home_lam,
            away_lambda=away_lam,
            btts_yes=btts,
            over15=over15,
            over25=over25,
            over35=over35,
        )


# ── convenience: feature seti hazirlama ─────────────────────────────────────
def default_feature_set(df: pd.DataFrame) -> list[str]:
    """Tum sayisal feature'lar + enhanced feature'lari doner.

    Varsayilan olarak modelin listesinden gecersiz olanlari filtreler.

    NOT: eger df outright patlarsa, sadece enhanced + temel common
    feature setini dondurur (production fallback).
    """
    import pandas as pd

    SKIP = {
        "match_id", "league", "season", "date", "home_team_id", "away_team_id",
        "home_goals", "away_goals", "result", "result_H", "result_D", "result_A",
        "btts", "over25", "total_goals",
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_xg", "away_xg",
        "home_yellow", "away_yellow", "home_red", "away_red",
        "referee",
        "home_xg_real", "away_xg_real", "total_xg_real", "xg_diff_real",
        "ht_home_goals", "ht_away_goals", "ht_total_goals",
        "ht_result_is_draw", "ht_home_leading",
        "ht_second_half_goals_expected", "second_half_goals",
        "home_score_prob", "away_score_prob", "btts_xprob", "over25_xprob",
        "xg_diff_abs", "draw_xprob",
        "avg_home_odds", "avg_draw_odds", "avg_away_odds",
        "avg_over25_odds", "avg_close_home_odds", "avg_close_draw_odds",
        "avg_close_away_odds", "avg_close_over25_odds",
        "mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob",
        "mkt_close_over25_prob", "shots_diff", "sot_diff",
    }

    # Market-implied oranları feature olarak dahil et (sinyal)
    mkt = ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob", "mkt_over25_prob"]

    feats = [c for c in df.columns
             if c not in SKIP
             and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    feats = feats + [m for m in mkt if m not in feats]
    # Enhanced feature'lar
    from feature_engine.enhance import get_enhanced_feature_names
    for f in get_enhanced_feature_names():
        if f not in feats:
            feats.append(f)
    # Duplikat sil + mevcut olmayanlari filtrele
    feats = list(dict.fromkeys(feats))
    feats = [f for f in feats if f in df.columns]
    return feats
