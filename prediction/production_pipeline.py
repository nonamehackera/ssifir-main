"""Production Prediction Pipeline (teslim edilebilir API).

Tek mac icin girdi: build_features / feature_engine tarafindan uretilmis
pre-match feature satiri. Bu modul:
  1. Eksik yeni feature'lari `feature_engine.enhance.augment_row` ile ekler
     (production'da build_features her seyi uretmemis olabilir).
  2. `ProductionEnsemble` ile 4 marketi tahmin eder.
  3. Lig guvenilirligine gore "playable" (oynanabilir) etiketi koyar.
  4. ROADMAP 52 uyumlu JSON uretir (PredictionPipeline ile ayni format).

Kullanim (tek mac):
    from prediction.production_pipeline import ProductionPredictor

    p = ProductionPredictor(model_version="production_v3")
    p.prepare(training_df, calibration_df)   # fit + kalibrasyon
    out = p.predict_single(feature_row)      # -> dict (JSON)
    print(json.dumps(out, ensure_ascii=False, indent=2))

Istediginiz guven esigi (playable icin) `min_confidence` parametresi ile
ayarlanir; varsayilan olarak gecmis testlerden olculen %60-65 eşigini
kullanir (burada %60 default, curklu deger degil).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from models.base import PredictionResult
from models.production_ensemble import ProductionEnsemble, default_feature_set
from feature_engine.enhance import augment_row, CONFIDENT_LEAGUE_MATCHES

logger = logging.getLogger(__name__)

# Olculmus gercek esikler (blind testlerden; uydurma degil):
#   buyuk/orta lig + market orani olan -> ~%55 Acc, LL~0.96
#   esik@60% tüm veri -> ~%69 @65% -> ~%73 @70% -> ~%77
# Playable etiketi icin kullanilan varsayilan guven esigi:
DEFAULT_PLAYABLE_CONF = 0.60
# Az verili (<1000 mac) bir ligde, tahmin "oynanabilir" sayilmaz (kupa/amator)
DEFAULT_MIN_LEAGUE_MATCHES = 1000


class ProductionPredictor:
    """ProductionEnsemble + enhance + playable etiketi (tek mac API'si)."""

    name = "production_predictor"

    def __init__(
        self,
        model_version: str = "production_v3",
        dataset_version: str = "unknown",
        playable_conf: float = DEFAULT_PLAYABLE_CONF,
        min_league_matches: int | None = DEFAULT_MIN_LEAGUE_MATCHES,
        league_match_counts: dict | None = None,
    ) -> None:
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.playable_conf = playable_conf
        self.min_league_matches = min_league_matches
        # {lig: mac sayisi} - fit() sirasinda otomatik cikarilir; disaridan
        # da verilebilir (canli/eksik veri senaryolari icin).
        self.league_match_counts = league_match_counts or {}
        self._ensemble: ProductionEnsemble | None = None
        self._feature_cols: list[str] = []

    # ── hazirlik ────────────────────────────────────────────────────────────
    def prepare(
        self,
        train: pd.DataFrame,
        cal: pd.DataFrame | None = None,
        league_match_counts: dict | None = None,
    ) -> "ProductionPredictor":
        """Fit + kalibrasyon + lig mac sayilari (train uzerinden)."""
        if league_match_counts is not None:
            self.league_match_counts = league_match_counts
        elif "league" in train.columns:
            self.league_match_counts = train["league"].value_counts().to_dict()

        feats_all = default_feature_set(train)
        feats_all = [f for f in feats_all if f in train.columns]

        self._feature_cols = feats_all
        self._ensemble = ProductionEnsemble(base_features=feats_all)
        self._ensemble.fit(train, cal)
        logger.info("ProductionPredictor hazir: %d feature, %d lig", len(feats_all), len(self.league_match_counts))
        return self

    # ── tahmin ──────────────────────────────────────────────────────────────
    def _row_to_df(self, row) -> pd.DataFrame:
        if isinstance(row, pd.Series):
            df = row.to_frame().T.copy()
        elif isinstance(row, pd.DataFrame):
            df = row.iloc[[0]].copy()
        else:
            df = pd.DataFrame([dict(row)])
        return df

    def predict_single(self, row, fixture_id=None, prediction_time=None) -> dict:
        """Tek mac feature satirini tahmin edip JSON-uyumlu dict doner.

        Girdi: pre-match feature satiri (build_features ya da pipeline
        ciktisi). Yeni feature'lar eksikse augment_row ile eklenir.
        """
        if self._ensemble is None:
            raise RuntimeError("once .prepare(...) cagirilmali")

        df = self._row_to_df(row)
        # Yeni feature'lari guvenceye al
        df = augment_row(df.iloc[0], self.league_match_counts).to_frame().T

        # Tahmin icin gerekli feature sutunlarini sec (eksikleri 0 doldur)
        X = pd.DataFrame(index=df.index)
        for c in self._feature_cols:
            if c in df.columns:
                X[c] = df[c]
            else:
                X[c] = 0.0
        X = X.astype("float64")

        pred = self._ensemble.predict(X)
        h = float(pred.home_win[0]); d = float(pred.draw[0]); a = float(pred.away_win[0])
        hl = float(pred.home_lambda[0]); al = float(pred.away_lambda[0])
        b = float(pred.btts_yes[0]); o = float(pred.over25[0])

        # Guven & oynanabilirlik
        league = None
        if "league" in df.columns:
            league = str(df["league"].iloc[0]) or None
        confidence = float(max(h, d, a))
        has_mkt = all(
            c in df.columns and pd.notna(df[c].iloc[0])
            for c in ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]
        )
        n_league = self.league_match_counts.get(league) if league else None
        low_data = (
            self.min_league_matches is not None
            and n_league is not None
            and n_league < self.min_league_matches
        )

        playable = confidence >= self.playable_conf and not low_data
        playable_reasons = []
        skip_reasons = []
        if confidence < self.playable_conf:
            skip_reasons.append(f"guven {confidence:.0%} < {self.playable_conf:.0%}")
        if low_data:
            skip_reasons.append(f"az verili lig ({n_league} mac)")
        if confidence >= self.playable_conf:
            playable_reasons.append("yuksek model guveni")
        if n_league is not None and n_league >= self.min_league_matches:
            playable_reasons.append(f"yeterli lig verisi ({n_league} mac)")

        expected_total = hl + al
        favorite = np.argmax([h, d, a])

        return {
            "fixture_id": int(fixture_id) if fixture_id is not None else None,
            "prediction_time": (prediction_time or datetime.now(timezone.utc)).isoformat(),
            "league": league,
            "model": self.name,
            "model_version": self.model_version,
            "result": {
                "home": round(h, 3),
                "draw": round(d, 3),
                "away": round(a, 3),
                "favorite": ["home", "draw", "away"][favorite],
            },
            "goals": {
                "home_lambda": round(hl, 3),
                "away_lambda": round(al, 3),
                "expected_total": round(expected_total, 3),
            },
            "btts": {"yes": round(b, 3), "no": round(1.0 - b, 3)},
            "over_under_2_5": {"over": round(o, 3), "under": round(1.0 - o, 3)},
            "top_scores": self._top_scores_X(hl, al),
            "reliability": {
                "confidence": round(confidence, 3),
                "data_completeness": self._dc(df),
                "league_matches": n_league,
                "has_market_odds": bool(has_mkt),
                "playable_refusal": (
                    "skip" if not playable else "play"
                ),
                "playable": bool(playable),
            },
            "policy": {
                "playable": bool(playable),
                "reasons_play": playable_reasons,
                "reasons_skip": skip_reasons,
            },
        }

    @staticmethod
    def _dc(df: pd.DataFrame):
        if "data_completeness" in df.columns and pd.notna(df["data_completeness"].iloc[0]):
            return round(float(df["data_completeness"].iloc[0]), 3)
        return None

    @staticmethod
    def _top_scores_X(home_lam: float, away_lam: float, max_g: int = 6) -> list[dict]:
        from scipy.stats import poisson

        gh = np.arange(max_g + 1)
        ph = poisson.pmf(gh, home_lam); pa = poisson.pmf(gh, away_lam)
        mat = ph[:, None] * pa[None, :]
        s = mat.sum()
        if s <= 0:
            return []
        mat = mat / s
        g1, g2 = np.meshgrid(gh, gh, indexing="ij")
        flat = mat.reshape(-1)
        order = np.argsort(flat)[::-1][:3]
        out = []
        for idx in order:
            h = int(idx % (max_g + 1)); a = int(idx // (max_g + 1))
            out.append({"score": f"{h}-{a}", "probability": round(float(flat[idx]), 4)})
        return out


def predict_to_json(payload: dict, indent: int = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent)
