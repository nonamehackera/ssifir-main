"""Production prediction pipeline (ROADMAP bolum 52, 35, 36, 37, 53).

Tek bir API:
    predictor.predict(features_row) -> ROADMAP 52 formatinda JSON

Ciktida:
  - 1X2 olasiliklari (Home/Draw/Away)
  - gol bekentileri (home_lambda, away_lambda, expected_total)
  - BTTS yes/no
  - Over 2.5
  - en yuksek exact score dagilimi (top N)
  - reliability / confidence bilgisi (ROADMAP 53: ucuk "93%" degil,
    olculmus degerler)

Bu sinif hem tek model (orn. sadece catboost) hem de ensemble'i
calistirip ROADMAP 52 cikti formatina donusturur. Tum olasiliklar
gelecek bilgisi icermez (feature engine zaten point-in-time garantili).
"""

from __future__ import annotations

import json
import logging
import numpy as np
from datetime import datetime, timezone

from models.base import PredictionResult

logger = logging.getLogger(__name__)


def _to_float(x) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _top_scores(score_matrix: list | None, n: int = 5) -> list[dict]:
    """Score matrix [(h, a, p), ...] -> ROADMAP 52 top_scores formati."""
    if not score_matrix:
        return []
    rows = score_matrix[0] if isinstance(score_matrix[0], list) else score_matrix
    out = []
    for h, a, p in rows[:n]:
        out.append({"score": f"{int(h)}-{int(a)}", "probability": round(_to_float(p), 4)})
    return out


class PredictionPipeline:
    """Model(ler)i sarar, ROADMAP 52 JSON ciktisi uretir.

    models: tek bir model VEYA liste (ensemble icin disaridan verilir).
    Tek model verilirse dogrudan onun ciktiı kullanilir.
    """

    def __init__(self, model, model_version: str = "unknown", dataset_version: str = "unknown") -> None:
        self.model = model
        self.model_version = model_version
        self.dataset_version = dataset_version

    def _predict_result(self, feats) -> PredictionResult:
        return self.model.predict(feats)

    def predict(self, feats, prediction_time: datetime | None = None, fixture_id=None) -> dict:
        """feats: tek satirlik feature DataFrame (veya 1 satir iceren DataFrame).
        Doner: ROADMAP 52 formatinda dict (JSON-serializable).
        """
        pred = self._predict_result(feats)
        home = np.asarray(pred.home_win, float)
        draw = np.asarray(pred.draw, float)
        away = np.asarray(pred.away_win, float)
        home_lam = np.asarray(pred.home_lambda, float) if pred.home_lambda is not None else np.array([np.nan])
        away_lam = np.asarray(pred.away_lambda, float) if pred.away_lambda is not None else np.array([np.nan])
        btts = np.asarray(pred.btts_yes, float) if pred.btts_yes is not None else np.array([np.nan])
        over15 = np.asarray(pred.over15, float) if pred.over15 is not None else np.array([np.nan])
        over25 = np.asarray(pred.over25, float) if pred.over25 is not None else np.array([np.nan])
        over35 = np.asarray(pred.over35, float) if pred.over35 is not None else np.array([np.nan])

        # probabilities to 1 (1X2)
        s = home + draw + away
        s = np.where(s <= 0, 1.0, s)
        home, draw, away = home / s, draw / s, away / s

        expected_total = _to_float(home_lam[0]) + _to_float(away_lam[0])

        out = {
            "fixture_id": int(fixture_id) if fixture_id is not None else None,
            "prediction_time": (prediction_time or datetime.now(timezone.utc)).isoformat(),
            "result": {
                "home": round(_to_float(home[0]), 3),
                "draw": round(_to_float(draw[0]), 3),
                "away": round(_to_float(away[0]), 3),
            },
            "goals": {
                "home_lambda": round(_to_float(home_lam[0]), 3),
                "away_lambda": round(_to_float(away_lam[0]), 3),
                "expected_total": round(expected_total, 3),
            },
            "btts": {
                "yes": round(_to_float(btts[0]), 3),
                "no": round(1.0 - _to_float(btts[0]), 3),
            },
            "over_under_1_5": {
                "over": round(_to_float(over15[0]), 3),
                "under": round(1.0 - _to_float(over15[0]), 3),
            },
            "over_under_2_5": {
                "over": round(_to_float(over25[0]), 3),
                "under": round(1.0 - _to_float(over25[0]), 3),
            },
            "over_under_3_5": {
                "over": round(_to_float(over35[0]), 3),
                "under": round(1.0 - _to_float(over35[0]), 3),
            },
            "top_scores": (_top_scores(pred.score_matrix, 3)
                          if pred.score_matrix else
                          self._score_matrix(_to_float(home_lam[0]), _to_float(away_lam[0]))),
            # ROADMAP 53: guven skoru ucuk bir sayi DEGIL,
            # olculmus sinyallerin ozeti
            "reliability": self._reliability(feats, home, draw, away),
        }
        return out

    @staticmethod
    def _score_matrix(home_lam: float, away_lam: float, max_g: int = 6) -> list[dict]:
        """Lambda'lardan basit (bagimsiz) Poisson score matrisi (ROADMAP 37).

        DC modeli score_matrix vermediginde (CatBoost/LightGBM gibi) exact
        score dagilimi buradan uretilir. Baglanti (rho) terimi icermez ama
        beklenen gol dagilimini makul sekilde yansitir.
        """
        from scipy.stats import poisson

        gh = np.arange(max_g + 1)
        ph = poisson.pmf(gh, home_lam)
        pa = poisson.pmf(gh, away_lam)
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
            h = int(idx % (max_g + 1))
            a = int(idx // (max_g + 1))
            out.append({"score": f"{h}-{a}", "probability": round(float(flat[idx]), 4)})
        return out

    @staticmethod
    def _reliability(feats, home, draw, away) -> dict:
        """ROADMAP 53: confidence = 93% gibi uydurma sayi uretme.
        Bunun yerine olculmus sinyaller."""
        try:
            dc = float(feats["data_completeness"].iloc[0])
        except Exception:
            dc = None
        # model agreement: en yuksek olasiligin digerlerinden ne kadar ayri
        probs = np.array([_to_float(home[0]), _to_float(draw[0]), _to_float(away[0])])
        spread = float(probs.max() - probs.min())
        return {
            "data_completeness": round(dc, 3) if dc is not None else None,
            "model_agreement_spread": round(spread, 3),
            "calibration": "n/a (pipeline duzeyinde olculmez; model bazli bakilmalı)",
            "note": "confidence tek bir yuzde DEGIL; yukaridaki sinyaller gercektir.",
        }


def predict_to_json(payload: dict, indent: int = 2) -> str:
    """Pipeline ciktisini JSON string'e cevirir."""
    return json.dumps(payload, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from ingestion.football_data.canonical import build_matches
    from feature_engine.engine import build_features
    from models.catboost.model import CatBoostModel

    m = build_matches()
    f = build_features(m)
    cut = int(len(f) * 0.8)
    train, test = f.iloc[:cut].copy(), f.iloc[cut:].reset_index(drop=True)

    model = CatBoostModel(with_odds=True, iterations=400, verbose=False,
                         draw_weight=4.0, time_decay=1.0).fit(train)
    pipe = PredictionPipeline(model, model_version="catboost_odds_v2", dataset_version="fd_2122_2425")
    # son maci tahmin et (gercek sonucu gormeden onceki feature)
    row = test.iloc[[-1]]
    out = pipe.predict(row, fixture_id=row["match_id"].iloc[0])
    print(predict_to_json(out))
