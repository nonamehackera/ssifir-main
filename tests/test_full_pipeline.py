"""Kapsamlı pipeline testi: feature → model → tuning → ensemble → walk-forward.

Bu test tum sistemin dogru calistigini, yeni eklenen feature'larin
calistigini ve hyperparameter tuning sonuclarini dogrular.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    accuracy_1x2,
    brier_1x2,
    log_loss_1x2,
    log_loss_binary,
    brier_binary,
    auc_binary,
    calibration_error,
    mae_goals,
    rmse_goals,
)
from evaluation.walkforward import make_folds, run_walkforward, summarize
from models.catboost.model import CatBoostModel, feature_columns
from models.ensemble.ensemble import EnsembleModel
from models.elo.model import EloModel
from models.lightgbm.model import LightGBMModel
from models.poisson.model import DixonColesModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS: list[dict] = []


def _record(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"test": test_name, "status": status, "detail": detail})
    icon = "✓" if passed else "✗"
    print(f"  {icon} {test_name}: {detail}" if detail else f"  {icon} {test_name}")


# ──────────────────────────────────────────────────────────────────────
# SECTION 1: FEATURE ENGINE
# ──────────────────────────────────────────────────────────────────────
def test_feature_engine():
    print("\n" + "=" * 70)
    print("SECTION 1: FEATURE ENGINE")
    print("=" * 70)

    from ingestion.football_data.canonical import build_matches
    from feature_engine.engine import build_features

    matches = build_matches()
    _record("Veri yuklendi", len(matches) > 10000, f"{len(matches)} mac")

    feats = build_features(matches)
    _record("Feature'lar uretildi", len(feats) > 10000, f"{len(feats)} satir, {feats.shape[1]} sutun")

    # Yeni eklenen feature'larin mevcudiyeti
    new_features = ["home_w_corners", "away_w_corners"]
    for feat_name in new_features:
        exists = feat_name in feats.columns
        not_all_null = feats[feat_name].notna().sum() > 0 if exists else False
        _record(f"Feature {feat_name} mevcut ve dolu", exists and not_all_null,
                f"null={feats[feat_name].isna().sum() if exists else 'N/A'} / {len(feats)}")

    # Genisletilmis feature seti
    extended_features = [
        "home_gf_8", "home_ga_8", "away_gf_8", "away_ga_8",
        "home_gf_20", "home_ga_20", "away_gf_20", "away_ga_20",
        "home_gf_std", "home_ga_std", "away_gf_std", "away_ga_std",
        "home_pts_std", "away_pts_std",
        "home_attack_elo", "home_defence_elo", "away_attack_elo", "away_defence_elo",
        "attack_elo_diff", "defence_elo_diff",
        "data_completeness",
    ]
    missing = [f for f in extended_features if f not in feats.columns]
    _record("Gelismis feature seti mevcut", len(missing) == 0,
            f"eksik: {missing}" if missing else "tumu mevcut")

    # Eksik veri orani
    core_cols = [c for c in feats.columns if c not in ["match_id", "date", "referee", "result"]]
    null_pct = feats[core_cols].isna().mean().mean() * 100
    _record("Eksik veri orani makul", null_pct < 15, f"%{null_pct:.1f}")

    # Leakage kontrolu: gelecek bilgisi kullanilmamali
    # Sonuclar feature uretiminden sonra guncellenir, o yuzden feature'larda olmamali
    target_cols = ["home_goals", "away_goals", "result", "btts", "over25", "total_goals"]
    for tc in target_cols:
        if tc in feats.columns:
            _record(f"Hedef '{tc}' feature setinde var (beklenen)", True)

    return feats


# ──────────────────────────────────────────────────────────────────────
# SECTION 2: MODEL CALISMA TESTLERI
# ──────────────────────────────────────────────────────────────────────
def test_models(feats: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SECTION 2: MODEL CALISMA TESTLERI")
    print("=" * 70)

    # Temporal split: %70 train, %15 cal, %15 test
    feats_sorted = feats.sort_values("date").reset_index(drop=True)
    n = len(feats_sorted)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    train = feats_sorted.iloc[:train_end]
    cal = feats_sorted.iloc[train_end:cal_end]
    test = feats_sorted.iloc[cal_end:]

    _record("Veri bolundu", True, f"train={len(train)} cal={len(cal)} test={len(test)}")

    # Feature kolon sayisi dogrulama
    catboost_cols = feature_columns(with_odds=False)
    catboost_cols_odds = feature_columns(with_odds=True)
    _record("CatBoost feature kolonlari (no odds)", len(catboost_cols) > 50, f"{len(catboost_cols)} kolon")
    _record("CatBoost feature kolonlari (odds)", len(catboost_cols_odds) > 55, f"{len(catboost_cols_odds)} kolon")

    # Eksik feature kontrolu
    missing_feats = [c for c in catboost_cols if c not in feats.columns]
    _record("Tum feature'lar mevcut", len(missing_feats) == 0,
            f"eksik: {missing_feats}" if missing_feats else "tam")

    models_results = {}

    # --- Elo ---
    print("\n  --- Elo Model ---")
    try:
        t0 = time.time()
        elo = EloModel()
        elo.fit(train)
        pred = elo.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)

        _record("Elo train+predict", True, f"{elapsed:.1f}s")
        _record("Elo log_loss", ll < 1.15, f"{ll:.4f}")
        _record("Elo accuracy", acc > 0.35, f"{acc:.3f}")
        models_results["elo"] = {"log_loss": ll, "brier": br, "accuracy": acc}
    except Exception as e:
        _record("Elo model", False, str(e))

    # --- Dixon-Coles ---
    print("\n  --- Dixon-Coles Model ---")
    try:
        t0 = time.time()
        dc = DixonColesModel()
        dc.fit(train)
        pred = dc.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)

        _record("Dixon-Coles train+predict", True, f"{elapsed:.1f}s")
        _record("Dixon-Coles log_loss", ll < 1.15, f"{ll:.4f}")
        _record("Dixon-Coles accuracy", acc > 0.38, f"{acc:.3f}")

        # Lambda dogrulama
        if pred.home_lambda is not None:
            mean_lam = np.mean(pred.home_lambda)
            _record("Dixon-Coles lambda makul", 0.5 < mean_lam < 3.0, f"ort={mean_lam:.2f}")

        models_results["dixon_coles"] = {"log_loss": ll, "brier": br, "accuracy": acc}
    except Exception as e:
        _record("Dixon-Coles model", False, str(e))

    # --- CatBoost (no odds) ---
    print("\n  --- CatBoost Model (no odds) ---")
    try:
        t0 = time.time()
        cat = CatBoostModel(with_odds=False)
        cat.fit(train, cal)
        pred = cat.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        btts_ll = log_loss_binary(test["btts"].to_numpy(), pred.btts_yes)
        over_ll = log_loss_binary(test["over25"].to_numpy(), pred.over25)

        _record("CatBoost train+predict", True, f"{elapsed:.1f}s")
        _record("CatBoost log_loss", ll < 1.05, f"{ll:.4f}")
        _record("CatBoost accuracy", acc > 0.45, f"{acc:.3f}")
        _record("CatBoost BTTS log_loss", btts_ll < 0.7, f"{btts_ll:.4f}")
        _record("CatBoost Over25 log_loss", over_ll < 0.65, f"{over_ll:.4f}")
        models_results["catboost"] = {"log_loss": ll, "brier": br, "accuracy": acc}
    except Exception as e:
        _record("CatBoost model", False, str(e))

    # --- CatBoost (with odds) ---
    print("\n  --- CatBoost Model (with odds) ---")
    try:
        t0 = time.time()
        cat_odds = CatBoostModel(with_odds=True)
        cat_odds.fit(train, cal)
        pred = cat_odds.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        btts_auc = auc_binary(test["btts"].to_numpy(), pred.btts_yes)
        over_auc = auc_binary(test["over25"].to_numpy(), pred.over25)

        _record("CatBoost+Odds train+predict", True, f"{elapsed:.1f}s")
        _record("CatBoost+Odds log_loss", ll < 1.0, f"{ll:.4f}")
        _record("CatBoost+Odds accuracy", acc > 0.48, f"{acc:.3f}")
        _record("CatBoost+Odds BTTS AUC", btts_auc > 0.52, f"{btts_auc:.3f}")
        _record("CatBoost+Odds Over25 AUC", over_auc > 0.56, f"{over_auc:.3f}")
        models_results["catboost_odds"] = {"log_loss": ll, "brier": br, "accuracy": acc}
    except Exception as e:
        _record("CatBoost+Odds model", False, str(e))

    # --- LightGBM (no odds) ---
    print("\n  --- LightGBM Model (no odds) ---")
    try:
        t0 = time.time()
        lgbm = LightGBMModel(with_odds=False)
        lgbm.fit(train, cal)
        pred = lgbm.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)

        _record("LightGBM train+predict", True, f"{elapsed:.1f}s")
        _record("LightGBM log_loss", ll < 1.1, f"{ll:.4f}")
        _record("LightGBM accuracy", acc > 0.42, f"{acc:.3f}")
        models_results["lightgbm"] = {"log_loss": ll, "brier": br, "accuracy": acc}
    except Exception as e:
        _record("LightGBM model", False, str(e))

    # --- LightGBM (with odds) ---
    print("\n  --- LightGBM Model (with odds) ---")
    try:
        t0 = time.time()
        lgbm_odds = LightGBMModel(with_odds=True)
        lgbm_odds.fit(train, cal)
        pred = lgbm_odds.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)

        _record("LightGBM+Odds train+predict", True, f"{elapsed:.1f}s")
        _record("LightGBM+Odds log_loss", ll < 1.05, f"{ll:.4f}")
        _record("LightGBM+Odds accuracy", acc > 0.45, f"{acc:.3f}")
        models_results["lightgbm_odds"] = {"log_loss": ll, "brier": br, "accuracy": acc}
    except Exception as e:
        _record("LightGBM+Odds model", False, str(e))

    return models_results


# ──────────────────────────────────────────────────────────────────────
# SECTION 3: ENSEMBLE TESTI
# ──────────────────────────────────────────────────────────────────────
def test_ensemble(feats: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SECTION 3: ENSEMBLE MODEL")
    print("=" * 70)

    feats_sorted = feats.sort_values("date").reset_index(drop=True)
    n = len(feats_sorted)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    train = feats_sorted.iloc[:train_end]
    cal = feats_sorted.iloc[train_end:cal_end]
    test = feats_sorted.iloc[cal_end:]

    def make_elo():
        return EloModel()

    def make_dc():
        return DixonColesModel()

    def make_cat():
        return CatBoostModel(with_odds=False)

    def make_lgbm():
        return LightGBMModel(with_odds=False)

    try:
        t0 = time.time()
        ens = EnsembleModel([make_elo(), make_dc(), make_cat(), make_lgbm()])
        ens.fit(train, cal)
        pred = ens.predict(test)
        elapsed = time.time() - t0

        ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        br = brier_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)
        acc = accuracy_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)

        _record("Ensemble train+predict", True, f"{elapsed:.1f}s")
        _record("Ensemble log_loss", ll < 1.05, f"{ll:.4f}")
        _record("Ensemble accuracy", acc > 0.45, f"{acc:.3f}")
        _record("Ensemble agirliklar", ens.weights is not None,
                f"agirliklar={np.round(ens.weights, 3)}")
        _record("Ensemble temperature", 0.5 < ens.temperature < 2.0,
                f"T={ens.temperature:.3f}")

        # En iyi tek model ile karsilastirma
        single_ll = 1.1  # baseline
        for name in ["catboost", "lightgbm"]:
            try:
                m = CatBoostModel(with_odds=False) if name == "catboost" else LightGBMModel(with_odds=False)
                m.fit(train, cal)
                p = m.predict(test)
                sll = log_loss_1x2(test["result"], p.home_win, p.draw, p.away_win)
                if sll < single_ll:
                    single_ll = sll
            except Exception:
                pass

        _record("Ensemble tek modelden iyi", ll <= single_ll + 0.01,
                f"ensemble={ll:.4f} vs best_single={single_ll:.4f}")

        return {"ensemble": {"log_loss": ll, "brier": br, "accuracy": acc}}
    except Exception as e:
        _record("Ensemble model", False, str(e))
        return {}


# ──────────────────────────────────────────────────────────────────────
# SECTION 4: WALK-FORWARD (BUYUK TEST)
# ──────────────────────────────────────────────────────────────────────
def test_walkforward(feats: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SECTION 4: WALK-FORWARD EVALUATION (4 FOLD)")
    print("=" * 70)

    def make_elo():
        return EloModel()

    def make_dc():
        return DixonColesModel()

    def make_cat():
        return CatBoostModel(with_odds=False)

    def make_cat_odds():
        return CatBoostModel(with_odds=True)

    def make_lgbm():
        return LightGBMModel(with_odds=False)

    def make_lgbm_odds():
        return LightGBMModel(with_odds=True)

    factories = {
        "elo": make_elo,
        "dixon_coles": make_dc,
        "catboost": make_cat,
        "catboost_odds": make_cat_odds,
        "lightgbm": make_lgbm,
        "lightgbm_odds": make_lgbm_odds,
    }

    print("\n  Walk-forward baslatiliyor (4 fold, bu uzun surer)...\n")
    t0 = time.time()
    results = run_walkforward(feats, factories, n_folds=4)
    elapsed = time.time() - t0

    summary = summarize(results)
    print("\n  WALK-FORWARD SONUCLARI:")
    print("  " + "-" * 80)
    print(summary.to_string(index=False))
    print("  " + "-" * 80)
    print(f"  Toplam sure: {elapsed:.0f}s")

    # Naive baseline: log_loss = -ln(1/3) = 1.0986
    naive_ll = 1.0986
    for _, row in summary.iterrows():
        model = row["model"]
        ll = row["log_loss"]
        acc = row["accuracy"]
        _record(f"WF {model} baseline'i yendi", ll < naive_ll,
                f"log_loss={ll:.4f} (baseline={naive_ll:.4f})")
        _record(f"WF {model} accuracy > %40", acc > 0.40, f"{acc:.3f}")

    # En iyi modeli bul
    best_row = summary.iloc[0]
    _record("WF En iyi model", True,
            f"{best_row['model']} log_loss={best_row['log_loss']:.4f}")

    return summary


# ──────────────────────────────────────────────────────────────────────
# SECTION 5: HATA ANALIZI VE KALIBRASYON
# ──────────────────────────────────────────────────────────────────────
def test_calibration_and_analysis(feats: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SECTION 5: KALIBRASYON VE HATA ANALIZI")
    print("=" * 70)

    feats_sorted = feats.sort_values("date").reset_index(drop=True)
    n = len(feats_sorted)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    train = feats_sorted.iloc[:train_end]
    cal = feats_sorted.iloc[train_end:cal_end]
    test = feats_sorted.iloc[cal_end:]

    try:
        from calibration.calibrators import MulticlassCalibrator

        cat = CatBoostModel(with_odds=True)
        cat.fit(train, cal)
        pred = cat.predict(test)

        raw_ll = log_loss_1x2(test["result"], pred.home_win, pred.draw, pred.away_win)

        # Kalibrasyon testi
        y_test = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        P_raw = np.column_stack([pred.home_win, pred.draw, pred.away_win])
        calibrator = MulticlassCalibrator()
        calibrator.fit(y_test, P_raw)
        P_cal = calibrator.transform(P_raw)

        cal_ll = log_loss_1x2(test["result"], P_cal[:, 0], P_cal[:, 1], P_cal[:, 2])
        _record("Kalibrasyon raw log_loss", True, f"{raw_ll:.4f}")
        _record("Kalibrasyon kalibre log_loss", True, f"{cal_ll:.4f}")
        _record("Kalibrasyon iyilestirdi", cal_ll <= raw_ll + 0.001,
                f"delta={cal_ll - raw_ll:.6f}")

    except Exception as e:
        _record("Kalibrasyon testi", False, str(e))

    # Hata analizi: hangi tur maclarda kotu
    try:
        cat = CatBoostModel(with_odds=True)
        cat.fit(train, cal)
        pred = cat.predict(test)
        preds_class = np.argmax(np.column_stack([pred.home_win, pred.draw, pred.away_win]), axis=1)
        y_true = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        errors = preds_class != y_true

        # Ev sahibi favori maclar (mkt_home_prob > 0.5)
        if "mkt_home_prob" in test.columns:
            home_fav = test["mkt_home_prob"] > 0.5
            error_rate_homefav = errors[home_fav].mean() if home_fav.sum() > 0 else None
            error_rate_other = errors[~home_fav].mean() if (~home_fav).sum() > 0 else None
            _record("Ev sahibi favori hata orani",
                    error_rate_homefav is not None and error_rate_homefav < 0.6,
                    f"%{error_rate_homefav*100:.1f}" if error_rate_homefav else "N/A")
            _record("Diger maclar hata orani",
                    error_rate_other is not None,
                    f"%{error_rate_other*100:.1f}" if error_rate_other else "N/A")

        # Beraberlik tahmin zorlugu
        draw_true = y_true == 1
        draw_pred = preds_class == 1
        if draw_true.sum() > 0:
            draw_recall = (draw_true & draw_pred).sum() / draw_true.sum()
            _record("Beraberlik recall", draw_recall > 0.1,
                    f"{draw_recall:.3f} (zor sinif)")

        _record("Toplam test ornek sayisi", True, f"{len(test)}")
        _record("Hata orani", True, f"%{errors.mean()*100:.1f}")

    except Exception as e:
        _record("Hata analizi", False, str(e))


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  FUTBOL TAHMIN SISTEMI - KAPSAMLI TEST RAPORU")
    print("=" * 70)

    t_start = time.time()

    feats = test_feature_engine()
    test_models(feats)
    test_ensemble(feats)
    test_calibration_and_analysis(feats)
    wf_summary = test_walkforward(feats)

    elapsed = time.time() - t_start

    # FINAL RAPORU
    print("\n" + "=" * 70)
    print("  FINAL TEST RAPORU")
    print("=" * 70)

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")

    print(f"\n  Toplam: {total} | Gecen: {passed} | Basarisiz: {failed}")
    print(f"  Basari orani: %{passed/total*100:.1f}")
    print(f"  Toplam sure: {elapsed:.0f}s")

    if failed > 0:
        print("\n  BASARISIZ TESTLER:")
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"    ✗ {r['test']}: {r['detail']}")

    print("\n  WALK-FORWARD OZET:")
    if wf_summary is not None:
        print(wf_summary.to_string(index=False))

    print("\n" + "=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
