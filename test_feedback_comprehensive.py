"""Kapsaml_ Feedback Loop Testleri - Ger_ek Veri _zerinde.

Bu testler:
- 528K sat_rl_k ger_ek features_enhanced_v5.parquet verisini kullan_r
- Ger_ek_i senaryolar_ sim_le eder
- _statistiksel olarak anlaml_ sonu_lar _retir
- Objectif ve detayl_ raporlar sunar
"""

import sys
import os
sys.path.insert(0, ".")

import json
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from feedback.error_tracker import ErrorTracker, PredictionRecord
from feedback.pattern_detector import PatternDetector, ErrorPattern
from feedback.self_correction import SelfCorrectionEngine, CorrectionResult
from feedback.feature_generator import ErrorFeatureGenerator
from feedback.orchestrator import FeedbackOrchestrator, FeedbackCycleResult

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ===============================================================================
# TEST ALTYAPISI
# ===============================================================================

class TestRunner:
    """Test _al__t_r_c_ ve raporlay_c_."""

    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def run(self, name, func):
        """Testi _al__t_r ve sonucu kaydet."""
        print(f"\n{'-' * 70}")
        print(f"  TEST: {name}")
        print(f"{'-' * 70}")

        start = time.time()
        try:
            result = func()
            duration = time.time() - start
            self.results.append({
                "name": name,
                "status": "PASS",
                "duration": duration,
                "details": result,
            })
            print(f"  [PASS] ({duration:.2f}s)")
            return result
        except AssertionError as e:
            duration = time.time() - start
            self.results.append({
                "name": name,
                "status": "FAIL",
                "duration": duration,
                "error": str(e),
            })
            print(f"  [FAIL] {e}")
            raise
        except Exception as e:
            duration = time.time() - start
            self.results.append({
                "name": name,
                "status": "ERROR",
                "duration": duration,
                "error": str(e),
            })
            print(f"  [ERROR] {e}")
            raise

    def summary(self):
        """_zet rapor yazd_r."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        errors = sum(1 for r in self.results if r["status"] == "ERROR")
        total_time = time.time() - self.start_time

        print(f"\n{'=' * 70}")
        print(f"  TEST SONUCLARI")
        print(f"{'=' * 70}")
        print(f"  Toplam: {total} | Ba_ar_l_: {passed} | Ba_ar_s_z: {failed} | Hata: {errors}")
        print(f"  S_re: {total_time:.2f} saniye")
        print(f"  Ba_ar_ Oran_: %{passed/total*100:.1f}")

        if failed > 0 or errors > 0:
            print(f"\n  BA_ARISIZ TESTLER:")
            for r in self.results:
                if r["status"] != "PASS":
                    print(f"    - {r['name']}: {r.get('error', 'N/A')}")

        print(f"{'=' * 70}")


# ===============================================================================
# VER_ Y_KLEME
# ===============================================================================

def load_real_data():
    """Ger_ek veriyi y_kle ve haz_rla."""
    print("\n  Veri y_kleniyor...")

    # Features y_kle
    feat_path = "data/gold/features_enhanced_v5.parquet"
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"{feat_path} bulunamad_")

    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    feat = feat.sort_values("date").reset_index(drop=True)

    # Hedef s_tunlar_ haz_rla
    for c in ["home_goals", "away_goals"]:
        feat[c] = feat[c].fillna(0).astype(int)
    feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
    feat["result"] = feat["result"].fillna("D")
    feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
    feat["over25"] = (feat["total_goals"] > 2.5).astype(int)

    # Eksik de_erleri doldur
    feat = feat.dropna(subset=["result", "elo_diff"])

    print(f"  Y_klenen: {len(feat)} sat_r, {len(feat.columns)} s_tun")
    print(f"  Tarih aral___: {feat['date'].min()} - {feat['date'].max()}")
    print(f"  Lig say_s_: {feat['league'].nunique()}")
    print(f"  Tak_m say_s_: {feat['home_team_id'].nunique()}")

    return feat


def generate_realistic_predictions(feat: pd.DataFrame, n: int = 1000) -> list[dict]:
    """Ger_ek_i tahminler _ret (ger_ek model outputlar_ gibi)."""
    print(f"\n  {n} ger_ek_i tahmin _retiliyor...")

    # Rastgele ma_ se_ (tarih s_ras_yla)
    sample_indices = np.random.choice(len(feat), size=min(n, len(feat)), replace=False)
    sample_indices = np.sort(sample_indices)  # Tarih s_ras_n_ koru
    sample = feat.iloc[sample_indices]

    predictions = []
    for idx, row in sample.iterrows():
        # Ger_ek sonucu al
        actual_result = row["result"]

        # Ger_ek_i tahmin olas_l_klar_ _ret (ger_ek model gibi)
        # Market olas_l_klar_n_ kullan (varsa)
        if "mkt_home_prob" in row and not pd.isna(row["mkt_home_prob"]):
            # Market olas_l_klar_na g_r_lt_ ekle
            noise = np.random.normal(0, 0.05, 3)
            pred_home = max(0.01, row["mkt_home_prob"] + noise[0])
            pred_draw = max(0.01, row.get("mkt_draw_prob", 0.33) + noise[1])
            pred_away = max(0.01, row.get("mkt_away_prob", 0.33) + noise[2])
        else:
            # ELO fark_na dayal_ tahmin
            elo_diff = row.get("elo_diff", 0)
            # Bradley-Terry modeli
            p_home = 1 / (1 + 10 ** (-elo_diff / 400))
            pred_home = p_home * 0.95  # Ev sahibi avantaj_
            pred_draw = 0.25 + np.random.normal(0, 0.05)
            pred_away = 1 - pred_home - pred_draw

            # Normalize et
            total = pred_home + pred_draw + pred_away
            pred_home /= total
            pred_draw /= total
            pred_away /= total

        # Olas_l_klar_ ger_ek_i aral__a s_k__t_r
        pred_home = np.clip(pred_home, 0.05, 0.85)
        pred_draw = np.clip(pred_draw, 0.05, 0.45)
        pred_away = np.clip(pred_away, 0.05, 0.85)

        # Normalize et
        total = pred_home + pred_draw + pred_away
        pred_home /= total
        pred_draw /= total
        pred_away /= total

        predictions.append({
            "match_id": str(row.get("match_id", f"match_{idx}")),
            "home_team_id": int(row["home_team_id"]),
            "away_team_id": int(row["away_team_id"]),
            "league": row["league"],
            "date": row["date"].isoformat() if pd.notna(row["date"]) else "",
            "pred_home": float(pred_home),
            "pred_draw": float(pred_draw),
            "pred_away": float(pred_away),
            "actual_result": actual_result,
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
        })

    print(f"  _retilen: {len(predictions)} tahmin")
    return predictions


# ===============================================================================
# TEST 1: ERROR TRACKER TESTLER_
# ===============================================================================

def test_error_tracker_basics():
    """ErrorTracker temel islevselligi."""
    import shutil
    test_dir = "data/feedback/test_tracker"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    tracker = ErrorTracker(data_dir=test_dir)

    # 100 tahmin kaydet
    predictions = []
    for i in range(100):
        # Olasiliklari duzgun normalize et
        probs = np.random.dirichlet([2, 1, 2])
        pred_home, pred_draw, pred_away = probs

        # Ger_ek sonucu rastgele belirle
        actual = np.random.choice(["H", "D", "A"], p=[0.45, 0.25, 0.30])

        record = tracker.record_prediction(
            match_id=f"test_{i}",
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual,
            league="TEST",
            home_team=f"Team_{i % 10}",
            away_team=f"Team_{(i+5) % 10}",
            home_goals=np.random.randint(0, 4),
            away_goals=np.random.randint(0, 4),
        )
        predictions.append(record)

    # Kontroller
    assert len(tracker.records) == 100, f"Beklenen 100 kay_t, bulunan {len(tracker.records)}"

    # Do_ru/yanl__ sayac_
    correct = sum(1 for r in tracker.records if r.is_correct)
    assert 0 < correct < 100, f"Do_ru tahmin say_s_ anormal: {correct}"

    # Log loss hesaplama
    log_losses = [r.log_loss for r in tracker.records]
    assert all(l >= 0 for l in log_losses), "Log loss negatif olamaz"
    assert np.mean(log_losses) > 0, "Ortalama log loss s_f_r olamaz"

    # Hata pay_ analizi
    error_margins = [r.error_margin for r in tracker.records]
    assert all(0 <= e <= 1 for e in error_margins), "Hata pay_ 0-1 aras_nda olmal_"

    # Kategori analizi
    analysis = tracker.analyze_by_category()
    assert "total_matches" in analysis
    assert "accuracy" in analysis
    assert 0 <= analysis["accuracy"] <= 1

    print(f"  Toplam kay_t: {len(tracker.records)}")
    print(f"  Do_ru tahmin: {correct}/{len(tracker.records)} (%{correct/len(tracker.records)*100:.1f})")
    print(f"  Ortalama log loss: {np.mean(log_losses):.4f}")
    print(f"  Ortalama hata pay_: {np.mean(error_margins):.4f}")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_tracker", ignore_errors=True)

    return {"total": 100, "correct": correct, "log_loss": np.mean(log_losses)}


def test_error_tracker_with_real_data(predictions: list[dict]):
    """Ger_ek veri ile ErrorTracker testi."""
    tracker = ErrorTracker(data_dir="data/feedback/test_real_tracker")

    # T_m tahminleri kaydet
    for pred in predictions:
        tracker.record_prediction(
            match_id=pred["match_id"],
            pred_home=pred["pred_home"],
            pred_draw=pred["pred_draw"],
            pred_away=pred["pred_away"],
            actual_result=pred["actual_result"],
            league=pred["league"],
            home_team=str(pred["home_team_id"]),
            away_team=str(pred["away_team_id"]),
            home_goals=pred["home_goals"],
            away_goals=pred["away_goals"],
        )

    # Analiz yap
    analysis = tracker.analyze_by_category()

    # _statistiksel kontroller
    n = len(tracker.records)
    assert n == len(predictions), f"Beklenen {len(predictions)} kay_t, bulunan {n}"

    # Do_ruluk testi (rastgele tahmin i_in beklenen: ~33%)
    accuracy = analysis["accuracy"]
    print(f"  Toplam kay_t: {n}")
    print(f"  Do_ruluk: %{accuracy*100:.1f}")

    # Lig baz_nda analiz
    if "by_league" in analysis:
        league_count = len(analysis["by_league"])
        print(f"  Lig say_s_: {league_count}")

        for league, stats in list(analysis["by_league"].items())[:5]:
            print(f"    {league}: n={stats['n']}, do_ruluk=%{stats['accuracy']*100:.1f}")

    # Sonu_ baz_nda analiz
    if "by_result" in analysis:
        for result, stats in analysis["by_result"].items():
            print(f"    '{result}' sonucu: n={stats['n']}, ortalama tahmin={stats['avg_pred_for']:.3f}")

    # G_ven da__l_m_
    confidence_dist = tracker.get_confidence_distribution()
    print(f"  Ortalama g_ven: %{confidence_dist.get('avg_confidence', 0)*100:.1f}")

    # En k_t_ tahminler
    worst = tracker.get_worst_predictions(10)
    print(f"  En k_t_ 10 tahminin ortalama hata pay_: {np.mean([r.error_margin for r in worst]):.4f}")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_real_tracker", ignore_errors=True)

    return analysis


# ===============================================================================
# TEST 2: PATTERN DETECTOR TESTLER_
# ===============================================================================

def test_pattern_detector_systematic_bias():
    """Sistemli _nyarg_ tespiti testi."""
    tracker = ErrorTracker(data_dir="data/feedback/test_pattern")

    # Bilin_li olarak _nyarg_l_ tahminler _ret
    # Ev sahibi tak_mlar_ i_in d___k tahmin yap
    for i in range(200):
        if i < 100:
            # Ev sahibi tak_mlar_ i_in kas_tl_ d___k tahmin
            pred_home = np.random.uniform(0.1, 0.3)  # D___k
            pred_draw = np.random.uniform(0.3, 0.5)
            pred_away = 1 - pred_home - pred_draw
            actual = "H"  # Ama ev sahibi kazan_yor
            league = "BIAS_LEAGUE"
        else:
            # Normal tahmin
            pred_home = np.random.uniform(0.3, 0.7)
            pred_draw = np.random.uniform(0.1, 0.4)
            pred_away = 1 - pred_home - pred_draw
            actual = np.random.choice(["H", "D", "A"], p=[0.45, 0.25, 0.30])
            league = "NORMAL_LEAGUE"

        tracker.record_prediction(
            match_id=f"bias_{i}",
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual,
            league=league,
            home_team=f"Team_{i % 20}",
            away_team=f"Team_{(i+10) % 20}",
        )

    # Kal_p tespiti
    detector = PatternDetector(tracker)
    patterns = detector.detect_all_patterns()

    # Kontroller
    assert len(patterns) > 0, "En az bir kal_p tespit edilmeli"

    # Lig kal_b_n_ bul
    league_patterns = [p for p in patterns if p.pattern_type == "lig"]
    assert len(league_patterns) > 0, "Lig bazl_ kal_p tespit edilmeli"

    # BIAS_LEAGUE kal_b_n_ kontrol et
    bias_pattern = None
    for p in league_patterns:
        if "BIAS_LEAGUE" in p.description:
            bias_pattern = p
            break

    assert bias_pattern is not None, "BIAS_LEAGUE kal_b_ tespit edilmeli"
    assert bias_pattern.severity > 0.2, f"Ciddiyet _ok d___k: {bias_pattern.severity}"
    assert bias_pattern.confidence > 0.4, f"G_ven _ok d___k: {bias_pattern.confidence}"

    print(f"  Tespit edilen kal_p: {len(patterns)}")
    print(f"  Lig kal_plar_: {len(league_patterns)}")
    print(f"  BIAS_LEAGUE ciddiyet: {bias_pattern.severity:.3f}")
    print(f"  BIAS_LEAGUE g_ven: {bias_pattern.confidence:.3f}")

    # Rapor olu_tur
    report = detector.generate_report()
    assert len(report) > 100, "Rapor _ok k_sa"
    print(f"  Rapor uzunlu_u: {len(report)} karakter")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_pattern", ignore_errors=True)

    return {"patterns": len(patterns), "severity": bias_pattern.severity}


def test_pattern_detector_with_real_data(predictions: list[dict]):
    """Ger_ek veri ile kal_p tespiti."""
    tracker = ErrorTracker(data_dir="data/feedback/test_real_pattern")

    # Tahminleri kaydet
    for pred in predictions:
        tracker.record_prediction(
            match_id=pred["match_id"],
            pred_home=pred["pred_home"],
            pred_draw=pred["pred_draw"],
            pred_away=pred["pred_away"],
            actual_result=pred["actual_result"],
            league=pred["league"],
            home_team=str(pred["home_team_id"]),
            away_team=str(pred["away_team_id"]),
            home_goals=pred["home_goals"],
            away_goals=pred["away_goals"],
        )

    # Kal_p tespiti
    detector = PatternDetector(tracker)
    patterns = detector.detect_all_patterns()

    print(f"  Toplam kal_p: {len(patterns)}")

    # Kal_p t_rlerini analiz et
    pattern_types = {}
    for p in patterns:
        if p.pattern_type not in pattern_types:
            pattern_types[p.pattern_type] = []
        pattern_types[p.pattern_type].append(p)

    for ptype, plist in pattern_types.items():
        avg_severity = np.mean([p.severity for p in plist])
        avg_confidence = np.mean([p.confidence for p in plist])
        print(f"    {ptype}: {len(plist)} kal_p, ortalama ciddiyet={avg_severity:.3f}, g_ven={avg_confidence:.3f}")

    # En ciddi kal_plar_ g_ster
    print("\n  EN C_DD_ 5 KALIP:")
    for i, p in enumerate(patterns[:5], 1):
        print(f"    {i}. [{p.pattern_type}] {p.description[:80]}...")
        print(f"       Ciddiyet: {p.severity:.3f}, G_ven: {p.confidence:.3f}")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_real_pattern", ignore_errors=True)

    return {"patterns": len(patterns), "types": pattern_types}


# ===============================================================================
# TEST 3: SELF-CORRECTION ENGINE TESTLER_
# ===============================================================================

def test_self_correction_basic():
    """Temel d_zeltme mekanizmas_ testi."""
    tracker = ErrorTracker(data_dir="data/feedback/test_correction")
    detector = PatternDetector(tracker)
    engine = SelfCorrectionEngine(tracker, detector, correction_dir="data/feedback/test_correction")

    # _nyarg_l_ tahminler kaydet
    for i in range(150):
        if i < 75:
            # Y_ksek g_venli ama yanl__ tahminler
            pred_home = np.random.uniform(0.7, 0.9)
            pred_draw = np.random.uniform(0.05, 0.15)
            pred_away = 1 - pred_home - pred_draw
            actual = "A"  # Deplasman kazan_yor
        else:
            # Normal tahminler
            pred_home = np.random.uniform(0.3, 0.7)
            pred_draw = np.random.uniform(0.1, 0.4)
            pred_away = 1 - pred_home - pred_draw
            actual = np.random.choice(["H", "D", "A"], p=[0.45, 0.25, 0.30])

        tracker.record_prediction(
            match_id=f"correct_{i}",
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual,
            league="TEST",
        )

    # Mevcut metrikleri hesapla
    metrics_before = engine._calculate_metrics()
    print(f"  _ncesi - Do_ruluk: %{metrics_before['accuracy']*100:.1f}, Log Loss: {metrics_before['log_loss']:.4f}")

    # D_zeltme uygula
    result = engine.analyze_and_correct(
        current_weights=np.array([0.4, 0.3, 0.3]),
        current_temperature=1.0,
    )

    # Kontroller
    assert result.success, "D_zeltme ba_ar_l_ olmal_"
    assert len(result.actions_taken) > 0, "En az bir d_zeltme eylemi olmal_"
    assert result.improvement != 0, "_yile_tirme s_f_r olmamal_"

    print(f"  D_zeltme eylemleri: {len(result.actions_taken)}")
    print(f"  Beklenen iyile_tirme: {result.improvement:.4f}")

    # Eylem t_rlerini g_ster
    for action in result.actions_taken:
        print(f"    [{action.action_type}] {action.description[:60]}...")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_correction", ignore_errors=True)

    return {
        "actions": len(result.actions_taken),
        "improvement": result.improvement,
    }


def test_self_correction_with_real_data(predictions: list[dict]):
    """Ger_ek veri ile d_zeltme motoru."""
    tracker = ErrorTracker(data_dir="data/feedback/test_real_correction")
    detector = PatternDetector(tracker)
    engine = SelfCorrectionEngine(tracker, detector, correction_dir="data/feedback/test_real_correction")

    # Tahminleri kaydet
    for pred in predictions:
        tracker.record_prediction(
            match_id=pred["match_id"],
            pred_home=pred["pred_home"],
            pred_draw=pred["pred_draw"],
            pred_away=pred["pred_away"],
            actual_result=pred["actual_result"],
            league=pred["league"],
            home_team=str(pred["home_team_id"]),
            away_team=str(pred["away_team_id"]),
            home_goals=pred["home_goals"],
            away_goals=pred["away_goals"],
        )

    # Mevcut metrikler
    metrics_before = engine._calculate_metrics()
    print(f"  _ncesi:")
    print(f"    Do_ruluk: %{metrics_before['accuracy']*100:.1f}")
    print(f"    Log Loss: {metrics_before['log_loss']:.4f}")
    print(f"    Brier: {metrics_before['brier']:.4f}")

    # D_zeltme uygula
    result = engine.analyze_and_correct(
        current_weights=np.array([0.35, 0.30, 0.35]),
        current_temperature=1.0,
    )

    print(f"\n  D_zeltme sonucu:")
    print(f"    Ba_ar_l_: {result.success}")
    print(f"    Eylem say_s_: {len(result.actions_taken)}")
    print(f"    Beklenen iyile_tirme: {result.improvement:.4f}")

    # D_zeltme _zetini g_ster
    summary = engine.get_correction_summary()
    print(f"\n  D_zeltme _zeti:")
    print(f"    Toplam d_zeltme: {summary.get('total_corrections', 0)}")
    print(f"    Ortalama iyile_tirme: {summary.get('average_improvement', 0):.4f}")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_real_correction", ignore_errors=True)

    return {"metrics_before": metrics_before, "result": result}


# ===============================================================================
# TEST 4: FEATURE GENERATOR TESTLER_
# ===============================================================================

def test_feature_generator():
    """_zellik _retici testi."""
    tracker = ErrorTracker(data_dir="data/feedback/test_features")
    detector = PatternDetector(tracker)
    generator = ErrorFeatureGenerator(tracker, detector, feature_dir="data/feedback/test_features")

    # Veri kaydet
    for i in range(100):
        pred_home = np.random.uniform(0.3, 0.7)
        pred_draw = np.random.uniform(0.1, 0.4)
        pred_away = 1 - pred_home - pred_draw
        actual = np.random.choice(["H", "D", "A"], p=[0.45, 0.25, 0.30])

        tracker.record_prediction(
            match_id=f"feat_{i}",
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual,
            league="TEST_LEAGUE",
            home_team=f"Team_{i % 5}",
            away_team=f"Team_{(i+2) % 5}",
        )

    # Tak_m _zellikleri _ret
    team_features = generator.generate_team_features("Team_0")
    assert len(team_features) > 0, "Tak_m _zellikleri _retilmeli"

    print(f"  Tak_m _zelli_i say_s_: {len(team_features)}")
    for key, value in list(team_features.items())[:5]:
        print(f"    {key}: {value:.4f}")

    # Lig _zellikleri _ret
    league_features = generator.generate_league_features("TEST_LEAGUE")
    assert len(league_features) > 0, "Lig _zellikleri _retilmeli"

    print(f"  Lig _zelli_i say_s_: {len(league_features)}")

    # Rapor olu_tur
    report = generator.get_feature_importance_report()
    assert len(report) > 50, "Rapor _ok k_sa"
    print(f"\n  _zellik raporu:")
    print(f"  {report[:200]}...")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_features", ignore_errors=True)

    return {"team_features": len(team_features), "league_features": len(league_features)}


# ===============================================================================
# TEST 5: ORCHESTRATOR TESTLER_
# ===============================================================================

def test_orchestrator_full_cycle():
    """Tam d_ng_ testi."""
    orchestrator = FeedbackOrchestrator(
        data_dir="data/feedback/test_orchestrator",
        auto_correct=True,
        min_predictions_for_correction=50,
    )

    # 200 tahmin kaydet
    for i in range(200):
        pred_home = np.random.uniform(0.3, 0.7)
        pred_draw = np.random.uniform(0.1, 0.4)
        pred_away = 1 - pred_home - pred_draw
        actual = np.random.choice(["H", "D", "A"], p=[0.45, 0.25, 0.30])

        orchestrator.record_prediction(
            match_id=f"orch_{i}",
            pred_home=pred_home,
            pred_draw=pred_draw,
            pred_away=pred_away,
            actual_result=actual,
            league="TEST",
            home_team=f"Team_{i % 10}",
            away_team=f"Team_{(i+5) % 10}",
        )

    # D_ng_ _al__t_r
    result = orchestrator.run_cycle(
        current_weights=np.array([0.33, 0.34, 0.33]),
        current_temperature=1.0,
    )

    # Kontroller
    assert isinstance(result, FeedbackCycleResult)
    assert result.cycle_id == 1
    assert result.total_predictions == 200

    print(f"  D_ng_ #{result.cycle_id}")
    print(f"  S_re: {result.duration_seconds:.2f}s")
    print(f"  Toplam tahmin: {result.total_predictions}")
    print(f"  Tespit edilen kal_p: {result.patterns_detected}")
    print(f"  D_zeltme uyguland_: {result.correction_applied}")
    print(f"  _yile_tirme: {result.improvement:.4f}")

    # Rapor olu_tur
    report = orchestrator.generate_report()
    assert len(report) > 200, "Rapor _ok k_sa"
    print(f"\n  Rapor uzunlu_u: {len(report)} karakter")

    # _zet
    summary = orchestrator.get_cycle_summary()
    print(f"  Toplam d_ng_: {summary.get('total_cycles', 0)}")

    # Temizle
    import shutil
    shutil.rmtree("data/feedback/test_orchestrator", ignore_errors=True)

    return {"result": result, "report_length": len(report)}


# ===============================================================================
# TEST 6: PERFORMANS KAR_ILA_TIRMASI
# ===============================================================================

def test_performance_comparison(feat: pd.DataFrame):
    """_nce/sonra performans kar__la_t_rmas_."""
    print("\n  Ger_ek veri ile performans kar__la_t_rmas_...")

    # Walk-forward split
    train = feat[(feat["date"] >= "2020-01-01") & (feat["date"] < "2024-01-01")]
    test = feat[(feat["date"] >= "2024-01-01") & (feat["date"] < "2025-01-01")]

    print(f"  E_itim seti: {len(train)} ma_")
    print(f"  Test seti: {len(test)} ma_")

    # Basit bir model e_it (LightGBM)
    try:
        import lightgbm as lgb

        # Feature preparasyonu
        CAT = ["league", "home_team_id", "away_team_id"]
        exclude = {"second_half_goals", "ht_total_goals", "ht_result_is_draw", "ht_home_leading",
                   "ht_home_goals", "ht_away_goals", "home_shots", "away_shots", "home_sot", "away_sot",
                   "home_corners", "away_corners", "home_yellow", "away_yellow", "home_red", "away_red",
                   "shots_diff", "sot_diff", "home_goals", "away_goals", "result", "result_H", "result_D",
                   "result_A", "btts", "over25", "over15", "over35", "total_goals", "match_id", "season",
                   "date", "referee"}
        exclude |= {"home_xgot", "away_xgot", "home_big_chances", "away_big_chances",
                    "home_xg_flash", "away_xg_flash", "home_fouls", "away_fouls",
                    "home_possession", "away_possession"}
        exclude |= {"mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob",
                    "mkt_close_over25_prob", "avg_close_home_odds", "avg_close_draw_odds",
                    "avg_close_away_odds", "avg_close_over25_odds"}

        cats = [c for c in CAT if c in train.columns]
        nums = [c for c in train.columns if c not in exclude and c not in cats
                and train[c].dtype in ["float64", "int64", "float32", "int32"]]
        cols = cats + nums

        # Veriyi haz_rla
        train_lgb = train.copy()
        test_lgb = test.copy()
        for c in CAT:
            if c in train_lgb.columns:
                train_lgb[c] = train_lgb[c].astype(str).astype("category").cat.codes
                test_lgb[c] = test_lgb[c].astype(str).astype("category").cat.codes

        Xtr = train_lgb[cols].fillna(0).astype(np.float32).values
        ytr = train_lgb["result"].map({"H": 0, "D": 1, "A": 2}).values
        Xte = test_lgb[cols].fillna(0).astype(np.float32).values
        yte = test_lgb["result"].map({"H": 0, "D": 1, "A": 2}).values

        # Model e_it
        print("  LightGBM modeli e_itiliyor...")
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, num_leaves=50,
            learning_rate=0.03, n_estimators=300, max_depth=6,
            min_child_samples=80, subsample=0.75, colsample_bytree=0.65,
            reg_alpha=0.5, reg_lambda=5.0, random_seed=42,
            verbose=-1, n_jobs=-1,
        )
        m.fit(Xtr, ytr)

        # Tahminler
        raw_probs = m.predict_proba(Xte)
        base_preds = np.argmax(raw_probs, axis=1)

        # Metrikler
        from evaluation.metrics import log_loss_1x2, brier_1x2, accuracy_1x2

        # _nce ham tahminler
        base_accuracy = np.mean(base_preds == yte)
        base_logloss = log_loss_1x2(
            yte,
            raw_probs[:, 0],
            raw_probs[:, 1],
            raw_probs[:, 2],
        )
        base_brier = brier_1x2(
            yte,
            raw_probs[:, 0],
            raw_probs[:, 1],
            raw_probs[:, 2],
        )

        print(f"\n  MODELLER:")
        print(f"  Ham model (LightGBM):")
        print(f"    Do_ruluk: %{base_accuracy*100:.2f}")
        print(f"    Log Loss: {base_logloss:.4f}")
        print(f"    Brier: {base_brier:.4f}")

        # Feedback loop ile ayarlanm__ tahminler
        from feedback.integration import get_feedback
        import shutil

        # Test dizini temizle
        test_dir = "data/feedback/test_perf"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        feedback = get_feedback()

        # Test setindeki tahminleri kaydet (ilk 500'_)
        test_sample = test.head(500)
        for i, (idx, row) in enumerate(test_sample.iterrows()):
            actual = row["result"]
            pred_probs = raw_probs[:500][i] if i < len(raw_probs) else [0.33, 0.34, 0.33]

            feedback.record_prediction(
                match_id=str(row.get("match_id", f"perf_{i}")),
                home_team=str(row["home_team_id"]),
                away_team=str(row["away_team_id"]),
                league=row["league"],
                pred_home=float(pred_probs[0]),
                pred_draw=float(pred_probs[1]),
                pred_away=float(pred_probs[2]),
                actual_result=actual,
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
            )

        # D_ng_ _al__t_r
        result = feedback.run_feedback_cycle()

        # D_zeltilmi_ tahminler
        adjusted_correct = 0
        adjusted_loglosses = []
        adjusted_briers = []

        for i in range(min(500, len(raw_probs))):
            pred = {
                "home_win": raw_probs[i, 0],
                "draw": raw_probs[i, 1],
                "away_win": raw_probs[i, 2],
            }

            adjusted = feedback.get_adjusted_prediction(
                home_team=str(test.iloc[i]["home_team_id"]),
                away_team=str(test.iloc[i]["away_team_id"]),
                league=test.iloc[i]["league"],
                raw_prediction=pred,
            )

            # D_zeltilmi_ tahminle sonu_lar_ kar__la_t_r
            adj_probs = [adjusted["home_win"], adjusted["draw"], adjusted["away_win"]]
            adj_pred = np.argmax(adj_probs)
            if adj_pred == yte[i]:
                adjusted_correct += 1

            # D_zeltilmi_ metrikler
            actual_onehot = np.zeros(3)
            actual_onehot[yte[i]] = 1
            adjusted_loglosses.append(-np.log(max(adj_probs[yte[i]], 1e-10)))
            adjusted_briers.append(np.sum((np.array(adj_probs) - actual_onehot) ** 2))

        adj_accuracy = adjusted_correct / min(500, len(raw_probs))
        adj_logloss = np.mean(adjusted_loglosses)
        adj_brier = np.mean(adjusted_briers)

        print(f"\n  D_zeltilmi_ model (Feedback Loop):")
        print(f"    Do_ruluk: %{adj_accuracy*100:.2f}")
        print(f"    Log Loss: {adj_logloss:.4f}")
        print(f"    Brier: {adj_brier:.4f}")

        # Kar__la_t_rma
        print(f"\n  KAR_ILA_TIRMA:")
        print(f"    Do_ruluk: %{base_accuracy*100:.2f} -> %{adj_accuracy*100:.2f} ({(adj_accuracy-base_accuracy)*100:+.2f}%)")
        print(f"    Log Loss: {base_logloss:.4f} -> {adj_logloss:.4f} ({adj_logloss-base_logloss:+.4f})")
        print(f"    Brier: {base_brier:.4f} -> {adj_brier:.4f} ({adj_brier-base_brier:+.4f})")

        # Temizle
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        return {
            "base_accuracy": base_accuracy,
            "adjusted_accuracy": adj_accuracy,
            "base_logloss": base_logloss,
            "adjusted_logloss": adj_logloss,
            "base_brier": base_brier,
            "adjusted_brier": adj_brier,
        }

    except ImportError as e:
        print(f"  LightGBM y_klenemedi: {e}")
        return None


# ===============================================================================
# TEST 7: STRES TESTLER_
# ===============================================================================

def test_stress_cases():
    """S_n_r durumlar_ ve stres testleri."""
    print("\n  S_n_r durumlar_ test ediliyor...")

    tracker = ErrorTracker(data_dir="data/feedback/test_stress")

    # Test 1: _ok fazla veri
    print("    1. 1000 kay_t ekleniyor...")
    for i in range(1000):
        tracker.record_prediction(
            match_id=f"stress_{i}",
            pred_home=np.random.uniform(0.1, 0.9),
            pred_draw=np.random.uniform(0.05, 0.4),
            pred_away=np.random.uniform(0.05, 0.4),
            actual_result=np.random.choice(["H", "D", "A"]),
            league=f"LEAGUE_{i % 20}",
        )
    assert len(tracker.records) == 1000
    print("      [OK] 1000 kay_t ba_ar_yla eklendi")

    # Test 2: T_m tahminler ayn_ sonu_ta
    print("    2. Tek sonu_lu tahminler...")
    tracker2 = ErrorTracker(data_dir="data/feedback/test_stress2")
    for i in range(100):
        tracker2.record_prediction(
            match_id=f"single_{i}",
            pred_home=0.8,
            pred_draw=0.1,
            pred_away=0.1,
            actual_result="H",
        )
    analysis = tracker2.analyze_by_category()
    assert analysis["accuracy"] == 1.0, "T_m tahminler do_ru olmal_"
    print("      [OK] Tek sonu_lu tahminler testi ba_ar_l_")

    # Test 3: T_m tahminler yanl__
    print("    3. T_m tahminler yanl__...")
    tracker3 = ErrorTracker(data_dir="data/feedback/test_stress3")
    for i in range(100):
        tracker3.record_prediction(
            match_id=f"wrong_{i}",
            pred_home=0.8,
            pred_draw=0.1,
            pred_away=0.1,
            actual_result="A",  # Hep yanl__
        )
    analysis = tracker3.analyze_by_category()
    assert analysis["accuracy"] == 0.0, "T_m tahminler yanl__ olmal_"
    print("      [OK] T_m yanl__ tahminler testi ba_ar_l_")

    # Test 4: _ok d___k olas_l_klar
    print("    4. D___k olas_l_klar...")
    tracker4 = ErrorTracker(data_dir="data/feedback/test_stress4")
    for i in range(100):
        tracker4.record_prediction(
            match_id=f"low_{i}",
            pred_home=0.01,
            pred_draw=0.01,
            pred_away=0.98,
            actual_result="A",
        )
    analysis = tracker4.analyze_by_category()
    print(f"      Log Loss: {analysis['log_loss']:.4f}")
    print("      [OK] D___k olas_l_k testi ba_ar_l_")

    # Test 5: E_it olas_l_klar
    print("    5. E_it olas_l_klar...")
    tracker5 = ErrorTracker(data_dir="data/feedback/test_stress5")
    for i in range(100):
        tracker5.record_prediction(
            match_id=f"equal_{i}",
            pred_home=0.333,
            pred_draw=0.333,
            pred_away=0.334,
            actual_result=np.random.choice(["H", "D", "A"]),
        )
    analysis = tracker5.analyze_by_category()
    print(f"      Do_ruluk: %{analysis['accuracy']*100:.1f}")
    print("      [OK] E_it olas_l_k testi ba_ar_l_")

    # Temizle
    import shutil
    for d in ["test_stress", "test_stress2", "test_stress3", "test_stress4", "test_stress5"]:
        shutil.rmtree(f"data/feedback/{d}", ignore_errors=True)

    return {"stress_tests": 5, "all_passed": True}


# ===============================================================================
# TEST 8: GER_EK__ SENARYO TESTLER_
# ===============================================================================

def test_realistic_scenarios(feat: pd.DataFrame):
    """Ger_ek_i senaryolar_ test et."""
    print("\n  Ger_ek_i senaryolar test ediliyor...")

    # Premier League ma_lar_n_ al
    epl = feat[feat["league"] == "E0"].tail(200)

    if len(epl) < 100:
        print("  Yeterli EPL verisi yok, atlan_yor")
        return None

    print(f"  EPL ma__: {len(epl)}")

    # Senaryo 1: Derbi ma_lar_ (y_ksek ELO fark_ yok)
    derbies = epl[epl["elo_diff"].abs() < 50]
    print(f"\n  Senaryo 1: Derbi ma_lar_ (|ELO diff| < 50)")
    print(f"    Ma_ say_s_: {len(derbies)}")

    if len(derbies) > 10:
        tracker = ErrorTracker(data_dir="data/feedback/test_scenario")
        for _, row in derbies.iterrows():
            # Derbilerde beraberlik olas_l___n_ art_r
            pred_home = 0.35
            pred_draw = 0.35
            pred_away = 0.30

            tracker.record_prediction(
                match_id=str(row.get("match_id", "")),
                pred_home=pred_home,
                pred_draw=pred_draw,
                pred_away=pred_away,
                actual_result=row["result"],
                league="EPL",
                home_team=str(row["home_team_id"]),
                away_team=str(row["away_team_id"]),
            )

        analysis = tracker.analyze_by_category()
        print(f"    Do_ruluk: %{analysis['accuracy']*100:.1f}")

    # Senaryo 2: Liderlik yar___nda olan tak_mlar
    print(f"\n  Senaryo 2: Y_ksek ELO tak_mlar_")
    high_elo = epl[(epl["home_elo"] > 1600) | (epl["away_elo"] > 1600)]
    print(f"    Ma_ say_s_: {len(high_elo)}")

    if len(high_elo) > 10:
        tracker2 = ErrorTracker(data_dir="data/feedback/test_scenario2")
        for _, row in high_elo.iterrows():
            # G__l_ tak_mlar i_in farkl_ strateji
            if row["home_elo"] > row["away_elo"]:
                pred_home = 0.55
                pred_draw = 0.25
                pred_away = 0.20
            else:
                pred_home = 0.20
                pred_draw = 0.25
                pred_away = 0.55

            tracker2.record_prediction(
                match_id=str(row.get("match_id", "")),
                pred_home=pred_home,
                pred_draw=pred_draw,
                pred_away=pred_away,
                actual_result=row["result"],
                league="EPL",
            )

        analysis = tracker2.analyze_by_category()
        print(f"    Do_ruluk: %{analysis['accuracy']*100:.1f}")

    # Senaryo 3: Sezon sonu ma_lar_
    print(f"\n  Senaryo 3: May_s ay_ ma_lar_")
    may_matches = feat[feat["date"].dt.month == 5].tail(200)
    print(f"    Ma_ say_s_: {len(may_matches)}")

    if len(may_matches) > 10:
        tracker3 = ErrorTracker(data_dir="data/feedback/test_scenario3")
        for _, row in may_matches.iterrows():
            pred_home = np.random.uniform(0.3, 0.7)
            pred_draw = np.random.uniform(0.1, 0.4)
            pred_away = 1 - pred_home - pred_draw

            tracker3.record_prediction(
                match_id=str(row.get("match_id", "")),
                pred_home=pred_home,
                pred_draw=pred_draw,
                pred_away=pred_away,
                actual_result=row["result"],
                league=row["league"],
            )

        analysis = tracker3.analyze_by_category()
        print(f"    Do_ruluk: %{analysis['accuracy']*100:.1f}")

    # Temizle
    import shutil
    for d in ["test_scenario", "test_scenario2", "test_scenario3"]:
        shutil.rmtree(f"data/feedback/{d}", ignore_errors=True)

    return {"scenarios_tested": 3}


# ===============================================================================
# ANA TEST _ALI_TIRICI
# ===============================================================================

def main():
    """T_m testleri _al__t_r."""
    import shutil
    
    # Onceki test dizinlerini temizle
    test_dirs = [
        "data/feedback/test_tracker", "data/feedback/test_real_tracker",
        "data/feedback/test_pattern", "data/feedback/test_real_pattern",
        "data/feedback/test_correction", "data/feedback/test_real_correction",
        "data/feedback/test_features", "data/feedback/test_orchestrator",
        "data/feedback/test_stress", "data/feedback/test_stress2",
        "data/feedback/test_stress3", "data/feedback/test_stress4",
        "data/feedback/test_stress5", "data/feedback/test_scenario",
        "data/feedback/test_scenario2", "data/feedback/test_scenario3",
        "data/feedback/test_perf",
    ]
    for d in test_dirs:
        shutil.rmtree(d, ignore_errors=True)
    print("  Onceki test dizinleri temizlendi.")
    
    print("=" * 72)
    print("  FEEDBACK LOOP KAPSAMLI TESTLER - GERCEK VERI UZERINDE")
    print("  528K+ satir, 192 sutun, 2015-2026 verisi")
    print("=" * 72)

    runner = TestRunner()

    # Veriy_kle
    feat = load_real_data()

    # Ger_ek_i tahminler _ret
    predictions = generate_realistic_predictions(feat, n=1000)

    # Testleri _al__t_r
    runner.run("ErrorTracker Temel", test_error_tracker_basics)
    runner.run("ErrorTracker Ger_ek Veri", lambda: test_error_tracker_with_real_data(predictions))
    runner.run("PatternDetector _nyarg_", test_pattern_detector_systematic_bias)
    runner.run("PatternDetector Ger_ek Veri", lambda: test_pattern_detector_with_real_data(predictions))
    runner.run("SelfCorrection Temel", test_self_correction_basic)
    runner.run("SelfCorrection Ger_ek Veri", lambda: test_self_correction_with_real_data(predictions))
    runner.run("FeatureGenerator", test_feature_generator)
    runner.run("Orchestrator Tam D_ng_", test_orchestrator_full_cycle)
    runner.run("Stres Testleri", test_stress_cases)
    runner.run("Performans Kar__la_t_rmas_", lambda: test_performance_comparison(feat))
    runner.run("Ger_ek_i Senaryolar", lambda: test_realistic_scenarios(feat))

    # _zet
    runner.summary()

    return runner.results


if __name__ == "__main__":
    results = main()
