"""Feedback Loop Örnek Kullanım Scripti.

Bu script, geri besleme döngüsünün nasıl kullanılacağını gösterir.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from feedback.integration import get_feedback


def example_usage():
    """Geri besleme döngüsü örnek kullanımı."""
    print("=" * 60)
    print("  FEEDBACK LOOP - ÖRNEK KULLANIM")
    print("=" * 60)

    # Geri besleme entegrasyonunu al
    feedback = get_feedback()

    # Örnek tahminler kaydet
    print("\n1. Tahminler kaydediliyor...")

    # Doğru tahminler
    feedback.record_prediction(
        match_id="match_001",
        home_team="Galatasaray",
        away_team="Fenerbahce",
        league="Super Lig",
        pred_home=0.45,
        pred_draw=0.25,
        pred_away=0.30,
        actual_result="H",
        home_goals=2,
        away_goals=1,
    )

    # Yanlış tahmin
    feedback.record_prediction(
        match_id="match_002",
        home_team="Barcelona",
        away_team="Real Madrid",
        league="La Liga",
        pred_home=0.50,
        pred_draw=0.25,
        pred_away=0.25,
        actual_result="A",  # Yanlış tahmin
        home_goals=1,
        away_goals=3,
    )

    # Başka bir yanlış tahmin
    feedback.record_prediction(
        match_id="match_003",
        home_team="Man City",
        away_team="Liverpool",
        league="Premier League",
        pred_home=0.60,
        pred_draw=0.20,
        pred_away=0.20,
        actual_result="D",  # Yanlış tahmin
        home_goals=1,
        away_goals=1,
    )

    print("   3 tahmin kaydedildi.")

    # Geri besleme döngüsü çalıştır
    print("\n2. Geri besleme döngüsü çalıştırılıyor...")
    result = feedback.run_feedback_cycle()
    print(f"   Döngü #{result['cycle_id']} tamamlandı.")
    print(f"   Tespit edilen kalıp: {result['patterns_detected']}")
    print(f"   Düzeltme uygulandı: {result['correction_applied']}")

    # Takım düzeltme önerileri
    print("\n3. Takım düzeltme önerileri...")
    galatasaray_features = feedback.get_team_correction("Galatasaray")
    print(f"   Galatasaray hata oranı: {galatasaray_features.get('Galatasaray_error_rate', 'N/A')}")

    # Rapor oluştur
    print("\n4. Tam rapor oluşturuluyor...")
    report = feedback.get_report()
    print(report)

    # Ayarlanmış tahmin örneği
    print("\n5. Ayarlanmış tahmin örneği...")
    raw_prediction = {
        "home_win": 0.55,
        "draw": 0.25,
        "away_win": 0.20,
    }

    adjusted = feedback.get_adjusted_prediction(
        home_team="Galatasaray",
        away_team="Fenerbahce",
        league="Super Lig",
        raw_prediction=raw_prediction,
    )

    print(f"   Ham tahmin: {raw_prediction}")
    print(f"   Ayarlanmış: home={adjusted['home_win']:.3f}, draw={adjusted['draw']:.3f}, away={adjusted['away_win']:.3f}")
    print(f"   Güven faktörü: {adjusted['confidence_factor']:.3f}")


if __name__ == "__main__":
    example_usage()
