"""FeedbackAwarePredictor gercek veri testi."""
import sys, os, shutil
sys.path.insert(0, ".")
from feedback.feedback_predictor import FeedbackAwarePredictor

test_dir = "data/feedback/test_feedback_real"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

fp = FeedbackAwarePredictor(data_dir=test_dir)

print("=== FEEDBACK AWARE PREDICTOR TEST ===")
print()

predictions = [
    (81, 82, "Super Lig", 0.45, 0.25, 0.30, 50, "H"),
    (81, 83, "Super Lig", 0.50, 0.25, 0.25, 100, "H"),
    (82, 83, "Super Lig", 0.40, 0.30, 0.30, 10, "D"),
    (1, 2, "EPL", 0.55, 0.25, 0.20, 150, "H"),
    (1, 3, "EPL", 0.60, 0.20, 0.20, 200, "A"),
    (2, 3, "EPL", 0.35, 0.30, 0.35, 5, "D"),
    (4, 5, "La Liga", 0.70, 0.20, 0.10, 300, "H"),
    (4, 6, "La Liga", 0.65, 0.25, 0.10, 250, "A"),
    (5, 6, "La Liga", 0.45, 0.30, 0.25, 50, "D"),
    (7, 8, "Bundesliga", 0.50, 0.25, 0.25, 80, "H"),
    (7, 9, "Bundesliga", 0.55, 0.25, 0.20, 120, "H"),
    (8, 9, "Bundesliga", 0.40, 0.30, 0.30, 20, "A"),
    (10, 11, "Serie A", 0.45, 0.30, 0.25, 30, "D"),
    (10, 12, "Serie A", 0.50, 0.25, 0.25, 60, "H"),
    (11, 12, "Serie A", 0.40, 0.35, 0.25, 10, "D"),
]

print("1. TAHMINLER KAYDEDILIYOR...")
for home_id, away_id, league, ph, pd_, pa, elo_diff, actual in predictions:
    fp.record_prediction(
        home_team_id=home_id,
        away_team_id=away_id,
        league=league,
        pred_home=ph,
        pred_draw=pd_,
        pred_away=pa,
        elo_diff=elo_diff,
        match_id=str(home_id) + "_" + str(away_id) + "_test",
    )
    fp.record_result(match_id=str(home_id) + "_" + str(away_id) + "_test", actual_result=actual)

print("   Toplam tahmin: " + str(len(predictions)))
print()

print("2. TAHMIN AYARLAMASI TESTLERI:")
print()

test_cases = [
    (81, 82, "Super Lig", 0.45, 0.25, 0.30, 50, "Galatasaray-Fenerbahce (derbi)"),
    (81, 83, "Super Lig", 0.55, 0.25, 0.20, 100, "Galatasaray-Besiktas"),
    (1, 2, "EPL", 0.50, 0.25, 0.25, 150, "ManU-Liverpool"),
    (4, 5, "La Liga", 0.60, 0.25, 0.15, 300, "Barca-Real Madrid"),
    (7, 8, "Bundesliga", 0.45, 0.30, 0.25, 80, "Bayern-Dortmund"),
]

for home_id, away_id, league, ph, pd_, pa, elo_diff, desc in test_cases:
    adj = fp.adjust_prediction(
        home_team_id=home_id,
        away_team_id=away_id,
        league=league,
        pred_home=ph,
        pred_draw=pd_,
        pred_away=pa,
        elo_diff=elo_diff,
    )

    print("   " + desc + ":")
    print("     Orijinal:  Ev=" + str(round(ph, 3)) + " Ber=" + str(round(pd_, 3)) + " Dep=" + str(round(pa, 3)))
    print("     Ayarlanmis: Ev=" + str(round(adj.adjusted_home, 3)) + " Ber=" + str(round(adj.adjusted_draw, 3)) + " Dep=" + str(round(adj.adjusted_away, 3)))
    print("     Degisim:   Ev=" + str(round(adj.home_adjustment, 3)) + " Ber=" + str(round(adj.draw_adjustment, 3)) + " Dep=" + str(round(adj.away_adjustment, 3)))
    print("     Uygulanan: " + str(adj.adjustments_applied))
    print("     Guven: " + str(round(adj.confidence_factor, 2)))
    print()

print("3. ISTATISTIKLER:")
stats = fp.get_stats()
for k, v in stats.items():
    print("   " + str(k) + ": " + str(v))
print()

print("4. TAKIM RAPORLARI:")
for team_id in [81, 82, 1, 4]:
    report = fp.get_team_report(team_id)
    if "error" not in report:
        print("   Takim " + str(team_id) + ": " + str(report["total_predictions"]) + " tahmin, home_bias=" + str(round(report["home_bias"], 3)))

print()
print("=== TEST TAMAMLANDI ===")

# Temizle
shutil.rmtree(test_dir, ignore_errors=True)
