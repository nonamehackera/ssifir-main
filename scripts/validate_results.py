"""Dogrulama: Tahminleri gercek dunya sonuclariyla karsilastirir.
Kullanici tarafindan verilen 20 mac tahminini + internetten bulunan sonuclari karsilastirir.
"""

import json
from datetime import datetime

# ── Kullanici tarafindan verilen 20 mac tahmini ──
MATCHES = [
    # (id, home, away, tahmin_home%, tahmin_draw%, tahmin_away%, best_market, best_pick, gercek_sonuc)
    (1, "Stoke City", "Norwich City", 43.4, 23.7, 32.9, "1X2", "Home", "1-0"),
    (2, "Birmingham City", "Southampton", 22.4, 30.3, 47.4, "1X2", "Away", "1-1"),
    (3, "Portsmouth", "Derby County", 48.0, 23.6, 28.4, "1X2", "Home", "0-2"),
    (4, "Preston North End", "Bristol City", 40.7, 26.6, 32.7, "1X2", "Home", "1-3"),
    (5, "West Ham United", "Wolverhampton", 56.0, 22.2, 21.8, "1X2", "Home", "4-2"),
    (6, "Swansea City", "Watford FC", 43.6, 26.5, 29.9, "1X2", "Home", "2-0"),
    (7, "Lincoln City", "Blackburn Rovers", 41.1, 26.5, 32.5, "1X2", "Home", "0-0"),
    (8, "Sheffield United", "Bolton", 53.0, 22.6, 24.3, "1X2", "Home", "3-2"),
    (9, "West Brom", "Charlton Athletic", 58.4, 22.0, 19.5, "1X2", "Home", "1-1"),
    (10, "Millwall FC", "Wrexham AFC", 58.5, 21.9, 19.5, "1X2", "Home", "0-3"),
    (11, "Burnley FC", "Middlesbrough", 41.6, 27.2, 31.3, "1X2", "Home", "1-1"),
    (12, "QPR", "Cardiff City", 62.5, 20.1, 17.4, "1X2", "Home", "2-1"),
    (13, "Toulouse FC", "Lille OSC", 54.8, 21.2, 24.1, "1X2", "Home", "0-1"),
    (14, "Leyton Orient", "Bromley FC", 50.3, 24.4, 25.3, "1X2", "Home", "4-0"),
    (15, "Ashton United", "FC United of Manchester", 40.8, 21.1, 38.2, "1X2", "Home", "2-0"),
    (16, "Curzon Ashton FC", "Spennymoor Town", 33.0, 23.1, 43.8, "1X2", "Away", "1-3"),
    (17, "Queen's Park", "Stenhousemuir", 57.7, 22.3, 20.0, "1X2", "Home", "1-1"),
    (18, "Boreham Wood FC", "Dorking Wanderers", 42.9, 25.5, 31.7, "1X2", "Home", "2-1"),
    (19, "Enfield Town FC", "Farnborough FC", 45.0, 21.3, 33.7, "1X2", "Home", "3-2"),
    (20, "Maidstone United", "Worthing FC", 46.7, 22.8, 30.4, "1X2", "Home", "0-0"),
]

def evaluate():
    results = []
    
    for m in MATCHES:
        id, home, away, h_prob, d_prob, a_prob, market, pick, score = m
        
        # Skoru parse et
        parts = score.split("-")
        h_goals = int(parts[0])
        a_goals = int(parts[1])
        
        # Gercek sonuc
        if h_goals > a_goals:
            gercek = "Home"
            gercek_str = "1"
        elif h_goals == a_goals:
            gercek = "Draw"
            gercek_str = "X"
        else:
            gercek = "Away"
            gercek_str = "2"
        
        # Modelin 1X2 tahmini
        if h_prob >= d_prob and h_prob >= a_prob:
            tahmin = "Home"
            tahmin_prob = h_prob
        elif d_prob >= h_prob and d_prob >= a_prob:
            tahmin = "Draw"
            tahmin_prob = d_prob
        else:
            tahmin = "Away"
            tahmin_prob = a_prob
        
        # Dogru mu?
        dogru = tahmin == gercek
        
        # Buyuk ihtimal mi? (>50%)
        buyuk_ihtimal = tahmin_prob > 50
        
        # Best bet dogru mu?
        if pick == "1" or pick == "Home":
            best_bet_dogru = gercek == "Home"
        elif pick == "X" or pick == "Draw":
            best_bet_dogru = gercek == "Draw"
        elif pick == "2" or pick == "Away":
            best_bet_dogru = gercek == "Away"
        else:
            best_bet_dogru = False
        
        # Gol fazla/az mi?
        toplam_gol = h_goals + a_goals
        ust_2_5 = toplam_gol > 2.5
        btts = h_goals > 0 and a_goals > 0
        
        results.append({
            "id": id,
            "mac": f"{home} vs {away}",
            "skor": score,
            "toplam_gol": toplam_gol,
            "tahmin_1X2": tahmin,
            "tahmin_prob": tahmin_prob,
            "gercek_sonuc": gercek,
            "dogru": dogru,
            "buyuk_ihtimal": buyuk_ihtimal,
            "best_bet": pick,
            "best_bet_dogru": best_bet_dogru,
            "ust_2_5": ust_2_5,
            "btts": btts,
        })
    
    return results

def print_report(results):
    print("=" * 80)
    print("  GERCEK DUNYA DOGRULAMA RAPORU - 20 MAC TAHMIN VERISI")
    print("=" * 80)
    print()
    
    # Genel istatistikler
    toplam = len(results)
    dogru_sayisi = sum(1 for r in results if r["dogru"])
    yanlis_sayisi = toplam - dogru_sayisi
    
    buyuk_ihtimal = [r for r in results if r["buyuk_ihtimal"]]
    buyuk_dogru = sum(1 for r in buyuk_ihtimal if r["dogru"])
    
    buyuk_degil = [r for r in results if not r["buyuk_ihtimal"]]
    kucuk_dogru = sum(1 for r in buyuk_degil if r["dogru"])
    
    best_bet_dogru = sum(1 for r in results if r["best_bet_dogru"])
    
    # 1X2 accuracy
    print(f"  1X2 ACCURACY: {dogru_sayisi}/{toplam} = %{dogru_sayisi/toplam*100:.1f}")
    print()
    
    # Best bet accuracy
    print(f"  BEST BET ACCURACY: {best_bet_dogru}/{toplam} = %{best_bet_dogru/toplam*100:.1f}")
    print()
    
    # Buyuk ihtimal analizi
    print("  BUYUK IHTIMAL ANALIZI (>50%):")
    if buyuk_ihtimal:
        print(f"    Buyuk ihtimal ile tahmin edilen: {len(buyuk_ihtimal)} mac")
        print(f"    Dogru: {buyuk_dogru}/{len(buyuk_ihtimal)} = %{buyuk_dogru/len(buyuk_ihtimal)*100:.1f}")
    if buyuk_degil:
        print(f"    Buyuk ihtimal olmayan (<50%): {len(buyuk_degil)} mac")
        print(f"    Dogru: {kucuk_dogru}/{len(buyuk_degil)} = %{kucuk_dogru/len(buyuk_degil)*100:.1f}")
    print()
    
    # Gol analizi
    toplam_goller = sum(r["toplam_gol"] for r in results)
    ust_2_5_sayisi = sum(1 for r in results if r["ust_2_5"])
    btts_sayisi = sum(1 for r in results if r["btts"])
    
    print(f"  GOL ANALIZI:")
    print(f"    Ortalama gol: {toplam_goller/toplam:.2f}")
    print(f"    Ust 2.5: {ust_2_5_sayisi}/{toplam} = %{ust_2_5_sayisi/toplam*100:.1f}")
    print(f"    BTTS: {btts_sayisi}/{toplam} = %{btts_sayisi/toplam*100:.1f}")
    print()
    
    # Sonuc dagilimi
    home_wins = sum(1 for r in results if r["gercek_sonuc"] == "Home")
    draws = sum(1 for r in results if r["gercek_sonuc"] == "Draw")
    away_wins = sum(1 for r in results if r["gercek_sonuc"] == "Away")
    
    print(f"  SONUC DAGILIMI:")
    print(f"    Home Win: {home_wins}/{toplam} = %{home_wins/toplam*100:.1f}")
    print(f"    Draw:     {draws}/{toplam} = %{draws/toplam*100:.1f}")
    print(f"    Away Win: {away_wins}/{toplam} = %{away_wins/toplam*100:.1f}")
    print()
    
    # Tahmin dagilimi
    tahmin_home = sum(1 for r in results if r["tahmin_1X2"] == "Home")
    tahmin_draw = sum(1 for r in results if r["tahmin_1X2"] == "Draw")
    tahmin_away = sum(1 for r in results if r["tahmin_1X2"] == "Away")
    
    print(f"  TAHMIN DAGILIMI:")
    print(f"    Home: {tahmin_home}/{toplam} = %{tahmin_home/toplam*100:.1f}")
    print(f"    Draw: {tahmin_draw}/{toplam} = %{tahmin_draw/toplam*100:.1f}")
    print(f"    Away: {tahmin_away}/{toplam} = %{tahmin_away/toplam*100:.1f}")
    print()
    
    # Her mac sonucu
    print("-" * 80)
    print("  MAC MAC SONUCLAR:")
    print("-" * 80)
    print()
    print(f"  {'Mac':<45} {'Skor':>6} {'Tahmin':>8} {'Gercek':>8} {'Dogru':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    
    for r in results:
        isaret = "OK" if r["dogru"] else "X"
        print(f"  {r['mac']:<45} {r['skor']:>6} {r['tahmin_1X2']:>8} {r['gercek_sonuc']:>8} {isaret:>6}")
    
    print()
    print("-" * 80)
    print()
    
    # Basari oranlari
    print("  BASARI ORANLARI:")
    
    # Home tahmin dogrulugu
    home_tahminler = [r for r in results if r["tahmin_1X2"] == "Home"]
    if home_tahminler:
        home_dogru = sum(1 for r in home_tahminler if r["dogru"])
        print(f"    Home tahmin dogrulugu: {home_dogru}/{len(home_tahminler)} = %{home_dogru/len(home_tahminler)*100:.1f}")
    
    # Draw tahmin dogrulugu
    draw_tahminler = [r for r in results if r["tahmin_1X2"] == "Draw"]
    if draw_tahminler:
        draw_dogru = sum(1 for r in draw_tahminler if r["dogru"])
        print(f"    Draw tahmin dogrulugu: {draw_dogru}/{len(draw_tahminler)} = %{draw_dogru/len(draw_tahminler)*100:.1f}")
    
    # Away tahmin dogrulugu
    away_tahminler = [r for r in results if r["tahmin_1X2"] == "Away"]
    if away_tahminler:
        away_dogru = sum(1 for r in away_tahminler if r["dogru"])
        print(f"    Away tahmin dogrulugu: {away_dogru}/{len(away_tahminler)} = %{away_dogru/len(away_tahminler)*100:.1f}")
    
    print()
    print("=" * 80)
    
    # Oncelik siralamasi
    print("  ONCELIK SIRASI:")
    print(f"    1. Home: %{tahmin_home/toplam*100:.1f} tahmin, %{home_wins/toplam*100:.1f} gercek sonuc")
    print(f"    2. Away: %{tahmin_away/toplam*100:.1f} tahmin, %{away_wins/toplam*100:.1f} gercek sonuc")
    print(f"    3. Draw: %{tahmin_draw/toplam*100:.1f} tahmin, %{draws/toplam*100:.1f} gercek sonuc")
    print()
    print("  DEGERLENDIRME:")
    
    # Dogru tahminlerin skor analizi
    dogrular = [r for r in results if r["dogru"]]
    yanlislar = [r for r in results if not r["dogru"]]
    
    if dogrular:
        dogru_ortalama_prob = sum(r["tahmin_prob"] for r in dogrular) / len(dogrular)
        print(f"    Dogru tahminlerde ortalama guven: %{dogru_ortalama_prob:.1f}")
    
    if yanlislar:
        yanlis_ortalama_prob = sum(r["tahmin_prob"] for r in yanlislar) / len(yanlislar)
        print(f"    Yanlis tahminlerde ortalama guven: %{yanlis_ortalama_prob:.1f}")
    
    print()
    
    # Logloss hesapla
    import math
    logloss = 0
    for r in results:
        # Gercek sonuca karsi tahmin olasiligi
        if r["gercek_sonuc"] == "Home":
            # Ev sahibi kazandi
            # But we need the actual home/draw/away probs from the data
            # Using h_prob, d_prob, a_prob from MATCHES
            m = MATCHES[r["id"]-1]
            prob = m[3] / 100  # home_prob
        elif r["gercek_sonuc"] == "Draw":
            m = MATCHES[r["id"]-1]
            prob = m[4] / 100  # draw_prob
        else:
            m = MATCHES[r["id"]-1]
            prob = m[5] / 100  # away_prob
        
        # Clamp to avoid log(0)
        prob = max(prob, 1e-7)
        logloss -= math.log(prob)
    
    logloss /= len(results)
    
    # Brier score
    # For 1X2, Brier = mean of (1 - p_correct)^2
    brier = 0
    for r in results:
        m = MATCHES[r["id"]-1]
        if r["gercek_sonuc"] == "Home":
            p_correct = m[3] / 100
        elif r["gercek_sonuc"] == "Draw":
            p_correct = m[4] / 100
        else:
            p_correct = m[5] / 100
        brier += (1 - p_correct) ** 2
    brier /= len(results)
    
    print(f"  KALIBRASYON METRIKLERI:")
    print(f"    LogLoss: {logloss:.4f}")
    print(f"    Brier Score: {brier:.4f}")
    print()
    
    # Kalibrasyon bucketlari
    print("  KALIBRASYON BUCKETLARI:")
    buckets = [(40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]
    for low, high in buckets:
        bucket = [r for r in results if low <= r["tahmin_prob"] < high]
        if bucket:
            bucket_dogru = sum(1 for r in bucket if r["dogru"])
            bucket_oran = bucket_dogru / len(bucket) * 100
            print(f"    %{low}-{high}: {len(bucket)} mac, {bucket_dogru} dogru = %{bucket_oran:.1f}")
    
    print()

if __name__ == "__main__":
    results = evaluate()
    print_report(results)
