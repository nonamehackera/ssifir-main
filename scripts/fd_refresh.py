"""1. football-data.co.uk'den 2025-26 sezonu CSV'lerini cek + tamamini indir.
   Mevcut bronze cache'i gunceller, oran + korner + sut verisini artirir.
"""
import sys, os, urllib.request, urllib.error, time
sys.path.insert(0, ".")

from pathlib import Path
from ingestion.football_data.canonical import FD_LEAGUES, FD_SEASONS, _download_csv
import pandas as pd

BRONZE = Path("data/bronze/football_data/canonical")
BRONZE.mkdir(parents=True, exist_ok=True)

def main():
    total_new = 0
    total_updated = 0

    for lg, lg_name in FD_LEAGUES.items():
        for season in FD_SEASONS:
            cache = BRONZE / f"{lg}_{season}.csv"
            # Once cache var mi kontrol et
            if cache.exists():
                # 2025-26 sezonu icin her zaman guncelle
                if season in ("2425", "2526"):
                    pass  # guncelleyelim
                else:
                    continue  # eskileri atla

            url = f"https://www.football-data.co.uk/mmz4281/{season}/{lg}.csv"
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    text = r.read().decode("utf-8-sig", "replace")
                cache.write_text(text, encoding="utf-8")
                total_new += 1
                # 100ms bekle (rate limit)
                time.sleep(0.1)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                print(f"HATA {lg} {season}: {e}")
            except Exception as e:
                print(f"HATA {lg} {season}: {e}")

    # Tum CSV'leri oku, istatistik ver
    print(f"\nIndirilen: {total_new} yeni CSV")

    # Bronzdaki tum CSV'leri analiz et
    csvs = list(BRONZE.glob("*.csv"))
    print(f"Bronz toplam CSV: {len(csvs)}")

    # Her CSV'deoran, korner, sut var mi?
    stats = {"has_odds": 0, "has_corners": 0, "has_shots": 0, "total_rows": 0, "years": set()}
    for fp in csvs:
        try:
            df = pd.read_csv(fp, nrows=5, encoding="latin-1")
            cols = set(df.columns)
            if any(c in cols for c in ["AvgH", "B365H", "BWH"]):
                stats["has_odds"] += 1
            if "HC" in cols:
                stats["has_corners"] += 1
            if "HS" in cols:
                stats["has_shots"] += 1
            # yil bilgisi
            parts = fp.stem.split("_")
            if len(parts) >= 2:
                stats["years"].add(parts[1])
        except:
            pass

    # Tam sayiyi hesapla
    all_rows = 0
    for fp in csvs:
        try:
            df = pd.read_csv(fp, encoding="latin-1")
            all_rows += len(df)
        except:
            pass

    print(f"\nOZET:")
    print(f"  Oran olan CSV: {stats['has_odds']}/{len(csvs)}")
    print(f"  Korner olan CSV: {stats['has_corners']}/{len(csvs)}")
    print(f"  Sut olan CSV: {stats['has_shots']}/{len(csvs)}")
    print(f"  Toplam satir: {all_rows}")
    print(f"  Sezonlar: {sorted(stats['years'])}")

    # Ornek: E0 2526
    e0 = BRONZE / "E0_2526.csv"
    if e0.exists():
        df = pd.read_csv(e0, nrows=5, encoding="latin-1")
        print(f"\nE0_2526 ornek:")
        print(f"  Kolonlar: {list(df.columns)}")
        if len(df) > 0:
            print(f"  Satir: {dict(df.iloc[0])}")

if __name__ == "__main__":
    main()
