"""FD CSV kolon analiz - gercek kolon isimleri"""
import pandas as pd, os, glob

csvs = glob.glob("data/raw/footballcsv/**/*.csv", recursive=True)

# Ornek dosyalari oku, kolon isimlerini gor
seen = set()
for fp in csvs[:50]:
    try:
        df = pd.read_csv(fp, encoding="latin-1", nrows=3, low_memory=False)
        bn = os.path.basename(fp)
        key = bn
        if key not in seen:
            seen.add(key)
            print(f"\n{bn} ({os.path.dirname(fp).split(os.sep)[-1]}):")
            print(f"  Kolonlar: {list(df.columns)}")
            if len(df) > 0:
                print(f"  Ornek satir: {dict(df.iloc[0])}")
    except Exception as e:
        pass

# football_data canonical - gercek kolon eslesmesi
print("\n\n=== canonical.py fd_leagues ===")
from ingestion.football_data.canonical import FD_LEAGUES
for k, v in list(FD_LEAGUES.items())[:5]:
    print(f"  {k}: {v}")
