"""Football-Data CSV analiz: kac mac, oran, korner, xG var mi?"""
import pandas as pd, os, glob, json

csvs = glob.glob("data/raw/footballcsv/**/*.csv", recursive=True)
print(f"Toplam CSV: {len(csvs)}")

# Her CSV'yi tara
rows_all = []
for fp in csvs:
    try:
        df = pd.read_csv(fp, encoding="latin-1", low_memory=False)
        cols = list(df.columns)
        # oran var mi?
        has_odds = any(c in cols for c in ["AvgH","B365H","BWH","IWH","PSH","AvgCH"])
        has_hc = "HC" in cols
        has_ht = "HTHG" in cols
        # league kodu
        basename = os.path.basename(fp).replace(".csv","")
        league = os.path.dirname(fp).replace("\\","/").split("/")[-1] + "/" + basename
        rows_all.append({
            "path": fp, "league": league, "rows": len(df),
            "has_odds": has_odds, "has_corners": has_hc, "has_ht": has_ht,
            "date_range": f"{df['Date'].iloc[0] if len(df)>0 else '?'} -> {df['Date'].iloc[-1] if len(df)>0 else '?'}",
            "cols": cols[:15],
        })
    except:
        pass

rdf = pd.DataFrame(rows_all)
print(f"Basarili okunan: {len(rdf)}")
print(f"Oran olan: {rdf['has_odds'].sum()}")
print(f"Korner olan: {rdf['has_corners'].sum()}")
print(f"HT gol olan: {rdf['has_ht'].sum()}")
print(f"Toplam mac: {rdf['rows'].sum()}")

# Buyuk ligler
print("\nBUYUK LIGLER (ornegi olan CSV'ler):")
big = rdf[rdf["has_odds"]].sort_values("rows", ascending=False).head(20)
for _, r in big.iterrows():
    print(f"  {r['league']:40s} {r['rows']:>6} mac  {r['date_range'][:50]}")

# Ortalama sut/korner kolonlari
sample = pd.read_csv(rdf[rdf["has_corners"]].iloc[0]["path"], encoding="latin-1", nrows=100)
stat_cols = [c for c in sample.columns if c in ["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR","B365H","B365D","B365A","AvgH","AvgD","AvgA","Avg>2.5","Avg<2.5","AvgCH","AvgCD","AvgCA"]]
print(f"\nORNK CSV kolonlari (istatistik + oran):")
print(f"  {stat_cols}")
