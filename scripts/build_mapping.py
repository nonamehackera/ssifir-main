import pandas as pd
tn = pd.read_csv("data/gold/team_names.csv")
f = pd.read_parquet("data/gold/features.parquet")

# Features'taki ID'lerin tiplerini kontrol et
print("Features ID type:", type(f["home_team_id"].iloc[0]))
print("NameCSV ID type:", type(tn["id"].iloc[0]))

# Feature ID'lerini string yaparak eslestir
f_sample = f["home_team_id"].unique()[:10]
for tid in f_sample:
    name = tn[tn["id"].astype(str) == str(tid)]
    if len(name) > 0:
        n = name.iloc[0]["name"]
        print(f"  {tid} -> {n}")
    else:
        print(f"  {tid} -> ???")

# Features'taki her ID icin isim bul
# Feature ID -> name mapping
mapping = {}
for tid in f["home_team_id"].unique():
    name = tn[tn["id"].astype(str) == str(tid)]
    if len(name) > 0:
        mapping[tid] = name.iloc[0]["name"]
    else:
        mapping[tid] = f"Team_{tid}"

print(f"\nBulunan: {sum(1 for v in mapping.values() if not v.startswith('Team_'))}/{len(mapping)}")

# Mapping'i kaydet
import json
with open("data/gold/team_id_to_name.json", "w") as fh:
    json.dump({str(k): v for k, v in mapping.items()}, fh)
print("Kaydedildi: data/gold/team_id_to_name.json")

# Ornek goster
for tid in list(mapping.keys())[:20]:
    print(f"  {tid}: {mapping[tid]}")
