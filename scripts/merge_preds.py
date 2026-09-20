import json

# Kök dizindeki predictions.json (13 mac)
with open(r'C:\Users\furka\Desktop\ssifir-main\predictions.json', 'r', encoding='utf-8') as f:
    root_preds = json.load(f)

# tahminler/predictions.json (10 mac)
with open(r'C:\Users\furka\Desktop\ssifir-main\tahminler\predictions.json', 'r', encoding='utf-8') as f:
    tahminler_preds = json.load(f)

# Eksik olanlari bul (home_id+away_id key olarak)
existing = set()
for p in tahminler_preds:
    key = (p.get('home_id'), p.get('away_id'))
    existing.add(key)

# Son kayitlari al (en guncel)
root_by_key = {}
for p in root_preds:
    key = (p.get('home_id'), p.get('away_id'))
    root_by_key[key] = p  # son olani saklar

missing = []
for key, p in root_by_key.items():
    if key not in existing:
        missing.append(p)
        print(f"Eksik: {p.get('home_name')} vs {p.get('away_name')}")

# Eksik olanlari ekle
tahminler_preds.extend(missing)

# Kaydet
with open(r'C:\Users\furka\Desktop\ssifir-main\tahminler\predictions.json', 'w', encoding='utf-8') as f:
    json.dump(tahminler_preds, f, ensure_ascii=False, indent=2)

print(f'\nToplam: {len(tahminler_preds)} tahmin (onceki: {len(tahminler_preds) - len(missing)}, eklenen: {len(missing)})')
