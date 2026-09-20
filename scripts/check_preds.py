import json

with open(r'C:\Users\furka\Desktop\ssifir-main\predictions.json', 'r', encoding='utf-8') as f:
    preds = json.load(f)

print(f'Toplam: {len(preds)} tahmin')
for p in preds:
    print(f"  {p.get('home_name')} vs {p.get('away_name')} - {p.get('timestamp')}")

# tahminler/predictions.json
with open(r'C:\Users\furka\Desktop\ssifir-main\tahminler\predictions.json', 'r', encoding='utf-8') as f:
    preds2 = json.load(f)

print(f'\ntahminler/predictions.json: {len(preds2)} tahmin')
# unique by home_id+away_id
seen = set()
for p in preds2:
    key = (p.get('home_id'), p.get('away_id'))
    if key not in seen:
        seen.add(key)
        print(f"  {p.get('home_name')} vs {p.get('away_name')} - {p.get('timestamp')}")
print(f'Benzersiz mac: {len(seen)}')

# root predictions.json unique
seen2 = set()
for p in preds:
    key = (p.get('home_id'), p.get('away_id'))
    if key not in seen2:
        seen2.add(key)
        print(f"  {p.get('home_name')} vs {p.get('away_name')} - {p.get('timestamp')}")
print(f'Root benzersiz mac: {len(seen2)}')

# birlesik
all_keys = seen | seen2
print(f'\nBirlesik benzersiz mac: {len(all_keys)}')
missing = all_keys - seen
if missing:
    print('tahminler/predictions.json da eksik olanlar:')
    for k in missing:
        for p in preds:
            if (p.get('home_id'), p.get('away_id')) == k:
                print(f"  {p.get('home_name')} vs {p.get('away_name')}")
