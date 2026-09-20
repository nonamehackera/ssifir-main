import json
preds = json.load(open("tahminler/ftms_tahminler_2026-09-07.json", "r", encoding="utf-8"))
for i, p in enumerate(preds):
    mk = p["match"]
    print(f"{i+1:>2}. {mk}")
