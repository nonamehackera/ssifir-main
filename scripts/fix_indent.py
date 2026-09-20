path = "web/app.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 1175..1207 (1-indexed) = 0-indexed 1174..1206
# Bu blok eski 'try' govdesinden kalma 12 fazla boslukla basliyor.
# Hepsinden baslangictaki 12 boslugu at.
for i in range(1174, 1207):
    if i < len(lines):
        lines[i] = lines[i][12:] if lines[i].startswith("            ") else lines[i]

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("indent fixed")
