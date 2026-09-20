"""ProductionPredictor (teslim API) end-to-end testi.

1. train/cal split -> prepare()
2. Son maçlardan birkaç feature satırı al -> predict_single()
3. playable/refusal etiketini dogrula (buyuk vs az verili lig)
4. Gercek sonuclarla karsilastir (son N mac son ocukligi)
5. JSON ciktisini goster
"""
import sys, os, warnings, time, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from feature_engine.enhance import enhance_features
from prediction.production_pipeline import ProductionPredictor, predict_to_json

t0 = time.time()
print("=" * 95)
print("  ProductionPredictor END-TO-END TESTI")
print("=" * 95)

# ─── 1. Veri ────────────────────────────────────────────────────────────────
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])
feat=enhance_features(feat)

lc=feat["league"].value_counts()
feat["liga_cat"]=feat["league"].map(lc).apply(lambda n:"buyuk" if n>=3000 else ("orta" if n>=1000 else "az"))

# Train/cal split (son 3 ay test disi)
TEST_START=pd.Timestamp("2026-08-01")
train=feat[feat["date"]<TEST_START].copy()
test=feat[feat["date"]>=TEST_START].copy()
n_cal=max(1000,int(len(train)*0.15))
cal=train.iloc[-n_cal:].copy()
tr=train.iloc[:-n_cal].copy()
print(f"\n  Train: {len(tr)} | Cal: {len(cal)} | Test: {len(test)}")

# ─── 2. prepare ─────────────────────────────────────────────────────────────
print("\n  ProductionPredictor hazirlaniyor...")
p = ProductionPredictor(model_version="test_prod", dataset_version="gold")
p.prepare(tr, cal)
print(f"  Feature: {len(p._feature_cols)} | Lig: {len(p.league_match_counts)}")

# ─── 3. GST. lig vs az verili (buyuk/orta/az) ───────────────────────────────
print("\n  Lig kategorisi dogrulama (playable etiketi):")
labels=["buyuk","orta","az"]
row_i=0
for cat in labels:
    m=test["liga_cat"]==cat
    if m.sum()==0: continue
    row=test[m].iloc[0]
    out=p.predict_single(row, fixture_id=str(row.get("match_id","?")),
                         prediction_time=pd.Timestamp.now())
    actual=row["result"]
    played_letter = {"home":"H","draw":"D","away":"A"}.get(out["result"]["favorite"],"?")
    ok="V" if played_letter==actual else "X"
    print(f"    {cat:>6s} lig ({row['league']}): fav={played_letter} gercek={actual} {ok} | "
          f"conf={out['reliability']['confidence']:.2f} | "
          f"policy={out['policy']['playable']} | n_lig={out['reliability']['league_matches']}")

# ─── 4. Son N mac - favori dogrulugu ────────────────────────────────────────
N=500
print(f"\n  Son {N} mac FAVORI dogrulugu:")
test_tail=test.tail(N).copy()
fc_left=0; fc_ok=0; play_ok=0; play_n=0; ref_ok=0; ref_n=0
for i in range(len(test_tail)):
    row=test_tail.iloc[i]
    out=p.predict_single(row)
    played_letter = {"home":"H","draw":"D","away":"A"}.get(out["result"]["favorite"],"?")
    fav=played_letter; actual=row["result"]
    fc_left+=1
    if fav==actual: fc_ok+=1
    if out["policy"]["playable"]:
        play_n+=1
        if fav==actual: play_ok+=1
    else:
        ref_n+=1
        if fav==actual: ref_ok+=1
print(f"    Tum favori:      {fc_ok/fc_left*100:.1f}% ({fc_ok}/{fc_left})")
if play_n>0: print(f"    PLAYABLE favori: {play_ok/play_n*100:.1f}% ({play_ok}/{play_n})  <-- dondugu secimler")
if ref_n>0: print(f"    'skip' favori:   {ref_ok/ref_n*100:.1f}% ({ref_ok}/{ref_n})  <-- donulmeyenler (kupa/amator/az verili)")

# ─── 5. Ornek JSON ──────────────────────────────────────────────────────────
print("\n  ORNEK JSON (tek mac):")
row=test_tail[test_tail["liga_cat"]=="buyuk"].iloc[0] if (test_tail["liga_cat"]=="buyuk").any() else test_tail.iloc[0]
out=p.predict_single(row)
print(predict_to_json(out)[:1500])

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("="*95)