"""ProductionEnsemble sinifini dogrudan test eder.

Gerçek kullanim senaryosu:
  - Feature_engine.enhance ile feature uretim
  - models.production_ensemble.ProductionEnsemble ile fit/predict
  - Blind test (son 6 ay)
  - Lig kategorileri (buyuk/orta/az) ayri ayri
  - Market orani OLAN vs OLMAYAN ayri ayri
"""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, mean_absolute_error

from feature_engine.enhance import enhance_features
from models.production_ensemble import ProductionEnsemble, default_feature_set

t0 = time.time()
print("=" * 95)
print("  ProductionEnsemble SINIF TESTI + BLIND + LIG KATEGORILERI")
print("=" * 95)

# ─── 1. VERI ────────────────────────────────────────────────────────────────
print("\n[1] Veri yukleniyor ve enhance ediliyor...")
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]: feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])
feat = enhance_features(feat)

# Lig kategorileri
lc = feat["league"].value_counts()
feat["liga_cat"] = feat["league"].map(lc).apply(lambda n: "buyuk" if n>=3000 else ("orta" if n>=1000 else "az"))
feat["has_mkt"] = feat["mkt_home_prob"].notna() & feat["mkt_draw_prob"].notna() & feat["mkt_away_prob"].notna()

# ─── 2. BLIND TEST (SON 6 AY) ───────────────────────────────────────────────
TEST_START = pd.Timestamp("2026-03-01")
train_blind = feat[feat["date"] < TEST_START].copy()
test_blind = feat[feat["date"] >= TEST_START].copy()
print(f"\n  Train: {len(train_blind)} ({train_blind['date'].min().date()}~{train_blind['date'].max().date()})")
print(f"  Test:  {len(test_blind)} ({test_blind['date'].min().date()}~{test_blind['date'].max().date()})")

n_cal = max(1000, int(len(train_blind)*0.15))
cal_blind = train_blind.iloc[-n_cal:].copy()
train_blind_fit = train_blind.iloc[:-n_cal].copy()

# Feature seti
feats_all = default_feature_set(train_blind_fit)
feats_all = [f for f in feats_all if f in train_blind_fit.columns]
print(f"\n  Feature sayisi: {len(feats_all)}")

# Model fit
print("\n[2] ProductionEnsemble fit ediliyor...")
ens = ProductionEnsemble(base_features=feats_all)
ens.fit(train_blind_fit)

# Predict
print("\n[3] Predict ediliyor (blind test)...")
pred = ens.predict(test_blind)

# ─── 3. GENEL SONUCLAR ──────────────────────────────────────────────────────
y_1x2 = test_blind["result"].map({"H":0,"D":1,"A":2}).to_numpy()
y_btts = test_blind["btts"].to_numpy()
y_o25 = test_blind["over25"].to_numpy()
y_total = (test_blind["home_goals"] + test_blind["away_goals"]).to_numpy()

ph = np.asarray(pred.home_win); pd_ = np.asarray(pred.draw); pa = np.asarray(pred.away_win)
probs = np.column_stack([ph,pd_,pa]); probs = probs / probs.sum(axis=1,keepdims=True)
pred_1x2 = np.argmax(probs, axis=1); conf = np.max(probs, axis=1)
ll = log_loss(y_1x2, probs); acc = accuracy_score(y_1x2, pred_1x2)
f1 = f1_score(y_1x2, pred_1x2, average="macro")
base_h = (y_1x2==0).mean()

pb = np.asarray(pred.btts_yes)
auc_b = roc_auc_score(y_btts, pb) if len(np.unique(y_btts))>1 else 0.5
po = np.asarray(pred.over25)
auc_o = roc_auc_score(y_o25, po) if len(np.unique(y_o25))>1 else 0.5

hl = np.asarray(pred.home_lambda); al = np.asarray(pred.away_lambda)
mae = mean_absolute_error(y_total, hl+al)
naive = train_blind_fit["total_goals"].mean()
mae_n = mean_absolute_error(y_total, np.full(len(y_total), naive))

print("\n" + "=" * 95)
print("  BLIND TEST GENEL SONUCLARI")
print("=" * 95)
print(f"\n  1X2:")
print(f"    LogLoss:   {ll:.4f} (random: {np.log(3):.4f})")
print(f"    Accuracy:  {acc:.3f} (baseline H: {base_h:.3f})")
print(f"    Macro-F1:  {f1:.3f}")
print(f"\n  BTTS:   AUC={auc_b:.4f}")
print(f"  Over2.5: AUC={auc_o:.4f}")
print(f"  Gol:    MAE={mae:.4f} (naive {mae_n:.4f})")

# Esik bazli
print(f"\n  ESIK BAZLI 1X2:")
for t in [0.50,0.55,0.60,0.65,0.70,0.75]:
    mask=conf>=t; n=mask.sum()
    if n>10:
        print(f"    @{t:.0%}: {accuracy_score(y_1x2[mask],pred_1x2[mask]):.3f} ({n} picks)")

# ─── 4. LIG KATEGORISI ──────────────────────────────────────────────────────
print(f"\n  LIG KATEGORISI BAZLI BLIND TEST:")
print(f"  {'Kategori':>8s} {'Mac':>6s} {'Acc':>7s} {'LL':>7s} {'BTTS':>7s} {'O25':>7s}")
print(f"  {'-'*50}")
for cat in ["buyuk","orta","az"]:
    m = test_blind["liga_cat"]==cat
    n = m.sum()
    if n<20: continue
    a = accuracy_score(y_1x2[m], pred_1x2[m])
    l = log_loss(y_1x2[m], probs[m])
    try: b = roc_auc_score(y_btts[m], pb[m]) if len(np.unique(y_btts[m]))>1 else 0.5
    except: b = 0.5
    try: o = roc_auc_score(y_o25[m], po[m]) if len(np.unique(y_o25[m]))>1 else 0.5
    except: o = 0.5
    print(f"  {cat:>8s} {n:6d} {a:6.3f} {l:7.4f} {b:6.3f} {o:6.3f}")

# ─── 5. MARKET ORANI VAR/YOK ────────────────────────────────────────────────
print(f"\n  MARKET ORANI OLAN vs OLMAYAN (blind):")
for mkt in [True, False]:
    m = test_blind["has_mkt"]==mkt
    n = m.sum()
    if n<10: continue
    a = accuracy_score(y_1x2[m], pred_1x2[m])
    l = log_loss(y_1x2[m], probs[m])
    lab = "oran var" if mkt else "oran yok"
    print(f"    {lab:>8s}: Acc={a:.3f} LL={l:.4f} (n={n})")

# ─── 6. ORNEKLER ────────────────────────────────────────────────────────────
print(f"\n  ORNEK TAHMINLER (son 15 mac):")
result_map = {0:"H",1:"D",2:"A"}
print(f"  {'#':>3s} {'Tarih':>10s} {'Lig':>20s} {'Tahmin':>6s} {'Gerçek':>6s} {'OK':>4s} {'Conf':>6s}")
print(f"  {'-'*55}")
n_ok = 0; n_tot = 0
for i in range(len(test_blind)-15, len(test_blind)):
    row = test_blind.iloc[i]
    t = result_map[pred_1x2[i]]; g = row["result"]
    ok = "V" if t==g else "X"
    if t==g: n_ok+=1
    n_tot+=1
    print(f"  {i+1:3d} {str(row['date'].date()):>10s} {str(row['league'])[:20]:>20s} {t:>6s} {g:>6s} {ok:>4s} {conf[i]:6.3f}")
print(f"\n  Son {n_tot} mac dogruluk: {n_ok}/{n_tot} ({n_ok/max(n_tot,1)*100:.0f}%)")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("=" * 95)
