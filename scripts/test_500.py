"""Gerçek out-of-sample test: son 500 stat'lı maç (FD), öncesiyle eğitim."""
import sys, time
sys.path.insert(0, "/home/hackerspg/Masaüstü/sıfır")

import numpy as np
import pandas as pd

from models.catboost.model import CatBoostModel

t0 = time.time()
feats = pd.read_parquet("data/gold/features.parquet").sort_values("date").reset_index(drop=True)
fd_mask = feats.league.str.match(r"^(E|SP|I|D|F|N|P|T|B|G|S)")
fd_idx = feats.index[fd_mask]
test_idx = fd_idx[-500:]
test = feats.loc[test_idx].copy()
train = feats.loc[feats.index < test_idx[0]].copy()
print(f"Eğitim: {len(train)} maç ({train.date.min().date()} -> {train.date.max().date()})")
print(f"TEST : {len(test)} maç ({test.date.min().date()} -> {test.date.max().date()})")
sys.stdout.flush()

model = CatBoostModel(with_odds=True, iterations=400, verbose=False)
model.fit(train)

pred = model.predict(test)

y = test.result.to_numpy()
p = np.column_stack([pred.home_win, pred.draw, pred.away_win])
y_cat = np.array(["H", "D", "A"])[p.argmax(axis=1)]
acc = (y_cat == y).mean()

print("\n=== 1X2 SONUÇ (doğruluk) ===")
print(f"GENEL DOĞRULUK: {acc*100:.1f}%  ({int((y_cat==y).sum())}/500)")
for cls in ["H", "D", "A"]:
    m = y == cls
    correct = (y_cat[m] == cls).sum()
    print(f"  {cls}: gerçek {m.sum():3d} maç, model {int((y_cat==cls).sum()):3d} kez {cls} dedi, "
          f"doğru {int(correct):3d} -> {correct/m.sum()*100:5.1f}%")

yb = test.btts.to_numpy()
pb = pred.btts_yes
pred_b = pb >= 0.5
for val, name in [(1, "Evet (BTTS)"), (0, "Hayır (no BTTS)")]:
    m = yb == val
    correct = (pred_b[m] == val).sum()
    print(f"  {name}: gerçek {m.sum():3d}, model {int((pred_b==val).sum()):3d}, doğru {int(correct):3d} -> {correct/m.sum()*100:5.1f}%")
print(f"  BTTS GENEL DOĞRULUK: {(pred_b==yb).mean()*100:.1f}%")

yo = test.over25.to_numpy()
po = pred.over25
pred_o = po >= 0.5
for val, name in [(1, "Üst (2.5+)"), (0, "Alt (0-2)")]:
    m = yo == val
    correct = (pred_o[m] == val).sum()
    print(f"  {name}: gerçek {m.sum():3d}, model {int((pred_o==val).sum()):3d}, doğru {int(correct):3d} -> {correct/m.sum()*100:5.1f}%")
print(f"  O2.5 GENEL DOĞRULUK: {(pred_o==yo).mean()*100:.1f}%")

mkt = test[["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]].to_numpy()
mkt_cat = np.array(["H", "D", "A"])[mkt.argmax(axis=1)]
print(f"\nKarşılaştırma (aynı 500 maç):")
print(f"  Market favorisi doğruluk: {(mkt_cat==y).mean()*100:.1f}%")
print(f"  Model doğruluk:           {acc*100:.1f}%")

hg_err = np.abs(pred.home_lambda - test.home_goals).mean()
ag_err = np.abs(pred.away_lambda - test.away_goals).mean()
print(f"\nGol tahmin hata ort. (MAE): ev {hg_err:.2f}, depl {ag_err:.2f} (gerçek ort {test.home_goals.mean():.2f}/{test.away_goals.mean():.2f})")

def ll(ps, ys):
    eps = 1e-9
    return -np.mean(np.log(np.clip(ps[np.arange(len(ys)), ys], eps, 1)))
yy = np.array([{"H":0,"D":1,"A":2}[v] for v in y])
print(f"1X2 log-loss: {ll(p, yy):.4f}")
print(f"süre: {time.time()-t0:.0f} sn")