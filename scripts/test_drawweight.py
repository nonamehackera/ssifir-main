"""FD-only eğitim + draw_weight deneyi, aynı 500 test maçı."""
import sys, time
sys.path.insert(0, "/home/hackerspg/Masaüstü/sıfır")

import numpy as np
import pandas as pd

from models.catboost.model import CatBoostModel

feats = pd.read_parquet("data/gold/features.parquet").sort_values("date").reset_index(drop=True)
fd_mask = feats.league.str.match(r"^(E|SP|I|D|F|N|P|T|B|G|S)")
fd_idx = feats.index[fd_mask]
test_idx = fd_idx[-500:]
test = feats.loc[test_idx].copy()
train = feats.loc[(feats.index < test_idx[0]) & fd_mask].copy()
print(f"FD eğitim: {len(train)} | test: {len(test)}")

y = test.result.to_numpy()
yb = test.btts.to_numpy()
yo = test.over25.to_numpy()
yy = np.array([{"H":0,"D":1,"A":2}[v] for v in y])

def ll(ps, ys):
    eps = 1e-9
    return -np.mean(np.log(np.clip(ps[np.arange(len(ys)), ys], eps, 1)))

mkt = test[["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]].to_numpy()
mkt_cat = np.array(["H", "D", "A"])[mkt.argmax(axis=1)]
print(f"MARKET favori doğruluğu: {(mkt_cat==y).mean()*100:.1f}%\n")

results = []
for dw in [1.2, 2.5, 4.0]:
    t0 = time.time()
    print(f"\n### draw_weight={dw} ###")
    model = CatBoostModel(with_odds=True, iterations=400, verbose=False, draw_weight=dw)
    model.fit(train)
    pred = model.predict(test)

    p = np.column_stack([pred.home_win, pred.draw, pred.away_win])
    y_cat = np.array(["H", "D", "A"])[p.argmax(axis=1)]
    acc = (y_cat == y).mean()
    d_recall = ((y_cat == "D") & (y == "D")).sum() / (y == "D").sum()

    pred_b = pred.btts_yes >= 0.5
    bacc = (pred_b == yb).mean()
    pred_o = pred.over25 >= 0.5
    oacc = (pred_o == yo).mean()

    print(f"1X2 genel: {acc*100:.1f}% | D recall: {d_recall*100:.1f}% | logloss: {ll(p, yy):.4f}")
    print(f"BTTS: {bacc*100:.1f}% | O2.5: {oacc*100:.1f}%")
    print(f"  H: {(y_cat==y)[y=='H'].mean()*100:.1f}%  D: {(y_cat==y)[y=='D'].mean()*100:.1f}%  A: {(y_cat==y)[y=='A'].mean()*100:.1f}%")
    print(f"  D dediği: {int((y_cat=='D').sum())} | doğru D: {int(((y_cat=='D')&(y=='D')).sum())}")
    print(f"  süre: {time.time()-t0:.0f} sn")
    results.append((dw, acc, d_recall, ll(p, yy), bacc, oacc))
    sys.stdout.flush()

print("\n=== ÖZET ===")
for dw, acc, dr, l, b, o in results:
    print(f"dw={dw}: 1X2 %{acc*100:.1f} | D recall %{dr*100:.1f} | logloss {l:.4f} | BTTS %{b*100:.1f} | O2.5 %{o*100:.1f}")