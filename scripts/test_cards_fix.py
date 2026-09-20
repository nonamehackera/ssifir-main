"""Duzeltilmis kart/korner modeli testi (leakage yok) - ayni 500 mac."""
import sys, time
sys.path.insert(0, "/home/hackerspg/Masaüstü/sıfır")

import numpy as np
import pandas as pd

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features

t0 = time.time()
m = build_matches()
fd = m[m.league.str.match(r"^(E|SP|I|D|F|N|P|T|B|G|S)")]
print(f"FD maclar: {len(fd)} | feature uretiliyor...")
f = build_features(fd)
f = f.sort_values("date").reset_index(drop=True)
print(f"features: {len(f)} satir, home_cards_5 var mi: {'home_cards_5' in f.columns} ({time.time()-t0:.0f} sn)")

test = f.tail(500).copy()
train = f.iloc[:-500].copy()

def report(name, pred_bool, y, naive):
    acc = (pred_bool == y).mean()
    print(f"  {name}: %{acc*100:.1f} (naive %{naive*100:.1f}) doğru {int((pred_bool==y).sum())}/500")
    return acc

# KART 4.5
yk = (train.home_yellow.fillna(0).add(train.away_yellow.fillna(0))
      .add(train.home_red.fillna(0)).add(train.away_red.fillna(0)) > 4.5).astype(int).to_numpy()
yke = (test.home_yellow.fillna(0).add(test.away_yellow.fillna(0))
       .add(test.home_red.fillna(0)).add(test.away_red.fillna(0)) > 4.5).astype(int).to_numpy()
naive_k = max(yke.mean(), 1 - yke.mean())

from models.markets.corners_cards import CardsModel, CornersModel
print("\n=== KART 4.5 (leakage düzeltildi) ===")
cm = CardsModel(family="negbin", use_ml=True).fit(train)
pr = cm.predict(test)
p_ok = pr.extra["over_3_5"]
# over_3_5 aslinda 4.5 ustune isaret mi? koda bak: 1 - total_dist[:,:4].sum = P(total>=4). total>=4 != >4.5.
# Kendi tahminimizi dogru uretelim: home+away dist convolution'dan P(total>4.5)
h = cm.home.predict_dist(cm.home.predict_mean(test))
a = cm.away.predict_dist(cm.away.predict_mean(test))
td = np.array([np.convolve(h[i], a[i]) for i in range(len(h))])
p45 = 1.0 - td[:, :5].sum(axis=1)
report("Kart4.5 (düzeltilmiş)", p45 >= 0.5, yke, naive_k)

# KORNER 9.5
print("\n=== KORNER 9.5 ===")
trc = train.home_corners.fillna(0).add(train.away_corners.fillna(0))
tec = test.home_corners.fillna(0).add(test.away_corners.fillna(0))
yc = (trc > 9.5).astype(int).to_numpy()
yce = (tec > 9.5).astype(int).to_numpy()
naive_c = max(yce.mean(), 1 - yce.mean())
cn = CornersModel(family="negbin", use_ml=True).fit(train)
prc = cn.predict(test)
report("Korner9.5 (negbin)", prc.extra["over_9_5"] >= 0.5, yce, naive_c)

print(f"\nToplam sure: {time.time()-t0:.0f} sn")