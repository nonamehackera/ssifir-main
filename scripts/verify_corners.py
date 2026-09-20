"""Yeni CornersModel (CatBoost) dogrulama - ayni 500 mac."""
import sys, time
sys.path.insert(0, "/home/hackerspg/Masaüstü/sıfır")

import numpy as np
import pandas as pd

from models.markets.corners_cards import CornersModel

t0 = time.time()
feats = pd.read_parquet("data/gold/features.parquet").sort_values("date").reset_index(drop=True)
fd_mask = feats.league.str.match(r"^(E|SP|I|D|F|N|P|T|B|G|S)")
fd_idx = feats.index[fd_mask]
test_idx = fd_idx[-500:]
test = feats.loc[test_idx].copy()
train = feats.loc[(feats.index < test_idx[0]) & fd_mask].copy()
print(f"train {len(train)} test 500")

trc = train.home_corners.fillna(0).add(train.away_corners.fillna(0))
tec = test.home_corners.fillna(0).add(test.away_corners.fillna(0))
yce = (tec > 9.5).astype(int).to_numpy()
naive = max(yce.mean(), 1 - yce.mean())

cn = CornersModel(family="negbin", use_ml=True).fit(train)
pr = cn.predict(test)
p = pr.extra["over_9_5"]
acc = ((p >= 0.5) == yce).mean()
brier = ((p - yce) ** 2).mean()
print(f"Korner9.5 CatBoost: %{acc*100:.1f} (naive %{naive*100:.1f}) | Brier {brier:.4f}")
print(f"sure: {time.time()-t0:.0f} sn")