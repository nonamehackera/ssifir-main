"""Tam piyasa testi: 1X2, BTTS, O1.5, O2.5, Korner 9.5, Kart 4.5 — FD-only, ayni 500 mac."""
import sys, time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from models.catboost.model import CAT_COLS, NUMERIC_FEATURES, ODDS_FEATURES

feats = pd.read_parquet("data/gold/features.parquet").sort_values("date").reset_index(drop=True)
fd_mask = feats.league.str.match(r"^(E|SP|I|D|F|N|P|T|B|G|S)")
fd_idx = feats.index[fd_mask]
test_idx = fd_idx[-500:]
test = feats.loc[test_idx].copy()
train = feats.loc[(feats.index < test_idx[0]) & fd_mask].copy()

def to_targets(df):
    tc = df.home_corners.add(df.away_corners)
    tk = df.home_yellow.add(df.away_yellow).add(df.home_red.fillna(0)).add(df.away_red.fillna(0))
    tg = df.home_goals.add(df.away_goals)
    return {
        "1x2": df.result.map({"H": 0, "D": 1, "A": 2}),
        "btts": df.btts,
        "o15": (tg >= 2).astype(int),
        "o25": (tg >= 3).astype(int),
        "corner95": (tc > 9.5).astype(int),
        "card45": (tk > 4.5).astype(int),
    }

yt, tt = to_targets(train), to_targets(test)

cols = CAT_COLS + NUMERIC_FEATURES + ODDS_FEATURES
cat_idx = [cols.index(c) for c in CAT_COLS]
Xtr, Xte = train[cols].copy(), test[cols].copy()

params = dict(iterations=400, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
              border_count=128, random_strength=1.0, bagging_temperature=0.8,
              random_seed=42, verbose=False, allow_writing_files=False)

def run_bin(name, ytr, yte, cw=None):
    m = CatBoostClassifier(loss_function="Logloss", cat_features=cat_idx,
                           class_weights=cw, **params)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = p >= 0.5
    base = max(yte.mean(), 1 - yte.mean())
    acc = (pred == yte).mean()
    brier = ((p - yte) ** 2).mean()
    print(f"  {name}: doğru {int((pred==yte).sum())}/500 -> %{acc*100:.1f} "
          f"(naive baz %{base*100:.1f}) | Brier {brier:.4f} | model üst deme oranı %{(pred==1).mean()*100:.0f}")
    return acc

print(f"\nEğitim: {len(train)} (FD) | Test: 500")
print("=== 1X2 ===")
m1 = CatBoostClassifier(loss_function="MultiClass", cat_features=cat_idx,
                        class_weights={0: 1, 1: 1.2, 2: 1}, **params)
m1.fit(Xtr, yt["1x2"])
p1 = m1.predict_proba(Xte)
yc = np.argmax(p1, axis=1)
y = yt_actual = test.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
print(f"  genel doğruluk: {(yc==y).mean()*100:.1f}% (naive %{max(pd.Series(y).value_counts(normalize=True)):.1f})")
for i, cls in enumerate(["H", "D", "A"]):
    mm = y == i
    print(f"    {cls}: doğru {int((yc[mm]==i).sum())}/{int(mm.sum())} -> %{(yc[mm]==i).mean()*100:.1f}")

mkt = test[["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]].to_numpy()
mkt_cat = np.argmax(mkt, axis=1)
print(f"  market favori: %{(mkt_cat==y).mean()*100:.1f}")
def ll(ps, ys):
    eps = 1e-9
    return -np.mean(np.log(np.clip(ps[np.arange(len(ys)), ys], eps, 1)))
print(f"  log-loss: {ll(p1, y):.4f}")

print("=== BTTS ===")
run_bin("BTTS", yt["btts"], tt["btts"])
print("=== Over/Under 1.5 ===")
run_bin("O1.5", yt["o15"], tt["o15"])
print("=== Over/Under 2.5 ===")
run_bin("O2.5", yt["o25"], tt["o25"])
print("=== Korner 9.5 ===")
run_bin("Korner9.5", yt["corner95"], tt["corner95"])
print("=== Kart 4.5 ===")
run_bin("Kart4.5", yt["card45"], tt["card45"])
print("bitti")