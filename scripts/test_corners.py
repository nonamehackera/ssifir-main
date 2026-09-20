"""Korner v2: tam feature seti + korner feature'lari + Poisson (NaN temiz)."""
import sys, time
sys.path.insert(0, "/home/hackerspg/Masaüstü/sıfır")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from models.catboost.model import CAT_COLS, NUMERIC_FEATURES, ODDS_FEATURES

t0 = time.time()
feats = pd.read_parquet("data/gold/features.parquet").sort_values("date").reset_index(drop=True)
fd_mask = feats.league.str.match(r"^(E|SP|I|D|F|N|P|T|B|G|S)")
fd_idx = feats.index[fd_mask]
test_idx = fd_idx[-500:]
test = feats.loc[test_idx].copy()
train = feats.loc[(feats.index < test_idx[0]) & fd_mask].copy()
fd = pd.concat([train, test]).reset_index(drop=True)

home_state: dict = {}
away_state: dict = {}
lg_state: dict = {}
rows = []
for _, m in fd.iterrows():
    h = home_state.get((m.league, m.home_team_id), [])
    a = away_state.get((m.league, m.away_team_id), [])
    hc, ac = m.home_corners, m.away_corners

    def stats(hist):
        cf = [x[0] for x in hist if x[0] is not None]
        ca = [x[1] for x in hist if x[1] is not None]
        return {
            "cf5": sum(cf[-5:]) / min(5, max(len(cf), 1)) if cf else np.nan,
            "ca5": sum(ca[-5:]) / min(5, max(len(ca), 1)) if ca else np.nan,
            "cf20": sum(cf[-20:]) / min(20, max(len(cf), 1)) if cf else np.nan,
            "ca20": sum(ca[-20:]) / min(20, max(len(ca), 1)) if ca else np.nan,
        }

    hs, as_ = stats(h), stats(a)
    lg = lg_state.get(m.league, [])
    rows.append({
        "h_cf5": hs["cf5"], "h_ca5": hs["ca5"], "h_cf20": hs["cf20"], "h_ca20": hs["ca20"],
        "a_cf5": as_["cf5"], "a_ca5": as_["ca5"], "a_cf20": as_["cf20"], "a_ca20": as_["ca20"],
        "cf_diff5": hs["cf5"] - as_["cf5"] if hs["cf5"] is not None and as_["cf5"] is not None else np.nan,
        "ca_diff5": hs["ca5"] - as_["ca5"] if hs["ca5"] is not None and as_["ca5"] is not None else np.nan,
        "exp_total5": (hs["cf5"] + hs["ca5"] + as_["cf5"] + as_["ca5"]) / 2
                     if all(v is not None for v in [hs["cf5"], hs["ca5"], as_["cf5"], as_["ca5"]]) else np.nan,
        "lg_corner_roll": np.mean(lg) if lg else np.nan,
        "tc": (hc + ac) if hc is not None and ac is not None else np.nan,
    })
    if hc is not None and ac is not None:
        home_state.setdefault((m.league, m.home_team_id), []).append((hc, ac))
        away_state.setdefault((m.league, m.away_team_id), []).append((ac, hc))
        lg_state.setdefault(m.league, []).append(hc + ac)

cf = pd.DataFrame(rows)
fd = pd.concat([fd, cf], axis=1)
print(f"feature hesaplandi: {time.time()-t0:.0f} sn")

train = fd.iloc[:-500].copy()
test = fd.iloc[-500:].copy()
ok = train.tc.notna()
yt = (train.loc[ok, "tc"] > 9.5).astype(int).to_numpy()
yte = (test.tc > 9.5).astype(int).to_numpy()
naive = max(yte.mean(), 1 - yte.mean())
print(f"test ust %{yte.mean()*100:.1f} | naive %{naive*100:.1f}")

corner_feats = ["h_cf5", "h_ca5", "h_cf20", "h_ca20", "a_cf5", "a_ca5", "a_cf20", "a_ca20",
                "cf_diff5", "ca_diff5", "exp_total5", "lg_corner_roll"]

def report(name, p):
    pred = p >= 0.5
    acc = (pred == yte).mean()
    brier = ((p - yte) ** 2).mean()
    eps = 1e-9
    ll_ = -np.mean(yte * np.log(np.clip(p, eps, 1)) + (1 - yte) * np.log(np.clip(1 - p, eps, 1)))
    print(f"  {name}: %{acc*100:.1f} (naive %{naive*100:.1f}) | Brier {brier:.4f} | logloss {ll_:.4f}")
    return acc

full = CAT_COLS + NUMERIC_FEATURES + ODDS_FEATURES
full_cat = [full.index(c) for c in CAT_COLS]
set_a = full + corner_feats
set_a_cat = [set_a.index(c) for c in CAT_COLS]

def fit_clf(fs, cidx, it, lr, d):
    X = train.loc[ok, fs]
    Xe = test[fs]
    m = CatBoostClassifier(loss_function="Logloss", cat_features=cidx, iterations=it,
                           learning_rate=lr, depth=d, verbose=False, allow_writing_files=False)
    m.fit(X, yt)
    return report(f"CatBoost {fs[0][:3]}.. lr={lr} d={d} it={it}", m.predict_proba(Xe)[:, 1])

print("\n=== Tam feature seti (85+odds) ===")
fit_clf(full, full_cat, 400, 0.03, 8)
print("\n=== Tam feature + korner feature'lar ===")
fit_clf(set_a, set_a_cat, 400, 0.03, 8)
fit_clf(set_a, set_a_cat, 500, 0.05, 6)

print("\n=== Poisson (toplam korner) ===")
X = train.loc[ok, set_a]
Xe = test[set_a]
r = CatBoostRegressor(loss_function="Poisson", cat_features=set_a_cat, iterations=400,
                      learning_rate=0.05, depth=5, verbose=False, allow_writing_files=False)
r.fit(X, train.loc[ok, "tc"].to_numpy())
lam = np.maximum(r.predict(Xe), 0.1)
from math import factorial
def pois_p(lam, k):
    return lam ** k / factorial(k) * np.exp(-lam)
p = np.array([1 - sum(pois_p(lam[i], k) for k in range(10)) for i in range(len(lam))])
report("Poisson toplam korner", p)
print(f"  ort lam: {lam.mean():.1f} (gercek ort {test.tc.mean():.1f})")

print("\n=== Poisson ev+deplasman ayri ayri ===")
for tgt, name in [("home_corners", "ev"), ("away_corners", "dep")]:
    ok2 = train[tgt].notna()
    r = CatBoostRegressor(loss_function="Poisson", cat_features=set_a_cat, iterations=400,
                          learning_rate=0.05, depth=5, verbose=False, allow_writing_files=False)
    r.fit(train.loc[ok2, set_a], train.loc[ok2, tgt].to_numpy())
    if tgt == "home_corners":
        lam_h = np.maximum(r.predict(Xe), 0.1)
    else:
        lam_a = np.maximum(r.predict(Xe), 0.1)
lam = lam_h + lam_a
p = np.array([1 - sum(pois_p(lam[i], k) for k in range(10)) for i in range(len(lam))])
report("Poisson ev+dep (toplam)", p)
print(f"  ort lam ev {lam_h.mean():.1f} dep {lam_a.mean():.1f} (gercek {test.home_corners.mean():.1f}/{test.away_corners.mean():.1f})")
print(f"\nToplam sure: {time.time()-t0:.0f} sn")