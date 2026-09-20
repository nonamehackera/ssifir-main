"""
train_v8.py — SIFIRDAN YENI EGITIM
- Walk-forward ensemble (2y train, 3mo test)
- Sadece prediction-time'da mevcut feature'lar (odds, xG, ht yok)
- Proper isotonic calibration per fold
- Ensemble of 2 LGBM models per market
- Diger piyasalar icin ayri model (over35, under25, double chance, corner)
- Web model.pkl olarak kaydet
"""
import sys, os, time, pickle, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson

t0 = time.time()

# ─────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────
print("=" * 72)
print("ADIM 1: Veri yukleniyor...")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold")
feat = pd.read_parquet(os.path.join(DATA, "features.parquet"))
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)

for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
feat["over35"] = (feat["total_goals"] > 3.5).astype(int)
feat["under25"] = 1 - feat["over25"]
feat = feat.dropna(subset=["result", "elo_diff"])

hc = pd.to_numeric(feat.get("home_corners"), errors="coerce").fillna(0)
ac = pd.to_numeric(feat.get("away_corners"), errors="coerce").fillna(0)
feat["total_corners"] = hc + ac
feat["corner_over85"] = (feat["total_corners"] > 8.5).astype(int)
feat["corner_reliable"] = feat["total_corners"] >= 5

print(f"  Toplam: {len(feat)} mac | Tarih: {feat['date'].min().date()} -> {feat['date'].max().date()}")

# ─────────────────────────────────────────────────────────
# 2. FEATURES — Sadece prediction-time'da mevcut olanlar
# ─────────────────────────────────────────────────────────
# SECILENLER: ELO, form, league stats, h2h, rest days
# ELKLANANLAR: odds, xG, ht stats ( bunlar sadece egitimde var)
SAFE_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_attack_elo", "home_defence_elo", "away_attack_elo", "away_defence_elo",
    "attack_elo_diff", "defence_elo_diff",
    "home_gf_5", "home_ga_5", "home_pts_5",
    "away_gf_5", "away_ga_5", "away_pts_5",
    "home_gf_3", "home_ga_3",
    "away_gf_3", "away_ga_3",
    "home_gf_8", "home_ga_8",
    "away_gf_8", "away_ga_8",
    "home_gf_20", "home_ga_20",
    "away_gf_20", "away_ga_20",
    "home_hgf_5", "home_hga_5", "home_hpts_5",
    "away_agf_5", "away_aga_5", "away_apts_5",
    "home_w_gf", "home_w_ga", "home_w_shots", "home_w_sot",
    "away_w_gf", "away_w_ga", "away_w_shots", "away_w_sot",
    "home_momentum", "away_momentum",
    "home_wins_last5", "away_wins_last5",
    "home_gdiff5", "away_gdiff5",
    "home_rest_days", "away_rest_days",
    "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts",
    "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "lg_corner_avg",
]

# Diger formlar (form_diff, gf_diff vs)
feat["form_diff"] = feat.get("home_pts_5", 0).fillna(0) - feat.get("away_pts_5", 0).fillna(0)
feat["gf_diff"] = feat.get("home_gf_5", 0).fillna(0) - feat.get("away_gf_5", 0).fillna(0)
feat["ga_diff"] = feat.get("home_ga_5", 0).fillna(0) - feat.get("away_ga_5", 0).fillna(0)
feat["elo_x_hform"] = feat["elo_diff"] * feat.get("home_pts_5", 0).fillna(0)
feat["home_strong"] = (feat["elo_diff"] > 100).astype(int)
feat["away_strong"] = (feat["elo_diff"] < -100).astype(int)
feat["draw_likely"] = ((feat["elo_diff"].abs() < 50) & (feat.get("lg_draw_rate", 0.25).fillna(0.25) > 0.25)).astype(int)
feat["shots_diff_5"] = feat.get("home_shots_5", 0).fillna(0) - feat.get("away_shots_5", 0).fillna(0)
feat["sot_diff_5"] = feat.get("home_sot_5", 0).fillna(0) - feat.get("away_sot_5", 0).fillna(0)
feat["corner_diff_5"] = feat.get("home_corners_5", 0).fillna(0) - feat.get("away_corners_5", 0).fillna(0)
feat["home_pts_std"] = feat.get("home_pts_std", 0).fillna(0)
feat["away_pts_std"] = feat.get("away_pts_std", 0).fillna(0)
feat["data_completeness"] = feat.get("data_completeness", 0.5).fillna(0.5)

SAFE_FEATURES += [
    "form_diff", "gf_diff", "ga_diff", "elo_x_hform",
    "home_strong", "away_strong", "draw_likely",
    "shots_diff_5", "sot_diff_5", "corner_diff_5",
    "home_pts_std", "away_pts_std", "data_completeness",
]

# Kategorik encoder — deployment'da kullanilamaz, cikar
# for c in ["league", "home_team_id", "away_team_id", "season"]:
#     if c in feat.columns:
#         feat[c + "_enc"] = feat[c].astype("category").cat.codes
#         SAFE_FEATURES.append(c + "_enc")

F = [c for c in SAFE_FEATURES if c in feat.columns]
print(f"  Feature sayisi: {len(F)}")

# ─────────────────────────────────────────────────────────
# 3. WALK-FORWARD
# ─────────────────────────────────────────────────────────
WINDOWS = [
    ("2023-01-01", "2023-04-01", "2023-Q1"),
    ("2023-04-01", "2023-07-01", "2023-Q2"),
    ("2023-07-01", "2023-10-01", "2023-Q3"),
    ("2023-10-01", "2024-01-01", "2023-Q4"),
    ("2024-01-01", "2024-04-01", "2024-Q1"),
    ("2024-04-01", "2024-07-01", "2024-Q2"),
    ("2024-07-01", "2024-10-01", "2024-Q3"),
    ("2024-10-01", "2025-01-01", "2024-Q4"),
    ("2025-01-01", "2025-04-01", "2025-Q1"),
    ("2025-04-01", "2025-07-01", "2025-Q2"),
    ("2025-07-01", "2025-10-01", "2025-Q3"),
]

# Model params — dengeli, overfit azaltmak icin kucuk
# draw_weight=4.0 ile beraberlik sinifina ek agirlik
LC = dict(objective="multiclass", num_class=3, num_leaves=35, learning_rate=0.02,
          n_estimators=400, max_depth=5, min_child_samples=100,
          subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
          verbose=-1, random_state=42, class_weight={0: 1, 1: 4.0, 2: 1})
LB = dict(objective="binary", num_leaves=30, learning_rate=0.03,
          n_estimators=350, max_depth=4, min_child_samples=120,
          subsample=0.65, colsample_bytree=0.5, reg_alpha=1.5, reg_lambda=15.0,
          verbose=-1, random_state=99)


def calibrate_3class(raw_cal, raw_test, y_cal):
    """3-class isotonic calibration. Returns (calibrated_probs, fitted_calibrators)."""
    cal = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal[:, c], (y_cal == c).astype(float))
        cal.append(ir)
    cal_test = np.column_stack([cal[c].predict(raw_test[:, c]) for c in range(3)])
    s = cal_test.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    cal_test /= s
    return cal_test, cal


def calibrate_binary(raw_cal, raw_test, y_cal):
    """Binary isotonic calibration."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, y_cal.astype(float))
    return ir.predict(raw_test)


# ─────────────────────────────────────────────────────────
# 4. WALK-FORWARD LOOP
# ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("WALK-FORWARD ENSEMBLE (2xLGB, 2y train, 3mo test)")
print("=" * 72)

RESULTS = []
ALL_CAL_1X2 = []
ALL_PRED_1X2 = []
ALL_TRUE_1X2 = []

# Son 2 yil full-ensemble egitimi icin
FULL_TRAIN_CUTOFF = pd.Timestamp("2024-09-01")

for vs, ve, lb in WINDOWS:
    vs_, ve_ = pd.Timestamp(vs), pd.Timestamp(ve)
    tm = (feat["date"] >= vs_ - pd.DateOffset(years=2)) & (feat["date"] < vs_)
    vm = (feat["date"] >= vs_) & (feat["date"] < ve_)
    tr = feat[tm].copy()
    va = feat[vm].copy()
    if len(tr) < 5000 or len(va) < 100:
        print(f"  {lb}: atlandi (train={len(tr)}, test={len(va)})")
        continue

    yr = tr["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    yv = va["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    # TIME DECAY: yeni maclar daha onemli
    t_days = (tr["date"].max() - tr["date"]).dt.days.clip(lower=0)
    decay = np.exp(-1.0 * t_days / 365.0).values
    sample_weight = decay

    # Split train for calibration
    sp = int(len(tr) * 0.85)
    Xt, Xv_ = tr[F].fillna(0).iloc[:sp], tr[F].fillna(0).iloc[sp:]
    Xte = va[F].fillna(0)

    # 1X2 Ensemble
    mA = lgb.LGBMClassifier(**LC)
    mA.fit(Xt, yr[:sp], sample_weight=sample_weight[:sp])
    mB = lgb.LGBMClassifier(**{**LC, "random_state": 99})
    mB.fit(Xt, yr[:sp], sample_weight=sample_weight[:sp])

    pA, _ = calibrate_3class(mA.predict_proba(Xv_), mA.predict_proba(Xte), yr[sp:])
    pB, _ = calibrate_3class(mB.predict_proba(Xv_), mB.predict_proba(Xte), yr[sp:])
    p1x2 = 0.5 * pA + 0.5 * pB
    s = p1x2.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    p1x2 /= s

    pk = np.argmax(p1x2, axis=1)
    a1 = (pk == yv).mean()
    conf = np.max(p1x2, axis=1)

    # Double chance
    csp = np.column_stack([p1x2[:, 0] + p1x2[:, 1], p1x2[:, 1] + p1x2[:, 2], p1x2[:, 0] + p1x2[:, 2]])
    csy = np.column_stack([(yv == 0) | (yv == 1), (yv == 1) | (yv == 2), (yv == 0) | (yv == 2)])
    acs = csy[np.arange(len(yv)), np.argmax(csp, axis=1)].mean()

    # BTTS
    bA = lgb.LGBMClassifier(**LB)
    bA.fit(Xt, tr["btts"].values[:sp], sample_weight=sample_weight[:sp])
    bB = lgb.LGBMClassifier(**{**LB, "random_state": 42})
    bB.fit(Xt, tr["btts"].values[:sp], sample_weight=sample_weight[:sp])
    bp = 0.5 * calibrate_binary(bA.predict_proba(Xv_)[:, 1], bA.predict_proba(Xte)[:, 1], tr["btts"].values[sp:]) + \
         0.5 * calibrate_binary(bB.predict_proba(Xv_)[:, 1], bB.predict_proba(Xte)[:, 1], tr["btts"].values[sp:])
    abt = ((bp > 0.5).astype(int) == va["btts"].values).mean()

    # Over 2.5
    oA = lgb.LGBMClassifier(**LB)
    oA.fit(Xt, tr["over25"].values[:sp], sample_weight=sample_weight[:sp])
    oB = lgb.LGBMClassifier(**{**LB, "random_state": 42})
    oB.fit(Xt, tr["over25"].values[:sp], sample_weight=sample_weight[:sp])
    op25 = 0.5 * calibrate_binary(oA.predict_proba(Xv_)[:, 1], oA.predict_proba(Xte)[:, 1], tr["over25"].values[sp:]) + \
           0.5 * calibrate_binary(oB.predict_proba(Xv_)[:, 1], oB.predict_proba(Xte)[:, 1], tr["over25"].values[sp:])
    ao2 = ((op25 > 0.5).astype(int) == va["over25"].values).mean()

    # Over 1.5
    o1m = lgb.LGBMClassifier(**LB)
    o1m.fit(Xt, tr["over15"].values[:sp], sample_weight=sample_weight[:sp])
    o1p = calibrate_binary(o1m.predict_proba(Xv_)[:, 1], o1m.predict_proba(Xte)[:, 1], tr["over15"].values[sp:])
    ao1 = ((o1p > 0.5).astype(int) == va["over15"].values).mean()

    # Over 3.5
    o3m = lgb.LGBMClassifier(**LB)
    o3m.fit(Xt, tr["over35"].values[:sp], sample_weight=sample_weight[:sp])
    o3p = calibrate_binary(o3m.predict_proba(Xv_)[:, 1], o3m.predict_proba(Xte)[:, 1], tr["over35"].values[sp:])
    ao3 = ((o3p > 0.5).astype(int) == va["over35"].values).mean()

    # Corner (Poisson)
    ctr = tr["corner_reliable"].values.astype(bool)
    cte = va["corner_reliable"].values.astype(bool)
    NF = [c for c in F if "_enc" not in c]
    cpr = np.full(len(va), 0.5)
    if ctr.sum() > 500 and cte.sum() > 10:
        m5 = lgb.LGBMRegressor(objective="poisson", num_leaves=25, learning_rate=0.02,
                                n_estimators=300, max_depth=4, min_child_samples=50,
                                subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
                                verbose=-1, random_state=42)
        m5.fit(tr.loc[ctr, NF].fillna(0), tr.loc[ctr, "total_corners"].values.astype(float))
        cor_lam = np.maximum(m5.predict(va.loc[cte, NF].fillna(0)), 0.5)
        cpr[cte] = 1 - poisson.cdf(8, cor_lam)

    acr = ((cpr[cte] > 0.5).astype(int) == va.loc[cte, "corner_over85"].values).mean() if cte.sum() > 10 else None

    ll = -np.mean(np.log(np.clip(p1x2[np.arange(len(yv)), yv], 1e-9, None)))

    R = {"lb": lb, "n": len(va), "a1": a1, "acs": acs, "abt": abt, "ao1": ao1, "ao2": ao2, "ao3": ao3,
         "ll": ll, "cn": int(cte.sum()), "acr": acr}
    RESULTS.append(R)
    ALL_CAL_1X2.append(conf)
    ALL_PRED_1X2.append(pk)
    ALL_TRUE_1X2.append(yv)

    print(f"  {lb}: n={len(va):>5} 1X2={a1*100:.1f}% CS={acs*100:.1f}% BT={abt*100:.1f}% "
          f"O15={ao1*100:.1f}% O25={ao2*100:.1f}% O35={ao3*100:.1f}% "
          f"COR={'%.1f%%'%(acr*100) if acr else 'N/A':>5}")

# ─────────────────────────────────────────────────────────
# 5. GENEL SONUCLAR
# ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("GENEL SONUCLAR (WF Agirlikli Ort)")
print("=" * 72)
df = pd.DataFrame(RESULTS)
w = df["n"] / df["n"].sum()
for co, nm in [("a1", "ENSEMBLE 1X2"), ("acs", "Cifte Sans"), ("abt", "BTTS"),
               ("ao1", "Ust 1.5"), ("ao2", "Ust 2.5"), ("ao3", "Ust 3.5")]:
    v = (df[co] * w).sum()
    print(f"  {nm:22s}: {v*100:>5.1f}%  (min={df[co].min()*100:.1f}% max={df[co].max()*100:.1f}%)")
cr = df[df["cn"] > 10]
if len(cr) > 0:
    cw = cr["cn"] / cr["cn"].sum()
    vc = (cr["acr"] * cw).sum()
    print(f"  {'Korner Ust 8.5':22s}: {vc*100:>5.1f}%  (n={int(cr['cn'].sum())})")
print(f"\n  Logloss ort: {df['ll'].mean():.4f}")

# Kalibrasyon ozeti
ac2 = np.concatenate(ALL_CAL_1X2)
ap2 = np.concatenate(ALL_PRED_1X2)
ay2 = np.concatenate(ALL_TRUE_1X2)
print(f"\n  Kalibrasyon (1X2):")
for lo in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    hi = lo + 0.05
    msk = (ac2 >= lo) & (ac2 < hi)
    if msk.sum() > 10:
        r = (ap2[msk] == ay2[msk]).mean()
        print(f"    {lo*100:.0f}-{hi*100:.0f}%  n={msk.sum():>5}  gercek={r*100:.1f}%  model={ac2[msk].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────
# 6. FINAL MODEL — son 2 yil tum veri ile
# ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("FINAL MODEL EGITIMI (son 2 yil, tum veri)")
print("=" * 72)

final_tr = feat[feat["date"] >= FULL_TRAIN_CUTOFF].copy()
print(f"  Final train: {len(final_tr)} mac")

final_tr = final_tr.dropna(subset=["result"])
for c in F:
    final_tr[c] = final_tr[c].fillna(0)

X_final = final_tr[F]

# TIME DECAY: yeni maclar daha onemli
t_days_f = (final_tr["date"].max() - final_tr["date"]).dt.days.clip(lower=0)
decay_f = np.exp(-1.0 * t_days_f / 365.0).values

# 1X2
y_final = final_tr["result"].map({"H": 0, "D": 1, "A": 2}).values
sp = int(len(X_final) * 0.85)
Xt_f, Xv_f = X_final.iloc[:sp], X_final.iloc[sp:]

m1x2_a = lgb.LGBMClassifier(**LC)
m1x2_a.fit(Xt_f, y_final[:sp], sample_weight=decay_f[:sp])
m1x2_b = lgb.LGBMClassifier(**{**LC, "random_state": 99})
m1x2_b.fit(Xt_f, y_final[:sp], sample_weight=decay_f[:sp])

# Calibration sets
p_cal_a = m1x2_a.predict_proba(Xv_f)
p_cal_b = m1x2_b.predict_proba(Xv_f)

# BTTS
y_btts = final_tr["btts"].values
mbtts_a = lgb.LGBMClassifier(**LB)
mbtts_a.fit(Xt_f, y_btts[:sp], sample_weight=decay_f[:sp])
mbtts_b = lgb.LGBMClassifier(**{**LB, "random_state": 42})
mbtts_b.fit(Xt_f, y_btts[:sp], sample_weight=decay_f[:sp])

# Over 2.5
y_o25 = final_tr["over25"].values
mo25_a = lgb.LGBMClassifier(**LB)
mo25_a.fit(Xt_f, y_o25[:sp], sample_weight=decay_f[:sp])
mo25_b = lgb.LGBMClassifier(**{**LB, "random_state": 42})
mo25_b.fit(Xt_f, y_o25[:sp], sample_weight=decay_f[:sp])

# Over 1.5
y_o15 = final_tr["over15"].values
mo15 = lgb.LGBMClassifier(**LB)
mo15.fit(Xt_f, y_o15[:sp], sample_weight=decay_f[:sp])

# Over 3.5
y_o35 = final_tr["over35"].values
mo35 = lgb.LGBMClassifier(**LB)
mo35.fit(Xt_f, y_o35[:sp], sample_weight=decay_f[:sp])

# Under 2.5
y_u25 = final_tr["under25"].values
mu25 = lgb.LGBMClassifier(**LB)
mu25.fit(Xt_f, y_u25[:sp], sample_weight=decay_f[:sp])

# Double chance
dcm = {}
for name, vals in [("1X", [0, 1]), ("X2", [1, 2]), ("12", [0, 2])]:
    y_dc = final_tr["result"].isin(vals).astype(int).values
    m = lgb.LGBMClassifier(**LB)
    m.fit(Xt_f, y_dc[:sp], sample_weight=decay_f[:sp])
    dcm[name] = m

# Corner Poisson
NF = [c for c in F if "_enc" not in c]
ct_mask = final_tr["corner_reliable"].values.astype(bool)
m_corner = None
if ct_mask.sum() > 500:
    m_corner = lgb.LGBMRegressor(objective="poisson", num_leaves=25, learning_rate=0.02,
                                  n_estimators=300, max_depth=4, min_child_samples=50,
                                  subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0,
                                  verbose=-1, random_state=42)
    m_corner.fit(final_tr.loc[ct_mask, NF].fillna(0), final_tr.loc[ct_mask, "total_corners"].values.astype(float),
                 sample_weight=decay_f[ct_mask])
    print(f"  Corner model: {ct_mask.sum()}可靠 veri")

# Poisson regressor (home/away goals)
mh_pois = lgb.LGBMRegressor(objective="poisson", num_leaves=30, learning_rate=0.02,
                             n_estimators=400, max_depth=5, min_child_samples=50,
                             subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0,
                             verbose=-1, random_state=42)
mh_pois.fit(Xt_f, final_tr["home_goals"].values[:sp].astype(float), sample_weight=decay_f[:sp])
ma_pois = lgb.LGBMRegressor(objective="poisson", num_leaves=30, learning_rate=0.02,
                             n_estimators=400, max_depth=5, min_child_samples=50,
                             subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0,
                             verbose=-1, random_state=99)
ma_pois.fit(Xt_f, final_tr["away_goals"].values[:sp].astype(float), sample_weight=decay_f[:sp])

print("  Modeller egitildi.")

# ─────────────────────────────────────────────────────────
# 7. FINAL TEST — son 3 ay
# ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("FINAL TEST (son 3 ay, en guncel veri)")
print("=" * 72)

test_start = pd.Timestamp("2026-06-01")
test_end = pd.Timestamp("2026-09-01")
test_mask = (feat["date"] >= test_start) & (feat["date"] < test_end)
test_data = feat[test_mask].copy()
print(f"  Test: {len(test_data)} mac ({test_data['date'].min().date()} -> {test_data['date'].max().date()})")

if len(test_data) > 0:
    X_test = test_data[F].fillna(0)

    # 1X2
    p_a = m1x2_a.predict_proba(X_test)
    p_b = m1x2_b.predict_proba(X_test)
    # Use validation calibration from final training
    pA_cal, cal_1x2_a = calibrate_3class(p_cal_a, p_a, y_final[sp:])
    pB_cal, cal_1x2_b = calibrate_3class(p_cal_b, p_b, y_final[sp:])
    p_final = 0.5 * pA_cal + 0.5 * pB_cal
    s = p_final.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    p_final /= s
    pk_f = np.argmax(p_final, axis=1)
    y_test = test_data["result"].map({"H": 0, "D": 1, "A": 2}).values
    acc_1x2 = (pk_f == y_test).mean()
    conf_f = np.max(p_final, axis=1)

    # BTTS
    bt_a = calibrate_binary(mbtts_a.predict_proba(Xv_f)[:, 1], mbtts_a.predict_proba(X_test)[:, 1], y_btts[sp:])
    bt_b = calibrate_binary(mbtts_b.predict_proba(Xv_f)[:, 1], mbtts_b.predict_proba(X_test)[:, 1], y_btts[sp:])
    bt_p = 0.5 * bt_a + 0.5 * bt_b
    acc_btts = ((bt_p > 0.5).astype(int) == test_data["btts"].values).mean()

    # Over 2.5
    o25_a = calibrate_binary(mo25_a.predict_proba(Xv_f)[:, 1], mo25_a.predict_proba(X_test)[:, 1], y_o25[sp:])
    o25_b = calibrate_binary(mo25_b.predict_proba(Xv_f)[:, 1], mo25_b.predict_proba(X_test)[:, 1], y_o25[sp:])
    o25_p = 0.5 * o25_a + 0.5 * o25_b
    acc_o25 = ((o25_p > 0.5).astype(int) == test_data["over25"].values).mean()

    # Over 1.5
    o15_p = calibrate_binary(mo15.predict_proba(Xv_f)[:, 1], mo15.predict_proba(X_test)[:, 1], y_o15[sp:])
    acc_o15 = ((o15_p > 0.5).astype(int) == test_data["over15"].values).mean()

    # Over 3.5
    o35_p = calibrate_binary(mo35.predict_proba(Xv_f)[:, 1], mo35.predict_proba(X_test)[:, 1], y_o35[sp:])
    acc_o35 = ((o35_p > 0.5).astype(int) == test_data["over35"].values).mean()

    # Logloss
    ll_f = -np.mean(np.log(np.clip(p_final[np.arange(len(y_test)), y_test], 1e-9, None)))

    print(f"  1X2:       {acc_1x2*100:.1f}%  (logloss={ll_f:.4f})")
    print(f"  BTTS:      {acc_btts*100:.1f}%")
    print(f"  O1.5:      {acc_o15*100:.1f}%")
    print(f"  O2.5:      {acc_o25*100:.1f}%")
    print(f"  O3.5:      {acc_o35*100:.1f}%")

    # Confidence breakdown
    print(f"\n  1X2 Kalibrasyon (final test):")
    for lo in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        hi = lo + 0.05
        msk = (conf_f >= lo) & (conf_f < hi)
        if msk.sum() > 5:
            r = (pk_f[msk] == y_test[msk]).mean()
            print(f"    {lo*100:.0f}-{hi*100:.0f}%  n={msk.sum():>4}  gercek={r*100:.1f}%  model={conf_f[msk].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────
# 8. SAVE — web model.pkl
# ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("KAYDET")
print("=" * 72)

model_data = {
    "m_1x2": (m1x2_a, m1x2_b, cal_1x2_a, cal_1x2_b),
    "m_btts": (mbtts_a, mbtts_b),
    "m_over25": (mo25_a, mo25_b),
    "m_over15": mo15,
    "m_over35": mo35,
    "m_under25": mu25,
    "m_1x": dcm.get("1X"),
    "m_x2": dcm.get("X2"),
    "m_12": dcm.get("12"),
    "m_corner_poisson": m_corner,
    "m_home_goals": mh_pois,
    "m_away_goals": ma_pois,
    "features": F,
    "version": "v8.1",
    "train_date": str(pd.Timestamp.now().date()),
    "train_n": len(final_tr),
    "draw_weight": 4.0,
    "time_decay": 1.0,
    "no_leak": True,
}

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(model_data, f)
print(f"  Kaydedildi: {out_path}")
print(f"  Sure: {time.time()-t0:.0f}s")
