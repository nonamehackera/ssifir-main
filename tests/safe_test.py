"""SAF TEST - Sadece mevcut feature ile, gurultu yok.
Her market icin net dogruluk raporu."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score, f1_score, mean_absolute_error

print("=" * 80)
print("  SAF DOGRULUK TESTI - SADECE MEVCUT FEATURE'LARLA")
print("=" * 80)

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]: feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

SKIP = {
    "match_id", "league", "season", "date", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "result", "result_H", "result_D", "result_A",
    "btts", "over25", "total_goals",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_xg", "away_xg",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "referee", "data_completeness",
    "home_xg_real", "away_xg_real", "total_xg_real", "xg_diff_real",
    "ht_home_goals", "ht_away_goals", "ht_total_goals",
    "ht_result_is_draw", "ht_home_leading", "ht_second_half_goals_expected",
    "second_half_goals",
    "home_score_prob", "away_score_prob", "btts_xprob", "over25_xprob",
    "xg_diff_abs", "draw_xprob",
    "avg_home_odds", "avg_draw_odds", "avg_away_odds",
    "avg_over25_odds", "avg_close_home_odds", "avg_close_draw_odds",
    "avg_close_away_odds", "avg_close_over25_odds",
    "mkt_close_home_prob", "mkt_close_draw_prob", "mkt_close_away_prob",
    "mkt_close_over25_prob", "shots_diff", "sot_diff",
}
FEATS = [c for c in feat.columns if c not in SKIP and feat[c].dtype in ["float64", "int64", "float32", "int32"]]

# 5-fold walk-forward (eskisiyle ayni parametreler)
N_FOLDS = 5
TEST_SIZE = 15000
MIN_TRAIN = 200000
STEP = 15000
folds = []
start = 0
total = len(feat)
while start + MIN_TRAIN + TEST_SIZE <= total:
    folds.append((start, start + MIN_TRAIN, start + MIN_TRAIN, start + MIN_TRAIN + TEST_SIZE))
    start += STEP

results_all = []

for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds[:N_FOLDS]):
    train = feat.iloc[tr_s:tr_e].copy()
    test = feat.iloc[te_s:te_e].copy()
    X_all = train[FEATS].fillna(0)
    X_test = test[FEATS].fillna(0)
    split = int(len(X_all) * 0.85)

    print(f"\n=== Fold {fold_i+1}/{N_FOLDS} ({test['date'].iloc[0].date()} ~ {test['date'].iloc[-1].date()}) ===")

    # 1X2
    y_1x2 = train["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    y_t_1x2 = test["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,
        learning_rate=0.015,n_estimators=500,max_depth=6,min_child_samples=100,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=42,verbose=-1,n_jobs=-1)
    m.fit(X_all.iloc[:split], y_1x2[:split])
    raw = m.predict_proba(X_all.iloc[split:])
    cal = [IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y_1x2[split:]==c).astype(float)) for c in range(3)]
    p = m.predict_proba(X_test)
    pc = np.column_stack([cal[c].predict(p[:,c]) for c in range(3)])
    s=pc.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); pc/=s
    pred = np.argmax(pc, axis=1)
    conf = np.max(pc, axis=1)

    ll = log_loss(y_t_1x2, pc)
    acc = accuracy_score(y_t_1x2, pred)
    f1 = f1_score(y_t_1x2, pred, average="macro")
    base_h = (y_t_1x2==0).mean()

    # Eşik bazli
    accs = {}
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = conf >= t
        n = mask.sum()
        if n > 10:
            accs[t] = (accuracy_score(y_t_1x2[mask], pred[mask]), n)

    print(f"  1X2: LL={ll:.4f} Acc={acc:.3f} F1={f1:.3f} base_H={base_h:.3f}")
    for t, (a, n) in accs.items():
        print(f"    @{t:.0%}: {a:.3f} ({n} picks)")

    results_all.append({"fold": fold_i+1, "market": "1X2", "ll": ll, "acc": acc, "f1": f1,
                         "base_h": base_h, "n": len(test)})

    # BTTS
    y_btts = train["btts"].to_numpy()
    y_t_btts = test["btts"].to_numpy()
    mb = lgb.LGBMClassifier(objective="binary",num_leaves=45,
        learning_rate=0.015,n_estimators=500,max_depth=6,min_child_samples=80,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=42,verbose=-1,n_jobs=-1)
    mb.fit(X_all.iloc[:split], y_btts[:split])
    raw_b = mb.predict_proba(X_all.iloc[split:])[:,1]
    cal_b = IsotonicRegression(out_of_bounds="clip").fit(raw_b, y_btts[split:].astype(float))
    pb = cal_b.predict(mb.predict_proba(X_test)[:,1])
    pred_b = (pb > 0.5).astype(int)
    ll_b = log_loss(y_t_btts, np.column_stack([1-pb, pb]))
    try: auc_b = roc_auc_score(y_t_btts, pb)
    except: auc_b = 0.5
    acc_b = accuracy_score(y_t_btts, pred_b)

    print(f"  BTTS: LL={ll_b:.4f} AUC={auc_b:.4f} Acc={acc_b:.3f} base={y_t_btts.mean():.3f}")
    results_all.append({"fold": fold_i+1, "market": "BTTS", "ll": ll_b, "auc": auc_b,
                         "acc": acc_b, "n": len(test)})

    # Over 2.5
    y_o25 = train["over25"].to_numpy()
    y_t_o25 = test["over25"].to_numpy()
    mo = lgb.LGBMClassifier(objective="binary",num_leaves=45,
        learning_rate=0.015,n_estimators=500,max_depth=6,min_child_samples=80,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=42,verbose=-1,n_jobs=-1)
    mo.fit(X_all.iloc[:split], y_o25[:split])
    raw_o = mo.predict_proba(X_all.iloc[split:])[:,1]
    cal_o = IsotonicRegression(out_of_bounds="clip").fit(raw_o, y_o25[split:].astype(float))
    po = cal_o.predict(mo.predict_proba(X_test)[:,1])
    pred_o = (po > 0.5).astype(int)
    ll_o = log_loss(y_t_o25, np.column_stack([1-po, po]))
    try: auc_o = roc_auc_score(y_t_o25, po)
    except: auc_o = 0.5
    acc_o = accuracy_score(y_t_o25, pred_o)

    print(f"  O25:  LL={ll_o:.4f} AUC={auc_o:.4f} Acc={acc_o:.3f} base={y_t_o25.mean():.3f}")
    results_all.append({"fold": fold_i+1, "market": "Over2.5", "ll": ll_o, "auc": auc_o,
                         "acc": acc_o, "n": len(test)})

    # Gol
    y_hg = train["home_goals"].to_numpy()
    y_ag = train["away_goals"].to_numpy()
    mhg = lgb.LGBMRegressor(objective="poisson",num_leaves=45,
        learning_rate=0.015,n_estimators=400,max_depth=6,min_child_samples=80,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=42,verbose=-1,n_jobs=-1)
    mhg.fit(X_all.iloc[:split], y_hg[:split])
    phg = np.maximum(mhg.predict(X_test), 0.05)
    mag = lgb.LGBMRegressor(objective="poisson",num_leaves=45,
        learning_rate=0.015,n_estimators=400,max_depth=6,min_child_samples=80,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=42,verbose=-1,n_jobs=-1)
    mag.fit(X_all.iloc[:split], y_ag[:split])
    pag = np.maximum(mag.predict(X_test), 0.05)
    y_t_hg = test["home_goals"].to_numpy()
    y_t_ag = test["away_goals"].to_numpy()
    mae_t = mean_absolute_error(y_t_hg+y_t_ag, phg+pag)
    naive = train["total_goals"].mean()
    mae_n = mean_absolute_error(y_t_hg+y_t_ag, np.full(len(y_t_hg), naive))

    print(f"  GOL:  MAE={mae_t:.4f} naive={mae_n:.4f} fark={mae_t-mae_n:+.4f}")
    results_all.append({"fold": fold_i+1, "market": "Goals", "mae": mae_t, "mae_naive": mae_n, "n": len(test)})

# OZET
print("\n" + "=" * 80)
print("  5-FOLD ORTALAMA SONUCLAR")
print("=" * 80)

df = pd.DataFrame(results_all)
for mkt in ["1X2", "BTTS", "Over2.5", "Goals"]:
    sub = df[df["market"] == mkt]
    if len(sub) == 0: continue
    print(f"\n  {mkt}:")
    for col in sub.columns:
        if col in ("fold", "market"): continue
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"    {col:15s}: ort={vals.mean():.4f}")

# Final gercek degerlendirme
print("\n" + "=" * 80)
print("  GERCEK DEGERLENDIRME - MODeller CALISIYOR MU?")
print("=" * 80)

sub_1x2 = df[df["market"]=="1X2"]
sub_btts = df[df["market"]=="BTTS"]
sub_o25 = df[df["market"]=="Over2.5"]
sub_goals = df[df["market"]=="Goals"]

issues = []

# 1X2
ll_mean = sub_1x2["ll"].mean()
acc_mean = sub_1x2["acc"].mean()
if ll_mean < np.log(3) * 0.95:
    print(f"  1X2: LogLoss {ll_mean:.4f} < random*0.95 ({np.log(3)*0.95:.4f}) - IYI")
elif ll_mean < np.log(3):
    print(f"  1X2: LogLoss {ll_mean:.4f} < random ({np.log(3):.4f}) - ZAYIF AMA CALISIYOR")
else:
    print(f"  1X2: LogLoss {ll_mean:.4f} >= random - CALISMIYOR")
    issues.append("1X2")

if acc_mean < 0.45:
    print(f"  1X2: Accuracy {acc_mean:.3f} < 0.45 - ZAYIF")

# BTTS
auc_btts_mean = sub_btts["auc"].mean()
if auc_btts_mean > 0.55:
    print(f"  BTTS: AUC {auc_btts_mean:.3f} > 0.55 - CALISIYOR")
elif auc_btts_mean > 0.52:
    print(f"  BTTS: AUC {auc_btts_mean:.3f} 0.52-0.55 - NAIF USTU")
else:
    print(f"  BTTS: AUC {auc_btts_mean:.3f} <= 0.52 - CALISMIYOR")
    issues.append("BTTS")

# Over 2.5
auc_o25_mean = sub_o25["auc"].mean()
if auc_o25_mean > 0.60:
    print(f"  O25:  AUC {auc_o25_mean:.3f} > 0.60 - EN IYI MARKET")
elif auc_o25_mean > 0.55:
    print(f"  O25:  AUC {auc_o25_mean:.3f} 0.55-0.60 - ZAYIF")
else:
    print(f"  O25:  AUC {auc_o25_mean:.3f} < 0.55 - CALISMIYOR")
    issues.append("O25")

# Gol
mae_mean = sub_goals["mae"].mean()
mae_n_mean = sub_goals["mae_naive"].mean()
if mae_mean < mae_n_mean * 0.97:
    print(f"  GOL:  MAE {mae_mean:.4f} < naive*0.97 ({mae_n_mean*0.97:.4f}) - IYI")
elif mae_mean < mae_n_mean:
    print(f"  GOL:  MAE {mae_mean:.4f} < naive ({mae_n_mean:.4f}) - ZAYIF")
else:
    print(f"  GOL:  MAE {mae_mean:.4f} >= naive - CALISMIYOR")
    issues.append("GOL")

if not issues:
    print(f"\n  TUM MARKETLER CALISIYOR (ama zayif)")
else:
    print(f"\n  {len(issues)} MARKET CALISMIYOR: {', '.join(issues)}")

print("\n" + "=" * 80)
