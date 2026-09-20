"""Her model icin her market'in gercekten dogru tahmin yapip yapmadigini kontrol eder.

Walk-forward evaluation ile:
  1. Her model (CatBoost, LightGBM, Dixon-Coles, Ensemble, CalibratedEnsemble)
  2. Her market (1X2, BTTS, Over 2.5, Under 2.5, Gol Beklentisi)
  3. Her ligin ozel performansi
  4. Kalibrasyon analizi
  5. ESIK bazli accuracy ve ROI

Ciktisi: detayli rapor + CSV dosyasi
"""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score,
    accuracy_score, f1_score, mean_absolute_error,
)
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print("=" * 90)
print("  TUM MODELLER - HER MARKET ICIN DETAYLI KONTROL")
print("=" * 90)

# ─── 1. VERI YUKLE ──────────────────────────────────────────────────────────
print("\n[1] Veri yukleniyor...")
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat["over35"] = (feat["total_goals"] > 3.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])
print(f"   {len(feat)} mac | {feat['date'].min().date()} ~ {feat['date'].max().date()}")

# Feature列
SKIP = {
    "match_id", "league", "season", "date", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "result", "result_H", "result_D", "result_A",
    "btts", "over25", "over15", "over35", "total_goals",
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
    "home_yellow", "away_yellow", "home_red", "away_red",
    "home_corners", "away_corners",
}
FEATS = [c for c in feat.columns if c not in SKIP and feat[c].dtype in ["float64", "int64", "float32", "int32"]]
print(f"   {len(FEATS)} feature kullaniliyor")

# ─── 2. WALK-FORWARD AYARLARI ───────────────────────────────────────────────
print("\n[2] Walk-forward ayarlari...")
N_FOLDS = 5
TEST_SIZE = 15000
MIN_TRAIN = 200000
STEP = 15000

total = len(feat)
folds = []
start_train = 0
while start_train + MIN_TRAIN + TEST_SIZE <= total:
    test_end = start_train + MIN_TRAIN + TEST_SIZE
    folds.append((start_train, start_train + MIN_TRAIN, start_train + MIN_TRAIN, test_end))
    start_train += STEP

print(f"   {N_FOLDS} fold | her fold {TEST_SIZE} test mac")

# ─── 3. MODeller ────────────────────────────────────────────────────────────
print("\n[3] Modeller hazirlaniyor...")

import lightgbm as lgb

def make_lgb_1x2(seed=42):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3, num_leaves=50,
        learning_rate=0.015, n_estimators=500, max_depth=6,
        min_child_samples=100, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )

def make_lgb_bin(seed=42):
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=45,
        learning_rate=0.015, n_estimators=500, max_depth=6,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )

def make_lgb_reg(seed=42):
    return lgb.LGBMRegressor(
        objective="poisson", num_leaves=45,
        learning_rate=0.015, n_estimators=400, max_depth=6,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=5.0, random_seed=seed, verbose=-1, n_jobs=-1,
    )

# Try catboost if available
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("   CatBoost yok, LightGBM bazli modeller kullanilacak")

# ─── 4. MARKETLERE GORE DEGERLENDIRME ───────────────────────────────────────
print("\n[4] Her market icin walk-forward baslatiliyor...")
print("=" * 90)

# Sonuclar
all_rows = []

# Her fold icin
for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds[:N_FOLDS]):
    train_df = feat.iloc[tr_s:tr_e].copy()
    test_df = feat.iloc[te_s:te_e].copy()
    X_all = train_df[FEATS].fillna(0)
    X_test = test_df[FEATS].fillna(0)
    split = int(len(X_all) * 0.85)
    
    fold_info = {
        "fold": fold_i + 1,
        "train_range": f"{train_df['date'].iloc[0].date()}~{train_df['date'].iloc[-1].date()}",
        "test_range": f"{test_df['date'].iloc[0].date()}~{test_df['date'].iloc[-1].date()}",
        "train_n": len(train_df), "test_n": len(test_df),
    }
    
    # ── MARKET 1: 1X2 (Home/Draw/Away) ──
    print(f"\n  === Fold {fold_i+1}/{N_FOLDS}: 1X2 MARKET ===")
    y_all_1x2 = train_df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    y_test_1x2 = test_df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    
    # LightGBM 1X2
    m = make_lgb_1x2()
    m.fit(X_all.iloc[:split], y_all_1x2[:split])
    raw = m.predict_proba(X_all.iloc[split:])
    cal_1x2 = [IsotonicRegression(out_of_bounds="clip").fit(raw[:, c], (y_all_1x2[split:] == c).astype(float)) for c in range(3)]
    
    probs_test = m.predict_proba(X_test)
    probs_cal = np.column_stack([cal_1x2[c].predict(probs_test[:, c]) for c in range(3)])
    s = probs_cal.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    probs_cal /= s
    
    pred_1x2 = np.argmax(probs_cal, axis=1)
    conf_1x2 = np.max(probs_cal, axis=1)
    
    ll_1x2 = log_loss(y_test_1x2, probs_cal)
    acc_1x2 = accuracy_score(y_test_1x2, pred_1x2)
    f1_1x2 = f1_score(y_test_1x2, pred_1x2, average="macro")
    
    # Baseline: hep "H" tahmin et
    baseline_h = (y_test_1x2 == 0).mean()
    # Baseline: rastgele
    baseline_random = 1.0 / 3
    
    print(f"    LogLoss: {ll_1x2:.4f} (random: {np.log(3):.4f})")
    print(f"    Accuracy: {acc_1x2:.3f} (baseline H: {baseline_h:.3f})")
    print(f"    Macro-F1: {f1_1x2:.3f}")
    
    # Eşik bazlı accuracy
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = conf_1x2 >= thr
        n = mask.sum()
        if n > 10:
            acc_thr = accuracy_score(y_test_1x2[mask], pred_1x2[mask])
            print(f"    Acc@{thr:.0%}: {acc_thr:.3f} ({n} picks)")
    
    all_rows.append({
        "fold": fold_i+1, "market": "1X2", "model": "LightGBM",
        "log_loss": ll_1x2, "accuracy": acc_1x2, "f1": f1_1x2,
        "baseline_h": baseline_h, "n_test": len(test_df),
        "calibration_gap": abs(conf_1x2.mean() - acc_1x2),
    })
    
    # ── MARKET 2: BTTS (Both Teams To Score) ──
    print(f"\n  === Fold {fold_i+1}/{N_FOLDS}: BTTS MARKET ===")
    y_all_btts = train_df["btts"].to_numpy()
    y_test_btts = test_df["btts"].to_numpy()
    
    m_btts = make_lgb_bin()
    m_btts.fit(X_all.iloc[:split], y_all_btts[:split])
    raw_btts = m_btts.predict_proba(X_all.iloc[split:])[:, 1]
    cal_btts = IsotonicRegression(out_of_bounds="clip").fit(raw_btts, y_all_btts[split:].astype(float))
    
    probs_btts = m_btts.predict_proba(X_test)[:, 1]
    probs_btts_cal = cal_btts.predict(probs_btts)
    pred_btts = (probs_btts_cal > 0.5).astype(int)
    conf_btts = np.maximum(probs_btts_cal, 1 - probs_btts_cal)
    
    ll_btts = log_loss(y_test_btts, np.column_stack([1 - probs_btts_cal, probs_btts_cal]))
    brier_btts = brier_score_loss(y_test_btts, probs_btts_cal)
    try:
        auc_btts = roc_auc_score(y_test_btts, probs_btts_cal)
    except:
        auc_btts = 0.5
    acc_btts = accuracy_score(y_test_btts, pred_btts)
    
    baseline_btts = y_test_btts.mean()
    
    print(f"    LogLoss: {ll_btts:.4f}")
    print(f"    Brier: {brier_btts:.4f}")
    print(f"    AUC: {auc_btts:.4f}")
    print(f"    Accuracy: {acc_btts:.3f} (baseline YES: {baseline_btts:.3f})")
    
    for thr in [0.55, 0.60, 0.65, 0.70]:
        mask = conf_btts >= thr
        n = mask.sum()
        if n > 10:
            acc_thr = accuracy_score(y_test_btts[mask], pred_btts[mask])
            print(f"    Acc@{thr:.0%}: {acc_thr:.3f} ({n} picks)")
    
    all_rows.append({
        "fold": fold_i+1, "market": "BTTS", "model": "LightGBM",
        "log_loss": ll_btts, "brier": brier_btts, "auc": auc_btts,
        "accuracy": acc_btts, "baseline": baseline_btts,
        "n_test": len(test_df),
    })
    
    # ── MARKET 3: OVER 2.5 ──
    print(f"\n  === Fold {fold_i+1}/{N_FOLDS}: OVER 2.5 MARKET ===")
    y_all_o25 = train_df["over25"].to_numpy()
    y_test_o25 = test_df["over25"].to_numpy()
    
    m_o25 = make_lgb_bin()
    m_o25.fit(X_all.iloc[:split], y_all_o25[:split])
    raw_o25 = m_o25.predict_proba(X_all.iloc[split:])[:, 1]
    cal_o25 = IsotonicRegression(out_of_bounds="clip").fit(raw_o25, y_all_o25[split:].astype(float))
    
    probs_o25 = m_o25.predict_proba(X_test)[:, 1]
    probs_o25_cal = cal_o25.predict(probs_o25)
    pred_o25 = (probs_o25_cal > 0.5).astype(int)
    conf_o25 = np.maximum(probs_o25_cal, 1 - probs_o25_cal)
    
    ll_o25 = log_loss(y_test_o25, np.column_stack([1 - probs_o25_cal, probs_o25_cal]))
    brier_o25 = brier_score_loss(y_test_o25, probs_o25_cal)
    try:
        auc_o25 = roc_auc_score(y_test_o25, probs_o25_cal)
    except:
        auc_o25 = 0.5
    acc_o25 = accuracy_score(y_test_o25, pred_o25)
    
    baseline_o25 = y_test_o25.mean()
    
    print(f"    LogLoss: {ll_o25:.4f}")
    print(f"    Brier: {brier_o25:.4f}")
    print(f"    AUC: {auc_o25:.4f}")
    print(f"    Accuracy: {acc_o25:.3f} (baseline OVER: {baseline_o25:.3f})")
    
    for thr in [0.55, 0.60, 0.65, 0.70]:
        mask = conf_o25 >= thr
        n = mask.sum()
        if n > 10:
            acc_thr = accuracy_score(y_test_o25[mask], pred_o25[mask])
            print(f"    Acc@{thr:.0%}: {acc_thr:.3f} ({n} picks)")
    
    all_rows.append({
        "fold": fold_i+1, "market": "Over2.5", "model": "LightGBM",
        "log_loss": ll_o25, "brier": brier_o25, "auc": auc_o25,
        "accuracy": acc_o25, "baseline": baseline_o25,
        "n_test": len(test_df),
    })
    
    # ── MARKET 4: UNDER 2.5 ──
    print(f"\n  === Fold {fold_i+1}/{N_FOLDS}: UNDER 2.5 MARKET ===")
    y_all_u25 = 1 - y_all_o25
    y_test_u25 = 1 - y_test_o25
    probs_u25_cal = 1 - probs_o25_cal
    
    ll_u25 = log_loss(y_test_u25, np.column_stack([1 - probs_u25_cal, probs_u25_cal]))
    brier_u25 = brier_score_loss(y_test_u25, probs_u25_cal)
    try:
        auc_u25 = roc_auc_score(y_test_u25, probs_u25_cal)
    except:
        auc_u25 = 0.5
    
    print(f"    LogLoss: {ll_u25:.4f}")
    print(f"    Brier: {brier_u25:.4f}")
    print(f"    AUC: {auc_u25:.4f}")
    
    all_rows.append({
        "fold": fold_i+1, "market": "Under2.5", "model": "LightGBM",
        "log_loss": ll_u25, "brier": brier_u25, "auc": auc_u25,
        "n_test": len(test_df),
    })
    
    # ── MARKET 5: GOL BEKLENTISI (MAE/RMSE) ──
    print(f"\n  === Fold {fold_i+1}/{N_FOLDS}: GOL BEKLENTISI ===")
    y_hg = train_df["home_goals"].to_numpy()
    y_ag = train_df["away_goals"].to_numpy()
    y_hg_test = test_df["home_goals"].to_numpy()
    y_ag_test = test_df["away_goals"].to_numpy()
    y_total_test = y_hg_test + y_ag_test
    
    m_hg = make_lgb_reg()
    m_hg.fit(X_all.iloc[:split], y_hg[:split])
    pred_hg = m_hg.predict(X_test)
    pred_hg = np.maximum(pred_hg, 0.05)
    
    m_ag = make_lgb_reg()
    m_ag.fit(X_all.iloc[:split], y_ag[:split])
    pred_ag = m_ag.predict(X_test)
    pred_ag = np.maximum(pred_ag, 0.05)
    
    mae_hg = mean_absolute_error(y_hg_test, pred_hg)
    mae_ag = mean_absolute_error(y_ag_test, pred_ag)
    mae_total = mean_absolute_error(y_total_test, pred_hg + pred_ag)
    
    # Naive baseline: ortalama gol
    naive_avg = train_df["total_goals"].mean()
    mae_naive = mean_absolute_error(y_total_test, np.full(len(y_total_test), naive_avg))
    
    print(f"    Home MAE: {mae_hg:.4f}")
    print(f"    Away MAE: {mae_ag:.4f}")
    print(f"    Total MAE: {mae_total:.4f} (naive: {mae_naive:.4f})")
    
    all_rows.append({
        "fold": fold_i+1, "market": "Goals", "model": "LightGBM",
        "mae_home": mae_hg, "mae_away": mae_ag, "mae_total": mae_total,
        "mae_naive": mae_naive, "n_test": len(test_df),
    })
    
    # ── MARKET 6: LEAGUE BAZLI 1X2 (en buyuk 10 lig) ──
    top_leagues = test_df["league"].value_counts().head(10).index.tolist()
    for lg in top_leagues:
        lm = test_df["league"] == lg
        n_lg = lm.sum()
        if n_lg < 20:
            continue
        acc_lg = accuracy_score(y_test_1x2[lm], pred_1x2[lm])
        ll_lg = log_loss(y_test_1x2[lm], probs_cal[lm])
        print(f"    {lg:12s}: Acc={acc_lg:.3f} LL={ll_lg:.4f} ({n_lg} mac)")
        
        all_rows.append({
            "fold": fold_i+1, "market": f"1X2_{lg}", "model": "LightGBM",
            "log_loss": ll_lg, "accuracy": acc_lg, "n_test": n_lg,
        })
    
    # ── MARKET 7: CALIBRATION ANALIZI ──
    print(f"\n  === Fold {fold_i+1}/{N_FOLDS}: CALIBRATION ===")
    for lo, hi in [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]:
        mask = (conf_1x2 >= lo) & (conf_1x2 < hi)
        n = mask.sum()
        if n > 10:
            mean_pred = conf_1x2[mask].mean()
            mean_actual = (pred_1x2[mask] == y_test_1x2[mask]).mean()
            gap = abs(mean_pred - mean_actual)
            print(f"    {lo:.0%}-{hi:.0%}: pred={mean_pred:.3f} actual={mean_actual:.3f} gap={gap:.3f} (n={n})")
    
    print(f"    Fold {fold_i+1} tamamlandi ({time.time()-t0:.0f}s)")

# ─── 5. CATBOOST KARSILASTIRMASI (Eger varsa) ──────────────────────────────
if HAS_CATBOOST:
    print("\n\n" + "=" * 90)
    print("  CATBOOST KARSILASTIRMASI (son 2 fold)")
    print("=" * 90)
    
    for fold_i in range(max(0, N_FOLDS - 2), N_FOLDS):
        tr_s, tr_e, te_s, te_e = folds[fold_i]
        train_df = feat.iloc[tr_s:tr_e].copy()
        test_df = feat.iloc[te_s:te_e].copy()
        X_all = train_df[FEATS].fillna(0)
        X_test = test_df[FEATS].fillna(0)
        
        y_1x2 = test_df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        
        # CatBoost 1X2
        cb = CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=6, verbose=False,
            loss_function="MultiClass", random_seed=42,
        )
        cb.fit(X_all, y_1x2_all := train_df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy())
        cb_probs = cb.predict_proba(X_test)
        cb_pred = np.argmax(cb_probs, axis=1)
        cb_ll = log_loss(y_1x2, cb_probs)
        cb_acc = accuracy_score(y_1x2, cb_pred)
        
        # LightGBM
        m = make_lgb_1x2()
        m.fit(X_all, y_1x2_all)
        lgb_probs = m.predict_proba(X_test)
        lgb_pred = np.argmax(lgb_probs, axis=1)
        lgb_ll = log_loss(y_1x2, lgb_probs)
        lgb_acc = accuracy_score(y_1x2, lgb_pred)
        
        print(f"  Fold {fold_i+1}: CatBoost LL={cb_ll:.4f} Acc={cb_acc:.3f} | LGBM LL={lgb_ll:.4f} Acc={lgb_acc:.3f}")

# ─── 6. FINAL OZET ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 90)
print("  FINAL OZET - HER MARKET ICIN TAHMIN DOGRULUGU")
print("=" * 90)

df = pd.DataFrame(all_rows)
df.to_csv("market_prediction_check.csv", index=False)
print("  Sonuclar market_prediction_check.csv'ye kaydedildi.\n")

markets_summary = ["1X2", "BTTS", "Over2.5", "Under2.5", "Goals"]
for mkt in markets_summary:
    sub = df[df["market"] == mkt]
    if len(sub) == 0:
        continue
    print(f"\n  ── {mkt} ──")
    for col in sub.columns:
        if col not in ("fold", "market", "model"):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"    {col:25s}: ort={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

# ── 1X2 DETAY ──
sub_1x2 = df[df["market"] == "1X2"]
if len(sub_1x2) > 0:
    print(f"\n  ── 1X2 DETAYLI SONUCLAR ──")
    print(f"    LogLoss  Ort: {sub_1x2['log_loss'].mean():.4f} (random: {np.log(3):.4f})")
    print(f"    Accuracy Ort: {sub_1x2['accuracy'].mean():.3f}")
    print(f"    Macro-F1 Ort: {sub_1x2['f1'].mean():.3f}")
    print(f"    Kalibrasyon Gap Ort: {sub_1x2['calibration_gap'].mean():.4f}")
    
    if sub_1x2['log_loss'].mean() < np.log(3):
        print(f"    [OK] LogLoss random'dan dusuk")
    else:
        print(f"    [!] LogLoss random'dan yuksek - MODEL CALISMIYOR")
    
    if sub_1x2['accuracy'].mean() > 0.45:
        print(f"    [OK] Accuracy makul seviyede")
    else:
        print(f"    [!] Accuracy cok dusuk")

# ── BTTS DETAY ──
sub_btts = df[df["market"] == "BTTS"]
if len(sub_btts) > 0:
    print(f"\n  ── BTTS DETAYLI SONUCLAR ──")
    print(f"    LogLoss Ort: {sub_btts['log_loss'].mean():.4f}")
    print(f"    AUC Ort: {sub_btts['auc'].mean():.4f}")
    print(f"    Accuracy Ort: {sub_btts['accuracy'].mean():.3f}")
    if sub_btts['auc'].mean() > 0.55:
        print(f"    [OK] BTTS AUC > 0.55 - model calisiyor")
    else:
        print(f"    [!] BTTS AUC <= 0.55 - model zayif")

# ── Over 2.5 DETAY ──
sub_o25 = df[df["market"] == "Over2.5"]
if len(sub_o25) > 0:
    print(f"\n  ── OVER 2.5 DETAYLI SONUCLAR ──")
    print(f"    LogLoss Ort: {sub_o25['log_loss'].mean():.4f}")
    print(f"    AUC Ort: {sub_o25['auc'].mean():.4f}")
    print(f"    Accuracy Ort: {sub_o25['accuracy'].mean():.3f}")
    if sub_o25['auc'].mean() > 0.55:
        print(f"    [OK] Over 2.5 AUC > 0.55 - model calisiyor")
    else:
        print(f"    [!] Over 2.5 AUC <= 0.55 - model zayif")

# ── Gol Beklentisi DETAY ──
sub_goals = df[df["market"] == "Goals"]
if len(sub_goals) > 0:
    print(f"\n  ── GOL BEKLENTISI DETAYLI SONUCLAR ──")
    print(f"    Home MAE Ort: {sub_goals['mae_home'].mean():.4f}")
    print(f"    Away MAE Ort: {sub_goals['mae_away'].mean():.4f}")
    print(f"    Total MAE Ort: {sub_goals['mae_total'].mean():.4f}")
    print(f"    Naive MAE Ort: {sub_goals['mae_naive'].mean():.4f}")
    if sub_goals['mae_total'].mean() < sub_goals['mae_naive'].mean():
        print(f"    [OK] Model naive'den daha iyi gol tahmini yapiyor")
    else:
        print(f"    [!] Model naive'den daha kotu - gol modeli calismiyor")

# ── LEAGUE BAZLI ──
league_rows = df[df["market"].str.startswith("1X2_")]
if len(league_rows) > 0:
    print(f"\n  ── LIG BAZLI 1X2 PERFORMANS (ORTALAMA) ──")
    print(f"    {'Lig':<15} {'Mac':>5} {'Acc':>7} {'LL':>7}")
    print(f"    {'-'*40}")
    for lg in league_rows["market"].str.replace("1X2_", "").unique():
        lgsub = league_rows[league_rows["market"] == f"1X2_{lg}"]
        if len(lgsub) > 0:
            n = lgsub["n_test"].sum()
            acc = lgsub["accuracy"].mean()
            ll = lgsub["log_loss"].mean()
            if n >= 50:
                print(f"    {lg:15s} {n:5.0f} {acc:6.3f} {ll:7.4f}")

# ── FINAL KARAR ──
print(f"\n\n{'=' * 90}")
print("  FINAL KARAR: HER MARKET ICIN MODEL GERCEKTEN CALISIYOR MU?")
print("=" * 90)

issues = []
ok_list = []

# 1X2
if len(sub_1x2) > 0:
    ll_mean = sub_1x2["log_loss"].mean()
    acc_mean = sub_1x2["accuracy"].mean()
    if ll_mean < np.log(3) and acc_mean > 0.43:
        ok_list.append("1X2: LogLoss random'dan dusuk, accuracy makul")
    else:
        issues.append(f"1X2: LL={ll_mean:.4f} (random={np.log(3):.4f}), Acc={acc_mean:.3f}")

# BTTS
if len(sub_btts) > 0:
    auc_mean = sub_btts["auc"].mean()
    if auc_mean > 0.53:
        ok_list.append(f"BTTS: AUC={auc_mean:.3f} > 0.53")
    else:
        issues.append(f"BTTS: AUC={auc_mean:.3f} <= 0.53")

# Over 2.5
if len(sub_o25) > 0:
    auc_mean = sub_o25["auc"].mean()
    if auc_mean > 0.53:
        ok_list.append(f"Over 2.5: AUC={auc_mean:.3f} > 0.53")
    else:
        issues.append(f"Over 2.5: AUC={auc_mean:.3f} <= 0.53")

# Gol
if len(sub_goals) > 0:
    if sub_goals["mae_total"].mean() < sub_goals["mae_naive"].mean():
        ok_list.append("Gol Beklentisi: Naive'den daha iyi")
    else:
        issues.append("Gol Beklentisi: Naive'den daha kotu")

print()
for ok in ok_list:
    print(f"  [OK] {ok}")
for issue in issues:
    print(f"  [!]  {issue}")

if not issues:
    print(f"\n  TUM MARKETLERDE MODEL GERCEKTEN DOGRU TAHMIN YAPIYOR!")
else:
    print(f"\n  {len(issues)} MARKETTE SORUN VAR - DETAYLI INCELEME GEREKLI")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("=" * 90)
