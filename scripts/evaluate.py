"""Kapsamli degerlendirme - Walk-Forward, Calibration, LogLoss, Brier, ROI, League."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

t0 = time.time()
print("="*80)
print("  KAPSAMLI DEGERLENDIRME SISTEMI")
print("="*80)

# ─── 1. VERI ────────────────────────────────────────────────────────────────
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over15"]=(feat["total_goals"]>1.5).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat["over35"]=(feat["total_goals"]>3.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

SKIP={"match_id","league","season","date","home_team_id","away_team_id",
    "home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over25","over15","over35","total_goals",
    "home_shots","away_shots","home_sot","away_sot",
    "home_corners","away_corners","home_xg","away_xg","home_yellow","away_yellow",
    "home_red","away_red","referee","data_completeness","home_xg_real","away_xg_real",
    "total_xg_real","xg_diff_real","ht_home_goals","ht_away_goals","ht_total_goals",
    "ht_result_is_draw","ht_home_leading","ht_second_half_goals_expected",
    "second_half_goals","home_score_prob","away_score_prob","btts_xprob","over25_xprob",
    "xg_diff_abs","draw_xprob","avg_home_odds","avg_draw_odds","avg_away_odds",
    "avg_over25_odds","avg_close_home_odds","avg_close_draw_odds",
    "avg_close_away_odds","avg_close_over25_odds","mkt_close_home_prob",
    "mkt_close_draw_prob","mkt_close_away_prob","mkt_close_over25_prob",
    "shots_diff","sot_diff"}
FEATS=[c for c in feat.columns if c not in SKIP and feat[c].dtype in ["float64","int64","float32","int32"]]
print(f"  Veri: {len(feat)} mac, {len(FEATS)} feature")
print(f"  Tarih: {feat['date'].min().date()} -> {feat['date'].max().date()}")

# ─── 2. WALK-FORWARD PARAMETERS ────────────────────────────────────────────
MIN_TRAIN = 300000
TEST_WINDOW = 25000
STEP = 25000

total = len(feat)
start_train = 0
folds = []
while start_train + MIN_TRAIN + TEST_WINDOW <= total:
    test_end = start_train + MIN_TRAIN + TEST_WINDOW
    folds.append((start_train, start_train + MIN_TRAIN, start_train + MIN_TRAIN, test_end))
    start_train += STEP

print(f"  Walk-Forward: {len(folds)} fold, her fold {TEST_WINDOW} test mac")
print(f"  Train baslangici: fold1={feat.iloc[folds[0][1]-1]['date'].date()}, son fold={feat.iloc[folds[-1][3]-1]['date'].date()}")

# ─── 3. MODEL ───────────────────────────────────────────────────────────────
def make_model(seed):
    return lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,
        learning_rate=0.015,n_estimators=500,max_depth=6,min_child_samples=100,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=seed,verbose=-1,n_jobs=-1)

def make_bin_model(seed):
    return lgb.LGBMClassifier(objective="binary",num_leaves=45,
        learning_rate=0.015,n_estimators=500,max_depth=6,min_child_samples=80,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
        random_seed=seed,verbose=-1,n_jobs=-1)

# ─── 4. WALK-FORWARD LOOP ──────────────────────────────────────────────────
all_results = []
markets = {
    "1X2": {"type":"multi", "col":"result", "map":{"H":0,"D":1,"A":2}},
    "BTTS": {"type":"bin", "col":"btts"},
    "O15": {"type":"bin", "col":"over15"},
    "O25": {"type":"bin", "col":"over25"},
    "O35": {"type":"bin", "col":"over35"},
}

for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
    train_df = feat.iloc[tr_start:tr_end].copy()
    test_df = feat.iloc[te_start:te_end].copy()
    
    X_all = train_df[FEATS].fillna(0)
    X_test = test_df[FEATS].fillna(0)
    split = int(len(X_all)*0.85)
    
    fold_res = {"fold": fold_i+1, 
                "train_date": f"{train_df['date'].iloc[-1].date()}",
                "test_range": f"{test_df['date'].iloc[0].date()}~{test_df['date'].iloc[-1].date()}",
                "train_n": len(train_df), "test_n": len(test_df)}
    
    for mkt, info in markets.items():
        if info["type"]=="multi":
            y_all = np.array([info["map"][r] for r in train_df[info["col"]]])
            y_test = np.array([info["map"][r] for r in test_df[info["col"]]])
        else:
            y_all = train_df[info["col"]].values
            y_test = test_df[info["col"]].values
        
        models = []
        for s in [42]:
            if info["type"]=="multi":
                m = make_model(s)
                m.fit(X_all.iloc[:split], y_all[:split])
                raw = m.predict_proba(X_all.iloc[split:])
                cal = [IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y_all[split:]==c).astype(float)) for c in range(3)]
            else:
                m = make_bin_model(s)
                m.fit(X_all.iloc[:split], y_all[:split])
                raw = m.predict_proba(X_all.iloc[split:])
                cal = IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw, y_all[split:].astype(float))
            models.append((m, cal))
        
        # Predict
        probs = []
        for m, cal in models:
            raw = m.predict_proba(X_test)
            if info["type"]=="multi":
                cp = np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
                s=cp.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); cp/=s
                probs.append(cp)
            else:
                probs.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
        prob = np.mean(probs, axis=0)
        
        if info["type"]=="multi":
            conf = np.max(prob, axis=1)
            pred = np.argmax(prob, axis=1)
            ll = log_loss(y_test, prob)
            brier = np.mean([(prob[i,y_test[i]]-1)**2 + sum(prob[i,j]**2 for j in range(3) if j!=y_test[i]) for i in range(len(y_test))])
        else:
            conf = np.maximum(prob, 1-prob)
            pred = (prob>0.5).astype(int)
            ll = log_loss(y_test, np.column_stack([1-prob, prob]))
            brier = brier_score_loss(y_test, prob)
        
        # Per-threshold accuracy
        thr_acc = {}
        for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
            mask = conf>=t
            n = mask.sum()
            if n>0:
                acc = (pred[mask]==y_test[mask]).mean()
                thr_acc[f"acc_{int(t*100)}"] = acc
                thr_acc[f"picks_{int(t*100)}"] = n
        
        # ROI at 60% (flat bet 1 unit)
        mask60 = conf>=0.60
        n60 = mask60.sum()
        roi60 = 0
        if n60 > 0 and info["type"]=="multi":
            roi60 = ((pred[mask60]==y_test[mask60]).sum() - n60) / n60
        elif n60 > 0:
            roi60 = ((pred[mask60]==y_test[mask60]).sum() - n60) / n60
        
        fold_res[f"{mkt}_ll"] = ll
        fold_res[f"{mkt}_brier"] = brier
        fold_res[f"{mkt}_conf_mean"] = conf.mean()
        fold_res[f"{mkt}_n60"] = n60
        fold_res[f"{mkt}_roi60"] = roi60
        fold_res.update({f"{mkt}_{k}":v for k,v in thr_acc.items()})
    
    all_results.append(fold_res)
    
    if fold_i % 5 == 0:
        elapsed = time.time()-t0
        print(f"  Fold {fold_i+1}/{len(folds)} tamam ({elapsed:.0f}s) | "
              f"1X2 ll={fold_res['1X2_ll']:.4f} O25 ll={fold_res['O25_ll']:.4f}")

df = pd.DataFrame(all_results)
print(f"\n  Walk-Forward tamam: {len(folds)} fold ({time.time()-t0:.0f}s)")

# ─── 5. AGGREGATED RESULTS ─────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  AGGREGATE SONUCLAR (Walk-Forward Ortalamasi)")
print("="*80)

for mkt in markets:
    print(f"\n  ── {mkt} ──")
    ll_mean = df[f"{mkt}_ll"].mean()
    brier_mean = df[f"{mkt}_brier"].mean()
    print(f"    LogLoss: {ll_mean:.4f} | Brier: {brier_mean:.4f}")
    print(f"    {'Esik':>6} {'Picks/ort':>10} {'Acc/ort':>10} {'ROI60':>8}")
    print(f"    {'-'*40}")
    for t in [50,55,60,65,70,75,80]:
        pk = df[f"{mkt}_picks_{t}"].mean() if f"{mkt}_picks_{t}" in df.columns else 0
        ac = df[f"{mkt}_acc_{t}"].mean() if f"{mkt}_acc_{t}" in df.columns else np.nan
        if t==60:
            roi = df[f"{mkt}_roi60"].mean()
            print(f"    {t:>5}% {pk:>10.0f} {ac*100:>9.1f}% {roi*100:>+7.1f}%")
        else:
            print(f"    {t:>5}% {pk:>10.0f} {ac*100:>9.1f}%")

# ─── 6. CALIBRATION ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  CALIBRATION ANALYSIS (son fold)")
print("="*80)

last_test = feat.iloc[folds[-1][2]:folds[-1][3]]
X_last = last_test[FEATS].fillna(0)

for mkt, info in markets.items():
    if info["type"]=="multi":
        y_all = np.array([info["map"][r] for r in train_df[info["col"]]])
        y_last = np.array([info["map"][r] for r in last_test[info["col"]]])
        m = make_model(42); m.fit(X_all.iloc[:split], y_all[:split])
        raw = m.predict_proba(X_last)
        prob = np.mean([cal[c].predict(raw[:,c]) for c in range(3)], axis=0) if False else raw
        conf = np.max(prob, axis=1)
        pred = np.argmax(prob, axis=1)
        
        print(f"\n  {mkt} Calibration:")
        print(f"    {'Bucket':>10} {'Mean Pred':>10} {'Mean Act':>10} {'N':>6} {'Gap':>8}")
        print(f"    {'-'*50}")
        for lo, hi in [(0.5,0.55),(0.55,0.6),(0.6,0.65),(0.65,0.7),(0.7,0.75),(0.75,0.8),(0.8,1.0)]:
            mask = (conf>=lo)&(conf<hi)
            n = mask.sum()
            if n>10:
                mp = conf[mask].mean()
                ma = (pred[mask]==y_last[mask]).mean()
                print(f"    {lo*100:.0f}-{hi*100:.0f}% {mp*100:>9.1f}% {ma*100:>9.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")
    else:
        y_all = train_df[info["col"]].values
        y_last = last_test[info["col"]].values
        m = make_bin_model(42); m.fit(X_all.iloc[:split], y_all[:split])
        raw = m.predict_proba(X_last)
        prob = raw[:,1]
        conf = np.maximum(prob, 1-prob)
        pred = (prob>0.5).astype(int)
        
        print(f"\n  {mkt} Calibration:")
        print(f"    {'Bucket':>10} {'Mean Pred':>10} {'Mean Act':>10} {'N':>6} {'Gap':>8}")
        print(f"    {'-'*50}")
        for lo, hi in [(0.5,0.55),(0.55,0.6),(0.6,0.65),(0.65,0.7),(0.7,0.75),(0.75,0.8),(0.8,1.0)]:
            mask = (conf>=lo)&(conf<hi)
            n = mask.sum()
            if n>10:
                mp = conf[mask].mean()
                ma = (pred[mask]==y_last[mask]).mean()
                print(f"    {lo*100:.0f}-{hi*100:.0f}% {mp*100:>9.1f}% {ma*100:>9.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")

# ─── 7. MARKET vs NO-MARKET ────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  MARKET vs NO-MARKET (Orani olan vs olmayan)")
print("="*80)

has_odds = feat["avg_home_odds"].notna() & feat["avg_draw_odds"].notna() & feat["avg_away_odds"].notna()
no_odds = ~has_odds
print(f"  Oran olan: {has_odds.sum()} | Olmayan: {no_odds.sum()}")

# Son fold uzerinde
te_mask = pd.Series(False, index=feat.index)
te_mask.iloc[folds[-1][2]:folds[-1][3]] = True
odds_in_test = te_mask & has_odds
no_odds_in_test = te_mask & no_odds

for mkt in ["1X2","O25"]:
    if mkt=="1X2":
        ll_odds = df["1X2_ll"].mean()
        ll_no = df["1X2_ll"].mean()  # same model, different data subsets
        print(f"  {mkt}: LogLoss={ll_odds:.4f}")
    else:
        print(f"  {mkt}: LogLoss={df['O25_ll'].mean():.4f}")

# ─── 8. LEAGUE BAZLI PERFORMANS ────────────────────────────────────────────
print(f"\n{'='*80}")
print("  LEAGUE BAZLI PERFORMANS (son fold)")
print("="*80)

# Son fold'da her ligin accuracy'si
last_leagues = last_test["league"].value_counts()
top_leagues = last_leagues[last_leagues>=50].index.tolist()

# Tekrar egit son fold icin
for mkt_name, mkt_info in [("1X2",markets["1X2"]),("O25",markets["O25"])]:
    if mkt_info["type"]=="multi":
        y_all = np.array([mkt_info["map"][r] for r in train_df[mkt_info["col"]]])
        y_last = np.array([mkt_info["map"][r] for r in last_test[mkt_info["col"]]])
        m = make_model(42); m.fit(X_all.iloc[:split], y_all[:split])
        raw = m.predict_proba(X_last)
        prob = raw
        conf = np.max(prob, axis=1)
        pred = np.argmax(prob, axis=1)
    else:
        y_all = train_df[mkt_info["col"]].values
        y_last = last_test[mkt_info["col"]].values
        m = make_bin_model(42); m.fit(X_all.iloc[:split], y_all[:split])
        raw = m.predict_proba(X_last)
        prob = raw[:,1]
        conf = np.maximum(prob, 1-prob)
        pred = (prob>0.5).astype(int)
    
    print(f"\n  {mkt_name} League Performans:")
    print(f"    {'League':<12} {'Mac':>5} {'Acc':>7} {'Conf':>7} {'Acc65+':>7} {'N65+':>5} {'Acc75+':>7} {'N75+':>5}")
    print(f"    {'-'*65}")
    for lg in top_leagues[:20]:
        lm = last_test["league"]==lg
        n = lm.sum()
        acc = (pred[lm]==y_last[lm]).mean()
        c_mean = conf[lm].mean()
        m65 = conf[lm]>=0.65; n65=m65.sum()
        acc65 = (pred[lm][m65]==y_last[lm][m65]).mean() if n65>10 else np.nan
        m75 = conf[lm]>=0.75; n75=m75.sum()
        acc75 = (pred[lm][m75]==y_last[lm][m75]).mean() if n75>5 else np.nan
        a65 = f"{acc65*100:.1f}%" if not np.isnan(acc65) else "  -  "
        a75 = f"{acc75*100:.1f}%" if not np.isnan(acc75) else "  -  "
        print(f"    {lg:<12} {n:>5} {acc*100:>6.1f}% {c_mean*100:>6.1f}% {a65:>7} {n65:>5} {a75:>7} {n75:>5}")

# ─── 9. ROI DETAYLI ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  ROI DETAYLI (Walk-Forward)")
print("="*80)

print(f"\n  Duz Bahis (1 birim) - threshold bazli ortalama ROI:")
print(f"  {'Market':<8} {'T55':>7} {'T60':>7} {'T65':>7} {'T70':>7} {'T75':>7} {'T80':>7}")
print(f"  {'-'*50}")
for mkt in markets:
    rois = []
    for t in [55,60,65,70,75,80]:
        col = f"{mkt}_acc_{t}"
        ncol = f"{mkt}_picks_{t}"
        if col in df.columns:
            accs = df[col].values
            ns = df[ncol].values
            valid = ~np.isnan(accs) & (ns>0)
            if valid.sum()>0:
                roi = np.average(accs[valid]-1, weights=ns[valid])
                rois.append(f"{roi*100:>+6.1f}%")
            else:
                rois.append("    -  ")
        else:
            rois.append("    -  ")
    print(f"  {mkt:<8} {''.join(r for r in rois)}")

# ─── 10. FINAL SUMMARY ─────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  FINAL OZET")
print("="*80)
print(f"  Walk-Forward: {len(folds)} fold")
print(f"  Test mac/fold: {TEST_WINDOW}")
print(f"  Toplam test: {sum(df['test_n']):,}")
print()
for mkt in markets:
    ll = df[f"{mkt}_ll"].mean()
    br = df[f"{mkt}_brier"].mean()
    cm = df[f"{mkt}_conf_mean"].mean()
    print(f"  {mkt:<8} LogLoss={ll:.4f} Brier={br:.4f} AvgConf={cm*100:.1f}%")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
