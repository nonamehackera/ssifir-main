"""FINAL TEST - Calibrated CatBoost, Value Bet, ROI Backtest."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

t0 = time.time()
print("="*80)
print("  FINAL TEST - CALIBRATED CATBOOST + ROI BACKTEST")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# 1. VERI
# ═══════════════════════════════════════════════════════════════════════════
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])
print(f"  Veri: {len(feat)} mac ({time.time()-t0:.0f}s)")

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading",
        "ht_home_goals","ht_away_goals","home_xg","away_xg","total_xg_real",
        "xg_diff_real","home_score_prob","away_score_prob","btts_xprob",
        "over25_xprob","draw_xprob","data_completeness",
        "home_shots","away_shots","home_sot","away_sot",
        "home_corners","away_corners","home_yellow","away_yellow","home_red","away_red",
        "avg_home_odds","avg_draw_odds","avg_away_odds","avg_over25_odds",
        "avg_close_home_odds","avg_close_draw_odds","avg_close_away_odds","avg_close_over25_odds",
        "mkt_close_home_prob","mkt_close_draw_prob","mkt_close_away_prob","mkt_close_over25_prob",
        "shots_diff","sot_diff","home_goals","away_goals","result","result_H","result_D","result_A",
        "btts","over25","over15","over35","total_goals","match_id","league","season","date",
        "home_team_id","away_team_id","referee","home_xg_real","away_xg_real"}

CAT_COLS = ["league","home_team_id","away_team_id","season"]
ALL_COLS_ORIG = [c for c in feat.columns if c not in LEAK and feat[c].dtype in ["float64","int64","float32","int32"]]
ALL_COLS = CAT_COLS + [c for c in ALL_COLS_ORIG if c not in CAT_COLS]
cat_idx = [ALL_COLS.index(c) for c in CAT_COLS]
print(f"  Features: {len(ALL_COLS)} ({len(CAT_COLS)} cat + {len(ALL_COLS)-len(CAT_COLS)} num)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD + CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════
MIN_TRAIN = 300000
TEST_W = 40000
STEP = 40000
folds = []
i = 0
while i + MIN_TRAIN + TEST_W <= len(feat):
    folds.append((i, i+MIN_TRAIN, i+MIN_TRAIN, i+MIN_TRAIN+TEST_W))
    i += STEP

print(f"  Walk-Forward: {len(folds)} fold, TEST={TEST_W}")

# Accumulators
all_1x2_y, all_1x2_prob, all_1x2_cal_prob = [], [], []
all_btts_y, all_btts_prob, all_btts_cal = [], [], []
all_o25_y, all_o25_prob, all_o25_cal = [], [], []
all_leagues, all_dates = [], []

for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
    train = feat.iloc[tr_s:tr_e]
    test = feat.iloc[te_s:te_e]
    
    Xtr = train[ALL_COLS]; Xte = test[ALL_COLS]
    y_1x2 = train["result"].map({"H":0,"D":1,"A":2}).values
    y_1x2_te = test["result"].map({"H":0,"D":1,"A":2}).values
    y_btts_tr = train["btts"].values; y_btts_te = test["btts"].values
    y_o25_tr = train["over25"].values; y_o25_te = test["over25"].values
    
    # Train with validation split
    sp = int(len(Xtr)*0.85)
    
    # 1X2
    clf = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
        border_count=64,random_strength=2,bagging_temperature=1,
        cat_features=cat_idx,loss_function="MultiClass",
        class_weights={0:1,1:1.3,2:1},random_seed=42,verbose=False,allow_writing_files=False)
    clf.fit(Xtr.iloc[:sp], y_1x2[:sp], use_best_model=True,
            eval_set=(Xtr.iloc[sp:], y_1x2[sp:]))
    raw_1x2 = clf.predict_proba(Xte)
    
    # Isotonic calibration on validation set
    raw_val = clf.predict_proba(Xtr.iloc[sp:])
    cal_1x2 = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_val[:,c], (y_1x2[sp:]==c).astype(float))
        cal_1x2.append(ir)
    
    # Calibrated probabilities (renormalize)
    cal_p = np.column_stack([cal_1x2[c].predict(raw_1x2[:,c]) for c in range(3)])
    cal_s = cal_p.sum(axis=1,keepdims=True)
    cal_s = np.where(cal_s==0,1,cal_s)
    cal_p /= cal_s
    
    all_1x2_y.extend(y_1x2_te)
    all_1x2_prob.append(raw_1x2)
    all_1x2_cal_prob.append(cal_p)
    all_leagues.extend(test["league"].tolist())
    all_dates.extend(test["date"].tolist())
    
    # BTTS
    clf_bt = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
        cat_features=cat_idx,loss_function="Logloss",random_seed=42,verbose=False,allow_writing_files=False)
    clf_bt.fit(Xtr.iloc[:sp], y_btts_tr[:sp], use_best_model=True,
               eval_set=(Xtr.iloc[sp:], y_btts_tr[sp:]))
    raw_bt = clf_bt.predict_proba(Xte)[:,1]
    raw_bt_val = clf_bt.predict_proba(Xtr.iloc[sp:])[:,1]
    
    ir_bt = IsotonicRegression(out_of_bounds="clip")
    ir_bt.fit(raw_bt_val, y_btts_tr[sp:].astype(float))
    cal_bt = ir_bt.predict(raw_bt)
    
    all_btts_y.extend(y_btts_te)
    all_btts_prob.append(raw_bt)
    all_btts_cal.append(cal_bt)
    
    # O25
    clf_o = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
        cat_features=cat_idx,loss_function="Logloss",random_seed=42,verbose=False,allow_writing_files=False)
    clf_o.fit(Xtr.iloc[:sp], y_o25_tr[:sp], use_best_model=True,
              eval_set=(Xtr.iloc[sp:], y_o25_tr[sp:]))
    raw_o = clf_o.predict_proba(Xte)[:,1]
    raw_o_val = clf_o.predict_proba(Xtr.iloc[sp:])[:,1]
    
    ir_o = IsotonicRegression(out_of_bounds="clip")
    ir_o.fit(raw_o_val, y_o25_tr[sp:].astype(float))
    cal_o = ir_o.predict(raw_o)
    
    all_o25_y.extend(y_o25_te)
    all_o25_prob.append(raw_o)
    all_o25_cal.append(cal_o)
    
    ll_raw = log_loss(y_1x2_te, raw_1x2)
    ll_cal = log_loss(y_1x2_te, cal_p)
    print(f"  Fold {fi+1}/{len(folds)} | Raw={ll_raw:.4f} Cal={ll_cal:.4f} ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 3. RAPOR
# ═══════════════════════════════════════════════════════════════════════════
y1 = np.array(all_1x2_y)
raw1 = np.vstack(all_1x2_prob)
cal1 = np.vstack(all_1x2_cal_prob)
c_raw = np.max(raw1,axis=1); p_raw = np.argmax(raw1,axis=1)
c_cal = np.max(cal1,axis=1); p_cal = np.argmax(cal1,axis=1)

yb = np.array(all_btts_y)
raw_b = np.concatenate(all_btts_prob)
cal_b = np.concatenate(all_btts_cal)

yo = np.array(all_o25_y)
raw_o = np.concatenate(all_o25_prob)
cal_o = np.concatenate(all_o25_cal)

N = len(y1)
print(f"\n{'='*80}")
print(f"  SONUCLAR ({N} TEST MAC)")
print("="*80)

# 1X2 - Raw vs Calibrated
for label, prob, conf, pred in [("RAW", raw1, c_raw, p_raw), ("CALIBRATED", cal1, c_cal, p_cal)]:
    ll = log_loss(y1, prob)
    print(f"\n  1X2 {label} (LogLoss={ll:.4f})")
    print(f"  {'Esik':>6} {'Picks':>7} {'Oneri%':>7} {'Dogru':>7} {'Acc':>7}")
    print(f"  {'-'*40}")
    for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
        m=conf>=t; n=m.sum()
        if n>0:
            c=(pred[m]==y1[m]).sum()
            print(f"  {t*100:>5.0f}% {n:>7} {n/N*100:>6.1f}% {c:>7} {c/n*100:>6.1f}%")

# BTTS
for label, raw, cal, yv in [("RAW", raw_b, cal_b, yb)]:
    ll_r = log_loss(yv, np.column_stack([1-raw,raw]))
    ll_c = log_loss(yv, np.column_stack([1-cal,cal]))
    print(f"\n  BTTS (Raw LogLoss={ll_r:.4f}, Cal LogLoss={ll_c:.4f})")
    for lbl2, prob in [("Raw",raw),("Cal",cal)]:
        conf = np.maximum(prob,1-prob)
        pred = (prob>0.5).astype(int)
        m60=conf>=0.60; n60=m60.sum()
        a60 = (pred[m60]==yv[m60]).mean()*100 if n60>10 else 0
        print(f"    {lbl2}: Acc={pred.mean()*100:.1f}% Acc60+={a60:.1f}% (n={n60})")

# O25
print(f"\n  O25 (Raw LogLoss={log_loss(yo,np.column_stack([1-raw_o,raw_o])):.4f}, Cal LogLoss={log_loss(yo,np.column_stack([1-cal_o,cal_o])):.4f})")
for lbl2, prob in [("Raw",raw_o),("Cal",cal_o)]:
    conf = np.maximum(prob,1-prob)
    pred = (prob>0.5).astype(int)
    m60=conf>=0.60; n60=m60.sum()
    a60 = (pred[m60]==yo[m60]).mean()*100 if n60>10 else 0
    print(f"    {lbl2}: Acc={pred.mean()*100:.1f}% Acc60+={a60:.1f}% (n={n60})")

# ═══════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION DETAY (CALIBRATED)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  CALIBRATION (CALIBRATED MODEL)")
print("="*80)
print(f"  {'Bucket':>10} {'Pred':>8} {'Actual':>8} {'N':>6} {'Gap':>8}")
for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.0)]:
    m=(c_cal>=lo)&(c_cal<hi); n=m.sum()
    if n>10:
        mp=c_cal[m].mean(); ma=(p_cal[m]==y1[m]).mean()
        print(f"  {lo*100:.0f}-{hi*100:.0f}% {mp*100:>7.1f}% {ma*100:>7.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 5. ROI BACKTEST (CALIBRATED + ODDS)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  ROI BACKTEST (CALIBRATED MODEL)")
print("="*80)

# Oranlari test setleriyle eslesir
all_odds_h, all_odds_d, all_odds_a = [], [], []
for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
    test = feat.iloc[te_s:te_e]
    all_odds_h.extend(test["avg_home_odds"].fillna(0).tolist())
    all_odds_d.extend(test["avg_draw_odds"].fillna(0).tolist())
    all_odds_a.extend(test["avg_away_odds"].fillna(0).tolist())

oh = np.array(all_odds_h)
od = np.array(all_odds_d)
oa = np.array(all_odds_a)
has_odds = (oh > 0) & (od > 0) & (oa > 0)

print(f"  Oran olan mac: {has_odds.sum()} / {N}")

# Implied probabilities
impl_h = np.where(has_odds, 1/oh, 0)
impl_d = np.where(has_odds, 1/od, 0)
impl_a = np.where(has_odds, 1/oa, 0)
impl_total = impl_h + impl_d + impl_a
margin = impl_total - 1

# Calibrated probs for odds matches
cal_h = cal1[:,0]; cal_d = cal1[:,1]; cal_a = cal1[:,2]

# Value = calibrated_prob - implied_prob
edge_h = cal_h - impl_h
edge_d = cal_d - impl_d
edge_a = cal_a - impl_a

# Stratejiler
print(f"\n  {'Strateji':<35} {'Bahis':>6} {'Kazanc':>8} {'ROI':>8} {'Yield':>8}")
print(f"  {'-'*65}")

# S1: Tum mac (en yüksek calibrated prob)
profit=0; nbet=0; wins=0
for i in range(N):
    if not has_odds[i]: continue
    nbet += 1
    best = np.argmax([cal_h[i], cal_d[i], cal_a[i]])
    actual = y1[i]
    if best == actual:
        wins += 1
        profit += [oh[i], od[i], oa[i]][best] - 1
    else:
        profit -= 1
print(f"  {'S1: Model max (calibrated)':<35} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}% {wins/nbet*100:>7.1f}%")

# S2: 60%+ calibrated confidence
m = c_cal >= 0.60
profit=0; nbet=0; wins=0
for i in np.where(m & has_odds)[0]:
    nbet += 1
    best = np.argmax([cal_h[i], cal_d[i], cal_a[i]])
    if best == y1[i]:
        wins += 1
        profit += [oh[i], od[i], oa[i]][best] - 1
    else:
        profit -= 1
if nbet>0: print(f"  {'S2: 60%+ calibrated':<35} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}% {wins/nbet*100:>7.1f}%")

# S3: 65%+ calibrated confidence
m = c_cal >= 0.65
profit=0; nbet=0; wins=0
for i in np.where(m & has_odds)[0]:
    nbet += 1
    best = np.argmax([cal_h[i], cal_d[i], cal_a[i]])
    if best == y1[i]:
        wins += 1
        profit += [oh[i], od[i], oa[i]][best] - 1
    else:
        profit -= 1
if nbet>0: print(f"  {'S3: 65%+ calibrated':<35} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}% {wins/nbet*100:>7.1f}%")

# S4: 70%+ calibrated confidence
m = c_cal >= 0.70
profit=0; nbet=0; wins=0
for i in np.where(m & has_odds)[0]:
    nbet += 1
    best = np.argmax([cal_h[i], cal_d[i], cal_a[i]])
    if best == y1[i]:
        wins += 1
        profit += [oh[i], od[i], oa[i]][best] - 1
    else:
        profit -= 1
if nbet>0: print(f"  {'S4: 70%+ calibrated':<35} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}% {wins/nbet*100:>7.1f}%")

# S5: Value bet - calibrated > implied + edge
for edge_thresh in [0.02, 0.05, 0.08, 0.10]:
    profit=0; nbet=0; wins=0
    for i in range(N):
        if not has_odds[i]: continue
        edges = [edge_h[i], edge_d[i], edge_a[i]]
        best = np.argmax(edges)
        if edges[best] > edge_thresh:
            nbet += 1
            if best == y1[i]:
                wins += 1
                profit += [oh[i], od[i], oa[i]][best] - 1
            else:
                profit -= 1
    if nbet>0:
        print(f"  {'S5: Value bet (edge>'+str(int(edge_thresh*100))+'%)':<35} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}% {wins/nbet*100:>7.1f}%")

# S6: Kelly criterion (oranli bahis)
profit=0; nbet=0; wins=0; total_risk=0
for i in range(N):
    if not has_odds[i]: continue
    edges = [edge_h[i], edge_d[i], edge_a[i]]
    probs = [cal_h[i], cal_d[i], cal_a[i]]
    odds = [oh[i], od[i], oa[i]]
    best = np.argmax(edges)
    if edges[best] > 0.02:
        # Kelly: f = (bp - q) / b where b=odds-1, p=model_prob, q=1-p
        b = odds[best] - 1
        p = probs[best]
        kelly_f = (b*p - (1-p)) / b
        kelly_f = max(0, min(kelly_f, 0.05))  # Max %5 bankroll per bet
        if kelly_f > 0:
            nbet += 1
            total_risk += kelly_f
            if best == y1[i]:
                wins += 1
                profit += kelly_f * (odds[best] - 1)
            else:
                profit -= kelly_f
if nbet>0: print(f"  {'S6: Kelly (%5 max)':<35} {nbet:>6} {profit:>+7.1f} {profit/total_risk*100:>+7.1f}% {wins/nbet*100:>7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 6. LIG BAZLI (CALIBRATED)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  LIG BAZLI (CALIBRATED 1X2)")
print("="*80)

leagues_arr = np.array(all_leagues)
lg_count = {}
for i in range(N):
    lg = leagues_arr[i]
    if lg not in lg_count: lg_count[lg] = {"n":0,"w":0,"c60":0,"w60":0,"c70":0,"w70":0}
    lg_count[lg]["n"] += 1
    if p_cal[i] == y1[i]: lg_count[lg]["w"] += 1
    if c_cal[i] >= 0.60:
        lg_count[lg]["c60"] += 1
        if p_cal[i] == y1[i]: lg_count[lg]["w60"] += 1
    if c_cal[i] >= 0.70:
        lg_count[lg]["c70"] += 1
        if p_cal[i] == y1[i]: lg_count[lg]["w70"] += 1

top_lgs = sorted(lg_count.items(), key=lambda x: -x[1]["n"])[:15]
print(f"  {'League':<15} {'Mac':>5} {'Acc':>7} {'Acc60+':>8} {'N60+':>5} {'Acc70+':>8} {'N70+':>5}")
print(f"  {'-'*60}")
for lg, d in top_lgs:
    acc = d["w"]/d["n"]*100
    a60 = d["w60"]/d["c60"]*100 if d["c60"]>10 else 0
    a70 = d["w70"]/d["c70"]*100 if d["c70"]>5 else 0
    n60 = d["c60"]
    n70 = d["c70"]
    a60s = f"{a60:.1f}%" if d["c60"]>10 else "  -  "
    a70s = f"{a70:.1f}%" if d["c70"]>5 else "  -  "
    print(f"  {lg:<15} {d['n']:>5} {acc:>6.1f}% {a60s:>8} {n60:>5} {a70s:>8} {n70:>5}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. DONEM BAZLI ROI
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  DONEM BAZLI ROI (60%+ value bet)")
print("="*80)

dates_arr = pd.to_datetime(np.array(all_dates))
periods = [("2019-2020", "2019-10-01", "2020-06-30"),
           ("2020-2021", "2020-07-01", "2021-06-30"),
           ("2021-2022", "2021-07-01", "2022-06-30"),
           ("2022-2023", "2022-07-01", "2023-06-30"),
           ("2023-2024", "2023-07-01", "2024-06-30"),
           ("2024-2025", "2024-07-01", "2025-06-30"),
           ("2025-2026", "2025-07-01", "2026-07-07")]

print(f"  {'Donem':<15} {'Mac':>6} {'Bahis':>6} {'Kazanc':>8} {'ROI':>8}")
print(f"  {'-'*50}")
for name, d1, d2 in periods:
    d1t = pd.Timestamp(d1); d2t = pd.Timestamp(d2)
    mask = (dates_arr>=d1t) & (dates_arr<d2t) & has_odds
    profit=0; nbet=0
    for i in np.where(mask)[0]:
        edges = [edge_h[i], edge_d[i], edge_a[i]]
        best = np.argmax(edges)
        if edges[best] > 0.02:
            nbet += 1
            if best == y1[i]:
                profit += [oh[i], od[i], oa[i]][best] - 1
            else:
                profit -= 1
    n_total = mask.sum()
    if nbet > 0:
        print(f"  {name:<15} {n_total:>6} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}%")

print(f"\nToplam: {time.time()-t0:.0f}s")
