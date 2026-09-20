"""Test fixes: leak removal, draw_weight, time decay."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

t0 = time.time()
print("="*80)
print("  TEST FIXES - LEAK REMOVAL + DRAW_WEIGHT + TIME DECAY")
print("="*80)

# 1. Load data
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

print(f"  Veri: {len(feat)} mac")

# 2. Feature set - LEAK-FREE (same as final_test.py LEAK set)
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

# Check for leaked features
leaked_in_cols = [c for c in ALL_COLS if c in LEAK]
if leaked_in_cols:
    print(f"  WARNING: Leaked features found: {leaked_in_cols}")
else:
    print(f"  OK: No leaked features in feature set")

# 3. Walk-Forward
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

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

for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
    train = feat.iloc[tr_s:tr_e]
    test = feat.iloc[te_s:te_e]
    
    Xtr = train[ALL_COLS]; Xte = test[ALL_COLS]
    y_1x2 = train["result"].map({"H":0,"D":1,"A":2}).values
    y_1x2_te = test["result"].map({"H":0,"D":1,"A":2}).values
    y_btts_tr = train["btts"].values; y_btts_te = test["btts"].values
    y_o25_tr = train["over25"].values; y_o25_te = test["over25"].values
    
    sp = int(len(Xtr)*0.85)
    
    # TIME DECAY weights
    t_days = (train["date"].max() - train["date"]).dt.days.clip(lower=0)
    decay = np.exp(-1.0 * t_days / 365.0).values
    draw_mask = (y_1x2 == 1)
    sw_1x2 = decay.copy()
    sw_1x2[draw_mask] *= 4.0  # draw_weight = 4.0
    
    # 1X2 - NEW MODEL (no leak, draw_weight=4.0, time_decay)
    clf = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
        border_count=64,random_strength=2,bagging_temperature=1,
        cat_features=cat_idx,loss_function="MultiClass",
        class_weights={0:1,1:4.0,2:1},random_seed=42,verbose=False,allow_writing_files=False)
    clf.fit(Xtr.iloc[:sp], y_1x2[:sp], use_best_model=True,
            eval_set=(Xtr.iloc[sp:], y_1x2[sp:]),
            sample_weight=sw_1x2[:sp])
    raw_1x2 = clf.predict_proba(Xte)
    
    # Isotonic calibration
    raw_val = clf.predict_proba(Xtr.iloc[sp:])
    cal_1x2 = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_val[:,c], (y_1x2[sp:]==c).astype(float))
        cal_1x2.append(ir)
    
    cal_p = np.column_stack([cal_1x2[c].predict(raw_1x2[:,c]) for c in range(3)])
    cal_s = cal_p.sum(axis=1,keepdims=True)
    cal_s = np.where(cal_s==0,1,cal_s)
    cal_p /= cal_s
    
    all_1x2_y.extend(y_1x2_te)
    all_1x2_prob.append(raw_1x2)
    all_1x2_cal_prob.append(cal_p)
    
    # BTTS
    clf_bt = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
        cat_features=cat_idx,loss_function="Logloss",random_seed=42,verbose=False,allow_writing_files=False)
    clf_bt.fit(Xtr.iloc[:sp], y_btts_tr[:sp], use_best_model=True,
               eval_set=(Xtr.iloc[sp:], y_btts_tr[sp:]),
               sample_weight=decay[:sp])
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
              eval_set=(Xtr.iloc[sp:], y_o25_tr[sp:]),
              sample_weight=decay[:sp])
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

# 4. REPORT
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

# 1X2
for label, prob, conf, pred in [("RAW", raw1, c_raw, p_raw), ("CALIBRATED", cal1, c_cal, p_cal)]:
    ll = log_loss(y1, prob)
    acc = (pred == y1).mean()
    print(f"\n  1X2 {label} (LogLoss={ll:.4f}, Acc={acc*100:.1f}%)")
    print(f"  {'Esik':>6} {'Picks':>7} {'Oneri%':>7} {'Dogru':>7} {'Acc':>7}")
    print(f"  {'-'*40}")
    for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
        m=conf>=t; n=m.sum()
        if n>0:
            c=(pred[m]==y1[m]).sum()
            print(f"  {t*100:>5.0f}% {n:>7} {n/N*100:>6.1f}% {c:>7} {c/n*100:>6.1f}%")

# BTTS
ll_r = log_loss(yb, np.column_stack([1-raw_b,raw_b]))
ll_c = log_loss(yb, np.column_stack([1-cal_b,cal_b]))
print(f"\n  BTTS (Raw LogLoss={ll_r:.4f}, Cal LogLoss={ll_c:.4f})")
for lbl2, prob in [("Raw",raw_b),("Cal",cal_b)]:
    conf = np.maximum(prob,1-prob)
    pred = (prob>0.5).astype(int)
    acc = (pred==yb).mean()
    m60=conf>=0.60; n60=m60.sum()
    a60 = (pred[m60]==yb[m60]).mean()*100 if n60>10 else 0
    print(f"    {lbl2}: Acc={acc*100:.1f}% Acc60+={a60:.1f}% (n={n60})")

# O25
ll_r = log_loss(yo, np.column_stack([1-raw_o,raw_o]))
ll_c = log_loss(yo, np.column_stack([1-cal_o,cal_o]))
print(f"\n  O25 (Raw LogLoss={ll_r:.4f}, Cal LogLoss={ll_c:.4f})")
for lbl2, prob in [("Raw",raw_o),("Cal",cal_o)]:
    conf = np.maximum(prob,1-prob)
    pred = (prob>0.5).astype(int)
    acc = (pred==yo).mean()
    m60=conf>=0.60; n60=m60.sum()
    a60 = (pred[m60]==yo[m60]).mean()*100 if n60>10 else 0
    print(f"    {lbl2}: Acc={acc*100:.1f}% Acc60+={a60:.1f}% (n={n60})")

# 5. COMPARISON TABLE
print(f"\n{'='*80}")
print("  ONCEKI vs YENI (1X2 CALIBRATED)")
print("="*80)
print(f"  {'Metrik':<20} {'Onceki*':>10} {'Yeni':>10} {'Degisim':>10}")
print(f"  {'-'*50}")

# Onceki sonuclari approximate olarak kullaniyoruz
# (user'in dedigine gore ~45% accuracy vardi)
old_acc = 45.0
new_acc = (p_cal == y1).mean() * 100
print(f"  {'Accuracy':<20} {old_acc:>9.1f}% {new_acc:>9.1f}% {new_acc-old_acc:>+9.1f}%")

old_ll = 1.1  # approximate
new_ll = log_loss(y1, cal1)
print(f"  {'LogLoss':<20} {old_ll:>10.4f} {new_ll:>10.4f} {new_ll-old_ll:>+10.4f}")

print(f"\n  * Onceki sonuclar user raporuna gore tahmini")
print(f"\nToplam: {time.time()-t0:.0f}s")
