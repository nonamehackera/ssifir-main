"""Quick test - sadece 1 fold ile degisikliklari dogrula."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

t0 = time.time()
print("="*80)
print("  QUICK TEST - 1 FOLD")
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

# 2. Feature set - LEAK-FREE
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
    print(f"  WARNING: Leaked features: {leaked_in_cols}")
else:
    print(f"  OK: No leaked features")

# 3. Son 1 fold'u al (daha hizli)
MIN_TRAIN = 300000
TEST_W = min(40000, len(feat) - MIN_TRAIN)
tr_e = MIN_TRAIN
te_s = MIN_TRAIN
te_e = min(MIN_TRAIN + TEST_W, len(feat))

train = feat.iloc[:tr_e]
test = feat.iloc[te_s:te_e]

print(f"  Train: {len(train)} | Test: {len(test)}")

Xtr = train[ALL_COLS]; Xte = test[ALL_COLS]
y_1x2 = train["result"].map({"H":0,"D":1,"A":2}).values
y_1x2_te = test["result"].map({"H":0,"D":1,"A":2}).values

sp = int(len(Xtr)*0.85)

# TIME DECAY weights
t_days = (train["date"].max() - train["date"]).dt.days.clip(lower=0)
decay = np.exp(-1.0 * t_days / 365.0).values
draw_mask = (y_1x2 == 1)
sw_1x2 = decay.copy()
sw_1x2[draw_mask] *= 4.0

from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

# NEW MODEL (no leak, draw_weight=4.0, time_decay)
print("\n  Training NEW model (no leak, draw_weight=4.0, time_decay)...")
clf = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
    border_count=64,random_strength=2,bagging_temperature=1,
    cat_features=cat_idx,loss_function="MultiClass",
    class_weights={0:1,1:4.0,2:1},random_seed=42,verbose=False,allow_writing_files=False)
clf.fit(Xtr.iloc[:sp], y_1x2[:sp], use_best_model=True,
        eval_set=(Xtr.iloc[sp:], y_1x2[sp:]),
        sample_weight=sw_1x2[:sp])
raw_1x2 = clf.predict_proba(Xte)

# Calibration
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

c_cal = np.max(cal_p,axis=1)
p_cal = np.argmax(cal_p,axis=1)

# RESULTS
N = len(y_1x2_te)
ll_raw = log_loss(y_1x2_te, raw_1x2)
ll_cal = log_loss(y_1x2_te, cal_p)
acc_raw = (np.argmax(raw_1x2,axis=1) == y_1x2_te).mean()
acc_cal = (p_cal == y_1x2_te).mean()

print(f"\n  SONUCLAR ({N} test mac)")
print(f"  {'='*50}")
print(f"  RAW:     LogLoss={ll_raw:.4f} Acc={acc_raw*100:.1f}%")
print(f"  CAL:     LogLoss={ll_cal:.4f} Acc={acc_cal*100:.1f}%")
print(f"  ONCEKI*: LogLoss=~1.1   Acc=~45%")

print(f"\n  1X2 CALIBRATED Detail:")
print(f"  {'Esik':>6} {'Picks':>7} {'Oneri%':>7} {'Dogru':>7} {'Acc':>7}")
print(f"  {'-'*40}")
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    m=c_cal>=t; n=m.sum()
    if n>0:
        c=(p_cal[m]==y_1x2_te[m]).sum()
        print(f"  {t*100:>5.0f}% {n:>7} {n/N*100:>6.1f}% {c:>7} {c/n*100:>6.1f}%")

# Feature importance
print(f"\n  TOP 15 FEATURE IMPORTANCE:")
imp = clf.get_feature_importance()
feat_imp = sorted(zip(ALL_COLS, imp), key=lambda x: -x[1])
for i, (name, score) in enumerate(feat_imp[:15]):
    print(f"  {i+1:>2}. {name:<30} {score:>6.2f}%")

# Leak features check
print(f"\n  Leak feature kontrol:")
for name in ["ht_total_goals","ht_result_is_draw","ht_home_leading","second_half_goals"]:
    if name in ALL_COLS:
        idx = ALL_COLS.index(name)
        print(f"  {name}: FEATURE SETTE! (idx={idx})")
    else:
        print(f"  {name}: yok (OK)")

print(f"\nToplam: {time.time()-t0:.0f}s")
