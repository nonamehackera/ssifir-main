"""SIZINTISIZ (leakage-free) model egitimi ve test."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()

print("="*70)
print("  SIZINTISIZ MODEL (LEAKAGE-FREE)")
print("="*70)

# 1. Veriyi yukle
all_m = pd.read_parquet("data/gold/matches_all.parquet")
all_m["date"] = pd.to_datetime(all_m["date"], errors="coerce")
all_m = all_m.sort_values("date").reset_index(drop=True)

# Son 3000 test, geri kalani train
test_size = 3000
test_m = all_m.tail(test_size).copy()
train_m = all_m.iloc[:-test_size].copy()

print(f"  Train: {len(train_m)} mac ({train_m['date'].min().date()} -> {train_m['date'].max().date()})")
print(f"  Test: {len(test_m)} mac ({test_m['date'].min().date()} -> {test_m['date'].max().date()})")

###############################################################################
# ELO + FORM: SADECE TRAIN VERISIYLE HESAPLA, TEST'E UYGULA
###############################################################################
print("\n  ELO + Form hesaplanıyor (sadece train)...")

elo_ratings = {}
K = 32
HOME_ADV = 50
form_history = {}

def get_elo(team):
    return elo_ratings.get(team, 1500)

def update_elo(winner, loser, draw=False):
    e_w, e_l = get_elo(winner), get_elo(loser)
    s_w, s_l = (0.5, 0.5) if draw else (1.0, 0.0)
    exp_w = 1 / (1 + 10 ** ((e_l - e_w - HOME_ADV) / 400))
    elo_ratings[winner] = e_w + K * (s_w - exp_w)
    elo_ratings[loser] = e_l + K * (s_l - (1 - exp_w))

def get_form(team, n=5):
    if team not in form_history or len(form_history[team]) == 0:
        return [0.0, 0.0, 0.0]
    r = form_history[team][-n:]
    return [np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), np.mean([x[2] for x in r])]

def compute_features(df, elo_r, form_h, is_train=True):
    """Her satir icin pre-match feature hesapla."""
    features = {k: [] for k in ["home_elo","away_elo","elo_diff",
        "home_gf_5","home_ga_5","home_pts_5","away_gf_5","away_ga_5","away_pts_5",
        "home_gf_3","home_ga_3","away_gf_3","away_ga_3",
        "home_w_gf","home_w_ga","away_w_gf","away_w_ga",
        "home_rest_days","away_rest_days","home_opp_elo","away_opp_elo"]}
    
    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        result = row["result"]
        
        h_elo = elo_r.get(home, 1500)
        a_elo = elo_r.get(away, 1500)
        features["home_elo"].append(h_elo)
        features["away_elo"].append(a_elo)
        features["elo_diff"].append(h_elo - a_elo + HOME_ADV)
        
        hf = get_form(home, 5)
        af = get_form(away, 5)
        features["home_gf_5"].append(hf[0])
        features["home_ga_5"].append(hf[1])
        features["home_pts_5"].append(hf[2])
        features["away_gf_5"].append(af[0])
        features["away_ga_5"].append(af[1])
        features["away_pts_5"].append(af[2])
        
        hf3 = get_form(home, 3)
        af3 = get_form(away, 3)
        features["home_gf_3"].append(hf3[0])
        features["home_ga_3"].append(hf3[1])
        features["away_gf_3"].append(af3[0])
        features["away_ga_3"].append(af3[1])
        
        features["home_w_gf"].append(hf[0])
        features["home_w_ga"].append(hf[1])
        features["away_w_gf"].append(af[0])
        features["away_w_ga"].append(af[1])
        features["home_rest_days"].append(5)
        features["away_rest_days"].append(5)
        features["home_opp_elo"].append(a_elo)
        features["away_opp_elo"].append(h_elo)
        
        if is_train:
            pts = 3 if result == "H" else (1 if result == "D" else 0)
            pts_a = 3 if result == "A" else (1 if result == "D" else 0)
            if home not in form_history: form_history[home] = []
            form_history[home].append((hg, ag, pts))
            if away not in form_history: form_history[away] = []
            form_history[away].append((ag, hg, pts_a))
            
            if result == "H":
                update_elo(home, away)
            elif result == "A":
                update_elo(away, home)
            else:
                update_elo(home, away, draw=True)
    
    for k, v in features.items():
        df[k] = v
    return df

train_m = compute_features(train_m, elo_ratings, form_history, is_train=True)
test_m = compute_features(test_m, elo_ratings, form_history, is_train=False)

# League avg (train only)
lg_stats = {}
for _, row in train_m.iterrows():
    lg = row["league"]
    if lg not in lg_stats: lg_stats[lg] = []
    total = row["home_goals"] + row["away_goals"]
    lg_stats[lg].append((total, row["home_goals"], row["away_goals"],
                        1 if row["result"]=="D" else 0,
                        1 if (row["home_goals"]>0 and row["away_goals"]>0) else 0,
                        1 if total>2.5 else 0))

lg_avg = {}
for lg, stats in lg_stats.items():
    if len(stats) >= 5:
        lg_avg[lg] = (np.mean([s[0] for s in stats]), np.mean([s[1] for s in stats]),
                      np.mean([s[2] for s in stats]), np.mean([s[3] for s in stats]),
                      np.mean([s[4] for s in stats]), np.mean([s[5] for s in stats]))
default_lg = (2.5, 1.3, 1.2, 0.25, 0.5, 0.45)

for df in [train_m, test_m]:
    df["lg_avg_goals"] = df["league"].map(lambda x: lg_avg.get(x, default_lg)[0])
    df["lg_home_goal_avg"] = df["league"].map(lambda x: lg_avg.get(x, default_lg)[1])
    df["lg_away_goal_avg"] = df["league"].map(lambda x: lg_avg.get(x, default_lg)[2])
    df["lg_draw_rate"] = df["league"].map(lambda x: lg_avg.get(x, default_lg)[3])
    df["lg_btts_rate"] = df["league"].map(lambda x: lg_avg.get(x, default_lg)[4])
    df["lg_over25_rate"] = df["league"].map(lambda x: lg_avg.get(x, default_lg)[5])
    df["mkt_home_prob"] = 1.0 / df["lg_home_goal_avg"].clip(lower=0.5)
    df["mkt_draw_prob"] = df["lg_draw_rate"]
    df["mkt_away_prob"] = 1.0 / df["lg_away_goal_avg"].clip(lower=0.5)
    df["mkt_over25_prob"] = df["lg_over25_rate"]
    df["home_score_prob"] = df["lg_home_goal_avg"] / df["lg_avg_goals"].clip(lower=1)
    df["away_score_prob"] = df["lg_away_goal_avg"] / df["lg_avg_goals"].clip(lower=1)
    df["btts_xprob"] = df["lg_btts_rate"]
    df["over25_xprob"] = df["lg_over25_rate"]
    df["draw_xprob"] = df["lg_draw_rate"]

print(f"  Train: {len(train_m)}, Test: {len(test_m)}")

###############################################################################
# MODEL
###############################################################################
FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_gf_5", "home_ga_5", "home_pts_5",
    "away_gf_5", "away_ga_5", "away_pts_5",
    "home_gf_3", "home_ga_3", "away_gf_3", "away_ga_3",
    "home_w_gf", "home_w_ga", "away_w_gf", "away_w_ga",
    "home_rest_days", "away_rest_days", "home_opp_elo", "away_opp_elo",
    "lg_avg_goals", "lg_home_goal_avg", "lg_away_goal_avg",
    "lg_draw_rate", "lg_btts_rate", "lg_over25_rate",
    "mkt_home_prob", "mkt_draw_prob", "mkt_away_prob", "mkt_over25_prob",
    "home_score_prob", "away_score_prob", "btts_xprob", "over25_xprob", "draw_xprob",
]
f_avail = [f for f in FEATURES if f in train_m.columns]
print(f"\n  Features: {len(f_avail)}")

train_c = train_m.dropna(subset=["result","btts","over25"]).copy()

def prep(df):
    return df[f_avail].copy()

# 1X2
print("\n  1X2 egitiliyor...")
X_tr = prep(train_c)
y_enc = np.array([{"H":0,"D":1,"A":2}[r] for r in train_c["result"]])

models_1x2 = []
for s in [42, 123, 456]:
    m = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=40,
        learning_rate=0.02, n_estimators=1000, max_depth=5, min_child_samples=80,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0,
        random_seed=s, verbose=-1)
    split = int(len(X_tr) * 0.8)
    m.fit(X_tr.iloc[:split], y_enc[:split])
    raw_cal = m.predict_proba(X_tr.iloc[split:])
    calibrators = []
    for cls in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_cal[:, cls], (y_enc[split:] == cls).astype(float))
        calibrators.append(ir)
    models_1x2.append((m, calibrators))

X_test = prep(test_m)
p1x2_list = []
for m, cal in models_1x2:
    raw = m.predict_proba(X_test)
    cal_p = np.column_stack([cal[c].predict(raw[:, c]) for c in range(3)])
    s = cal_p.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    cal_p /= s
    p1x2_list.append(cal_p)
p1x2 = np.mean(p1x2_list, axis=0)
pw = np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
     np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
conf1x2 = np.max(p1x2, axis=1)

# BTTS
print("  BTTS egitiliyor...")
models_btts = []
for s in [42, 123, 456]:
    m = lgb.LGBMClassifier(objective="binary", num_leaves=36, learning_rate=0.02,
        n_estimators=800, max_depth=5, min_child_samples=60, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1)
    split = int(len(X_tr) * 0.8)
    m.fit(X_tr.iloc[:split], train_c["btts"].values[:split])
    raw_cal = m.predict_proba(X_tr.iloc[split:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, train_c["btts"].values[split:].astype(float))
    models_btts.append((m, ir))

pbt_list = []
for m, ir in models_btts:
    pbt_list.append(ir.predict(m.predict_proba(X_test)[:, 1]))
pbt_p = np.mean(pbt_list, axis=0)
pbtth = (pbt_p > 0.5).astype(int)
confbtts = np.maximum(pbt_p, 1-pbt_p)

# GOL UST 2.5
print("  Gol UST 2.5 egitiliyor...")
models_o25 = []
for s in [42, 123, 456]:
    m = lgb.LGBMClassifier(objective="binary", num_leaves=36, learning_rate=0.02,
        n_estimators=800, max_depth=5, min_child_samples=60, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0, random_seed=s, verbose=-1)
    split = int(len(X_tr) * 0.8)
    m.fit(X_tr.iloc[:split], train_c["over25"].values[:split])
    raw_cal = m.predict_proba(X_tr.iloc[split:])[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, train_c["over25"].values[split:].astype(float))
    models_o25.append((m, ir))

po2_list = []
for m, ir in models_o25:
    po2_list.append(ir.predict(m.predict_proba(X_test)[:, 1]))
po2_p = np.mean(po2_list, axis=0)
po2ens = (po2_p > 0.5).astype(int)
confou25 = np.maximum(po2_p, 1-po2_p)

###############################################################################
# RAPOR
###############################################################################
y_result = test_m["result"].values
y_btts = test_m["btts"].values.astype(int)
y_over25 = test_m["over25"].values.astype(int)

def acc(p,a): return f"{(p==a).sum()}/{len(a)} = {(p==a).sum()/len(a)*100:.1f}%"
sec = lambda t: print(f"\n{'='*70}\n  {t}\n{'='*70}")

sec(f"TEST: {len(test_m)} MAC ({test_m['date'].min().date()} -> {test_m['date'].max().date()})")
print(f"  Train: {len(train_m)} mac")
print(f"  Test: {len(test_m)} mac")

# Kaynak dagilimi
print(f"\n  Test seti kaynaklari:")
for src in ["HF_", "XG_", "TM_", "FC_"]:
    n = test_m[test_m["match_id"].str.startswith(src, na=False)]
    if len(n) > 0:
        print(f"    {src}: {len(n)}")
fd_test = test_m[~test_m["match_id"].str.startswith(("HF_","XG_","TM_","FC_"), na=False)]
if len(fd_test) > 0:
    print(f"    FD/diger: {len(fd_test)}")

for name, preds, conf, actual in [
    ("1X2", pw, conf1x2, y_result),
    ("BTTS", pbtth, confbtts, y_btts),
    ("GOL UST 2.5", po2ens, confou25, y_over25),
]:
    sec(name)
    print(f"  Toplam: {acc(preds, actual)}")
    print(f"\n  {'Guven':>10} {'Adet':>6} {'Dogru':>6} {'Yanlis':>6} {'Acc':>8}")
    print("  " + "-"*45)
    for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.01)]:
        m=(conf>=lo)&(conf<hi); n=m.sum()
        if n>0:
            c=(preds[m]==actual[m]).sum()
            print(f"  {lo*100:>4.0f}-{hi*100:<4.0f}%  {n:>5}  {c:>5}  {n-c:>5}  {c/n*100:>7.1f}%")

sec("70%+ GUVEN OZETI")
for name, preds, conf, actual in [
    ("1X2", pw, conf1x2, y_result),
    ("BTTS", pbtth, confbtts, y_btts),
    ("GOL UST 2.5", po2ens, confou25, y_over25),
]:
    m70 = conf >= 0.70
    n = m70.sum()
    if n > 0:
        c = (preds[m70] == actual[m70]).sum()
        print(f"  {name:<20} {n:>5} oneri / {len(preds):>5} toplam  ->  {c}/{n} = {c/n*100:.1f}%")
    else:
        print(f"  {name:<20} 0 oneri")

sec("80%+ GUVEN OZETI")
for name, preds, conf, actual in [
    ("1X2", pw, conf1x2, y_result),
    ("BTTS", pbtth, confbtts, y_btts),
    ("GOL UST 2.5", po2ens, confou25, y_over25),
]:
    m80 = conf >= 0.80
    n = m80.sum()
    if n > 0:
        c = (preds[m80] == actual[m80]).sum()
        print(f"  {name:<20} {n:>5} oneri / {len(preds):>5} toplam  ->  {c}/{n} = {c/n*100:.1f}%")
    else:
        print(f"  {name:<20} 0 oneri")

sec("90%+ GUVEN OZETI")
for name, preds, conf, actual in [
    ("1X2", pw, conf1x2, y_result),
    ("BTTS", pbtth, confbtts, y_btts),
    ("GOL UST 2.5", po2ens, confou25, y_over25),
]:
    m90 = conf >= 0.90
    n = m90.sum()
    if n > 0:
        c = (preds[m90] == actual[m90]).sum()
        print(f"  {name:<20} {n:>5} oneri / {len(preds):>5} toplam  ->  {c}/{n} = {c/n*100:.1f}%")
    else:
        print(f"  {name:<20} 0 oneri")

# Feature importance
sec("FEATURE IMPORTANCE (1X2)")
imp = models_1x2[0][0].feature_importances_
fi = sorted(zip(f_avail, imp), key=lambda x: -x[1])
for fname, fimp in fi[:15]:
    print(f"  {fname:<30} {fimp:>6}")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
