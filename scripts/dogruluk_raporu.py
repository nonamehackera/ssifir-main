"""DOGALLIK: Her piyasa icin kac/2000 dogru?"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat["over15"] = (feat["total_goals"] > 1.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

hc = pd.to_numeric(feat["home_corners"], errors="coerce").fillna(0)
ac = pd.to_numeric(feat["away_corners"], errors="coerce").fillna(0)
feat["total_corners"] = hc + ac
feat["corner_over85"] = (feat["total_corners"] > 8.5).astype(int)
feat["corner_reliable"] = feat["total_corners"] >= 5

test = feat.iloc[-2000:].reset_index(drop=True)
train = feat.iloc[:-2000].reset_index(drop=True)

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading",
        "home_xg","away_xg","total_xg_real","xg_diff_real","home_score_prob",
        "away_score_prob","btts_xprob","over25_xprob","draw_xprob","xg_diff_abs"}
NUM = ["home_elo","away_elo","elo_diff","home_gf_5","home_ga_5","home_pts_5",
       "home_shots_5","home_sot_5","home_corners_5","home_w_gf","home_w_ga",
       "home_w_shots","home_w_sot","home_w_corners","home_hgf_5","home_hga_5",
       "home_hpts_5","home_gf_3","home_ga_3","home_pts_3","home_gf_8","home_ga_8",
       "home_gf_20","home_ga_20","home_gf_std","home_ga_std","home_pts_std",
       "away_gf_5","away_ga_5","away_pts_5","away_shots_5","away_sot_5",
       "away_corners_5","away_w_gf","away_w_ga","away_w_shots","away_w_sot",
       "away_w_corners","away_agf_5","away_aga_5","away_apts_5","away_gf_3",
       "away_ga_3","away_pts_3","away_gf_8","away_ga_8","away_gf_20","away_ga_20",
       "away_gf_std","away_ga_std","away_pts_std","home_attack_elo","home_defence_elo",
       "away_attack_elo","away_defence_elo","attack_elo_diff","defence_elo_diff",
       "home_rest_days","away_rest_days","h2h_home_win","h2h_draw","h2h_away_win",
       "h2h_goals_avg","h2h_btts","home_opp_elo","away_opp_elo","lg_avg_goals",
       "lg_home_goal_avg","lg_away_goal_avg","lg_draw_rate","lg_btts_rate",
       "lg_over25_rate","lg_corner_avg","lg_card_avg","data_completeness",
       "mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob",
       "home_momentum","away_momentum","home_wins_last5","home_draws_last5",
       "away_wins_last5","away_draws_last5","home_form_std","away_form_std",
       "home_gdiff5","away_gdiff5","home_gf_5_real","home_ga_5_real",
       "away_gf_5_real","away_ga_5_real"]
SAFE = [c for c in NUM if c not in LEAK and c in train.columns]
for c in ["league","home_team_id","away_team_id","season"]:
    if c in train.columns:
        combined = pd.concat([train[c], test[c]], axis=0).astype("category").cat.codes
        train[c+"_enc"] = combined[:len(train)].values
        test[c+"_enc"] = combined[len(train):].values
        SAFE.append(c+"_enc")

X_tr = train[SAFE].fillna(0).values
X_te = test[SAFE].fillna(0).values
sp = int(len(X_tr) * 0.85)
LGB = dict(num_leaves=40, learning_rate=0.03, n_estimators=600, max_depth=5,
           min_child_samples=80, subsample=0.7, colsample_bytree=0.6,
           reg_alpha=0.5, reg_lambda=8.0, verbose=-1, random_state=42)

# 1X2
y1x2 = train["result"].map({"H":0,"D":1,"A":2}).to_numpy()
m1 = lgb.LGBMClassifier(objective="multiclass", num_class=3, **LGB)
m1.fit(X_tr[:sp], y1x2[:sp])
raw = m1.predict_proba(X_te)
# calibrate
val_raw = m1.predict_proba(X_tr[sp:])
cal = [IsotonicRegression(out_of_bounds="clip").fit(val_raw[:,c], (y1x2[sp:]==c).astype(float)) for c in range(3)]
ph = cal[0].predict(raw[:,0]); pd_ = cal[1].predict(raw[:,1]); pa = cal[2].predict(raw[:,2])
s = ph+pd_+pa; s=np.where(s==0,1,s); ph/=s; pd_/=s; pa/=s

# BTTS
m2 = lgb.LGBMClassifier(objective="binary", **LGB)
m2.fit(X_tr[:sp], train["btts"].values[:sp])
raw_bt = m2.predict_proba(X_tr[sp:])[:,1]
cal_bt = IsotonicRegression(out_of_bounds="clip").fit(raw_bt, train["btts"].values[sp:].astype(float))
btts_p = cal_bt.predict(m2.predict_proba(X_te)[:,1])

# O15
m3 = lgb.LGBMClassifier(objective="binary", **LGB)
m3.fit(X_tr[:sp], train["over15"].values[:sp])
raw_o15 = m3.predict_proba(X_tr[sp:])[:,1]
cal_o15 = IsotonicRegression(out_of_bounds="clip").fit(raw_o15, train["over15"].values[sp:].astype(float))
o15_p = cal_o15.predict(m3.predict_proba(X_te)[:,1])

# O25
m4 = lgb.LGBMClassifier(objective="binary", **LGB)
m4.fit(X_tr[:sp], train["over25"].values[:sp])
raw_o25 = m4.predict_proba(X_tr[sp:])[:,1]
cal_o25 = IsotonicRegression(out_of_bounds="clip").fit(raw_o25, train["over25"].values[sp:].astype(float))
o25_p = cal_o25.predict(m4.predict_proba(X_te)[:,1])

# Corner
cte = test["corner_reliable"].to_numpy().astype(bool)
ctr = train["corner_reliable"].to_numpy().astype(bool)
cor_p = np.full(len(test), 0.5)
if ctr.sum() > 500:
    m5 = lgb.LGBMRegressor(objective="poisson", num_leaves=30, learning_rate=0.02,
        n_estimators=400, max_depth=5, min_child_samples=40, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=8.0, verbose=-1, random_state=42)
    m5.fit(X_tr[ctr], train.loc[ctr,"total_corners"].values.astype(float))
    lam = np.maximum(m5.predict(X_te[cte]), 0.5)
    cor_p[cte] = 1 - poisson.cdf(8, lam)

# Gercek degerler
yr = test["result"].map({"H":0,"D":1,"A":2}).to_numpy()
ybt = test["btts"].to_numpy()
yo15 = test["over15"].to_numpy()
yo25 = test["over25"].to_numpy()
ycor = test["corner_over85"].to_numpy()
pick1x2 = np.argmax(np.column_stack([ph, pd_, pa]), axis=1)

N = 2000
print("=" * 60)
print(f"  2000 MAC TAHMIN DOGRULUK RAPORU")
print("=" * 60)

# 1X2
d1x2 = (pick1x2 == yr).sum()
print(f"\n1X2 (Mac Sonucu H/D/A)")
print(f"  Dogru: {d1x2} / {N}  =  %{d1x2/N*100:.1f}")
print(f"  Yanlis: {N - d1x2}")

# Cifte Sans
cs_1x = ((yr == 0) | (yr == 1))  # 1X dogru mu?
cs_x2 = ((yr == 1) | (yr == 2))  # X2 dogru mu?
cs_12 = ((yr == 0) | (yr == 2))  # 12 dogru mu?
# Model hangisini secti? -> en yuksek olasiliga gore cifte sans sec
p_1x = ph + pd_  # 1X olasiligi
p_x2 = pd_ + pa  # X2 olasiligi
p_12 = ph + pa   # 12 olasiligi
pick_cs = np.argmax(np.column_stack([p_1x, p_x2, p_12]), axis=1)
cs_true = np.column_stack([cs_1x, cs_x2, cs_12])
cs_dogru = 0
for i in range(N):
    if cs_true[i, pick_cs[i]]:
        cs_dogru += 1
print(f"\nCIFTE SANS (en yuksek olasilikli secilen)")
print(f"  Dogru: {cs_dogru} / {N}  =  %{cs_dogru/N*100:.1f}")
print(f"  Yanlis: {N - cs_dogru}")

# BTTS
bt_dogru = ((btts_p > 0.5).astype(int) == ybt).sum()
print(f"\nBTTS (Karsilikli Gol Var/Yok)")
print(f"  Dogru: {bt_dogru} / {N}  =  %{bt_dogru/N*100:.1f}")
print(f"  Yanlis: {N - bt_dogru}")

# Toplam Gol (O1.5 + O2.5 ayri ayri)
o15_dogru = ((o15_p > 0.5).astype(int) == yo15).sum()
o25_dogru = ((o25_p > 0.5).astype(int) == yo25).sum()
print(f"\nTOPLAM GOL UST/ALT")
print(f"  Ust 1.5  Dogru: {o15_dogru} / {N}  =  %{o15_dogru/N*100:.1f}")
print(f"  Ust 2.5  Dogru: {o25_dogru} / {N}  =  %{o25_dogru/N*100:.1f}")

# Korner (sadece guvenilir olanlar)
if cte.sum() > 0:
    cor_pred = (cor_p[cte] > 0.5).astype(int)
    cor_true = ycor[cte]
    cor_dogru = (cor_pred == cor_true).sum()
    cor_n = cte.sum()
    print(f"\nTOPLAM KORNER UST/ALT 8.5")
    print(f"  Dogru: {cor_dogru} / {cor_n}  =  %{cor_dogru/cor_n*100:.1f}  (guvenilir veri)")
    print(f"  Not: {N - cor_n} macta guvenilir korner verisi yok")
else:
    print(f"\nTOPLAM KORNER: Yeterli veri yok!")

print(f"\n{'='*60}")
print("OZET TABLOSU")
print(f"{'='*60}")
print(f"{'Piyasa':<28} {'Dogru':<10} {'Oran':<10}")
print(f"{'-'*48}")
print(f"{'1X2 (H/D/A)':<28} {d1x2:>5}/{N:<5} %{d1x2/N*100:.1f}")
print(f"{'Cifte Sans':<28} {cs_dogru:>5}/{N:<5} %{cs_dogru/N*100:.1f}")
print(f"{'BTTS':<28} {bt_dogru:>5}/{N:<5} %{bt_dogru/N*100:.1f}")
print(f"{'Toplam Gol Ust 1.5':<28} {o15_dogru:>5}/{N:<5} %{o15_dogru/N*100:.1f}")
print(f"{'Toplam Gol Ust 2.5':<28} {o25_dogru:>5}/{N:<5} %{o25_dogru/N*100:.1f}")
if cte.sum() > 0:
    print(f"{'Korner Ust 8.5':<28} {cor_dogru:>5}/{cor_n:<5} %{cor_dogru/cor_n*100:.1f}")
print(f"{'-'*48}")

# Rastgele tahmin olsaydi?
print(f"\nRastgele tahmin olsaydi:")
print(f"  1X2: ~33%  = ~667/{N}")
print(f"  BTTS: ~50% = ~1000/{N}")
print(f"  O1.5: ~78% = ~1560/{N} (baseline)")
print(f"  O2.5: ~58% = ~1156/{N} (baseline)")
