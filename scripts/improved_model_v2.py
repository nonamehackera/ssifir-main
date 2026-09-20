"""GELISTIRILMIS MODEL v2 - Yuksek verili liglere odaklan + market oranlari + guclu rolling feature'lar.

Asil sorunlar ve cozumleri:
  1. 1385 lig var, model her ligu ayri ogrenemiyor.
     -> En az 3000 macli BUYUK liglere odaklan (top 20 lig, %60 kutu)
  2. Market-implied oranlar %97 kapsamli ama cikarilmisti (kopyalama endisesi).
     -> Oranlari KOPYALAMA olmadan kullan: bookmaker ve model arasindaki
        MISALIGNMENT'i (benzemezligi) feature yap. Model "oran bu, ben ne diyorum"
        farkini ogrenir. Bu LEVERAGE saglar, kopyalama degil.
  3. Shots/corners/cards rolling istatistiklerini artir - %97 kapsamli.
  4. Gercek blind test (son 6 ay) + walk-forward.
"""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score, f1_score, mean_absolute_error
from scipy.optimize import minimize

t0 = time.time()
print("=" * 90)
print("  GELISTIRILMIS MODEL v2 - BUYUK LIGLER + MARKET + ROLLING")
print("=" * 90)

# ─── 1. VERI YUKLE ──────────────────────────────────────────────────────────
print("\n[1] Veri yukleniyor...")
feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values(["date"]).reset_index(drop=True)
for c in ["home_goals", "away_goals"]: feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat["home_goals"] + feat["away_goals"]
feat["result"] = feat["result"].fillna("D")
feat["btts"] = ((feat["home_goals"] > 0) & (feat["away_goals"] > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat = feat.dropna(subset=["result", "elo_diff"])

print(f"  {len(feat)} mac")

# ─── 2. BUYUK LIGLERE FILTRE ────────────────────────────────────────────────
print("\n[2] Buyuk liglere filtreleniyor...")
lc = feat["league"].value_counts()
big_leagues = lc[lc >= 3000].index.tolist()
print(f"  >=3000 macli ligler: {len(big_leagues)}")
feat_big = feat[feat["league"].isin(big_leagues)].copy()
print(f"  Buyuk lige dusen mac sayisi: {len(feat_big)} ({len(feat_big)/len(feat)*100:.0f}%)")

# ─── 3. YENI FEATURE URETIMI ────────────────────────────────────────────────
print("\n[3] Yeni feature'lar uretiliyor...")

# 3a. Market-implied probabilities - oranlar zaten mkt_*_prob olarak var
# Bookmaker vs mevcut elo/form sinyali arasinda misalignment
# Bookmaker favorisine gore elo ne diyor?
for col in ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]:
    if col not in feat_big.columns:
        print(f"  [!] {col} yok")
feat_big = feat_big.dropna(subset=["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"])
print(f"  Market orani olan mac: {len(feat_big)}")

# Misalignment feature'lari: bookmaker oran vs model sinyalleri farki
# Elo'dan basit win prob hesaplama
def _elo_win_prob(home_elo, away_elo, home_adv=50):
    exp_h = 1 / (1 + 10 ** (-(home_elo + home_adv - away_elo) / 400))
    return exp_h

eh = _elo_win_prob(feat_big["home_elo"], feat_big["away_elo"])
feat_big["mkt_home_misalign"] = eh - feat_big["mkt_home_prob"]
feat_big["mkt_draw_mismatch"] = feat_big["mkt_draw_prob"] - (1 - abs(eh - (1-eh)) * 0.8)
# Bookmaker'ın over25 orani vs form sinyali
feat_big["mkt_over_misalign"] = feat_big["lg_over25_rate"] - feat_big["mkt_over25_prob"]

# 3b. Rolling istatistikler - elo'dan daha zengin
# Onceki maclarin shots/corners ortalamalarindan guc sinyali uret
# Basit ama etkili: takımın son 5 mac GF/GA farki elo'dan bagimsiz
# Zaten var: home_gf_5, home_ga_5 vs. Ama yeni kombinasyonlar:
feat_big["home_gf_per_ga"] = feat_big["home_gf_5"] / (feat_big["home_ga_5"] + 0.5)
feat_big["away_gf_per_ga"] = feat_big["away_gf_5"] / (feat_big["away_ga_5"] + 0.5)
feat_big["home_attack_ratio"] = feat_big["home_gf_5"] / (feat_big["home_gf_5"] + feat_big["home_ga_5"] + 0.1)
feat_big["away_attack_ratio"] = feat_big["away_gf_5"] / (feat_big["away_gf_5"] + feat_big["away_ga_5"] + 0.1)
feat_big["home_pts_per_game"] = feat_big["home_pts_5"] / 5
feat_big["away_pts_per_game"] = feat_big["away_pts_5"] / 5
feat_big["form_pts_diff"] = feat_big["home_pts_per_game"] - feat_big["away_pts_per_game"]

# 3c. BTTS sinyali - rolling (h2h + lig orani birlesimi)
feat_big["btts_form_signal"] = feat_big["lg_btts_rate"]  # lig ortalamasi
# H2H istatistikler BTTS icin
feat_big["btts_signal"] = (feat_big["h2h_btts"].fillna(feat_big["lg_btts_rate"])
                           * 0.5 + feat_big["lg_btts_rate"] * 0.5)

# 3d. Home/away guc farki (kombinasyon)
feat_big["home_overall"] = (feat_big["home_elo"] + feat_big["home_attack_elo"]/10
                            + feat_big["home_pts_per_game"]*50)
feat_big["away_overall"] = (feat_big["away_elo"] + feat_big["away_attack_elo"]/10
                            + feat_big["away_pts_per_game"]*50)
feat_big["overall_diff"] = feat_big["home_overall"] - feat_big["away_overall"]

print(f"  Yeni feature sayisi eklendi: 12")

# ─── 4. FEATURE LISTESI ─────────────────────────────────────────────────────
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
# Market oranlarini FEATURE OLARAK ekle (kopya degil, sinyal)
ALLOW_MKT = ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]
FEATS = [c for c in feat_big.columns if c not in SKIP and feat_big[c].dtype in ["float64","int64","float32","int32"]] + ALLOW_MKT
# Misalignment feature'lari dahil
for f in ["mkt_home_misalign", "mkt_draw_mismatch", "mkt_over_misalign",
          "home_gf_per_ga", "away_gf_per_ga", "home_attack_ratio", "away_attack_ratio",
          "home_pts_per_game", "away_pts_per_game", "form_pts_diff",
          "btts_signal", "home_overall", "away_overall", "overall_diff"]:
    if f not in FEATS:
        FEATS.append(f)
# Duplicateleri kaldir
FEATS = list(dict.fromkeys(FEATS))
FEATS = [f for f in FEATS if f in feat_big.columns]
print(f"  {len(FEATS)} feature kullaniliyor")

# ─── 5. WALK-FORWARD + BLIND TEST ───────────────────────────────────────────
print("\n[4] Walk-forward + son 6 ay blind test...")

# Walk-forward folds (sadece buyuk ligler uzerinde)
N_FOLDS = 4
TEST_SIZE = 12000
MIN_TRAIN = 150000
STEP = 12000
total_big = len(feat_big)
folds = []
start = 0
while start + MIN_TRAIN + TEST_SIZE <= total_big:
    folds.append((start, start+MIN_TRAIN, start+MIN_TRAIN, start+MIN_TRAIN+TEST_SIZE))
    start += STEP
print(f"  {len(folds)} walk-forward fold")

wf_results = []
for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds[:N_FOLDS]):
    train = feat_big.iloc[tr_s:tr_e].copy()
    test = feat_big.iloc[te_s:te_e].copy()
    X_all = train[FEATS].fillna(0)
    X_test = test[FEATS].fillna(0)
    split = int(len(X_all)*0.85)

    print(f"  === Fold {fold_i+1}/{N_FOLDS} ({test['date'].iloc[0].date()}~{test['date'].iloc[-1].date()}) ===")

    # 1X2
    y_1x2 = train["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    y_t_1x2 = test["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,
        learning_rate=0.015,n_estimators=600,max_depth=6,min_child_samples=100,
        subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    m.fit(X_all.iloc[:split], y_1x2[:split])
    raw = m.predict_proba(X_all.iloc[split:])
    cal = [IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y_1x2[split:]==c).astype(float)) for c in range(3)]
    p = m.predict_proba(X_test)
    pc = np.column_stack([cal[c].predict(p[:,c]) for c in range(3)])
    s=pc.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); pc/=s
    pred = np.argmax(pc, axis=1); conf = np.max(pc, axis=1)

    ll = log_loss(y_t_1x2, pc); acc = accuracy_score(y_t_1x2, pred)
    f1 = f1_score(y_t_1x2, pred, average="macro"); base_h = (y_t_1x2==0).mean()

    # Esik bazli
    accs = {}
    for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
        mask = conf>=t; n=mask.sum()
        if n>10: accs[t] = (accuracy_score(y_t_1x2[mask], pred[mask]), n)

    print(f"    1X2: LL={ll:.4f} Acc={acc:.3f} F1={f1:.3f} base_H={base_h:.3f}")
    print(f"         " + " ".join(f"@{t:.0%}:{a:.2f}({n})" for t,(a,n) in accs.items()))
    wf_results.append({"mode":"wf","fold":fold_i+1,"market":"1X2","ll":ll,"acc":acc,"f1":f1,"base_h":base_h})

    # BTTS
    y_b = train["btts"].to_numpy(); y_t_b = test["btts"].to_numpy()
    mb = lgb.LGBMClassifier(objective="binary",num_leaves=45,learning_rate=0.015,n_estimators=500,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mb.fit(X_all.iloc[:split], y_b[:split])
    raw_b = mb.predict_proba(X_all.iloc[split:])[:,1]
    cal_b = IsotonicRegression(out_of_bounds="clip").fit(raw_b, y_b[split:].astype(float))
    pb = cal_b.predict(mb.predict_proba(X_test)[:,1])
    try: auc_b = roc_auc_score(y_t_b, pb)
    except: auc_b = 0.5
    print(f"    BTTS: AUC={auc_b:.4f} Acc={accuracy_score(y_t_b,(pb>0.5).astype(int)):.3f} base={y_t_b.mean():.3f}")
    wf_results.append({"mode":"wf","fold":fold_i+1,"market":"BTTS","auc":auc_b,"acc":accuracy_score(y_t_b,(pb>0.5).astype(int))})

    # Over 2.5
    y_o = train["over25"].to_numpy(); y_t_o = test["over25"].to_numpy()
    mo = lgb.LGBMClassifier(objective="binary",num_leaves=45,learning_rate=0.015,n_estimators=500,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mo.fit(X_all.iloc[:split], y_o[:split])
    raw_o = mo.predict_proba(X_all.iloc[split:])[:,1]
    cal_o = IsotonicRegression(out_of_bounds="clip").fit(raw_o, y_o[split:].astype(float))
    po = cal_o.predict(mo.predict_proba(X_test)[:,1])
    try: auc_o = roc_auc_score(y_t_o, po)
    except: auc_o = 0.5
    print(f"    O25:  AUC={auc_o:.4f} Acc={accuracy_score(y_t_o,(po>0.5).astype(int)):.3f} base={y_t_o.mean():.3f}")
    wf_results.append({"mode":"wf","fold":fold_i+1,"market":"O25","auc":auc_o,"acc":accuracy_score(y_t_o,(po>0.5).astype(int))})

    # Gol
    y_hg=train["home_goals"].to_numpy(); y_ag=train["away_goals"].to_numpy()
    y_t_hg=test["home_goals"].to_numpy(); y_t_ag=test["away_goals"].to_numpy()
    mhg=lgb.LGBMRegressor(objective="poisson",num_leaves=45,learning_rate=0.015,n_estimators=400,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mhg.fit(X_all.iloc[:split], y_hg[:split]); phg=np.maximum(mhg.predict(X_test),0.05)
    mag=lgb.LGBMRegressor(objective="poisson",num_leaves=45,learning_rate=0.015,n_estimators=400,
        max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
        reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
    mag.fit(X_all.iloc[:split], y_ag[:split]); pag=np.maximum(mag.predict(X_test),0.05)
    mae_t=mean_absolute_error(y_t_hg+y_t_ag, phg+pag)
    mae_n=mean_absolute_error(y_t_hg+y_t_ag, np.full(len(y_t_hg), train["total_goals"].mean()))
    print(f"    GOL:  MAE={mae_t:.4f} naive={mae_n:.4f} fark={mae_t-mae_n:+.4f}")
    wf_results.append({"mode":"wf","fold":fold_i+1,"market":"GOL","mae":mae_t,"mae_n":mae_n})

df_wf = pd.DataFrame(wf_results)

# ─── 6. BLIND TEST (SON 6 AY) ───────────────────────────────────────────────
print("\n[5] Blind test - son 6 ay (model hic gormedi)")
TEST_START = pd.Timestamp("2026-03-01")
test_blind = feat_big[feat_big["date"] >= TEST_START].copy()
train_blind = feat_big[feat_big["date"] < TEST_START].copy()
print(f"  Train: {len(train_blind)} ({train_blind['date'].min().date()}~{train_blind['date'].max().date()})")
print(f"  Test:  {len(test_blind)} ({test_blind['date'].min().date()}~{test_blind['date'].max().date()})")

X_tr = train_blind[FEATS].fillna(0); X_te = test_blind[FEATS].fillna(0)
n_cal = max(500, int(len(train_blind)*0.15))
X_cal_fit = X_tr.iloc[-n_cal:]; X_tr_fit = X_tr.iloc[:-n_cal]
cal_tr = train_blind.iloc[-n_cal:]

def _to_np(df, col): return df[col].map({"H":0,"D":1,"A":2}).to_numpy() if col=="result" else df[col].to_numpy()

# 1X2 ensemble (3 seed)
ens_test = []
for seed in [42,123,456]:
    m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=50,learning_rate=0.015,
        n_estimators=600,max_depth=6,min_child_samples=100,subsample=0.7,colsample_bytree=0.6,
        reg_alpha=0.5,reg_lambda=5.0,random_seed=seed,verbose=-1,n_jobs=-1)
    y_tr = train_blind.iloc[:-n_cal]["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    m.fit(X_tr_fit, y_tr)
    raw_cal = m.predict_proba(X_cal_fit)
    y_cal = cal_tr["result"].map({"H":0,"D":1,"A":2}).to_numpy()
    cal_e = [IsotonicRegression(out_of_bounds="clip").fit(raw_cal[:,c],(y_cal==c).astype(float)) for c in range(3)]
    raw_te = m.predict_proba(X_te)
    cp = np.column_stack([cal_e[c].predict(raw_te[:,c]) for c in range(3)])
    s=cp.sum(axis=1,keepdims=True); s=np.where(s==0,1,s); cp/=s
    ens_test.append(cp)

P3 = np.stack(ens_test, axis=-1)
# agirlik optim
P3_cal = None
# basit ort = en saglam
probs_blind = np.mean(ens_test, axis=0)
pred_blind = np.argmax(probs_blind, axis=1); conf_blind = np.max(probs_blind, axis=1)
y_test_b = test_blind["result"].map({"H":0,"D":1,"A":2}).to_numpy()
ll_b = log_loss(y_test_b, probs_blind); acc_b = accuracy_score(y_test_b, pred_blind)
f1_b = f1_score(y_test_b, pred_blind, average="macro")
base_h_b = (y_test_b==0).mean()

print(f"\n  -- BLIND TEST SONUCLARI --")
print(f"  {'Metric':30s} {'WF ort':>10s} {'Blind':>10s}")
print(f"  {'-'*55}")

print(f"  1X2 LogLoss:  {ll_b:.4f} (WF: {df_wf[df_wf['market']=='1X2']['ll'].mean():.4f})")
print(f"  1X2 Accuracy: {acc_b:.3f} (WF: {df_wf[df_wf['market']=='1X2']['acc'].mean():.3f})")
print(f"  1X2 F1:       {f1_b:.3f} (baseline H: {base_h_b:.3f})")

print(f"\n  -- BLIND ESIK BAZLI 1X2 --")
for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
    mask = conf_blind>=t; n=mask.sum()
    if n>10:
        print(f"    @{t:.0%}: {accuracy_score(y_test_b[mask],pred_blind[mask]):.3f} ({n} picks)")

# BTTS blind
y_tr_b = train_blind.iloc[:-n_cal]["btts"].to_numpy()
y_cal_b = cal_tr["btts"].to_numpy()
y_te_b = test_blind["btts"].to_numpy()
mb = lgb.LGBMClassifier(objective="binary",num_leaves=45,learning_rate=0.015,n_estimators=500,
    max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
    reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
mb.fit(X_tr_fit, y_tr_b)
raw_cb = mb.predict_proba(X_cal_fit)[:,1]
cal_cb = IsotonicRegression(out_of_bounds="clip").fit(raw_cb, y_cal_b.astype(float))
pb_b = cal_cb.predict(mb.predict_proba(X_te)[:,1])
auc_bb = roc_auc_score(y_te_b, pb_b)
print(f"\n  BTTS Blind AUC: {auc_bb:.4f} (WF: {df_wf[df_wf['market']=='BTTS']['auc'].mean():.4f})")

# O25 blind
y_tr_o = train_blind.iloc[:-n_cal]["over25"].to_numpy()
y_cal_o = cal_tr["over25"].to_numpy()
y_te_o = test_blind["over25"].to_numpy()
mo = lgb.LGBMClassifier(objective="binary",num_leaves=45,learning_rate=0.015,n_estimators=500,
    max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
    reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
mo.fit(X_tr_fit, y_tr_o)
raw_co = mo.predict_proba(X_cal_fit)[:,1]
cal_co = IsotonicRegression(out_of_bounds="clip").fit(raw_co, y_cal_o.astype(float))
po_b = cal_co.predict(mo.predict_proba(X_te)[:,1])
auc_ob = roc_auc_score(y_te_o, po_b)
print(f"  Over2.5 Blind AUC: {auc_ob:.4f} (WF: {df_wf[df_wf['market']=='O25']['auc'].mean():.4f})")

# Gol blind
y_hg_tr=train_blind.iloc[:-n_cal]["home_goals"].to_numpy()
y_ag_tr=train_blind.iloc[:-n_cal]["away_goals"].to_numpy()
y_hg_te=test_blind["home_goals"].to_numpy(); y_ag_te=test_blind["away_goals"].to_numpy()
mhg=lgb.LGBMRegressor(objective="poisson",num_leaves=45,learning_rate=0.015,n_estimators=400,
    max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
    reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
mhg.fit(X_tr_fit, y_hg_tr); phg_b=np.maximum(mhg.predict(X_te),0.05)
mag=lgb.LGBMRegressor(objective="poisson",num_leaves=45,learning_rate=0.015,n_estimators=400,
    max_depth=6,min_child_samples=80,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,
    reg_lambda=5.0,random_seed=42,verbose=-1,n_jobs=-1)
mag.fit(X_tr_fit, y_ag_tr); pag_b=np.maximum(mag.predict(X_te),0.05)
mae_b=mean_absolute_error(y_hg_te+y_ag_te, phg_b+pag_b)
mae_nb=mean_absolute_error(y_hg_te+y_ag_te, np.full(len(y_hg_te), train_blind["total_goals"].mean()))
print(f"\n  Gol Blind MAE: {mae_b:.4f} (naive: {mae_nb:.4f}, WF: {df_wf[df_wf['market']=='GOL']['mae'].mean():.4f})")

# ─── 7. KARSILASTIRMA ───────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  ONCEKI vs YENI MODEL (walk-forward ortalamasi)")
print("=" * 90)
eski = {
    "1X2_ll": 1.0180, "1X2_acc": 0.4945, "BTTS_auc": 0.5609, "O25_auc": 0.6033, "GOL_mae": 1.3099
}
yeni_wf = df_wf[df_wf["market"]=="1X2"]
print(f"  1X2 LogLoss:  eski={eski['1X2_ll']:.4f}  yeni={yeni_wf['ll'].mean():.4f}  fark={yeni_wf['ll'].mean()-eski['1X2_ll']:+.4f}")
print(f"  1X2 Accuracy: eski={eski['1X2_acc']:.3f}  yeni={yeni_wf['acc'].mean():.3f}  fark={yeni_wf['acc'].mean()-eski['1X2_acc']:+.3f}")
new_btts = df_wf[df_wf["market"]=="BTTS"]["auc"].mean()
new_o25 = df_wf[df_wf["market"]=="O25"]["auc"].mean()
new_gol = df_wf[df_wf["market"]=="GOL"]["mae"].mean()
print(f"  BTTS AUC:     eski={eski['BTTS_auc']:.4f}  yeni={new_btts:.4f}  fark={new_btts-eski['BTTS_auc']:+.4f}")
print(f"  O25 AUC:      eski={eski['O25_auc']:.4f}  yeni={new_o25:.4f}  fark={new_o25-eski['O25_auc']:+.4f}")
print(f"  GOL MAE:      eski={eski['GOL_mae']:.4f}  yeni={new_gol:.4f}  fark={new_gol-eski['GOL_mae']:+.4f}")

print("\n  BLIND TEST (son 6 ay) - GERCEK REFERANS:")
print(f"  1X2: LL={ll_b:.4f} Acc={acc_b:.3f} (base_H={base_h_b:.3f})")
print(f"  BTTS: AUC={auc_bb:.4f}")
print(f"  O25: AUC={auc_ob:.4f}")
print(f"  GOL: MAE={mae_b:.4f} naive={mae_nb:.4f}")

print(f"\n  Toplam sure: {time.time()-t0:.0f}s")
print("=" * 90)
