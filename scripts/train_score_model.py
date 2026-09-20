"""SKOR DAGILIMI MODELI (Expected Goals + Poisson).

Yaklasim:
  1. Ev sahibi ve deplasman icin BEKLENEN GOL (xG_home, xG_away) regresyonu
     -> LightGBM regressor (tum zengin feature'larla)
  2. BTTS (var/yok) classification
  3. Korner icin BEKLENEN KORNER (kc_home, kc_away) regresyonu
  4. Poisson dagilimi ile skor matrisi -> tum pazarlarin olasiligi

Cikti: hem 'ust/alt' hem de 'kac gol korner bekleniyor' raporu.
Ayrica 3000 mac uzerinde GERCEK tutma orani.

Usage: python scripts/train_score_model.py
"""
import sys, os, time, warnings, math, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

df = pd.read_parquet("data/gold/features_master.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
N = 3000
test = df.tail(N).copy()
train = df.iloc[:-N].copy()
P(f"Train: {len(train):,} | Test: {len(test):,}")

SKIP = {"match_id","index","round","league","league_name","season","date","home_team","away_team",
    "home_team_id","away_team_id","home_goals","away_goals","result","result_H","result_D","result_A",
    "btts","over05","over15","over25","over35","over45","total_goals","under05","under15","under25",
    "under35","under45","double_1X","double_X2","double_12","total_corners","home_shots","away_shots",
    "home_sot","away_sot","home_xg","away_xg","home_corners","away_corners","ht_home_goals","ht_away_goals",
    "ht_total_goals","home_fouls","away_fouls","home_yellow","away_yellow","home_red","away_red",
    "referee","stadium","source","_src","attendance","home_minutes","away_minutes","home_assists","away_assists",
    "sot_diff","shot_diff","corner_diff","xg_diff","xg_home","xg_away","imp_home","imp_draw","imp_away",
    "h2h_home_win_rate","home_form_gf_5","home_form_ga_5","home_form_pts_5","home_form_gf_20","home_form_ga_20",
    "home_form_pts_20","away_form_gf_5","away_form_ga_5","away_form_pts_5","away_form_gf_20","away_form_ga_20",
    "away_form_pts_20","home_home_gf_10","home_home_ga_10","home_home_pts_10","away_away_gf_10","away_away_ga_10",
    "away_away_pts_10","lg_gpg","lg_hwin","lg_draw","cor_over_75","cor_over_85","cor_over_95","cor_over_105",
    "cor_under_75","cor_under_85","cor_under_95","cor_under_105"}
FEATS = [c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
for f in FEATS:
    df[f] = pd.to_numeric(df[f], errors="coerce")
sp = int(len(train)*0.85)

def train_reg(X, y, strong=False):
    ms=[]
    for s in [42,123,456]:
        if strong:
            # gol/korner regresyonu icin daha az regularization (underbias onlemek)
            m=lgb.LGBMRegressor(num_leaves=64,learning_rate=0.03,n_estimators=700,max_depth=8,
                min_child_samples=20,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=0.5,
                random_seed=s,verbose=-1,n_jobs=-1)
        else:
            m=lgb.LGBMRegressor(num_leaves=48,learning_rate=0.02,n_estimators=500,max_depth=6,
                min_child_samples=50,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.3,reg_lambda=4.0,
                random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp], y[:sp])
        ms.append(m)
    return ms

def pred_reg(ms, X):
    return np.mean([m.predict(X) for m in ms], axis=0)

def train_clf(col, feats=FEATS):
    tc=train.dropna(subset=[col])
    X=tc[feats].fillna(0); y=tc[col].values.astype(int)
    ms=[]
    valp=[]
    for s in [42,123,456]:
        m=lgb.LGBMClassifier(objective="binary",num_leaves=48,learning_rate=0.02,n_estimators=500,
            max_depth=6,min_child_samples=50,subsample=0.7,colsample_bytree=0.6,reg_alpha=0.3,
            reg_lambda=4.0,random_seed=s,verbose=-1,n_jobs=-1)
        m.fit(X.iloc[:sp],y[:sp])
        raw=m.predict_proba(X.iloc[sp:])
        cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1],y[sp:].astype(float))
        ms.append((m,cal))
        valp.append(cal.predict(raw[:,1]))
    return ms, np.mean(valp,axis=0), y[sp:]  # (models, val_probs, val_true)

def pred_clf(ms, X):
    ps=[]
    for m,cal in ms:
        raw=m.predict_proba(X)
        ps.append(cal.predict(raw[:,1]))
    return np.mean(ps,axis=0)

# ---- 1. EXPECTED GOALS (xG_home, xG_away) ----
P("  1. Expected Goals (ev/deplasman xG) regresyonu...")
tr_h = train.dropna(subset=["home_goals"]); Xh=tr_h[FEATS].fillna(0)
m_h = train_reg(Xh, tr_h["home_goals"].values, strong=True)
m_a = train_reg(train.dropna(subset=["away_goals"])[FEATS].fillna(0), train.dropna(subset=["away_goals"])["away_goals"].values, strong=True)
xg_home = np.clip(pred_reg(m_h, test[FEATS].fillna(0)), 0.1, 8)
xg_away = np.clip(pred_reg(m_a, test[FEATS].fillna(0)), 0.1, 8)
P(f"    bitti ({time.time()-t0:.0f}s)")

# ---- 2. EXPECTED CORNERS ----
P("  2. Expected Corners regresyonu...")
tr_c = train.dropna(subset=["home_corners"])
Xc = tr_c[FEATS].fillna(0)
m_ch = train_reg(Xc, tr_c["home_corners"].values, strong=True)
m_ca = train_reg(train.dropna(subset=["away_corners"])[FEATS].fillna(0),
                 train.dropna(subset=["away_corners"])["away_corners"].values, strong=True)
kc_home = np.clip(pred_reg(m_ch, test[FEATS].fillna(0)), 0.1, 20)
kc_away = np.clip(pred_reg(m_ca, test[FEATS].fillna(0)), 0.1, 20)
P(f"    bitti ({time.time()-t0:.0f}s)")

# ---- 3. BTTS ----
P("  3. BTTS (var/yok)...")
m_btts, _, _ = train_clf("btts")
p_btts = pred_clf(m_btts, test[FEATS].fillna(0))
P(f"    bitti ({time.time()-t0:.0f}s)")

# ---- 3b. DRAW CLASSIFIER (beraberlik tahmini) ----
P("  3b. Draw classifier (D mi?)...")
train["is_draw"] = (train["result"]=="D").astype(int)
m_draw, draw_valp, draw_valy = train_clf("is_draw")
p_draw = pred_clf(m_draw, test[FEATS].fillna(0))
# OPTIMAL ESİK (validation'da en yuksek F1)
from sklearn.metrics import f1_score
best_th, best_f1 = 0.5, 0
for t in np.arange(0.3, 0.75, 0.01):
    f1 = f1_score(draw_valy, (draw_valp>=t).astype(int))
    if f1 > best_f1:
        best_f1, best_th = f1, t
P(f"    optimal draw esigi: {best_th:.2f} (F1={best_f1:.3f})")

# ---- 4. POISSON SKOR MATRISI ----
P("  4. Poisson skor dagilimi hesaplaniyor...")
def poisson_pmf(lam, k):
    return np.exp(-lam) * (lam**k) / np.array([math.factorial(i) for i in range(len(k))])

maxg = 12
rows=[]
for i in range(N):
    lh=xg_home[i]; la=xg_away[i]
    ph=np.array([np.exp(-lh)*(lh**k)/math.factorial(k) for k in range(maxg)])
    pa=np.array([np.exp(-la)*(la**k)/math.factorial(k) for k in range(maxg)])
    mat=np.outer(ph,pa)  # [home_goals][away_goals]
    total=mat.sum(axis=(0,1))
    mat/=total
    # EN OLASI SKOR (mode of distribution)
    hi,ai=np.unravel_index(np.argmax(mat), mat.shape)
    # pazar olasiliklari
    rows.append({
        "xg_home":lh,"xg_away":la,
        "p_1x2_H": np.trace(mat, offset=1),
        "p_1x2_D": np.trace(mat),
        "p_1x2_A": np.trace(mat, offset=-1),
        "p_over05": 1-mat[0,0],
        "p_over15": 1-(mat[:2,:2].sum()),
        "p_over25": 1-(mat[:3,:3].sum()),
        "p_over35": 1-(mat[:4,:4].sum()),
        "p_btts": (mat[1:,1:].sum()),
        "exp_total_goals": lh+la,
        "pred_h": int(hi), "pred_a": int(ai),
    })
score_df = pd.DataFrame(rows)
P(f"    bitti ({time.time()-t0:.0f}s)")

# ---- 5. KORNER POISSON ----
P("  5. Korner dagilimi...")
kc_total = kc_home + kc_away
rows_k=[]
for i in range(N):
    lam=kc_total[i]
    pk=np.array([np.exp(-lam)*(lam**k)/math.factorial(k) for k in range(25)])
    pk=pk/pk.sum()
    ki=np.argmax(pk)  # en olasi korner sayisi
    rows_k.append({
        "kc_home":kc_home[i],"kc_away":kc_away[i],
        "p_cor_over75": 1-(pk[:8].sum()),
        "p_cor_over85": 1-(pk[:9].sum()),
        "p_cor_over95": 1-(pk[:10].sum()),
        "exp_corners": lam,
        "pred_corners": int(ki),
    })
kor_df = pd.DataFrame(rows_k)
P(f"    bitti ({time.time()-t0:.0f}s)")

# ============ RAPOR (SENIN MANTIGIN: tahmin edilen skora gore alt/ust) ============
P(f"\n{'='*78}\n  SKOR DAGILIMI MODELI - GERCEK TUTMA ORANI (3000 mac)\n  [Gol: SADECE 1.5 | Korner: SADECE 7.5 / 8.5 | Cifte Sans | BTTS]\n  Mantik: model tahmin ettigi gol/korner sayisina gore ALT/UST der\n  (orn. 1-0 tahmin etti -> 1 gol -> 1.5 ALT)\n  Yesil (>>>) = o pazardaki EN YUKSEK tutma ihtimali\n{'='*78}")

# REAL (test) degerleri
real_over15 = test["over15"].values
real_btts   = test["btts"].values.astype(int)
real_cor75  = test["cor_over_75"].values
real_cor85  = test["cor_over_85"].values
# Cifte sans: D yoksa 1X (ev veya beraberlik), A yoksa X2, H/A varsa 12
real_1x  = ((test["result"]!="A").astype(int)).values
real_x2  = ((test["result"]!="H").astype(int)).values
real_12  = ((test["result"]=="H")|(test["result"]=="A")).astype(int).values

# TAHMIN EDILEN DEGERE GORE KARAR
pred_total = score_df["pred_h"] + score_df["pred_a"]          # tahmin edilen toplam gol
pred_g15 = (pred_total >= 2).astype(int)                      # 1.5 ust = 2+ gol
pred_cor75 = (kor_df["pred_corners"] >= 8).astype(int)        # 7.5 ust = 8+ korner
pred_cor85 = (kor_df["pred_corners"] >= 9).astype(int)        # 8.5 ust = 9+ korner
pred_btts = (score_df["p_btts"] >= 0.5).astype(int)
# KAZANAN (1X2) - DRAW-AWARE: optimal esikle D der
def winner_draw_aware(i):
    if p_draw[i] > best_th:   # beraberlik olasiligi yuksek (optimal esik)
        return "D"
    # degilse Poisson dagilimindan en yuksek olasilikli (H veya A)
    if score_df["p_1x2_H"].iloc[i] >= score_df["p_1x2_A"].iloc[i]:
        return "H"
    return "A"
pred_winner = np.array([winner_draw_aware(i) for i in range(N)])
# Cifte sans: her birini ayri degerlendir (12 gercekte en cok tutan olmali)
p_1x = score_df["p_1x2_H"] + score_df["p_1x2_D"]
p_x2 = score_df["p_1x2_D"] + score_df["p_1x2_A"]
p_12 = score_df["p_1x2_H"] + score_df["p_1x2_A"]
# en yuksek P olan cifte sans secimi (dogru: 12 de dahil)
def choose_double(i):
    probs={"1X":p_1x.iloc[i],"X2":p_x2.iloc[i],"12":p_12.iloc[i]}
    return max(probs, key=probs.get)
pred_double = [choose_double(i) for i in range(N)]
pred_1x_v = np.array([1 if d in ("1X","12") else 0 for d in pred_double])
pred_x2_v = np.array([1 if d in ("X2","12") else 0 for d in pred_double])
pred_12_v = np.array([1 if d=="12" else 0 for d in pred_double])

def hold_rate(pred, real):
    n=len(pred); c=(pred==real).sum(); return n,c,c/n*100

# TUM PAZARLAR INDIRILMIS HOLD
all_res = {
    "Gol Ust 1.5": hold_rate(pred_g15, real_over15),
    "Gol Alt 1.5": hold_rate(1-pred_g15, 1-real_over15),
    "BTTS Var": hold_rate(pred_btts, real_btts),
    "BTTS Yok": hold_rate(1-pred_btts, 1-real_btts),
    "Korner Ust 7.5": hold_rate(pred_cor75, real_cor75),
    "Korner Alt 7.5": hold_rate(1-pred_cor75, 1-real_cor75),
    "Korner Ust 8.5": hold_rate(pred_cor85, real_cor85),
    "Korner Alt 8.5": hold_rate(1-pred_cor85, 1-real_cor85),
    "Cifte Sans 1X": hold_rate(pred_1x_v, real_1x),
    "Cifte Sans X2": hold_rate(pred_x2_v, real_x2),
    "Cifte Sans 12": hold_rate(pred_12_v, real_12),
    "Kazanan (H)": hold_rate((pred_winner=="H").astype(int), (test["result"]=="H").astype(int).values),
    "Kazanan (D)": hold_rate((pred_winner=="D").astype(int), (test["result"]=="D").astype(int).values),
    "Kazanan (A)": hold_rate((pred_winner=="A").astype(int), (test["result"]=="A").astype(int).values),
}
# EN YUKSEK TUTMA IHTIMALI OLAN PAZARLAR (grup bazinda)
groups = {
    "GOL 1.5": ["Gol Ust 1.5","Gol Alt 1.5"],
    "BTTS": ["BTTS Var","BTTS Yok"],
    "KORNER 7.5": ["Korner Ust 7.5","Korner Alt 7.5"],
    "KORNER 8.5": ["Korner Ust 8.5","Korner Alt 8.5"],
    "CIFTE SANS": ["Cifte Sans 1X","Cifte Sans X2","Cifte Sans 12"],
    "KAZANAN": ["Kazanan (H)","Kazanan (D)","Kazanan (A)"],
}
best = {}
for g, members in groups.items():
    best[g] = max(members, key=lambda m: all_res[m][2])

P(f"\n  {'PAZAR':<22}{'TAHMIN':>8}{'TUTAN':>8}{'ORAN':>8}  NOT")
P(f"  {'-'*52}")
for name,(n,c,acc) in sorted(all_res.items(), key=lambda x:-x[1][2]):
    mark = "  >>> EN IYI" if name in best.values() else ""
    P(f"  {name:<22}{n:>8}{c:>8}{acc:>7.1f}%{mark}")

# DAGILIM: model kac mac ust / alt dedi
P(f"\n{'='*78}\n  MODELIN TAHMİN ETTİĞİ YÖN DAĞILIMI (senin istediğin: hem alt hem üst)\n{'='*78}")
for label,pred in [("Gol 1.5",pred_g15),("Korner 7.5",pred_cor75),("Korner 8.5",pred_cor85)]:
    ust=(pred==1).sum(); alt=(pred==0).sum()
    P(f"  {label:<12} ÜST dedi: {ust:>4} maç | ALT dedi: {alt:>4} maç")

# ORNEK MAÇLAR - tahmin edilen skor + her pazarın tahmini + en iyi yesil
P(f"\n{'='*78}\n  ORNEK MAÇLAR (ilk 20) - MODEL TAHMİNİ\n{'='*78}")
P(f"  {'#':>3} {'SKOR':>5} | 1.5:{'':>5} BTTS:{'':>5} 7.5:{'':>5} 8.5:{'':>5} DC:{'':>4} WIN:{'':>3}")
for i in range(20):
    r=score_df.iloc[i]; k=kor_df.iloc[i]
    g15="ÜST" if pred_g15.iloc[i]==1 else "ALT"
    bt="Var" if pred_btts.iloc[i]==1 else "Yok"
    c75="ÜST" if pred_cor75.iloc[i]==1 else "ALT"
    c85="ÜST" if pred_cor85.iloc[i]==1 else "ALT"
    dc=pred_double[i]
    win=pred_winner[i]
    # bu mac icin en iyi pazari bul (grup bazinda)
    mac_best=[]
    if f"Gol {'Ust' if pred_g15.iloc[i]==1 else 'Alt'} 1.5" in best.values(): mac_best.append("1.5")
    if f"BTTS {'Var' if pred_btts.iloc[i]==1 else 'Yok'}" in best.values(): mac_best.append("BTTS")
    if f"Korner {'Ust' if pred_cor75.iloc[i]==1 else 'Alt'} 7.5" in best.values(): mac_best.append("7.5")
    if f"Korner {'Ust' if pred_cor85.iloc[i]==1 else 'Alt'} 8.5" in best.values(): mac_best.append("8.5")
    if f"Cifte Sans {dc}" in best.values(): mac_best.append("DC")
    if f"Kazanan ({win})" in best.values(): mac_best.append("WIN")
    green = "✓" if mac_best else ""
    P(f"  {i:>3} {str(r['pred_h'])+'-'+str(r['pred_a']):>5} | {g15:>5} {bt:>5} {c75:>5} {c85:>5} {dc:>4} {win:>3} {green}")

# ============ WEB ICIN MODEL KAYDET ============
P(f"\n[WEB] Model web icin kaydediliyor (web_model.pkl)...")
web_model = {
    "m_h": m_h, "m_a": m_a,            # xG regressor'lar (ev/deplasman)
    "m_ch": m_ch, "m_ca": m_ca,        # korner regressor'lar
    "m_btts": m_btts,                  # btts classifier
    "m_draw": m_draw,                  # draw classifier
    "m_draw_th": best_th,              # optimal draw esigi
    "feats": FEATS,
}
with open("web_model.pkl","wb") as f:
    pickle.dump(web_model, f)
P(f"[WEB] web_model.pkl kaydedildi ({len(FEATS)} feature)")

P(f"\nToplam süre: {time.time()-t0:.0f}s")
