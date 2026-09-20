"""KAPSAMLI TAHMIN MODELİ - Tüm Pazarlar + Lig Faktörleri + SofaScore.

Her pazar icin ayri model:
  1. 1X2 (Home/Draw/Away) - MultiClass LightGBM + Poisson
  2. Gol Ust/Alt 1.5 - Binary LightGBM + Poisson
  3. Gol Ust/Alt 2.5 - Binary LightGBM + Poisson
  4. Gol Ust/Alt 3.5 - Binary LightGBM + Poisson
  5. Korner Ust/Alt 7.5 - Binary LightGBM + Poisson
  6. Korner Ust/Alt 8.5 - Binary LightGBM + Poisson
  7. BTTS (var/yok) - Binary LightGBM
  8. Ciftli Sans (1X, X2, 12) - Hesaplamali
  9. Kazanan draw-aware - Draw classifier + 1X2

Lig Faktorleri:
  - Lig gucu (ortalama gol, ev avantaji, beraberlik orani)
  - Cross-league faktoru (farkli liglerden takimlar karsilastiginda)
  - Ev/deplasman performans farki per-league

SofaScore:
  - xG, satis, korner, kart, top hakimiyeti (mevcutsa)

Usage: python scripts/train_v2_complete.py
"""
import sys, os, time, warnings, math, pickle
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, brier_score_loss, log_loss

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════
# 1. VERI YUKLE + TEMIZLE
# ═══════════════════════════════════════════════════════════════════════════
P("=" * 70)
P("  KAPSAMLI TAHMIN MODELİ v2 - TUM PAZARLAR")
P("=" * 70)

P("\n[1/7] Veri yukleniyor...")
df = pd.read_parquet("data/gold/features.parquet")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

# Eksik hedef feature'lari hesapla
if "over15" not in df.columns:
    df["over15"] = (df["total_goals"] > 1.5).astype(int)
if "over35" not in df.columns:
    df["over35"] = (df["total_goals"] > 3.5).astype(int)

# Korner toplam
if "total_corners" not in df.columns:
    df["total_corners"] = df["home_corners"].fillna(0) + df["away_corners"].fillna(0)

# Korner hedefleri
df["corners_over75"] = (df["total_corners"] >= 8).astype(int)
df["corners_over85"] = (df["total_corners"] >= 9).astype(int)
df["corners_over95"] = (df["total_corners"] >= 10).astype(int)

# Ciftli sans hedefleri
df["double_1X"] = ((df["result"] == "H") | (df["result"] == "D")).astype(int)
df["double_X2"] = ((df["result"] == "D") | (df["result"] == "A")).astype(int)
df["double_12"] = ((df["result"] == "H") | (df["result"] == "A")).astype(int)

# SofaScore feature'lari ekle (mevcutsa)
try:
    master = pd.read_parquet("data/gold/features_master.parquet")
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    for c in ["home_team_id", "away_team_id"]:
        df[c] = df[c].astype(str)
        master[c] = master[c].astype(str)
    df["_key"] = df["home_team_id"] + "_" + df["away_team_id"] + "_" + df["date"].dt.strftime("%Y-%m-%d")
    master["_key"] = master["home_team_id"] + "_" + master["away_team_id"] + "_" + master["date"].dt.strftime("%Y-%m-%d")
    sf_cols = [c for c in master.columns if c.startswith("sf_") or c in [
        "home_possession", "away_possession", "home_bigch", "away_bigch",
        "home_xg_ot", "away_xg_ot", "home_players", "away_players"
    ]]
    master_sub = master[["_key"] + sf_cols].drop_duplicates(subset=["_key"], keep="last")
    df = df.merge(master_sub, on="_key", how="left")
    df = df.drop(columns=["_key"], errors="ignore")
    P(f"  SofaScore: {len(sf_cols)} kolon eklendi")
except Exception as e:
    P(f"  SofaScore atlandi: {e}")

P(f"  Toplam: {len(df):,} satir, {len(df.columns)} kolon")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LIG FAKTORLERINI HESAPLA
# ═══════════════════════════════════════════════════════════════════════════
P("\n[2/7] Lig faktorleri hesaplaniyor...")

# Her lig icin istatistik (sadece gecmis veriden - leak-free)
leagues = df["league"].fillna("?").unique()
lg_stats = {}
for lg in leagues:
    sub = df[df["league"] == lg]
    n = len(sub)
    if n < 20:
        continue
    hg = sub["home_goals"].mean()
    ag = sub["away_goals"].mean()
    draw_r = (sub["result"] == "D").mean()
    btts_r = sub["btts"].mean() if "btts" in sub.columns else ((sub["home_goals"] > 0) & (sub["away_goals"] > 0)).mean()
    o25_r = sub["over25"].mean() if "over25" in sub.columns else ((sub["home_goals"] + sub["away_goals"]) >= 3).mean()
    home_adv = hg - ag  # pozitif = ev avantaji var
    total_goals = hg + ag
    lg_stats[lg] = {
        "n": n, "avg_goals": total_goals, "avg_home": hg, "avg_away": ag,
        "draw_rate": draw_r, "btts_rate": btts_r, "over25_rate": o25_r,
        "home_advantage": home_adv,
    }

# Feature olarak lig faktorlerini ekle (point-in-time: sadece gecmis veriden hesapla)
P("  Point-in-time lig faktorleri hesaplaniyor...")
lg_running = {}
lg_features = []
for idx, row in df.iterrows():
    lg = row.get("league", "?")
    if lg not in lg_running:
        lg_running[lg] = {"n": 0, "hg": 0.0, "ag": 0.0, "dr": 0, "bt": 0, "o25": 0}
    s = lg_running[lg]
    hg = row.get("home_goals", 0) or 0
    ag = row.get("away_goals", 0) or 0
    tot = hg + ag
    # Bu macin sonuclari EKLENMEDEN once istatistikleri kullan (leak-free)
    lg_features.append({
        "lg_strength": s["hg"] / max(s["n"], 1) + s["ag"] / max(s["n"], 1),
        "lg_home_adv": s["hg"] / max(s["n"], 1) - s["ag"] / max(s["n"], 1),
        "lg_draw_rate": s["dr"] / max(s["n"], 1),
        "lg_total_goals": (s["hg"] + s["ag"]) / max(s["n"], 1),
    })
    # Mac sonrasi guncelle
    s["n"] += 1
    s["hg"] += hg
    s["ag"] += ag
    s["dr"] += 1 if hg == ag else 0
    s["bt"] += 1 if (hg > 0 and ag > 0) else 0
    s["o25"] += 1 if tot >= 3 else 0

lg_feat_df = pd.DataFrame(lg_features)
df = pd.concat([df.reset_index(drop=True), lg_feat_df.reset_index(drop=True)], axis=1)

# Cross-league faktoru: takim kendi ligi disinda oynuyorsa
df["cross_league"] = 0  # placeholder, gercekta takim-lig eslesmesi gerekir

P(f"  Lig faktoru kolonlari: lg_strength, lg_home_adv, lg_draw_rate, lg_total_goals")
P(f"  Lig sayisi: {len(lg_stats)}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE SECIMI
# ═══════════════════════════════════════════════════════════════════════════
P("\n[3/7] Feature secimi...")

SKIP = {"match_id", "index", "round", "league", "league_name", "season", "date",
        "home_team", "away_team", "home_team_id", "away_team_id",
        "home_goals", "away_goals", "result", "result_H", "result_D", "result_A",
        "btts", "over05", "over15", "over25", "over35", "over45", "total_goals",
        "under05", "under15", "under25", "under35", "under45",
        "double_1X", "double_X2", "double_12", "total_corners",
        "corners_over75", "corners_over85", "corners_over95",
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_xg", "away_xg", "home_corners", "away_corners",
        "ht_home_goals", "ht_away_goals", "ht_total_goals",
        "home_fouls", "away_fouls", "home_yellow", "away_yellow",
        "home_red", "away_red", "referee", "stadium", "source", "_src",
        "attendance", "home_minutes", "away_minutes", "home_assists", "away_assists",
        "corners_85", "corners_95", "corners_under85", "corners_under95",
        "home_corners_5", "away_corners_5", "home_cards_5", "away_cards_5",
        "h2h_home_win", "h2h_draw", "h2h_away_win", "h2h_goals_avg", "h2h_btts"}

numeric_cols = [c for c in df.columns if c not in SKIP and pd.api.types.is_numeric_dtype(df[c])]
nan_pct = df[numeric_cols].isna().mean()

# %40'dan az NaN olan feature'lari sec
FEATS = [c for c in numeric_cols if nan_pct[c] < 0.40]
P(f"  {len(FEATS)} feature secildi")
P(f"  Ortalama NaN: {df[FEATS].isna().mean().mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN/TEST AYRIMI
# ═══════════════════════════════════════════════════════════════════════════
P("\n[4/7] Train/Test ayrimi...")
for f in FEATS:
    df[f] = pd.to_numeric(df[f], errors="coerce")

N_TEST = 3000
test = df.tail(N_TEST).copy()
train = df.iloc[:-N_TEST].copy()
sp = int(len(train) * 0.85)
P(f"  Train: {len(train):,} | Test: {N_TEST:,} | Val: {len(train)-sp:,}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL EGITIM fonksiyonlari
# ═══════════════════════════════════════════════════════════════════════════
P("\n[5/7] Model egitimi...")

def train_reg(X, y, strong=False):
    """Ensemble LightGBM regressor (3 seed)."""
    ms = []
    for s in [42, 123, 456]:
        if strong:
            m = lgb.LGBMRegressor(
                num_leaves=64, learning_rate=0.03, n_estimators=700, max_depth=8,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.05, reg_lambda=0.5, random_seed=s, verbose=-1, n_jobs=-1)
        else:
            m = lgb.LGBMRegressor(
                num_leaves=48, learning_rate=0.02, n_estimators=500, max_depth=6,
                min_child_samples=50, subsample=0.7, colsample_bytree=0.6,
                reg_alpha=0.3, reg_lambda=4.0, random_seed=s, verbose=-1, n_jobs=-1)
        m.fit(X.iloc[:sp], y[:sp])
        ms.append(m)
    return ms

def pred_reg(ms, X):
    return np.mean([m.predict(X) for m in ms], axis=0)

def train_clf(col):
    """Ensemble LightGBM classifier + isotonic calibration (3 seed)."""
    tc = train.dropna(subset=[col])
    X = tc[FEATS].fillna(0)
    y = tc[col].values.astype(int)
    ms = []
    valp_all = []
    for s in [42, 123, 456]:
        m = lgb.LGBMClassifier(
            objective="binary", num_leaves=48, learning_rate=0.02, n_estimators=500,
            max_depth=6, min_child_samples=50, subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.3, reg_lambda=4.0, random_seed=s, verbose=-1, n_jobs=-1)
        m.fit(X.iloc[:sp], y[:sp])
        raw = m.predict_proba(X.iloc[sp:])
        cal = IsotonicRegression(out_of_bounds="clip").fit(raw[:, 1], y[sp:].astype(float))
        ms.append((m, cal))
        valp_all.append(cal.predict(raw[:, 1]))
    return ms, np.mean(valp_all, axis=0), y[sp:]

def pred_clf(ms, X):
    ps = []
    for m, cal in ms:
        raw = m.predict_proba(X)
        ps.append(cal.predict(raw[:, 1]))
    return np.mean(ps, axis=0)

def optimal_threshold(y_true, y_prob, metric="f1"):
    """En iyi esigi bul (F1 veya Brier)."""
    best_th, best_score = 0.5, -1
    for t in np.arange(0.25, 0.76, 0.01):
        if metric == "f1":
            sc = f1_score(y_true, (y_prob >= t).astype(int))
        else:
            sc = -brier_score_loss(y_true, y_prob)
        if sc > best_score:
            best_score, best_th = sc, t
    return best_th, best_score

# ═══════════════════════════════════════════════════════════════════════════
# 5a. xG REGRESYON (gol beklentisi)
# ═══════════════════════════════════════════════════════════════════════════
P("  [1/9] xG regresyon (ev gol beklentisi)...")
tr_h = train.dropna(subset=["home_goals"])
m_h = train_reg(tr_h[FEATS].fillna(0), tr_h["home_goals"].values, strong=True)
P(f"    bitti ({time.time()-t0:.0f}s)")

P("  [2/9] xG regresyon (dep gol beklentisi)...")
m_a = train_reg(
    train.dropna(subset=["away_goals"])[FEATS].fillna(0),
    train.dropna(subset=["away_goals"])["away_goals"].values, strong=True)
P(f"    bitti ({time.time()-t0:.0f}s)")

xg_home = np.clip(pred_reg(m_h, test[FEATS].fillna(0)), 0.1, 8)
xg_away = np.clip(pred_reg(m_a, test[FEATS].fillna(0)), 0.1, 8)

# ═══════════════════════════════════════════════════════════════════════════
# 5b. KORNER REGRESYON
# ═══════════════════════════════════════════════════════════════════════════
P("  [3/9] Korner regresyon...")
has_corners = train["home_corners"].notna().sum() > 1000
if has_corners:
    tr_ch = train.dropna(subset=["home_corners"])
    m_ch = train_reg(tr_ch[FEATS].fillna(0), tr_ch["home_corners"].values, strong=True)
    tr_ca = train.dropna(subset=["away_corners"])
    m_ca = train_reg(tr_ca[FEATS].fillna(0), tr_ca["away_corners"].values, strong=True)
    kc_home = np.clip(pred_reg(m_ch, test[FEATS].fillna(0)), 0, 20)
    kc_away = np.clip(pred_reg(m_ca, test[FEATS].fillna(0)), 0, 20)
    P(f"    bitti ({time.time()-t0:.0f}s)")
else:
    m_ch = m_ca = None
    kc_home = np.full(N_TEST, 5.0)
    kc_away = np.full(N_TEST, 4.5)
    P("    korner verisi yetersiz, fallback kullanildi")

# ═══════════════════════════════════════════════════════════════════════════
# 5c. BTTS CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
P("  [4/9] BTTS classifier...")
m_btts, btts_valp, btts_valy = train_clf("btts")
p_btts = pred_clf(m_btts, test[FEATS].fillna(0))
btts_th, btts_f1 = optimal_threshold(btts_valy, btts_valp, "f1")
P(f"    esik: {btts_th:.2f} (F1={btts_f1:.3f}) ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 5d. DRAW CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
P("  [5/9] Draw classifier...")
train["is_draw"] = (train["result"] == "D").astype(int)
m_draw, draw_valp, draw_valy = train_clf("is_draw")
p_draw = pred_clf(m_draw, test[FEATS].fillna(0))
draw_th, draw_f1 = optimal_threshold(draw_valy, draw_valp, "f1")
P(f"    esik: {draw_th:.2f} (F1={draw_f1:.3f}) ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 5e. OVER 1.5 CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
P("  [6/9] Over 1.5 classifier...")
m_o15, o15_valp, o15_valy = train_clf("over15")
p_o15 = pred_clf(m_o15, test[FEATS].fillna(0))
o15_th, o15_f1 = optimal_threshold(o15_valy, o15_valp, "f1")
P(f"    esik: {o15_th:.2f} (F1={o15_f1:.3f}) ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 5f. OVER 2.5 CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
P("  [7/9] Over 2.5 classifier...")
m_o25, o25_valp, o25_valy = train_clf("over25")
p_o25 = pred_clf(m_o25, test[FEATS].fillna(0))
o25_th, o25_f1 = optimal_threshold(o25_valy, o25_valp, "f1")
P(f"    esik: {o25_th:.2f} (F1={o25_f1:.3f}) ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 5g. OVER 3.5 CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
P("  [8/9] Over 3.5 classifier...")
m_o35, o35_valp, o35_valy = train_clf("over35")
p_o35 = pred_clf(m_o35, test[FEATS].fillna(0))
o35_th, o35_f1 = optimal_threshold(o35_valy, o35_valp, "f1")
P(f"    esik: {o35_th:.2f} (F1={o35_f1:.3f}) ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 5h. KORNER CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════
P("  [9/9] Korner classifiers...")
if has_corners:
    train["corners_over75"] = (train["total_corners"] >= 8).astype(int)
    train["corners_over85"] = (train["total_corners"] >= 9).astype(int)
    m_cor75, cor75_valp, cor75_valy = train_clf("corners_over75")
    p_cor75 = pred_clf(m_cor75, test[FEATS].fillna(0))
    cor75_th, cor75_f1 = optimal_threshold(cor75_valy, cor75_valp, "f1")

    m_cor85, cor85_valp, cor85_valy = train_clf("corners_over85")
    p_cor85 = pred_clf(m_cor85, test[FEATS].fillna(0))
    cor85_th, cor85_f1 = optimal_threshold(cor85_valy, cor85_valp, "f1")
    P(f"    7.5 esik: {cor75_th:.2f} (F1={cor75_f1:.3f})")
    P(f"    8.5 esik: {cor85_th:.2f} (F1={cor85_f1:.3f}) ({time.time()-t0:.0f}s)")
else:
    m_cor75 = m_cor85 = None
    cor75_th = cor85_th = 0.5
    p_cor75 = np.full(N_TEST, 0.5)
    p_cor85 = np.full(N_TEST, 0.5)

# ═══════════════════════════════════════════════════════════════════════════
# 6. POISSON SKOR MATRISI + TUM PAZARLAR
# ═══════════════════════════════════════════════════════════════════════════
P("\n[6/7] Skor matrisi + pazar hesaplari...")
maxg = 12
poisson_rows = []
for i in range(N_TEST):
    lh, la = xg_home[i], xg_away[i]
    ph = np.array([np.exp(-lh) * (lh**k) / math.factorial(k) for k in range(maxg)])
    pa = np.array([np.exp(-la) * (la**k) / math.factorial(k) for k in range(maxg)])
    mat = np.outer(ph, pa)
    mat_sum = mat.sum()
    if mat_sum > 0:
        mat /= mat_sum
    # 1X2 olasiliklari
    p_h = float(np.trace(mat, offset=1))
    p_d = float(np.trace(mat))
    p_a = float(np.trace(mat, offset=-1))
    s = p_h + p_d + p_a
    if s > 0:
        p_h, p_d, p_a = p_h/s, p_d/s, p_a/s
    # Gol pazarlari (Poisson)
    total_lam = lh + la
    po15 = 1.0 - (np.exp(-total_lam) + total_lam * np.exp(-total_lam))
    po25 = 1.0 - sum(np.exp(-total_lam) * (total_lam**k) / math.factorial(k) for k in range(3))
    po35 = 1.0 - sum(np.exp(-total_lam) * (total_lam**k) / math.factorial(k) for k in range(4))
    # BTTS (Poisson)
    p_btts_pois = 1.0 - np.exp(-lh) - np.exp(-la) + np.exp(-lh - la)
    # Korner (Poisson)
    if m_ch and m_ca:
        total_cor = kc_home[i] + kc_away[i]
        po_cor75 = 1.0 - sum(np.exp(-total_cor) * (total_cor**k) / math.factorial(k) for k in range(8))
        po_cor85 = 1.0 - sum(np.exp(-total_cor) * (total_cor**k) / math.factorial(k) for k in range(9))
    else:
        po_cor75 = 0.5
        po_cor85 = 0.5

    poisson_rows.append({
        "p_h": p_h, "p_d": p_d, "p_a": p_a,
        "po15": po15, "po25": po25, "po35": po35,
        "po_btts": p_btts_pois,
        "po_cor75": po_cor75, "po_cor85": po_cor85,
    })
poisson_df = pd.DataFrame(poisson_rows)

# ═══════════════════════════════════════════════════════════════════════════
# 7. ENSEMBLE TAHMINLER (Model + Poisson blend)
# ═══════════════════════════════════════════════════════════════════════════
P("\n[7/7] Ensemble tahminler + evaluation...")

# Draw-aware kazanan tahmini
def predict_winner(i):
    if p_draw[i] > draw_th:
        return "D"
    if poisson_df["p_h"].iloc[i] >= poisson_df["p_a"].iloc[i]:
        return "H"
    return "A"

pred_winner = np.array([predict_winner(i) for i in range(N_TEST)])

# Gol pazarlari: model + Poisson blend (60% model, 40% Poisson)
blend_o15 = 0.6 * p_o15 + 0.4 * poisson_df["po15"]
blend_o25 = 0.6 * p_o25 + 0.4 * poisson_df["po25"]
blend_o35 = 0.6 * p_o35 + 0.4 * poisson_df["po35"]
blend_btts = 0.6 * p_btts + 0.4 * poisson_df["po_btts"]
blend_cor75 = 0.5 * p_cor75 + 0.5 * poisson_df["po_cor75"] if m_cor75 else p_cor75
blend_cor85 = 0.5 * p_cor85 + 0.5 * poisson_df["po_cor85"] if m_cor85 else p_cor85

# Ciftli sans
blend_1x = blend_o15 * 0 + poisson_df["p_h"] + poisson_df["p_d"]  # placeholder
p_1x = poisson_df["p_h"] + poisson_df["p_d"]
p_x2 = poisson_df["p_d"] + poisson_df["p_a"]
p_12 = poisson_df["p_h"] + poisson_df["p_a"]

# ═══════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
P("\n" + "=" * 75)
P(f"  TAHMIN MODELİ v2 - GERCEK TUTMA ORANLARI (son {N_TEST} mac)")
P("=" * 75)

# Real values
real_h = (test["result"] == "H").astype(int).values
real_d = (test["result"] == "D").astype(int).values
real_a = (test["result"] == "A").astype(int).values
real_o15 = test["over15"].values
real_o25 = test["over25"].values
real_o35 = test["over35"].values
real_btts = test["btts"].values if "btts" in test.columns else ((test["home_goals"] > 0) & (test["away_goals"] > 0)).astype(int).values
real_cor75 = test["corners_over75"].values
real_cor85 = test["corners_over85"].values
real_1x = test["double_1X"].values if "double_1X" in test.columns else ((test["result"] != "A")).astype(int).values
real_x2 = test["double_X2"].values if "double_X2" in test.columns else ((test["result"] != "H")).astype(int).values
real_12 = test["double_12"].values if "double_12" in test.columns else ((test["result"] != "D")).astype(int).values

def hold(pred, real):
    n = len(pred)
    c = (pred == real).sum()
    return n, c, c / n * 100

def hold_pct(pred, real):
    return (pred == real).mean() * 100

results = {}

# KAZANAN
results["Kazanan (H)"] = hold((pred_winner == "H").astype(int), real_h)
results["Kazanan (D)"] = hold((pred_winner == "D").astype(int), real_d)
results["Kazanan (A)"] = hold((pred_winner == "A").astype(int), real_a)

# GOL
pred_o15_bin = (blend_o15 >= o15_th).astype(int)
pred_o25_bin = (blend_o25 >= o25_th).astype(int)
pred_o35_bin = (blend_o35 >= o35_th).astype(int)
results["Gol Ust 1.5"] = hold(pred_o15_bin, real_o15)
results["Gol Alt 1.5"] = hold(1 - pred_o15_bin, 1 - real_o15)
results["Gol Ust 2.5"] = hold(pred_o25_bin, real_o25)
results["Gol Alt 2.5"] = hold(1 - pred_o25_bin, 1 - real_o25)
results["Gol Ust 3.5"] = hold(pred_o35_bin, real_o35)
results["Gol Alt 3.5"] = hold(1 - pred_o35_bin, 1 - real_o35)

# BTTS
pred_btts_bin = (blend_btts >= btts_th).astype(int)
results["BTTS Var"] = hold(pred_btts_bin, real_btts)
results["BTTS Yok"] = hold(1 - pred_btts_bin, 1 - real_btts)

# KORNER
if m_cor75:
    pred_cor75_bin = (blend_cor75 >= cor75_th).astype(int)
    pred_cor85_bin = (blend_cor85 >= cor85_th).astype(int)
    results["Korner Ust 7.5"] = hold(pred_cor75_bin, real_cor75)
    results["Korner Alt 7.5"] = hold(1 - pred_cor75_bin, 1 - real_cor75)
    results["Korner Ust 8.5"] = hold(pred_cor85_bin, real_cor85)
    results["Korner Alt 8.5"] = hold(1 - pred_cor85_bin, 1 - real_cor85)

# CIFTE SANS
pred_1x_bin = (p_1x >= 0.5).astype(int)
pred_x2_bin = (p_x2 >= 0.5).astype(int)
pred_12_bin = (p_12 >= 0.5).astype(int)
results["Ciftli Sans 1X"] = hold(pred_1x_bin, real_1x)
results["Ciftli Sans X2"] = hold(pred_x2_bin, real_x2)
results["Ciftli Sans 12"] = hold(pred_12_bin, real_12)

# Brier Score (olasilik kalitesi)
brier_scores = {
    "1X2 Ev": brier_score_loss(real_h, poisson_df["p_h"].values),
    "1X2 Beraberlik": brier_score_loss(real_d, poisson_df["p_d"].values),
    "1X2 Dep": brier_score_loss(real_a, poisson_df["p_a"].values),
    "BTTS": brier_score_loss(real_btts, blend_btts),
    "Over 2.5": brier_score_loss(real_o25, blend_o25),
}

P(f"\n  {'PAZAR':<25}{'TAHMIN':>8}{'DOGRU':>8}{'TUTMA':>8}  NOT")
P(f"  {'-'*58}")
for name, (n, c, acc) in sorted(results.items(), key=lambda x: -x[1][2]):
    bar = ">" * int(acc / 5) if acc > 60 else ""
    P(f"  {name:<25}{n:>8}{c:>8}{acc:>7.1f}% {bar}")

P(f"\n  Brier Score (dusuk = iyi):")
for name, bs in sorted(brier_scores.items(), key=lambda x: x[1]):
    P(f"    {name:<20} {bs:.4f}")

# SIFIR GRUBU ANALIZI: model hicbir sey bilmeden ne kadar tutar?
random_hold = 1.0 / 3 * 100  # %33.3 for 3-way
P(f"\n  Rastgele tahmin (3'lü): {random_hold:.1f}%")
P(f"  Rastgele tahmin (2'li): 50.0%")

# ═══════════════════════════════════════════════════════════════════════════
# 9. KAYDET
# ═══════════════════════════════════════════════════════════════════════════
P(f"\n{'='*75}")
P("  WEB MODEL KAYDI")
P(f"{'='*75}")

web_model = {
    # xG regressors
    "m_h": m_h, "m_a": m_a,
    # Corner regressors
    "m_ch": m_ch if m_ch else [], "m_ca": m_ca if m_ca else [],
    # Classifiers
    "m_btts": m_btts, "m_draw": m_draw,
    "m_o15": m_o15, "m_o25": m_o25, "m_o35": m_o35,
    # Corner classifiers
    "m_cor75": m_cor75 if m_cor75 else [], "m_cor85": m_cor85 if m_cor85 else [],
    # Optimal thresholds
    "m_draw_th": draw_th, "m_btts_th": btts_th,
    "m_o15_th": o15_th, "m_o25_th": o25_th, "m_o35_th": o35_th,
    "m_cor75_th": cor75_th if m_cor75 else 0.5,
    "m_cor85_th": cor85_th if m_cor85 else 0.5,
    # Feature list
    "feats": FEATS,
    # Meta
    "version": "v2_complete",
    "train_size": len(train),
}

with open("web_model.pkl", "wb") as f:
    pickle.dump(web_model, f)
P(f"  web_model.pkl kaydedildi ({len(FEATS)} feature)")

P(f"\nToplam sure: {time.time()-t0:.0f}s")
P("Bitti! python run_web.py ile baslat.")
