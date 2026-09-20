"""Kapsamli data audit - 10 suphe tek tek kontrol."""
import sys, os, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

t0 = time.time()
print("="*80)
print("  KAPSAMLI DATA + MODEL AUDIT")
print("="*80)

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])
print(f"  Veri: {len(feat)} mac\n")

# ═══════════════════════════════════════════════════════════════════════════
# 1. ZAMANSAL VERI SIZINTISI - Tarih siralamasi dogru mu?
# ═══════════════════════════════════════════════════════════════════════════
print("="*80)
print("  1. ZAMANSAL SIZINTI KONTROLU")
print("="*80)

# Tarih sirasiyla mi?
dates = feat["date"].values
is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1) if pd.notna(dates[i]) and pd.notna(dates[i+1]))
print(f"  Tarih sirali mi: {is_sorted}")

# ELO hesaplama zamani - bir macin ELO'su o mactan onceki maclardan gelmeli
# Test: fold 1'deki takimlarin ELO'su fold 2'deki tarihten once hesaplanmis mi?
# Basit test: Ilk 10K mac icin ELO realism check
print(f"\n  ELO Realism Check:")
print(f"    Train baslangici: {feat['date'].iloc[0].date()}")
print(f"    Test baslangici: {feat['date'].iloc[-5000:].iloc[0].date()}")
print(f"    Train sonu: {feat['date'].iloc[-5001].date()}")
print(f"    Test sonu: {feat['date'].iloc[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAIN/TEST SPLIT - Neden 55K?
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  2. TRAIN/TEST SPLIT ANALIZI")
print("="*80)

# Her yil kac mac var
feat["year"] = feat["date"].dt.year
yearly = feat.groupby("year").size()
print("  Yillik mac sayisi:")
for y, n in yearly.items():
    print(f"    {y}: {n:>7} mac")

# Dogru walk-forward: tum veriyi kullan
MIN_TRAIN = 200000
TEST_W = 30000
STEP = 30000
folds = []
i = 0
while i + MIN_TRAIN + TEST_W <= len(feat):
    folds.append((i, i+MIN_TRAIN, i+MIN_TRAIN, i+MIN_TRAIN+TEST_W))
    i += STEP

print(f"\n  Dogru WF: {len(folds)} fold, MIN_TRAIN={MIN_TRAIN}, TEST={TEST_W}")
for fi, (ts,te,tes,tee) in enumerate(folds):
    print(f"    Fold {fi+1}: train {feat['date'].iloc[ts].date()}~{feat['date'].iloc[te-1].date()} | test {feat['date'].iloc[tes].date()}~{feat['date'].iloc[tee-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. TUREV FEATURE SIZINTISI
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  3. TUREV FEATURE SIZINTI KONTROLU")
print("="*80)

# CatBoost'un kullandigi feature'larda hedef bilgisi var mi?
suspects = [
    "home_gf_5","home_ga_5","home_pts_5",
    "home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_gf_8","home_ga_8","away_gf_8","away_ga_8",
    "home_gf_20","home_ga_20","away_gf_20","away_ga_20",
    "home_attack_elo","home_defence_elo","away_attack_elo","away_defence_elo",
    "lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg",
    "lg_draw_rate","lg_btts_rate","lg_over25_rate",
]

print("  Feature <-> Target correlation (Buyuk = supheli):")
for col in suspects:
    if col not in feat.columns: continue
    corr_total = feat[col].corr(feat["total_goals"])
    corr_result = feat[col].corr(feat["result"].map({"H":0,"D":1,"A":2}))
    if abs(corr_total) > 0.3 or abs(corr_result) > 0.3:
        print(f"    {col:<30} corr(gol)={corr_total:>+.3f} corr(result)={corr_result:>+.3f} ** SUPHELI **")
    elif abs(corr_total) > 0.15 or abs(corr_result) > 0.15:
        print(f"    {col:<30} corr(gol)={corr_total:>+.3f} corr(result)={corr_result:>+.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  4. CLASS IMBALANCE")
print("="*80)

res_dist = feat["result"].value_counts(normalize=True)
print("  1X2 dagilimi:")
for r, p in res_dist.items():
    print(f"    {r}: {p*100:.1f}%")

btts_dist = feat["btts"].value_counts(normalize=True)
print(f"  BTTS dagilimi:")
print(f"    Hayir: {btts_dist[0]*100:.1f}%")
print(f"    Evet: {btts_dist[1]*100:.1f}%")

o25_dist = feat["over25"].value_counts(normalize=True)
print(f"  O2.5 dagilimi:")
print(f"    Alt: {o25_dist[0]*100:.1f}%")
print(f"    Ust: {o25_dist[1]*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 5. LIGLER ARASI DAGILIM
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  5. LIGLER ARASI DAGILIM (Top 15)")
print("="*80)

lg_count = feat["league"].value_counts()
total = len(feat)
cumsum = 0
print(f"  {'League':<12} {'Mac':>7} {'Oran':>7} {'Kumulatif':>10}")
print(f"  {'-'*40}")
for lg, n in lg_count.head(15).items():
    cumsum += n
    print(f"  {lg:<12} {n:>7} {n/total*100:>6.1f}% {cumsum/total*100:>9.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 6. TAKIM GORUNMUSLUGU
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  6. TAKIM GORUNMUSLUGU PROBLEMI")
print("="*80)

# Train'de kac farkli takim var
train = feat.iloc[:-30000]
test = feat.iloc[-30000:]

train_teams = set(train["home_team_id"].unique()) | set(train["away_team_id"].unique())
test_teams = set(test["home_team_id"].unique()) | set(test["away_team_id"].unique())

new_teams = test_teams - train_teams
overlap = test_teams & train_teams

print(f"  Train takimlari: {len(train_teams)}")
print(f"  Test takimlari: {len(test_teams)}")
print(f"  Ortak: {len(overlap)}")
print(f"  YENI takimlar (test'te): {len(new_teams)}")
if new_teams:
    print(f"  Ornek: {list(new_teams)[:10]}")

# Takim bazli mac sayisi
team_counts = {}
for _, r in feat.iterrows():
    team_counts[r["home_team_id"]] = team_counts.get(r["home_team_id"], 0) + 1
    team_counts[r["away_team_id"]] = team_counts.get(r["away_team_id"], 0) + 1

counts = sorted(team_counts.values())
print(f"\n  Takim basina mac: min={counts[0]}, median={np.median(counts):.0f}, max={counts[-1]}")
print(f"  <20 mac olan takimlar: {sum(1 for c in counts if c < 20)}")
print(f"  <50 mac olan takimlar: {sum(1 for c in counts if c < 50)}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. MARKET ODDS KARSILASTIRMASI
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  7. MARKET ODDS ANALIZI")
print("="*80)

has_odds = feat["avg_home_odds"].notna()
print(f"  Oran olan mac: {has_odds.sum()} ({has_odds.mean()*100:.1f}%)")
print(f"  Oran olmayan mac: {(~has_odds).sum()} ({(~has_odds).mean()*100:.1f}%)")

# Basit bahis stratejisi: en yuksek orani oyna (no model)
# Value = model_prob > implied_prob
if has_odds.sum() > 0:
    odds_df = feat[has_odds].copy()
    odds_df["implied_home"] = 1/odds_df["avg_home_odds"]
    odds_df["implied_draw"] = 1/odds_df["avg_draw_odds"]
    odds_df["implied_away"] = 1/odds_df["avg_away_odds"]
    odds_df["implied_total"] = odds_df["implied_home"]+odds_df["implied_draw"]+odds_df["implied_away"]
    odds_df["margin"] = odds_df["implied_total"] - 1
    
    print(f"\n  Bookmaker marji: ort={odds_df['margin'].mean()*100:.1f}% min={odds_df['margin'].min()*100:.1f}% max={odds_df['margin'].max()*100:.1f}%")
    
    # Her sonuc icin implied probability vs gercek
    print(f"\n  Implied vs Gercek:")
    for col, res in [("implied_home","H"),("implied_draw","D"),("implied_away","A")]:
        impl = odds_df[col].mean()
        real = (odds_df["result"]==res).mean()
        print(f"    {res}: implied={impl*100:.1f}% gercek={real*100:.1f}% fark={((real-impl)/impl*100):+.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 8. CONFIDENCE vs GERCEKLIK (Calibration detay)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  8. CONFIDENCE GUVENILIRLIGI")
print("="*80)

# Son 30K test uzerinde basit model
from catboost import CatBoostClassifier

LEAK = {"second_half_goals","ht_total_goals","ht_result_is_draw","ht_home_leading",
        "ht_home_goals","ht_away_goals","home_xg","away_xg","total_xg_real",
        "xg_diff_real","home_score_prob","away_score_prob","btts_xprob",
        "over25_xprob","draw_xprob","data_completeness"}

CAT_COLS = ["league","home_team_id","away_team_id","season"]
# Ilk 30 feature ile hizli test
NUM_SUBSET = ["home_elo","away_elo","elo_diff","home_gf_5","home_ga_5","home_pts_5",
    "away_gf_5","away_ga_5","away_pts_5","home_gf_3","home_ga_3","away_gf_3","away_ga_3",
    "home_attack_elo","home_defence_elo","away_attack_elo","away_defence_elo",
    "attack_elo_diff","defence_elo_diff","home_rest_days","away_rest_days",
    "h2h_home_win","h2h_draw","h2h_away_win","h2h_goals_avg","h2h_btts",
    "lg_avg_goals","lg_home_goal_avg","lg_away_goal_avg","lg_draw_rate","lg_btts_rate"]
NUM_SUBSET = [c for c in NUM_SUBSET if c in feat.columns]
ALL_COLS = CAT_COLS + NUM_SUBSET
cat_idx = [ALL_COLS.index(c) for c in CAT_COLS]

Xtr = train[ALL_COLS]; Xte = test[ALL_COLS]
ytr = train["result"].map({"H":0,"D":1,"A":2}).values
yte = test["result"].map({"H":0,"D":1,"A":2}).values

clf = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,l2_leaf_reg=10,
    cat_features=cat_idx,loss_function="MultiClass",class_weights={0:1,1:1.3,2:1},
    random_seed=42,verbose=False,allow_writing_files=False)
clf.fit(Xtr, ytr, use_best_model=True, eval_set=(Xte, yte))
prob = clf.predict_proba(Xte)
conf = np.max(prob, axis=1)
pred = np.argmax(prob, axis=1)

print(f"  Confidence Calibration (30K test):")
print(f"    {'Bucket':>10} {'Pred':>8} {'Actual':>8} {'N':>6} {'Gap':>8} {'CalErr':>8}")
for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.0)]:
    m=(conf>=lo)&(conf<hi); n=m.sum()
    if n>10:
        mp=conf[m].mean(); ma=(pred[m]==yte[m]).mean()
        print(f"    {lo*100:.0f}-{hi*100:.0f}% {mp*100:>7.1f}% {ma*100:>7.1f}% {n:>6} {abs(mp-ma)*100:>+7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 9. FARKLI DONEM TEST
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  9. FARKLI DONEM PERFORMANSI")
print("="*80)

# Her donem icin ayri test
test_periods = [
    ("2024-01 ~ 2024-06", feat["date"]>="2024-01-01", feat["date"]<"2024-07-01"),
    ("2024-07 ~ 2025-01", feat["date"]>="2024-07-01", feat["date"]<"2025-01-01"),
    ("2025-01 ~ 2025-06", feat["date"]>="2025-01-01", feat["date"]<"2025-07-01"),
    ("2025-07 ~ 2026-01", feat["date"]>="2025-07-01", feat["date"]<"2026-01-01"),
    ("2026-01 ~ 2026-07", feat["date"]>="2026-01-01", feat["date"]<"2026-07-07"),
]

print(f"  {'Donem':<22} {'Mac':>6} {'Acc':>7} {'Conf60+':>8} {'Acc60+':>8} {'Acc70+':>8}")
print(f"  {'-'*65}")

for name, m1, m2 in test_periods:
    mask = m1 & m2
    t = feat[mask]
    if len(t) < 100: continue
    Xt = t[ALL_COLS].fillna(0)
    yt = t["result"].map({"H":0,"D":1,"A":2}).values
    p = clf.predict_proba(Xt)
    c = np.max(p, axis=1)
    pr = np.argmax(p, axis=1)
    acc = (pr==yt).mean()
    m60 = c>=0.60; n60=m60.sum()
    a60 = (pr[m60]==yt[m60]).mean()*100 if n60>10 else 0
    m70 = c>=0.70; n70=m70.sum()
    a70 = (pr[m70]==yt[m70]).mean()*100 if n70>10 else 0
    print(f"  {name:<22} {len(t):>6} {acc*100:>6.1f}% {n60/len(t)*100:>7.1f}% {a60:>7.1f}% {a70:>7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 10. ROI ANALIZI
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  10. ROI BACKTEST (Duz Bahis)")
print("="*80)

# Basit strateji: model 1X2'de en yüksek güvenli tahmini oyna
# Duz oranda bahis: 1 birim her bahis
# ROI = (dogru * kazanc - toplam bahis) / toplam bahis
# Kazanc = (odds - 1) dogru ise, -1 yanlis ise

if has_odds.sum() > 1000:
    # Test doneminde oran olan maclar
    test_odds = feat.tail(30000)
    test_odds = test_odds[test_odds["avg_home_odds"].notna()]
    
    Xt = test_odds[ALL_COLS].fillna(0)
    yt = test_odds["result"].map({"H":0,"D":1,"A":2}).values
    p = clf.predict_proba(Xt)
    c = np.max(p, axis=1)
    pr = np.argmax(p, axis=1)
    
    odds_h = test_odds["avg_home_odds"].values
    odds_d = test_odds["avg_draw_odds"].values
    odds_a = test_odds["avg_away_odds"].values
    
    print(f"\n  Test: {len(test_odds)} mac (oran olan)")
    print(f"  {'Strateji':<30} {'Bahis':>6} {'Kazanc':>8} {'ROI':>8}")
    print(f"  {'-'*55}")
    
    # Strateji 1: Her maci oyna (modelin en yuksek olasiligi)
    profit = 0; nbet = 0
    for i in range(len(yt)):
        nbet += 1
        if pr[i] == yt[i]:
            if yt[i]==0: profit += odds_h[i]-1
            elif yt[i]==1: profit += odds_d[i]-1
            else: profit += odds_a[i]-1
        else:
            profit -= 1
    print(f"  {'Tum mac (model max)':<30} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}%")
    
    # Strateji 2: Sadece 60%+ guvenle
    m = c>=0.60
    profit=0; nbet=0
    for i in np.where(m)[0]:
        nbet += 1
        if pr[i] == yt[i]:
            if yt[i]==0: profit += odds_h[i]-1
            elif yt[i]==1: profit += odds_d[i]-1
            else: profit += odds_a[i]-1
        else:
            profit -= 1
    if nbet>0:
        print(f"  {'60%+ guvenle':<30} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}%")
    
    # Strateji 3: Sadece 70%+ guvenle
    m = c>=0.70
    profit=0; nbet=0
    for i in np.where(m)[0]:
        nbet += 1
        if pr[i] == yt[i]:
            if yt[i]==0: profit += odds_h[i]-1
            elif yt[i]==1: profit += odds_d[i]-1
            else: profit += odds_a[i]-1
        else:
            profit -= 1
    if nbet>0:
        print(f"  {'70%+ guvenle':<30} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}%")
    
    # Strateji 4: Value bet (model_prob > implied_prob + edge)
    profit=0; nbet=0
    for i in range(len(yt)):
        implied = [1/odds_h[i], 1/odds_d[i], 1/odds_a[i]]
        edge = p[i] - np.array(implied)
        best = np.argmax(edge)
        if edge[best] > 0.05:  # %5+ value
            nbet += 1
            if best == yt[i]:
                if yt[i]==0: profit += odds_h[i]-1
                elif yt[i]==1: profit += odds_d[i]-1
                else: profit += odds_a[i]-1
            else:
                profit -= 1
    if nbet>0:
        print(f"  {'Value bet (5%+ edge)':<30} {nbet:>6} {profit:>+7.1f} {profit/nbet*100:>+7.1f}%")

print(f"\nToplam: {time.time()-t0:.0f}s")
