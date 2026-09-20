"""Yeni sofascore verisini dahil ederek TUM PAZARLAR icin egitim + 3000 mac simulasyonu.

PAZARLAR:
  1X2, Cifte Sans (1X/X2/12), BTTS Var/Yok,
  Gol Alt/Ust 1.5 / 2.5 / 3.5, Korner Alt/Ust 8.5 / 9.5

AKIS:
  1. matches_all.parquet + sofascore_matches.parquet birlestir
  2. build_features ile zengin feature seti (ELO/form/xg)
  3. Son 3000 mac test, gerisi train
  4. Her pazar icin LightGBM + isotonic calibration (3 seed ensemble)
  5. Guven esigine gore tutma orani raporu

Usage: python scripts/train_with_sofascore.py [--resume-features]
"""
import sys, os, argparse, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["SKIP_STATS_PREDICTOR"] = "1"  # hizli: istatistik tahmini atla

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

P = lambda *a, **k: print(*a, flush=True, **k)
t0 = time.time()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-features", action="store_true",
                    help="Hazir features_enriched_sofa.parquet varsa onu kullan")
    args = ap.parse_args()

    # ---------- 1. VERI BIRLESTIRME ----------
    P("="*72)
    P(" 1. VERI YUKLEME VE BIRLESTIRME")
    P("="*72)
    base = pd.read_parquet("data/gold/matches_all.parquet")
    sofa = pd.read_parquet("data/bronze/sofascore_matches.parquet")
    P(f"  Mevcut (matches_all): {len(base):,} mac")
    P(f"  Yeni (sofascore):     {len(sofa):,} mac")

    # kolon uyumu: sofa'dan base ile ortak kolonlari al, eksikleri NaN
    common = [c for c in base.columns if c in sofa.columns]
    sofa_c = sofa[common].copy()
    # league on eki zaten SOF|... -> cakismaz
    combined = pd.concat([base, sofa_c], ignore_index=True)
    P(f"  Birlestirildi: {len(combined):,} mac")
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("date").reset_index(drop=True)
    P(f"  Tarih araligi: {combined['date'].min().date()} -> {combined['date'].max().date()}")

    # ---------- 2. FEATURES ----------
    feat_path = "data/gold/features_fast.parquet"
    if args.resume_features and os.path.exists(feat_path):
        P(f"\n  [resume] features yukleniyor: {feat_path}")
        feat = pd.read_parquet(feat_path)
    else:
        # build_features_fast.py ile uretildi (hizli)
        if not os.path.exists(feat_path):
            P(f"\n  HATA: {feat_path} yok! Once: python scripts/build_features_fast.py")
            sys.exit(1)
        feat = pd.read_parquet(feat_path)
        P(f"\n  features_fast yuklendi: {feat.shape}")

    # ---------- 3. HEDEFLER ----------
    P(f"\n{'='*72}\n 3. HEDEFLER VE TRAIN/TEST AYRIMI\n{'='*72}")
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    feat = feat.sort_values("date").reset_index(drop=True)
    feat["total_goals"] = feat["home_goals"].fillna(0).astype(int) + feat["away_goals"].fillna(0).astype(int)
    feat["btts"] = ((feat["home_goals"].fillna(0).astype(int)>0)&(feat["away_goals"].fillna(0).astype(int)>0)).astype(int)
    feat["over15"] = (feat["total_goals"]>1.5).astype(int)
    feat["over25"] = (feat["total_goals"]>2.5).astype(int)
    feat["over35"] = (feat["total_goals"]>3.5).astype(int)
    feat["under15"] = 1-feat["over15"]; feat["under25"] = 1-feat["over25"]; feat["under35"] = 1-feat["over35"]
    feat["double_1X"] = ((feat["result"]=="H")|(feat["result"]=="D")).astype(int)
    feat["double_X2"] = ((feat["result"]=="D")|(feat["result"]=="A")).astype(int)
    feat["double_12"] = ((feat["result"]=="H")|(feat["result"]=="A")).astype(int)
    # korner (mevcut veride var; sofascore'da kismen)
    if "home_corners" in feat.columns and "away_corners" in feat.columns:
        feat["total_corners"] = feat["home_corners"].fillna(0).values + feat["away_corners"].fillna(0).values
        feat["corners_85"] = (feat["total_corners"]>=9).astype(int)
        feat["corners_95"] = (feat["total_corners"]>=10).astype(int)
        feat["corners_under85"] = 1-feat["corners_85"]
        feat["corners_under95"] = 1-feat["corners_95"]
        have_corner = feat["total_corners"].notna().sum()
    else:
        have_corner = 0
    P(f"  Korner verisi: {have_corner:,} mac")

    N = 3000
    test_m = feat.tail(N).copy()
    train_m = feat.iloc[:-N].copy()
    P(f"  Train: {len(train_m):,} | Test (son {N}): {len(test_m):,}")
    P(f"  Test tarih: {test_m['date'].min().date()} -> {test_m['date'].max().date()}")
    # test seti kaynak dagilimi
    sofa_in_test = test_m["league"].astype(str).str.startswith("SOF|").sum()
    P(f"  Test icinde SOFASCORE mac: {sofa_in_test:,} / {N}")

    # ---------- 4. FEATURE SECIMI ----------
    SKIP = {"match_id","league","season","date","home_team_id","away_team_id",
        "home_goals","away_goals","result","result_H","result_D","result_A",
        "btts","over25","over15","over35","total_goals","under15","under25","under35",
        "double_1X","double_X2","double_12",
        "corners_85","corners_95","corners_under85","corners_under95","total_corners",
        "home_shots","away_shots","home_sot","away_sot","home_xg","away_xg",
        "home_yellow","away_yellow","home_red","away_red","referee","data_completeness",
        "home_corners","away_corners","ht_home_goals","ht_away_goals","ht_total_goals",
        "sot_diff","shots_diff"}
    FEATS = [c for c in feat.columns if c not in SKIP and pd.api.types.is_numeric_dtype(feat[c])]
    P(f"  Kullanilan feature: {len(FEATS)}")

    sp = int(len(train_m)*0.85)

    # ---------- 5. MODEL EGITIM ----------
    def quick_train(X, y, task="binary"):
        ms = []
        for s in [42,123,456]:
            if task=="multiclass":
                m = lgb.LGBMClassifier(objective="multiclass",num_class=3,num_leaves=40,
                    learning_rate=0.02,n_estimators=400,max_depth=5,min_child_samples=80,
                    subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
                    random_seed=s,verbose=-1,n_jobs=-1)
            else:
                m = lgb.LGBMClassifier(objective="binary",num_leaves=40,
                    learning_rate=0.02,n_estimators=400,max_depth=5,min_child_samples=60,
                    subsample=0.7,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=5.0,
                    random_seed=s,verbose=-1,n_jobs=-1)
            m.fit(X.iloc[:sp], y[:sp])
            raw = m.predict_proba(X.iloc[sp:])
            if task=="multiclass":
                cal=[IsotonicRegression(out_of_bounds="clip").fit(raw[:,c],(y[sp:]==c).astype(float)) for c in range(3)]
            else:
                cal=IsotonicRegression(out_of_bounds="clip").fit(raw[:,1] if raw.ndim>1 else raw,y[sp:].astype(float))
            ms.append((m,cal))
        return ms

    def quick_pred(ms, X, task="binary"):
        ps=[]
        for m,cal in ms:
            raw=m.predict_proba(X)
            if task=="multiclass":
                cp=np.column_stack([cal[c].predict(raw[:,c]) for c in range(3)])
                ss=cp.sum(axis=1,keepdims=True); ss=np.where(ss==0,1,ss); cp/=ss; ps.append(cp)
            else:
                ps.append(cal.predict(raw[:,1] if raw.ndim>1 else raw))
        return np.mean(ps,axis=0)

    def train_binary(col, feats=FEATS):
        tc=train_m.dropna(subset=[col])
        X=tc[feats].fillna(0); y=tc[col].values.astype(int)
        ms=quick_train(X,y,"binary")
        p=quick_pred(ms,test_m[feats].fillna(0))
        return (p>0.5).astype(int), np.maximum(p,1-p)

    def section(t):
        P(f"\n{'='*72}\n  {t}\n{'='*72}")

    results = {}
    P(f"\n{'='*72}\n 4. EGITIM (tum pazarlar)\n{'='*72}")

    # 1X2
    P("  1X2...")
    tc=train_m.dropna(subset=["result"])
    X_tr=tc[FEATS].fillna(0); y_r=np.array([{"H":0,"D":1,"A":2}[r] for r in tc["result"]])
    m1x2=quick_train(X_tr,y_r,"multiclass")
    p1x2=quick_pred(m1x2,test_m[FEATS].fillna(0),"multiclass")
    pw=np.where((p1x2[:,0]>p1x2[:,1])&(p1x2[:,0]>p1x2[:,2]),"H",
        np.where((p1x2[:,2]>p1x2[:,1])&(p1x2[:,2]>p1x2[:,0]),"A","D"))
    c1=np.max(p1x2,axis=1)
    results["1X2"]=(pw,c1,test_m["result"].values)
    P(f"    bitti ({time.time()-t0:.0f}s)")

    # Cifte sans
    for name,col in [("Cifte Sans 1X","double_1X"),("Cifte Sans X2","double_X2"),("Cifte Sans 12","double_12")]:
        P(f"  {name}...")
        pr,cf=train_binary(col)
        results[name]=(pr,cf,test_m[col].values)
    # BTTS - GELISTIRILMIS: xg/sut/lig feature'lari ekle
    P("  BTTS Var (xg/sut feature'li)...")
    btts_feats = FEATS + [c for c in ["home_xg","away_xg","home_shots","away_shots",
        "home_sot","away_sot","lg_btts_rate","btts_xprob"] if c in feat.columns and c not in FEATS]
    pb,cfb=train_binary("btts", btts_feats)
    results["BTTS Var"]=(pb,cfb,test_m["btts"].values.astype(int))
    results["BTTS Yok"]=(1-pb,cfb,1-test_m["btts"].values.astype(int))
    # Gol alt/ust
    for name,col in [("Gol Ust 1.5","over15"),("Gol Alt 1.5","under15"),
                     ("Gol Ust 2.5","over25"),("Gol Alt 2.5","under25"),
                     ("Gol Ust 3.5","over35"),("Gol Alt 3.5","under35")]:
        P(f"  {name}...")
        pr,cf=train_binary(col)
        results[name]=(pr,cf,test_m[col].values.astype(int))
    # Korner (sadece verisi olanlarda anlamli)
    corner_feats=[f for f in FEATS if "corner" not in f.lower()]
    if have_corner>1000:
        for name,col in [("Korner Ust 8.5","corners_85"),("Korner Alt 8.5","corners_under85"),
                         ("Korner Ust 9.5","corners_95"),("Korner Alt 9.5","corners_under95")]:
            P(f"  {name}...")
            pr,cf=train_binary(col,corner_feats)
            # sadece korner verisi olan test maclarinda degerlendir
            mask=test_m["total_corners"].notna().values
            results[name]=(pr[mask],cf[mask],test_m[col].values.astype(int)[mask])
    P(f"    egitim bitti ({time.time()-t0:.0f}s)")

    # ---------- 6. RAPOR ----------
    def pr_table(preds,conf,act):
        P(f"  {'Esik':>6} {'Oneri':>7} {'Oran%':>7} {'Dogru':>6} {'Acc':>7}")
        P(f"  {'-'*44}")
        for th in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]:
            m=conf>=th; n=m.sum()
            if n>0:
                c=(preds[m]==act[m]).sum()
                P(f"  {th*100:>5.0f}% {n:>7} {n/len(preds)*100:>6.1f}% {c:>6} {c/n*100:>6.1f}%")

    section("1X2 - MAC SONUCU")
    pr_table(*results["1X2"])

    section("CIFTE SANS TAHMINLERI")
    for k in ["Cifte Sans 1X","Cifte Sans X2","Cifte Sans 12"]:
        section(k)
        pr_table(*results[k])

    section("KARSILIKLI GOL (BTTS)")
    section("BTTS Var")
    pr_table(*results["BTTS Var"])
    section("BTTS Yok")
    pr_table(*results["BTTS Yok"])

    section("TOPLAM GOL ALT / UST")
    for k in ["Gol Ust 1.5","Gol Alt 1.5","Gol Ust 2.5","Gol Alt 2.5","Gol Ust 3.5","Gol Alt 3.5"]:
        section(k)
        pr_table(*results[k])

    if have_corner>1000:
        section("TOPLAM KORNER ALT / UST")
        for k in ["Korner Ust 8.5","Korner Alt 8.5","Korner Ust 9.5","Korner Alt 9.5"]:
            if k in results:
                section(k)
                pr_table(*results[k])

    section("OZET - 70%+ GUVEN (TUM PAZARLAR)")
    P(f"  {'PAZAR':<22}{'ONERI':>7}{'DOGRU':>7}{'ACC':>8}")
    P(f"  {'-'*46}")
    for k,(pr_,cf_,ac_) in results.items():
        m=cf_>=0.70; n=m.sum()
        if n>0:
            c=(pr_[m]==ac_[m]).sum()
            P(f"  {k:<22}{n:>7}{c:>7}{c/n*100:>7.1f}%")
        else:
            P(f"  {k:<22}{'0':>7}{'-':>7}{'-':>8}")

    section("OZET - 75%+ GUVEN (TUM PAZARLAR)")
    P(f"  {'PAZAR':<22}{'ONERI':>7}{'DOGRU':>7}{'ACC':>8}")
    P(f"  {'-'*46}")
    for k,(pr_,cf_,ac_) in results.items():
        m=cf_>=0.75; n=m.sum()
        if n>0:
            c=(pr_[m]==ac_[m]).sum()
            P(f"  {k:<22}{n:>7}{c:>7}{c/n*100:>7.1f}%")
        else:
            P(f"  {k:<22}{'0':>7}{'-':>7}{'-':>8}")

    P(f"\nToplam sure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
