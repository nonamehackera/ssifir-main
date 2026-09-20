"""GURULTU ANALIZI: Enhanced feature'lar hangi mevcut feature ile correlated?
Hangi feature gercekten YENI bilgi getiriyor?"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

feat=pd.read_parquet("data/gold/features_enhanced.parquet").sort_values("date").reset_index(drop=True)
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

enhanced=["mkt_home_misalign","mkt_draw_mismatch","mkt_over_misalign",
    "home_gf_per_ga","away_gf_per_ga","home_attack_ratio","away_attack_ratio",
    "home_pts_per_game","away_pts_per_game","form_pts_diff",
    "btts_signal","home_overall","away_overall","overall_diff","lg_confidence_weight"]

existing=[
    "mkt_home_prob","mkt_draw_prob","mkt_away_prob","mkt_over25_prob",
    "elo_diff","home_elo","away_elo","attack_elo_diff","defence_elo_diff",
    "home_gf_5","home_ga_5","home_pts_5","away_gf_5","away_ga_5","away_pts_5",
    "lg_btts_rate","lg_over25_rate","lg_draw_rate",
    "home_attack_elo","away_attack_elo","home_defence_elo","away_defence_elo",
    "home_hgf_5","home_hga_5","away_agf_5","away_aga_5",
    "home_w_gf","home_w_ga","away_w_gf","away_w_ga",
]

# Her enhanced feature icin en yuksek korelasyonlu mevcut feature
print("="*80)
print("  ENHANCED FEATURE KORELASYON ANALIZI")
print("  Her enhanced feature'in mevcut feature'lardan en cok benzendigi")
print("="*80)

# Hedef: result (1X2), btts, over25
y_r=feat["result"].map({"H":0,"D":1,"A":2}).values
y_bt=feat["btts"].values
y_o25=feat["over25"].values

# Her enhanced feature'in target ile correlation'u
print(f"\n  {'Enhanced Feature':<25s} {'Target Corr':>12s} {'En Benzer Existing':>25s} {'Max Corr':>10s} {'YENi bilgi?':>12s}")
print(f"  {'-'*85}")

for ef in enhanced:
    if ef not in feat.columns: continue
    ev=feat[ef].fillna(0).values
    # Target correlation (point-biserial for btts/o25)
    corr_bt=np.corrcoef(ev,y_bt)[0,1] if np.std(ev)>0 else 0
    corr_o25=np.corrcoef(ev,y_o25)[0,1] if np.std(ev)>0 else 0
    best_target=max(abs(corr_bt),abs(corr_o25))
    target_name="BTTS" if abs(corr_bt)>abs(corr_o25) else "O25"

    # Mevcut feature'lardan en cok korelasyonlu olan
    best_corr=0; best_name=""
    for xf in existing:
        if xf not in feat.columns: continue
        xv=feat[xf].fillna(0).values
        if np.std(ev)==0 or np.std(xv)==0: continue
        c=abs(np.corrcoef(ev,xv)[0,1])
        if c>best_corr:
            best_corr=c; best_name=xf

    is_new="YENi" if best_corr<0.7 else ("CORR" if best_corr<0.9 else "AYNI")
    print(f"  {ef:<25s} {target_name}={best_target:.3f}   {best_name:<25s} {best_corr:.3f}     {is_new}")

# YENi bilgi getiren feature'lar
print(f"\n  YENi BILGI GETIRENLER (corr < 0.7):")
for ef in enhanced:
    if ef not in feat.columns: continue
    ev=feat[ef].fillna(0).values
    best_corr=0
    for xf in existing:
        if xf not in feat.columns: continue
        xv=feat[xf].fillna(0).values
        if np.std(ev)==0 or np.std(xv)==0: continue
        c=abs(np.corrcoef(ev,xv)[0,1])
        if c>best_corr: best_corr=c
    if best_corr<0.7:
        # Target ile unique bilgi
        corr_bt=abs(np.corrcoef(ev,y_bt)[0,1])
        corr_o25=abs(np.corrcoef(ev,y_o25)[0,1])
        corr_r=abs(np.corrcoef(ev,y_r)[0,1]) if np.std(y_r)>0 else 0
        print(f"    {ef:<25s} BTTS_corr={corr_bt:.3f}  O25_corr={corr_o25:.3f}  Result_corr={corr_r:.3f}  vs_existing={best_corr:.3f}")

print(f"\n  BOZUCU/AYNI feature'lar (corr > 0.9):")
for ef in enhanced:
    if ef not in feat.columns: continue
    ev=feat[ef].fillna(0).values
    for xf in existing:
        if xf not in feat.columns: continue
        xv=feat[xf].fillna(0).values
        if np.std(ev)==0 or np.std(xv)==0: continue
        c=abs(np.corrcoef(ev,xv)[0,1])
        if c>0.9:
            print(f"    {ef:<25s} <-> {xf:<25s} corr={c:.3f}  (AYNI BILGI)")
