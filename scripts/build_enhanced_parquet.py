"""Mevcut features.parquet'a enhanced feature'ları ekle + enhanced parquet kaydet."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
from feature_engine.enhance import enhance_features

feat = pd.read_parquet("data/gold/features.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
for c in ["home_goals","away_goals"]: feat[c]=feat[c].fillna(0).astype(int)
feat["total_goals"]=feat["home_goals"]+feat["away_goals"]
feat["result"]=feat["result"].fillna("D")
feat["btts"]=((feat["home_goals"]>0)&(feat["away_goals"]>0)).astype(int)
feat["over25"]=(feat["total_goals"]>2.5).astype(int)
feat=feat.dropna(subset=["result","elo_diff"])

feat = enhance_features(feat)

new_enhanced = ["mkt_home_misalign","mkt_draw_mismatch","mkt_over_misalign",
                "home_gf_per_ga","away_gf_per_ga","home_attack_ratio","away_attack_ratio",
                "home_pts_per_game","away_pts_per_game","form_pts_diff","btts_signal",
                "home_overall","away_overall","overall_diff","lg_confidence_weight"]
for c in new_enhanced:
    if c in feat.columns:
        non_zero = (feat[c] != 0).sum()
        print(f"  {c}: {non_zero} non-zero / {len(feat)}")

feat.to_parquet("data/gold/features_enhanced.parquet", index=False)
print(f"\nKaydedildi: features_enhanced.parquet ({len(feat)} satir, {len(feat.columns)} sutun)")
