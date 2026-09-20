"""Team index'i bir kez hesapla ve team_index.pkl olarak kaydet.
Web acilisinda bu dosya aninda yuklenir (~saniyeler)."""
import os, sys, json, pickle
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import web.app as a

print("[BUILD] features_web yukleniyor...", flush=True)
feat = pd.read_parquet("data/gold/features_web.parquet")
feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
for _idc in ["home_team_id", "away_team_id"]:
    if _idc in feat.columns:
        feat[_idc] = pd.to_numeric(feat[_idc], errors="coerce")
feat = feat.dropna(subset=["home_team_id", "away_team_id"])
feat["home_team_id"] = feat["home_team_id"].astype(int)
feat["away_team_id"] = feat["away_team_id"].astype(int)
feat = feat.sort_values("date").reset_index(drop=True)
for c in ["home_goals", "away_goals"]:
    if c in feat.columns:
        feat[c] = feat[c].fillna(0).astype(int)
feat["total_goals"] = feat.get("home_goals", 0) + feat.get("away_goals", 0)
feat["result"] = feat.get("result", pd.Series(["D"] * len(feat))).fillna("D")
feat["btts"] = ((feat.get("home_goals", 0) > 0) & (feat.get("away_goals", 0) > 0)).astype(int)
feat["over25"] = (feat["total_goals"] > 2.5).astype(int)
feat = feat.dropna(subset=["result"])

name_map = {}
name_path = "data/gold/team_id_to_name.json"
if os.path.exists(name_path):
    with open(name_path) as fh:
        raw_nm = json.load(fh)
        name_map = {str(k): v for k, v in raw_nm.items()}

print("[BUILD] team index build ediliyor (bu ~5dk suruyor)...", flush=True)
n_t, n_l = a._build_indices_from_feat(feat, name_map)
print(f"[BUILD] TAMAM: {n_t} takim, {n_l} lig -> team_index.pkl", flush=True)

# _build_indices_from_feat zaten pickle'ladi, ama emin olmak icin tekrar yaz
cache_path = os.path.join(os.path.dirname(__file__), "..", "team_index.pkl")
with open(cache_path, "wb") as fh:
    pickle.dump({"team_index": a.TEAM_INDEX, "team_id_map": a.TEAM_ID_MAP,
                 "league_teams": a.LEAGUE_TEAMS}, fh)
print(f"[BUILD] team_index.pkl kaydedildi: {os.path.getsize(cache_path)/1e6:.1f}MB", flush=True)
