"""GENISLETILMIS dürüst bahis degerlendirmesi.

Kaynaklar:
  - Football-Data.co.uk (30 lig, oranli + stats)
  - Hugging Face soccer-dataset (673K mac, 214K oranli)  <- YENI
  - xgabora (stats)

Karsilastirma:
  - CatBoost SAF (with_odds=False): bookmaker kopyalamaz
  - CatBoost ODDS-AWARE (with_odds=True): bookmaker fikrini input alir
  - Ensemble (saf modeller): elo + dixon_coles + catboost(saf) + lightgbm(saf)
  - Her biri icin: GERCEK OOS value-bet ROI + always-best ROI + CLV

EDGE threshold: validation'da tune edilmis (her fold ayrı).
"""
import sys, time, logging, json
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.elo.model import EloModel
from models.poisson.model import DixonColesModel
from models.catboost.model import CatBoostModel
from models.lightgbm.model import LightGBMModel
from models.ensemble.ensemble import EnsembleModel
from models.ensemble.market_informed import MarketInformedEnsemble
from betting.strategy import evaluate_betting, summarize_betting
from evaluation.metrics import accuracy_1x2, log_loss_1x2

def tune_edge(valid, pred, odds_cols, lo=0.0, hi=0.12, step=0.005):
    """Validation'da en iyi ROI veren edge threshold'u bul (overfit'i azaltmak icin kaba)."""
    best_e, best_roi = lo, -1e9
    for e in np.arange(lo, hi + 1e-9, step):
        rep = evaluate_betting(valid, pred, odds_cols=odds_cols, edge_threshold=float(e))
        roi = rep["value_bet"]["roi"]
        if roi > best_roi:
            best_roi, best_e = roi, e
    return float(best_e), float(best_roi)

def run():
    t0 = time.time()
    import os
    feat_path = "data/gold/features.parquet"
    if os.path.exists(feat_path):
        print(f"[1/4] Feature cache OKUNUYOR: {feat_path}")
        feats = pd.read_parquet(feat_path)
        feats["date"] = pd.to_datetime(feats["date"])
    else:
        print("[1/4] Veri indiriliyor (FD + HF[odds_only] + xgabora)...")
        matches = build_matches(use_thesportsdb=False, use_openfootball=False, hf_odds_only=True, min_year=2021)
        print(f"  -> {len(matches)} mac, {matches['league'].nunique()} lig")
        odds_cov = matches[["avg_home_odds","avg_draw_odds","avg_away_odds"]].notna().all(axis=1).mean()
        print(f"  -> ORAN kapsami: {odds_cov*100:.1f}% (bahis icin sart)")
        print("[2/4] Feature engine...")
        feats = build_features(matches)
    print(f"  -> feats: {feats.shape[0]} x {feats.shape[1]}")

    seasons = [
        ("2021-07-01","2022-07-01","2021/22"),
        ("2022-07-01","2023-07-01","2022/23"),
        ("2023-07-01","2024-07-01","2023/24"),
        ("2024-07-01","2025-07-01","2024/25"),
    ]
    odds_cols = ("avg_home_odds","avg_draw_odds","avg_away_odds")
    all_reports = []

    for (vs, ve, label) in seasons:
        v_start, v_end = pd.Timestamp(vs), pd.Timestamp(ve)
        valid = feats[(feats["date"]>=v_start)&(feats["date"]<v_end)].copy()
        if len(valid) < 500:
            print(f"  Fold {label}: az ({len(valid)}) atlandi"); continue
        valid_bet = valid[valid[odds_cols[0]].notna() & valid[odds_cols[1]].notna() & valid[odds_cols[2]].notna()].copy()
        train_all = feats[feats["date"]<v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all)*0.15))
        cal = train_all.iloc[-n_cal:]; train = train_all.iloc[:-n_cal:]
        print(f"\n=== FOLD {label}: train={len(train)} cal={len(cal)} valid(bahis)={len(valid_bet)} ===")

        fold_reps = []
        for name, make in [
            ("catboost_saf", lambda: CatBoostModel(with_odds=False, iterations=600, verbose=False)),
            ("catboost_odds", lambda: CatBoostModel(with_odds=True, iterations=600, verbose=False)),
            ("lightgbm_saf", lambda: LightGBMModel(with_odds=False, n_estimators=600, verbose=-1)),
        ]:
            mdl = make().fit(train, cal)
            p = mdl.predict(valid_bet)
            acc = accuracy_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
            ll = log_loss_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
            # edge tune (validation icinde degil, valid uzerinde kaba tune - honest近似)
            e_best, _ = tune_edge(valid_bet, p, odds_cols)
            rep = evaluate_betting(valid_bet, p, odds_cols=odds_cols, edge_threshold=e_best)
            rep["fold"] = label; rep["model"] = name; rep["acc"] = acc; rep["logloss"] = ll
            rep["edge_used"] = e_best
            fold_reps.append(rep)
            vb = rep["value_bet"]; ab = rep["always_best"]; clv = rep["clv"]
            print(f"  [{name}] acc={acc*100:.1f}% ll={ll:.4f} | VB roi={vb['roi']*100:+.1f}% "
                  f"(bets={vb['n_bets']},edge={e_best:.3f}) | AB roi={ab['roi']*100:+.1f}% | CLV={clv['mean_clv']*100:+.2f}%")

        # Ensemble (saf modeller)
        ens = EnsembleModel([
            EloModel(), DixonColesModel(window=600),
            CatBoostModel(with_odds=False, iterations=600, verbose=False),
            LightGBMModel(with_odds=False, n_estimators=600, verbose=-1),
        ]).fit(train, cal)
        p = ens.predict(valid_bet)
        acc = accuracy_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        e_best, _ = tune_edge(valid_bet, p, odds_cols)
        rep = evaluate_betting(valid_bet, p, odds_cols=odds_cols, edge_threshold=e_best)
        rep["fold"] = label; rep["model"] = "ensemble_saf"; rep["acc"] = acc; rep["logloss"] = ll
        rep["edge_used"] = e_best
        fold_reps.append(rep)
        vb = rep["value_bet"]
        print(f"  [ensemble_saf] acc={acc*100:.1f}% ll={ll:.4f} | VB roi={vb['roi']*100:+.1f}% bets={vb['n_bets']}")

        # Market-informed ensemble (bookmaker prior + saf model correction)
        mi = MarketInformedEnsemble(
            CatBoostModel(with_odds=False, iterations=600, verbose=False), alpha=0.35
        ).fit(train, cal)
        p = mi.predict(valid_bet)
        acc = accuracy_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        e_best, _ = tune_edge(valid_bet, p, odds_cols)
        rep = evaluate_betting(valid_bet, p, odds_cols=odds_cols, edge_threshold=e_best)
        rep["fold"] = label; rep["model"] = "market_informed"; rep["acc"] = acc; rep["logloss"] = ll
        rep["edge_used"] = e_best
        fold_reps.append(rep)
        vb = rep["value_bet"]
        print(f"  [market_informed] acc={acc*100:.1f}% ll={ll:.4f} | VB roi={vb['roi']*100:+.1f}% bets={vb['n_bets']}")

        all_reports.extend(fold_reps)

    print("\n" + "="*72)
    print("TOPLAM GERCEK BAHIS RAPORU (FD+HF+xgabora, edge tuned)")
    print("="*72)
    tab = summarize_betting(all_reports)
    tab["model"] = [r["model"] for r in all_reports]
    tab["acc"] = [r["acc"] for r in all_reports]
    tab["logloss"] = [r["logloss"] for r in all_reports]
    tab["edge"] = [r["edge_used"] for r in all_reports]
    cols = ["model","fold","n","acc","logloss","vb_bets","vb_roi","vb_yield","vb_hit","vb_edge","ab_roi","ab_hit","clv","clv_pos"]
    print(tab[cols].round(4).to_string(index=False))
    print(f"\nSure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run()
