"""HONEST bahis degerlendirmesi (overfit'siz).

FARK: tune_edge KULLANMIYORUZ. Sabit edge_threshold=0.05 ile out-of-sample
ROI olcuyoruz. Boylece model validasyonu "gormeden" gercek ROI'yi goruruz.
Sadece 2 model (en iyiler) -> hizli.
"""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.catboost.model import CatBoostModel
from models.ensemble.market_informed import MarketInformedEnsemble
from betting.strategy import evaluate_betting, summarize_betting
from evaluation.metrics import accuracy_1x2, log_loss_1x2

EDGE = 0.05  # SABIT -> honest OOS

def run():
    t0 = time.time()
    import os
    fp = "data/gold/features.parquet"
    if os.path.exists(fp):
        feats = pd.read_parquet(fp); feats["date"] = pd.to_datetime(feats["date"])
    else:
        m = build_matches(use_thesportsdb=False, use_openfootball=False, hf_odds_only=True, min_year=2021)
        feats = build_features(m)
    print(f"feats: {feats.shape[0]} x {feats.shape[1]}")

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
        valid_bet = valid[valid[odds_cols[0]].notna() & valid[odds_cols[1]].notna() & valid[odds_cols[2]].notna()].copy()
        if len(valid_bet) < 500:
            print(f"  Fold {label}: az ({len(valid_bet)}) atlandi"); continue
        train_all = feats[feats["date"]<v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all)*0.15))
        cal = train_all.iloc[-n_cal:]; train = train_all.iloc[:-n_cal:]
        print(f"\n=== FOLD {label}: train={len(train)} cal={len(cal)} valid(bahis)={len(valid_bet)} ===")

        # Model 1: catboost odds-aware
        m1 = CatBoostModel(with_odds=True, iterations=600, verbose=False).fit(train, cal)
        p = m1.predict(valid_bet)
        acc = accuracy_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        rep = evaluate_betting(valid_bet, p, odds_cols=odds_cols, edge_threshold=EDGE)
        rep["fold"]=label; rep["model"]="catboost_odds"; rep["acc"]=acc; rep["logloss"]=ll; rep["edge_used"]=EDGE
        all_reports.append(rep)
        vb=rep["value_bet"]; ab=rep["always_best"]
        print(f"  [catboost_odds] acc={acc*100:.1f}% ll={ll:.4f} | VB roi={vb['roi']*100:+.1f}% (bets={vb['n_bets']}) | AB roi={ab['roi']*100:+.1f}% | CLV={rep['clv']['mean_clv']*100:+.2f}%")

        # Model 2: market-informed ensemble
        m2 = MarketInformedEnsemble(CatBoostModel(with_odds=False, iterations=600, verbose=False), alpha=0.35).fit(train, cal)
        p = m2.predict(valid_bet)
        acc = accuracy_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid_bet["result"], p.home_win, p.draw, p.away_win)
        rep = evaluate_betting(valid_bet, p, odds_cols=odds_cols, edge_threshold=EDGE)
        rep["fold"]=label; rep["model"]="market_informed"; rep["acc"]=acc; rep["logloss"]=ll; rep["edge_used"]=EDGE
        all_reports.append(rep)
        vb=rep["value_bet"]; ab=rep["always_best"]
        print(f"  [market_informed] acc={acc*100:.1f}% ll={ll:.4f} | VB roi={vb['roi']*100:+.1f}% (bets={vb['n_bets']}) | AB roi={ab['roi']*100:+.1f}% | CLV={rep['clv']['mean_clv']*100:+.2f}%")

    print("\n" + "="*72)
    print(f"HONEST OOS RAPOR (edge SABIT={EDGE}, tune YOK)")
    print("="*72)
    tab = summarize_betting(all_reports)
    tab["model"]=[r["model"] for r in all_reports]
    tab["acc"]=[r["acc"] for r in all_reports]
    tab["logloss"]=[r["logloss"] for r in all_reports]
    cols=["model","fold","n","acc","logloss","vb_bets","vb_roi","vb_yield","vb_hit","vb_edge","ab_roi","ab_hit","clv","clv_pos"]
    print(tab[cols].round(4).to_string(index=False))
    # ortalama
    for mod in ["catboost_odds","market_informed"]:
        sub=[r for r in all_reports if r["model"]==mod]
        print(f"\n{mod}: ort VB roi={np.mean([r['value_bet']['roi'] for r in sub])*100:+.1f}% | ort AB roi={np.mean([r['always_best']['roi'] for r in sub])*100:+.1f}% | ort acc={np.mean([r['acc'] for r in sub])*100:.1f}%")
    print(f"\nSure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run()
