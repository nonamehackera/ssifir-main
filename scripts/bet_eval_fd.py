"""GERCEK dürüst bahis degerlendirmesi - Football-Data.co.uk (oranli veri).

1) FD verisini indir (30 lig, 15 sezon, ortalama oranlar dahil)
2) Feature engine calistir (point-in-time)
3) Walk-forward: saf sinyal modelleri (odds KOPYALAMA YOK) + ensemble
4) Her fold'da GERCEK OOS bahis ROI/yield/hit/CLV raporu

AMAÇ: 'model hatali tahmin ediyor / bahsi oynayamiyoruz' iddiasini
gercek sayilarla kanitlamak veya çürütmek.
"""
import sys, time, logging, json
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.elo.model import EloModel
from models.poisson.model import DixonColesModel
from models.catboost.model import CatBoostModel
from models.ensemble.ensemble import EnsembleModel
from betting.strategy import evaluate_betting, summarize_betting
from evaluation.metrics import accuracy_1x2, log_loss_1x2

# FD sadece (HF/TSDB/OF devre disi -> agir dataset yukleme yok)
import ingestion.football_data.canonical as C
_orig = C.build_matches
def fd_only(leagues=None, seasons=None, **kw):
    kw.update(use_huggingface=False, use_thesportsdb=False, use_openfootball=False,
              use_fbref=False)
    return _orig(leagues=leagues, seasons=seasons, **kw)
C.build_matches = fd_only

def run():
    t0 = time.time()
    print("[1/4] FD verisi indiriliyor (30 lig, 15 sezon)...")
    matches = build_matches()
    print(f"  -> {len(matches)} mac, {matches['league'].nunique()} lig, "
          f"{matches['date'].min().date()} .. {matches['date'].max().date()}")
    odds_cov = matches[["avg_home_odds","avg_draw_odds","avg_away_odds"]].notna().all(axis=1).mean()
    print(f"  -> oran kapsami: {odds_cov*100:.1f}% (bahis icin sart)")

    print("[2/4] Feature engine (point-in-time)...")
    feats = build_features(matches)
    print(f"  -> {feats.shape[0]} satir x {feats.shape[1]} kolon")

    # Walk-forward: son 4 sezon (2021/22 -> 2024/25)
    seasons = [
        ("2021-07-01","2022-07-01","2021/22"),
        ("2022-07-01","2023-07-01","2022/23"),
        ("2023-07-01","2024-07-01","2023/24"),
        ("2024-07-01","2025-07-01","2024/25"),
    ]
    reports = []
    for (vs, ve, label) in seasons:
        v_start, v_end = pd.Timestamp(vs), pd.Timestamp(ve)
        valid = feats[(feats["date"]>=v_start)&(feats["date"]<v_end)].copy()
        if len(valid) < 500:
            print(f"  Fold {label}: az ({len(valid)}) atlandi"); continue
        train_all = feats[feats["date"]<v_start].sort_values("date").reset_index(drop=True)
        n_cal = max(200, int(len(train_all)*0.15))
        cal = train_all.iloc[-n_cal:]; train = train_all.iloc[:-n_cal:]
        print(f"\n=== FOLD {label}: train={len(train)} cal={len(cal)} valid={len(valid)} ===")

        # SADECE SAF SINYAL (with_odds=False) -> bookmaker kopyalamasin
        cat = CatBoostModel(with_odds=False, iterations=800, verbose=False).fit(train, cal)
        p = cat.predict(valid)
        acc = accuracy_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        ll = log_loss_1x2(valid["result"], p.home_win, p.draw, p.away_win)
        print(f"  CatBoost(saf): acc={acc*100:.1f}% logloss={ll:.4f}")

        rep = evaluate_betting(valid, p, edge_threshold=0.03)
        rep["fold"] = label
        reports.append(rep)
        vb = rep["value_bet"]; ab = rep["always_best"]; clv = rep["clv"]
        print(f"  VALUE BET (edge>3%): bets={vb['n_bets']} roi={vb['roi']*100:+.1f}% "
              f"yield={vb['yield']*100:+.2f}% hit={vb['hit_rate']*100:.1f}% avg_edge={vb['avg_edge']*100:.1f}%")
        print(f"  ALWAYS BEST     : bets={ab['n_bets']} roi={ab['roi']*100:+.1f}% hit={ab['hit_rate']*100:.1f}%")
        print(f"  CLV             : mean={clv['mean_clv']*100:+.2f}% pozitif_oran={clv['positive_rate']*100:.1f}%")

    print("\n" + "="*70)
    print("TOPLAM GERCEK BAHIS RAPORU (FD verisi, saf sinyal modelleri)")
    print("="*70)
    tab = summarize_betting(reports)
    print(tab.round(4).to_string(index=False))
    if reports:
        def m(k,sub): return np.mean([r[sub][k] for r in reports])
        print(f"\nOrtalama VALUE BET ROI: {m('roi','value_bet')*100:+.1f}%  (pozitif = KAZANC)")
        print(f"Ortalama VALUE BET yield: {m('yield','value_bet')*100:+.2f}%")
        print(f"Ortalama ALWAYS-BEST ROI: {m('roi','always_best')*100:+.1f}%  (bookie gibi oyna)")
        print(f"Ortalama CLV: {np.mean([r['clv']['mean_clv'] for r in reports])*100:+.2f}%")
    print(f"\nSure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run()
