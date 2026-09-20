"""Smoke test for Elo + Dixon-Coles on a chronological holdout."""
import numpy as np
import pandas as pd

from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features
from models.elo.model import EloModel
from models.poisson.model import DixonColesModel
from evaluation.metrics import log_loss_1x2, brier_1x2, accuracy_1x2


def main():
    m = build_matches()
    f = build_features(m)
    # chronological split: last 20% as test, before that as "train"
    cut = int(len(f) * 0.8)
    train, test = f.iloc[:cut], f.iloc[cut:].reset_index(drop=True)
    print(f"train={len(train)} test={len(test)}")

    elo = EloModel().fit(train)
    ep = elo.predict(test)
    print("ELO  logloss=%.4f brier=%.4f acc=%.3f" % (
        log_loss_1x2(test["result"], ep.home_win, ep.draw, ep.away_win),
        brier_1x2(test["result"], ep.home_win, ep.draw, ep.away_win),
        accuracy_1x2(test["result"], ep.home_win, ep.draw, ep.away_win),
    ))

    dc = DixonColesModel(window=500).fit(train)
    dcp = dc.predict(test)
    print("DC   logloss=%.4f brier=%.4f acc=%.3f mean_lam=%.2f/%.2f" % (
        log_loss_1x2(test["result"], dcp.home_win, dcp.draw, dcp.away_win),
        brier_1x2(test["result"], dcp.home_win, dcp.draw, dcp.away_win),
        accuracy_1x2(test["result"], dcp.home_win, dcp.draw, dcp.away_win),
        float(np.mean(dcp.home_lambda)), float(np.mean(dcp.away_lambda)),
    ))
    # sanity: probabilities sum to 1
    s = np.array(ep.home_win) + np.array(ep.draw) + np.array(ep.away_win)
    print("ELO sum-range:", round(s.min(),4), round(s.max(),4))
    s2 = np.array(dcp.home_win) + np.array(dcp.draw) + np.array(dcp.away_win)
    print("DC  sum-range:", round(s2.min(),4), round(s2.max(),4))


if __name__ == "__main__":
    main()
