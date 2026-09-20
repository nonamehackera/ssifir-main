"""Walk-forward model karsilastirmasi (ROADMAP bolum 49, 50).

Kullanım:
    .venv/bin/python -m jobs._eval_wf --folds 4 --quick
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from evaluation.walkforward import run_walkforward, summarize
from ingestion.football_data.canonical import build_matches
from feature_engine.engine import build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Daha az iterasyon (hizli smoke)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("eval_wf")

    matches = build_matches()
    feats = build_features(matches)
    logger.info("Feature kumesi: %d satir x %d kolon", *feats.shape)

    iters = 400 if args.quick else 800

    def make_elo():
        from models.elo.model import EloModel
        return EloModel()

    def make_dc():
        from models.poisson.model import DixonColesModel
        return DixonColesModel(window=500)

    def make_cat():
        from models.catboost.model import CatBoostModel
        return CatBoostModel(iterations=iters, verbose=False)

    def make_cat_odds():
        from models.catboost.model import CatBoostModel
        return CatBoostModel(with_odds=True, iterations=iters, verbose=False)

    def make_lgbm():
        from models.lightgbm.model import LightGBMModel
        return LightGBMModel(n_estimators=iters)

    def make_ens():
        from models.ensemble.ensemble import EnsembleModel
        return EnsembleModel([make_elo(), make_dc(), make_cat(), make_lgbm()])

    factories = {
        "elo": make_elo,
        "dixon_coles": make_dc,
        "catboost": make_cat,
        "catboost_odds": make_cat_odds,
        "lightgbm": make_lgbm,
        "ensemble": make_ens,
    }

    results = run_walkforward(feats, factories, n_folds=args.folds)
    if not results:
        logger.error("Hicbir fold calismadi.")
        return

    table = summarize(results)
    print("\n=== WALK-FORWARD SONUCLARI (fold ortalamalari) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = table.iloc[0]
    print(f"\nEn iyi model: {best['model']} (logloss {best['log_loss']:.4f})")


if __name__ == "__main__":
    main()