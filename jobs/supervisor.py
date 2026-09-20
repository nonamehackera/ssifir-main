"""Sistemi ayaga kaldiran ana scheduler (ROADMAP 69, 86 + haftalik guncelleme).

Calistir:
  ./.venv/bin/python -m jobs.supervisor
    -> haftalik guncelleme + gunluk drift check scheduler baslatir

Veya sadece bir kez:
  ./.venv/bin/python -m jobs.weekly_update --force
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from jobs.scheduler import SimpleScheduler
from jobs.weekly_update import run_weekly_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-once", action="store_true", help="sadece bir kez guncelle, scheduler'i baslatma")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-odds", dest="with_odds", action="store_false", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.run_once:
        out = run_weekly_update(force=args.force, with_odds=args.with_odds)
        print("RUN ONCE:", out)
        return

    sched = SimpleScheduler()
    # haftalik tam guncelleme (604800 sn = 7 gun)
    sched.add_job("weekly_update", lambda: run_weekly_update(with_odds=args.with_odds),
                  interval_seconds=604800)
    # gunluk hafif drift kontrolu (86400 sn)
    from monitoring.drift import check_latest_drift, maybe_retrain

    def _drift_check():
        rep = check_latest_drift(retrain_fn=lambda: run_weekly_update(force=True, with_odds=args.with_odds))
        if rep:
            maybe_retrain(rep, retrain_fn=lambda: run_weekly_update(force=True, with_odds=args.with_odds))

    sched.add_job("daily_drift", _drift_check, interval_seconds=86400)
    sched.start()
    logging.info("Supervisor calisiyor. Ctrl+C ile durdurabilirsiniz.")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        sched.stop()
        logging.info("Supervisor durdu.")


if __name__ == "__main__":
    main()
