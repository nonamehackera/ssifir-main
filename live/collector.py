"""Canli veri toplama dongusu (ROADMAP bolum 61, 62, 22, 23).

Tasarim:
  - LiveSource: bir macin canli state'ini getirir (gercek API veya mock)
  - LiveCollector: belirli aralikla poll eder, her snapshot'i DB'ye yazar
    (live_snapshots tablosu) + MatchLifecycle.ingest_live ile canli tahmin uretir
  - rate-limit: score/events 15-30s, stats 30-60s (ROADMAP 61)

Gercek kaynak (API-Football) ucretli + canli mac gerektirir; bu yuzden
MockLiveSource ile test edilir. Gercek kaynak icin LiveSource interface'i
uygulanir (ingestion/api_football altina).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from db.engine import SessionLocal
from db.models import LiveSnapshot
from live.model import live_predict

logger = logging.getLogger(__name__)


class LiveSource:
    """Canli mac state kaynagi interface'i."""

    def fetch(self, fixture_id: int) -> dict | None:
        """Son canli state'i dondurur ya da None (mac baslamadi/bitti)."""
        raise NotImplementedError


class MockLiveSource(LiveSource):
    """Test icin deterministik canli state uretir (gelecege BAKMAZ)."""

    def __init__(self, scripted: list[dict]):
        # scripted: [{minute, home_goals, away_goals, xg_home, xg_away,
        #             live_odds_home, live_odds_draw, live_odds_away}, ...]
        self._script = sorted(scripted, key=lambda s: s["minute"])
        self._idx = 0

    def fetch(self, fixture_id: int) -> dict | None:
        if self._idx >= len(self._script):
            return None
        snap = self._script[self._idx]
        self._idx += 1
        return snap


class LiveCollector:
    def __init__(self, source: LiveSource, fixture_id: int,
                 predict_fn=None, interval: float = 5.0, max_minute: int = 95):
        self.source = source
        self.fixture_id = fixture_id
        self.predict_fn = predict_fn  # (fixture_id, snapshot) -> canli tahmin
        self.interval = interval
        self.max_minute = max_minute
        self._pre_lambda = (1.4, 1.1)

    def set_pre_lambda(self, home: float, away: float) -> None:
        self._pre_lambda = (home, away)

    def _save_snapshot(self, snap: dict) -> None:
        sess = SessionLocal()
        try:
            sess.add(LiveSnapshot(
                fixture_id=self.fixture_id,
                captured_at=datetime.now(timezone.utc),
                minute=snap.get("minute"),
                home_goals=snap.get("home_goals"),
                away_goals=snap.get("away_goals"),
                xg_home=snap.get("xg_home"), xg_away=snap.get("xg_away"),
                live_odds_home=snap.get("live_odds_home"),
                live_odds_draw=snap.get("live_odds_draw"),
                live_odds_away=snap.get("live_odds_away"),
                source="mock",
            ))
            sess.commit()
        finally:
            sess.close()

    def run_once(self) -> dict | None:
        snap = self.source.fetch(self.fixture_id)
        if snap is None:
            return None
        self._save_snapshot(snap)  # ROADMAP 22: her degisimde snapshot
        if self.predict_fn:
            pred = self.predict_fn(self.fixture_id, snap)
            return pred
        # varsayilan: live_predict ile
        return live_predict(
            self._pre_lambda[0], self._pre_lambda[1],
            minute=snap.get("minute", 0),
            xg_home=snap.get("xg_home"), xg_away=snap.get("xg_away"),
            score_home=snap.get("home_goals", 0), score_away=snap.get("away_goals", 0),
            live_odds_home=snap.get("live_odds_home"),
            live_odds_draw=snap.get("live_odds_draw"),
            live_odds_away=snap.get("live_odds_away"),
        )

    def run(self, max_iterations: int | None = None) -> list[dict]:
        """Senkron dongu (test icin). Gercek sistemde ayri thread/apscheduler."""
        out = []
        it = 0
        while True:
            pred = self.run_once()
            if pred is None:
                break
            out.append(pred)
            it += 1
            if max_iterations and it >= max_iterations:
                break
            if pred.get("minute", 0) >= self.max_minute:
                break
            time.sleep(self.interval)
        return out
