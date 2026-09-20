"""Pre-match tahmin yasam dongusu (ROADMAP bolum 40).

Bir mac icin tahmin zamanina gore snapshot'lar:
  T-48h, T-24h, T-6h, T-90m, T-60m, T-45m(lineup), T-30m, T-15m, KICKOFF

Her biri AYRI prediction_id ile kaydedilir (ROADMAP 40):
  12345_T-24H, 12345_T-90M, 12345_T-45M, 12345_T-15M

Boylece "hangi bilgi gelince model ne kadar degisti" izlenir.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Snapshot noktalari: (etiket, kickoff'a kalan sure)
SNAPSHOT_POINTS = [
    ("T-48H", timedelta(hours=48)),
    ("T-24H", timedelta(hours=24)),
    ("T-6H", timedelta(hours=6)),
    ("T-90M", timedelta(minutes=90)),
    ("T-60M", timedelta(minutes=60)),
    ("T-45M", timedelta(minutes=45)),  # lineup gelir
    ("T-30M", timedelta(minutes=30)),
    ("T-15M", timedelta(minutes=15)),
    ("KO", timedelta(minutes=0)),
]


def snapshot_schedule(kickoff_at: datetime) -> list[tuple[str, datetime]]:
    """Kickoff'a gore her snapshot'in prediction_time'ini uretir."""
    if kickoff_at.tzinfo is None:
        kickoff_at = kickoff_at.replace(tzinfo=timezone.utc)
    out = []
    for label, delta in SNAPSHOT_POINTS:
        out.append((label, kickoff_at - delta))
    return out


def prediction_id(fixture_id: int, snapshot_label: str) -> str:
    return f"{fixture_id}_{snapshot_label}"


def emit_snapshots(fixture_id: int, kickoff_at: datetime, predict_fn) -> list[dict]:
    """Her snapshot noktasinda predict_fn(features_at_time) cagirir.

    predict_fn: (snapshot_label, prediction_time) -> pipeline_payload
    Doner: herbir snapshot icin {prediction_id, prediction_time, snapshot_label, payload}
    """
    schedule = snapshot_schedule(kickoff_at)
    emitted = []
    for label, pt in schedule:
        payload = predict_fn(label, pt)
        emitted.append({
            "prediction_id": prediction_id(fixture_id, label),
            "prediction_time": pt,
            "snapshot_label": label,
            "payload": payload,
        })
    return emitted


def stability(reports: list[dict]) -> float:
    """Tahmin stabilitesi (ROADMAP 53 reliability icin).

    Ard ardaya snapshot'lar arasindaki 1X2 olasilik degisiminin ortalamasi.
    Dusuk degisim = yuksek stabilite.
    """
    if len(reports) < 2:
        return 0.0
    diffs = []
    prev = None
    for r in reports:
        res = r["payload"].get("result", {})
        cur = (res.get("home", 0), res.get("draw", 0), res.get("away", 0))
        if prev is not None:
            diffs.append(sum(abs(a - b) for a, b in zip(cur, prev)))
        prev = cur
    return float(np_mean(diffs)) if diffs else 0.0


def np_mean(x):
    import numpy as np
    return np.mean(x)
