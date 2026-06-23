from __future__ import annotations

import pandas as pd

from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock


def test_snapshot_clock_emits_five_minute_frames_for_one_market_day():
    clock = SnapshotClock()

    snapshots = list(clock.iter_trade_date("2026-01-02"))

    assert len(snapshots) == 78
    assert snapshots[0] == pd.Timestamp("2026-01-02T14:35:00Z")
    assert snapshots[-1] == pd.Timestamp("2026-01-02T21:00:00Z")


def test_snapshot_clock_iterates_across_trade_dates():
    clock = SnapshotClock()

    snapshots = list(clock.iter_range("2026-01-02", "2026-01-05"))

    assert snapshots[0] == pd.Timestamp("2026-01-02T14:35:00Z")
    assert snapshots[-1] == pd.Timestamp("2026-01-05T21:00:00Z")
    assert len(snapshots) == 156
