from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window


def test_dtw_window_disables_before_minimum_lookback():
    session_open = pd.Timestamp("2026-01-02T14:30:00Z")
    snapshot_time = pd.Timestamp("2026-01-02T14:35:00Z")

    result = compute_effective_dtw_window(
        snapshot_time=snapshot_time,
        session_open=session_open,
        min_minutes=5,
        max_minutes=30,
        target_min_overlap_points=8,
        min_overlap_floor_points=5,
    )

    assert result["enabled"] is True
    assert result["effective_lookback_minutes"] == 5
    assert result["dtw_mode"] == "warmup"
    assert result["effective_min_overlap_points"] == 5
    assert result["window_confidence"] > 0.0


def test_dtw_window_scales_confidence_until_maximum_lookback():
    session_open = pd.Timestamp("2026-01-02T14:30:00Z")
    snapshot_time = pd.Timestamp("2026-01-02T14:50:00Z")

    result = compute_effective_dtw_window(
        snapshot_time=snapshot_time,
        session_open=session_open,
        min_minutes=5,
        max_minutes=30,
        target_min_overlap_points=8,
        min_overlap_floor_points=5,
    )

    assert result["enabled"] is True
    assert result["effective_lookback_minutes"] == 20
    assert result["dtw_mode"] == "early"
    assert result["effective_min_overlap_points"] == 8
    assert result["window_confidence"] > 0.5


def test_dtw_window_reaches_full_mode_after_maximum_lookback():
    session_open = pd.Timestamp("2026-01-02T14:30:00Z")
    snapshot_time = pd.Timestamp("2026-01-02T15:05:00Z")

    result = compute_effective_dtw_window(
        snapshot_time=snapshot_time,
        session_open=session_open,
        min_minutes=5,
        max_minutes=30,
        target_min_overlap_points=8,
        min_overlap_floor_points=5,
    )

    assert result["enabled"] is True
    assert result["effective_lookback_minutes"] == 30
    assert result["dtw_mode"] == "full"
    assert result["effective_min_overlap_points"] == 8
    assert result["window_confidence"] == 1.0
