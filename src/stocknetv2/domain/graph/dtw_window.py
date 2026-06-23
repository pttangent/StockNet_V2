from __future__ import annotations

import pandas as pd


def compute_effective_dtw_window(
    *,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
    min_minutes: int = 5,
    max_minutes: int = 30,
    target_min_overlap_points: int = 8,
    min_overlap_floor_points: int = 5,
) -> dict[str, int | float | bool | str]:
    available = int((snapshot_time - session_open).total_seconds() // 60)
    effective_lookback_minutes = max(available, 0)
    effective_min_overlap_points = min(target_min_overlap_points, effective_lookback_minutes)

    if available < min_overlap_floor_points:
        return {
            "enabled": False,
            "effective_lookback_minutes": effective_lookback_minutes,
            "effective_min_overlap_points": effective_min_overlap_points,
            "dtw_mode": "warmup",
            "window_confidence": 0.0,
        }

    effective = min(available, max_minutes)

    if effective < target_min_overlap_points:
        warmup_span = max(1, target_min_overlap_points - min_overlap_floor_points)
        confidence = 0.2 + 0.3 * ((effective - min_overlap_floor_points) / warmup_span)
        mode = "warmup"
    elif effective < max_minutes:
        early_span = max(1, max_minutes - min_minutes)
        confidence = 0.5 + 0.5 * ((effective - min_minutes) / early_span)
        mode = "early"
    else:
        confidence = 1.0
        mode = "full"

    return {
        "enabled": True,
        "effective_lookback_minutes": effective,
        "effective_min_overlap_points": min(target_min_overlap_points, effective),
        "dtw_mode": mode,
        "window_confidence": round(confidence, 10),
    }
