from __future__ import annotations

import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate
from stocknetv2.application.services.lifecycle_service import LifecycleRecord, LifecycleService


def _candidate(theme_instance_id: str, theme_path_id: str, members: list[str]) -> ConsensusThemeCandidate:
    return ConsensusThemeCandidate(
        theme_instance_id=theme_instance_id,
        theme_path_id=theme_path_id,
        members=members,
        source_layers=["return_corr_graph", "flow_alignment_graph"],
        consensus_score=0.8,
        structure_score=0.8,
        cross_layer_consensus_score=0.8,
        flow_support_score=1.0,
        dtw_flow_support_score=0.0,
        volume_support_score=0.0,
        large_trade_support_score=0.0,
        stability_score=0.0,
        semantic_coherence_score=0.0,
        theme_quality_score=0.8,
        theme_quality_breakdown_json='{"version":"test"}',
    )


def _record(theme_path_id: str, theme_instance_id: str) -> LifecycleRecord:
    return LifecycleRecord(
        theme_path_id=theme_path_id,
        theme_instance_id=theme_instance_id,
        timestamp=pd.Timestamp("2026-01-02T14:45:00Z"),
        event_type="birth",
        age_frames=3,
        duration_minutes=15,
        match_score=1.0,
        previous_theme_instance_id=None,
        member_retention=1.0,
        status="active",
    )


def test_lifecycle_service_marks_split_when_one_previous_theme_branches():
    service = LifecycleService()
    previous = [_candidate("prev_theme", "path_prev", ["AAA", "BBB", "CCC", "DDD"])]
    previous_records = {"prev_theme": _record("path_prev", "prev_theme")}
    current = [
        _candidate("curr_theme_1", "path_new_1", ["AAA", "BBB"]),
        _candidate("curr_theme_2", "path_new_2", ["CCC", "DDD"]),
    ]

    assigned_candidates, records = service.assign_paths(
        candidates=current,
        previous_candidates=previous,
        previous_lifecycle_records=previous_records,
        timestamp=pd.Timestamp("2026-01-02T14:50:00Z"),
        frame_minutes=5,
    )

    assert assigned_candidates[0].theme_path_id == "path_prev"
    assert records[0].event_type == "continuation"
    assert records[1].event_type == "split"
    assert records[1].transition_parent_path_id == "path_prev"
    assert records[1].transition_kind == "split"


def test_lifecycle_service_marks_merge_when_current_theme_absorbs_multiple_previous_paths():
    service = LifecycleService()
    previous = [
        _candidate("prev_theme_1", "path_prev_1", ["AAA", "BBB"]),
        _candidate("prev_theme_2", "path_prev_2", ["CCC", "DDD"]),
    ]
    previous_records = {
        "prev_theme_1": _record("path_prev_1", "prev_theme_1"),
        "prev_theme_2": _record("path_prev_2", "prev_theme_2"),
    }
    current = [_candidate("curr_theme_1", "path_new_1", ["AAA", "BBB", "CCC", "DDD"])]

    assigned_candidates, records = service.assign_paths(
        candidates=current,
        previous_candidates=previous,
        previous_lifecycle_records=previous_records,
        timestamp=pd.Timestamp("2026-01-02T14:50:00Z"),
        frame_minutes=5,
    )

    assert assigned_candidates[0].theme_path_id == "path_prev_1"
    assert records[0].event_type == "merge"
    assert records[0].transition_parent_path_id == "path_prev_2"
    assert records[0].transition_child_path_id == "path_prev_1"
    assert records[0].transition_kind == "merge"
