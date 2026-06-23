from __future__ import annotations

import pandas as pd

from stocknetv2.application.services.temporal_edge_replay_service import (
    TemporalEdgeReplayService,
)
from stocknetv2.domain.graph.edge import GraphEdge


def test_temporal_edge_replay_service_builds_active_state_for_new_edge():
    service = TemporalEdgeReplayService(alpha=0.6, ttl_frames=2)

    states, state_map = service.replay(
        run_id="run_001",
        snapshot_id="snapshot_001",
        trade_date="2025-01-02",
        timestamp=pd.Timestamp("2025-01-02T14:35:00Z"),
        layer_edges={
            "return_corr_graph": [
                GraphEdge(
                    graph_layer="return_corr_graph",
                    edge_type="return_correlation",
                    source_symbol="AAA",
                    target_symbol="BBB",
                    snapshot_time=pd.Timestamp("2025-01-02T14:35:00Z"),
                    weight=0.8,
                    raw_score=0.8,
                    support_points=8,
                )
            ]
        },
        previous_states={},
    )

    assert len(states) == 1
    assert states[0].state == "active"
    assert states[0].presence_count == 1
    assert states[0].temporal_score == 0.8
    assert ("return_corr_graph", "AAA", "BBB") in state_map


def test_temporal_edge_replay_service_decays_missing_edge_within_ttl():
    service = TemporalEdgeReplayService(alpha=0.6, ttl_frames=2)
    timestamp = pd.Timestamp("2025-01-02T14:40:00Z")
    previous_states, state_map = service.replay(
        run_id="run_001",
        snapshot_id="snapshot_001",
        trade_date="2025-01-02",
        timestamp=pd.Timestamp("2025-01-02T14:35:00Z"),
        layer_edges={
            "return_corr_graph": [
                GraphEdge(
                    graph_layer="return_corr_graph",
                    edge_type="return_correlation",
                    source_symbol="AAA",
                    target_symbol="BBB",
                    snapshot_time=pd.Timestamp("2025-01-02T14:35:00Z"),
                    weight=0.8,
                    raw_score=0.8,
                    support_points=8,
                )
            ]
        },
        previous_states={},
    )

    decayed_states, next_state_map = service.replay(
        run_id="run_001",
        snapshot_id="snapshot_002",
        trade_date="2025-01-02",
        timestamp=timestamp,
        layer_edges={"return_corr_graph": []},
        previous_states=state_map,
    )

    assert len(decayed_states) == 1
    assert decayed_states[0].state == "cooldown"
    assert decayed_states[0].missing_frames == 1
    assert decayed_states[0].temporal_score == 0.32
    assert ("return_corr_graph", "AAA", "BBB") in next_state_map
