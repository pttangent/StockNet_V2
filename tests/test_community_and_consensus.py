from __future__ import annotations

import pandas as pd
import pytest

from stocknetv2.application.services.consensus_service import ConsensusService
from stocknetv2.domain.community.community import Community
from stocknetv2.domain.community.consensus_matrix import build_consensus_matrix
from stocknetv2.domain.community.detector import detect_communities_from_edges
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import select_topk_pair_indices


def _edge(source: str, target: str, weight: float, layer: str = "return_corr_graph") -> GraphEdge:
    return GraphEdge(
        graph_layer=layer,
        edge_type="test",
        source_symbol=source,
        target_symbol=target,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        weight=weight,
        raw_score=weight,
        support_points=4,
    )


def test_detect_communities_from_edges_finds_connected_components():
    communities = detect_communities_from_edges(
        [_edge("AAA", "BBB", 0.9), _edge("BBB", "CCC", 0.8), _edge("DDD", "EEE", 0.95)],
        min_members=2,
    )

    member_sets = [set(community.members) for community in communities]
    assert {"AAA", "BBB", "CCC"} in member_sets
    assert {"DDD", "EEE"} in member_sets


def test_weighted_leiden_request_fails_instead_of_silently_falling_back(monkeypatch):
    import stocknetv2.domain.community.detector as detector

    monkeypatch.setattr(detector, "ig", None)
    monkeypatch.setattr(detector, "leidenalg", None)
    monkeypatch.setattr(detector, "_LEIDEN_IMPORT_ERROR", ImportError("missing test dependency"))

    with pytest.raises(RuntimeError, match="automatic connected-components fallback is disabled"):
        detector.detect_communities_from_edges(
            [_edge("AAA", "BBB", 0.9)],
            min_members=2,
            algorithm="weighted_leiden",
            fallback_algorithm="connected_components",
        )


def test_build_consensus_matrix_aggregates_weighted_coassignment():
    layer_communities = {
        "return_corr_graph": [["AAA", "BBB"], ["CCC", "DDD"]],
        "flow_alignment_graph": [["AAA", "BBB", "CCC"]],
    }
    layer_weights = {
        "return_corr_graph": 0.25,
        "flow_alignment_graph": 0.20,
    }

    matrix = build_consensus_matrix(layer_communities, layer_weights)

    assert matrix.loc["AAA", "BBB"] == 0.45
    assert matrix.loc["AAA", "CCC"] == 0.20
    assert matrix.loc["AAA", "DDD"] == 0.0


def test_select_topk_pair_indices_can_require_reciprocal_neighbors():
    matrix = pd.DataFrame(
        [
            [0.0, 0.95, 0.90],
            [0.95, 0.0, 0.10],
            [0.40, 0.89, 0.0],
        ]
    ).to_numpy()

    pair_indices = select_topk_pair_indices(
        matrix,
        min_score=0.5,
        top_k_per_symbol=2,
        reciprocal_top_k=1,
        degree_cap=6,
    )

    assert pair_indices == {(0, 1)}


def test_detect_communities_marks_market_mode_for_large_universes():
    edges = [_edge(f"S{i:03d}", f"S{i + 1:03d}", 0.9) for i in range(60)]

    communities = detect_communities_from_edges(
        edges,
        min_members=2,
        algorithm="connected_components",
        universe_symbol_count=100,
        market_mode_max_member_ratio=0.15,
    )

    assert len(communities) == 1
    assert communities[0].is_market_mode is True


def test_consensus_service_filters_single_family_clusters():
    service = ConsensusService()

    candidates = service.build_consensus_themes(
        run_id="run_001",
        snapshot_id="snapshot_001",
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        layer_communities={
            "flow_alignment_graph": [Community(members=["AAA", "BBB", "CCC"])],
            "dtw_trade_flow_similarity_graph": [Community(members=["AAA", "BBB", "CCC"])],
        },
    )

    assert candidates == []
