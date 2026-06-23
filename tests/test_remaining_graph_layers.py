from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.dtw_return_similarity import build_dtw_return_similarity_edges
from stocknetv2.domain.graph.dtw_trade_flow_similarity import build_dtw_trade_flow_similarity_edges
from stocknetv2.domain.graph.flow_alignment import build_flow_alignment_edges
from stocknetv2.domain.graph.large_trade_alignment import build_large_trade_alignment_edges
from stocknetv2.domain.graph.volume_expansion import build_volume_expansion_edges


def _timestamp_range(periods: int) -> list[pd.Timestamp]:
    return list(pd.date_range("2026-01-02T14:31:00Z", periods=periods, freq="1min"))


def _shape(periods: int) -> list[float]:
    return [0.01 * ((index % 5) - 2) + 0.001 * index for index in range(periods)]


def test_dtw_return_similarity_builds_confident_edge_after_minimum_window():
    timestamps = _timestamp_range(20)
    base = _shape(20)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "ret_1m": base + [value * 1.01 + 0.0001 for value in base] + list(reversed(base)),
        }
    )

    edges = build_dtw_return_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
        min_overlap_points=8,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.graph_layer == "dtw_return_similarity_graph"
    assert {edge.source_symbol, edge.target_symbol} == {"AAA", "BBB"}
    assert edge.edge_confidence == 0.8
    assert edge.effective_lookback_minutes == 20
    assert edge.support_points >= 8


def test_dtw_return_similarity_torch_cpu_matches_default_backend():
    timestamps = _timestamp_range(20)
    base = _shape(20)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "ret_1m": base + [value * 1.01 + 0.0001 for value in base] + list(reversed(base)),
        }
    )

    cpu_edges = build_dtw_return_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
        min_overlap_points=8,
        backend="cpu_python",
    )
    torch_edges = build_dtw_return_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
        min_overlap_points=8,
        backend="torch_cpu",
        torch_batch_pair_threshold=1,
    )

    assert len(cpu_edges) == len(torch_edges) == 1
    assert {cpu_edges[0].source_symbol, cpu_edges[0].target_symbol} == {
        torch_edges[0].source_symbol,
        torch_edges[0].target_symbol,
    }
    assert abs(cpu_edges[0].weight - torch_edges[0].weight) < 1e-8


def test_dtw_return_rejects_pairs_without_shared_timestamps():
    left_times = list(pd.date_range("2026-01-02T14:31:00Z", periods=10, freq="2min"))
    right_times = list(pd.date_range("2026-01-02T14:32:00Z", periods=10, freq="2min"))
    features = pd.DataFrame(
        {
            "timestamp": left_times + right_times,
            "symbol": ["AAA"] * 10 + ["BBB"] * 10,
            "ret_1m": _shape(10) + _shape(10),
        }
    )

    edges = build_dtw_return_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:51:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.0,
        top_k_per_symbol=1,
        min_overlap_points=8,
    )

    assert edges == []


def test_dtw_return_rejects_constant_series():
    timestamps = _timestamp_range(12)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["AAA"] * 12 + ["BBB"] * 12,
            "ret_1m": [0.01] * 12 + [0.011] * 12,
        }
    )

    edges = build_dtw_return_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:43:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.0,
        top_k_per_symbol=1,
        min_overlap_points=8,
        min_variance=1e-8,
    )

    assert edges == []


def test_flow_alignment_graph_uses_signed_flow_alignment():
    timestamps = _timestamp_range(4)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "flow_impulse_score": [1.0, 2.0, 3.0, 4.0] + [1.1, 2.1, 3.1, 4.1] + [4.0, 3.0, 2.0, 1.0],
            "imbalance_z": [1.0, 1.0, 1.0, 1.0] + [1.0, 1.0, 1.0, 1.0] + [-1.0, -1.0, -1.0, -1.0],
        }
    )

    edges = build_flow_alignment_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_flow_alignment_graph_aligns_on_shared_timestamps_when_series_lengths_differ():
    timestamps = _timestamp_range(4)
    features = pd.DataFrame(
        {
            "timestamp": [
                timestamps[1],
                timestamps[2],
                timestamps[3],
                timestamps[0],
                timestamps[1],
                timestamps[2],
                timestamps[3],
            ],
            "symbol": ["AAA"] * 3 + ["BBB"] * 4,
            "flow_impulse_score": [1.0, 2.0, 3.0, 9.0, 1.1, 2.1, 3.1],
            "imbalance_z": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        }
    )

    edges = build_flow_alignment_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_flow_alignment_torch_cpu_matches_numpy_backend():
    timestamps = _timestamp_range(6)
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 6 + ["BBB"] * 6 + ["CCC"] * 6,
            "flow_impulse_score": [1.0, 2.0, 3.0, 4.0, 3.5, 3.0]
            + [1.1, 2.1, 3.1, 4.1, 3.6, 3.1]
            + [4.0, 3.0, 2.0, 1.0, 1.2, 0.8],
            "imbalance_z": [1.0, 1.0, 1.0, 1.0, 0.8, 0.6]
            + [1.0, 1.0, 1.0, 1.0, 0.8, 0.6]
            + [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6],
        }
    )

    numpy_edges = build_flow_alignment_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:37:00Z"),
        min_score=0.9,
        top_k_per_symbol=1,
        backend="cpu_numpy",
    )
    torch_edges = build_flow_alignment_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:37:00Z"),
        min_score=0.9,
        top_k_per_symbol=1,
        backend="torch_cpu",
        torch_device="cpu",
    )

    assert len(numpy_edges) == len(torch_edges) == 1
    assert {numpy_edges[0].source_symbol, numpy_edges[0].target_symbol} == {
        torch_edges[0].source_symbol,
        torch_edges[0].target_symbol,
    }
    assert abs(numpy_edges[0].weight - torch_edges[0].weight) < 1e-8
    assert torch_edges[0].calculation_backend == "torch_cpu_v1"


def test_dtw_trade_flow_similarity_builds_edge_from_flow_shape():
    timestamps = _timestamp_range(20)
    base = [float(index % 6) + 0.1 * index for index in range(20)]
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20,
            "flow_impulse_score": base + [value * 1.01 for value in base],
            "imbalance_z": [value * 0.2 for value in base] + [value * 0.202 for value in base],
            "large_trade_ratio_z": [value * 0.1 for value in base] + [value * 0.101 for value in base],
        }
    )

    edges = build_dtw_trade_flow_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
        min_overlap_points=8,
    )

    assert len(edges) == 1
    assert edges[0].graph_layer == "dtw_trade_flow_similarity_graph"
    assert edges[0].edge_confidence == 0.8
    assert edges[0].support_points >= 8


def test_dtw_trade_flow_similarity_torch_cpu_matches_default_backend():
    timestamps = _timestamp_range(20)
    base = [float(index % 6) + 0.1 * index for index in range(20)]
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20,
            "flow_impulse_score": base + [value * 1.01 for value in base],
            "imbalance_z": [value * 0.2 for value in base] + [value * 0.202 for value in base],
            "large_trade_ratio_z": [value * 0.1 for value in base] + [value * 0.101 for value in base],
        }
    )

    cpu_edges = build_dtw_trade_flow_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
        min_overlap_points=8,
        backend="cpu_python",
    )
    torch_edges = build_dtw_trade_flow_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=1,
        min_overlap_points=8,
        backend="torch_cpu",
        torch_batch_pair_threshold=1,
    )

    assert len(cpu_edges) == len(torch_edges) == 1
    assert {cpu_edges[0].source_symbol, cpu_edges[0].target_symbol} == {
        torch_edges[0].source_symbol,
        torch_edges[0].target_symbol,
    }
    assert abs(cpu_edges[0].weight - torch_edges[0].weight) < 1e-8


def test_dtw_trade_flow_rejects_single_point_overlap():
    timestamps = _timestamp_range(10)
    features = pd.DataFrame(
        {
            "timestamp": timestamps + [timestamps[-1]],
            "symbol": ["AAA"] * 10 + ["BBB"],
            "flow_impulse_score": list(range(10)) + [9.0],
            "imbalance_z": [value * 0.1 for value in range(10)] + [0.9],
            "large_trade_ratio_z": [value * 0.2 for value in range(10)] + [1.8],
        }
    )

    edges = build_dtw_trade_flow_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:42:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.0,
        top_k_per_symbol=1,
        min_overlap_points=8,
    )

    assert edges == []


def test_volume_expansion_graph_aligns_on_shared_timestamps_when_series_lengths_differ():
    timestamps = _timestamp_range(4)
    frame = pd.DataFrame(
        {
            "timestamp": [
                timestamps[1],
                timestamps[2],
                timestamps[3],
                timestamps[0],
                timestamps[1],
                timestamps[2],
                timestamps[3],
            ],
            "symbol": ["AAA"] * 3 + ["BBB"] * 4,
            "volume_z_12": [2.0, 2.5, 3.0, 0.1, 2.1, 2.6, 3.1],
        }
    )

    edges = build_volume_expansion_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_volume_expansion_graph_uses_volume_z_and_coexpansion():
    timestamps = _timestamp_range(4)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "volume_z_12": [2.0, 2.5, 3.0, 3.5] + [2.1, 2.6, 3.1, 3.6] + [0.1, 0.2, 0.3, 0.4],
        }
    )

    edges = build_volume_expansion_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}


def test_volume_expansion_torch_cpu_matches_numpy_backend():
    timestamps = _timestamp_range(6)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 6 + ["BBB"] * 6 + ["CCC"] * 6,
            "volume_z_12": [2.0, 2.5, 3.0, 3.5, 2.8, 2.2]
            + [2.1, 2.6, 3.1, 3.6, 2.9, 2.3]
            + [0.1, 0.2, 0.3, 0.4, 0.3, 0.2],
        }
    )

    numpy_edges = build_volume_expansion_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:37:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
        backend="cpu_numpy",
    )
    torch_edges = build_volume_expansion_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:37:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
        backend="torch_cpu",
        torch_device="cpu",
    )

    assert len(numpy_edges) == len(torch_edges) == 1
    assert {numpy_edges[0].source_symbol, numpy_edges[0].target_symbol} == {
        torch_edges[0].source_symbol,
        torch_edges[0].target_symbol,
    }
    assert abs(numpy_edges[0].weight - torch_edges[0].weight) < 1e-8
    assert torch_edges[0].calculation_backend == "torch_cpu_v1"


def test_large_trade_alignment_graph_uses_large_trade_ratio():
    timestamps = _timestamp_range(4)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "large_trade_ratio_z": [2.0, 2.5, 3.0, 3.5] + [2.1, 2.6, 3.1, 3.6] + [0.1, 0.2, 0.3, 0.4],
        }
    )

    edges = build_large_trade_alignment_edges(
        feature_frame=frame,
        snapshot_time=pd.Timestamp("2026-01-02T14:35:00Z"),
        min_score=0.9,
        threshold=1.5,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    assert {edges[0].source_symbol, edges[0].target_symbol} == {"AAA", "BBB"}
