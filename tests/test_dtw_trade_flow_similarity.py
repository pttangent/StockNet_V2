from __future__ import annotations

import pandas as pd
import pytest

from stocknetv2.domain.graph.dtw_trade_flow_similarity import build_dtw_trade_flow_similarity_edges


def _timestamp_range(num_points: int) -> list[pd.Timestamp]:
    base_time = pd.Timestamp("2026-01-02T14:30:00Z")
    return [base_time + pd.Timedelta(minutes=i) for i in range(num_points)]


def test_dtw_trade_flow_similarity_batch_computes_correctly():
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
        backend="torch_cpu",
        torch_batch_pair_threshold=1,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.graph_layer == "dtw_trade_flow_similarity_graph"
    assert edge.calculation_backend == "torch_cpu"
    assert edge.weight > 0.9


def test_dtw_trade_flow_similarity_chunking():
    timestamps = _timestamp_range(20)
    base = [float(index % 6) + 0.1 * index for index in range(20)]
    features = pd.DataFrame(
        {
            "timestamp": timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "flow_impulse_score": base + [value * 1.01 for value in base] + [value * 0.99 for value in base],
            "imbalance_z": [value * 0.2 for value in base] + [value * 0.202 for value in base] + [value * 0.198 for value in base],
            "large_trade_ratio_z": [value * 0.1 for value in base] + [value * 0.101 for value in base] + [value * 0.099 for value in base],
        }
    )

    # Run with a tiny chunk size of 2 to force multiple chunks
    edges = build_dtw_trade_flow_similarity_edges(
        features_1m=features,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        min_similarity=0.9,
        top_k_per_symbol=2,
        min_overlap_points=8,
        backend="torch_cpu",
        torch_batch_pair_threshold=1,
        torch_gpu_chunk_size=2,
    )

    assert len(edges) > 0
    for edge in edges:
        assert edge.calculation_backend == "torch_cpu"

