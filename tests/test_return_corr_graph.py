from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.return_corr import build_return_corr_edges


def test_return_corr_graph_builds_thresholded_edges():
    return_window = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03, 0.04],
            "BBB": [0.011, 0.021, 0.031, 0.041],
            "CCC": [-0.01, -0.02, -0.03, -0.04],
        }
    )

    edges = build_return_corr_edges(
        return_window=return_window,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        min_correlation=0.9,
        top_k_per_symbol=1,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.graph_layer == "return_corr_graph"
    assert {edge.source_symbol, edge.target_symbol} == {"AAA", "BBB"}
    assert edge.weight >= 0.9
    assert edge.support_points == 4


def test_return_corr_graph_skips_pairs_below_threshold():
    return_window = pd.DataFrame(
        {
            "AAA": [0.01, 0.00, 0.03, 0.02],
            "BBB": [-0.01, 0.02, -0.03, 0.01],
        }
    )

    edges = build_return_corr_edges(
        return_window=return_window,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        min_correlation=0.5,
        top_k_per_symbol=1,
    )

    assert edges == []


def test_return_corr_torch_cpu_matches_numpy_backend():
    return_window = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03, 0.04],
            "BBB": [0.011, 0.021, 0.031, 0.041],
            "CCC": [-0.01, -0.02, -0.03, -0.04],
        }
    )

    numpy_edges = build_return_corr_edges(
        return_window=return_window,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        min_correlation=0.9,
        top_k_per_symbol=1,
        backend="cpu_numpy",
    )
    torch_edges = build_return_corr_edges(
        return_window=return_window,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        min_correlation=0.9,
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
