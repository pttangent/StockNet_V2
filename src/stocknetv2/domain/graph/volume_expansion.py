from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_above_threshold_ratio,
    compute_overlap_counts,
    compute_pairwise_correlation_matrix,
    select_topk_pair_indices,
)
from stocknetv2.domain.graph.torch_graph_backend import (
    compute_activity_metrics_torch,
    resolve_graph_backend,
    resolve_graph_torch_device,
)

ACTIVITY_LAYER_LOOKBACK_MINUTES = 60


def build_volume_expansion_edges(
    *,
    feature_frame: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_score: float,
    threshold: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
    backend: str = "cpu_numpy",
    torch_device: str = "auto",
) -> list[GraphEdge]:
    value_matrix = build_pivot_matrix(
        feature_frame,
        value_column="volume_z_12",
        snapshot_time=snapshot_time,
        minutes=ACTIVITY_LAYER_LOOKBACK_MINUTES,
    )
    return _build_activity_edges(
        value_matrix=value_matrix,
        snapshot_time=snapshot_time,
        graph_layer="volume_expansion_graph",
        edge_type="volume_expansion",
        min_score=min_score,
        threshold=threshold,
        top_k_per_symbol=top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
        backend=backend,
        torch_device=torch_device,
    )


def _build_activity_edges(
    *,
    value_matrix: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    graph_layer: str,
    edge_type: str,
    min_score: float,
    threshold: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None,
    degree_cap: int | None,
    backend: str,
    torch_device: str,
) -> list[GraphEdge]:
    if value_matrix.empty:
        return []
    active_columns = value_matrix.columns[(value_matrix > threshold).any(axis=0)]
    if len(active_columns) < 2:
        return []
    value_matrix = value_matrix.loc[:, active_columns].copy()

    effective_backend = resolve_graph_backend(requested_backend=backend, torch_device=torch_device)
    if effective_backend == "cpu_numpy":
        correlation_matrix = compute_pairwise_correlation_matrix(value_matrix)
        co_expansion_matrix = compute_above_threshold_ratio(value_matrix, threshold)
        overlap_counts = compute_overlap_counts(value_matrix)
        calculation_backend = "cpu_numpy_v1"
    else:
        correlation_matrix, co_expansion_matrix, overlap_counts = compute_activity_metrics_torch(
            value_matrix,
            threshold=threshold,
            device=resolve_graph_torch_device(effective_backend=effective_backend, torch_device=torch_device),
        )
        calculation_backend = f"{effective_backend}_v1"
    score_matrix = 0.5 * correlation_matrix + 0.5 * co_expansion_matrix
    symbols = value_matrix.columns.tolist()

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        score_matrix,
        min_score=min_score,
        top_k_per_symbol=top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    ):
        score = float(score_matrix[left_index, right_index])
        edges.append(
            GraphEdge(
                graph_layer=graph_layer,
                edge_type=edge_type,
                source_symbol=symbols[left_index],
                target_symbol=symbols[right_index],
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=int(overlap_counts[left_index, right_index]),
                calculation_backend=calculation_backend,
            )
        )
    return edges
