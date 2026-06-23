from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.dtw_backend import compute_dtw_similarity_scores
from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_pairwise_correlation_matrix,
    select_topk_pair_indices,
    zscore_frame_columns,
)


def build_dtw_return_similarity_edges(
    *,
    features_1m: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
    min_similarity: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
    min_overlap_points: int = 8,
    min_overlap_floor_points: int = 5,
    min_variance: float = 1e-8,
    warmup_min_minutes: int = 5,
    max_lookback_minutes: int = 30,
    backend: str = "cpu_python",
    torch_device: str = "auto",
    torch_batch_pair_threshold: int = 1024,
    torch_activation_pair_threshold: int | None = None,
    torch_gpu_chunk_size: int = 8192,
) -> list[GraphEdge]:
    window_info = compute_effective_dtw_window(
        snapshot_time=snapshot_time,
        session_open=session_open,
        min_minutes=warmup_min_minutes,
        max_minutes=max_lookback_minutes,
        target_min_overlap_points=min_overlap_points,
        min_overlap_floor_points=min_overlap_floor_points,
    )
    if not window_info["enabled"]:
        return []
    effective_min_overlap_points = int(window_info["effective_min_overlap_points"])

    matrix = build_pivot_matrix(
        features_1m,
        value_column="ret_1m",
        snapshot_time=snapshot_time,
        minutes=int(window_info["effective_lookback_minutes"]),
    )
    if matrix.empty:
        return []

    normalized_matrix = zscore_frame_columns(matrix)
    coarse_matrix = compute_pairwise_correlation_matrix(
        normalized_matrix,
        min_periods=effective_min_overlap_points,
        min_variance=min_variance,
    )
    symbols = matrix.columns.tolist()
    pair_records: list[tuple[str, str, list[float], list[float], int]] = []
    for left_index, right_index in select_topk_pair_indices(
        coarse_matrix,
        min_score=-1.0,
        top_k_per_symbol=max(top_k_per_symbol * 4, top_k_per_symbol),
        reciprocal_top_k=None,
        degree_cap=None,
    ):
        left_symbol = symbols[left_index]
        right_symbol = symbols[right_index]
        aligned = matrix.loc[:, [left_symbol, right_symbol]].dropna()
        if len(aligned) < effective_min_overlap_points:
            continue

        left_std = float(aligned[left_symbol].std(ddof=0))
        right_std = float(aligned[right_symbol].std(ddof=0))
        if left_std < min_variance or right_std < min_variance:
            continue

        left_values = ((aligned[left_symbol] - aligned[left_symbol].mean()) / left_std).astype(float).tolist()
        right_values = ((aligned[right_symbol] - aligned[right_symbol].mean()) / right_std).astype(float).tolist()
        pair_records.append((left_symbol, right_symbol, left_values, right_values, len(aligned)))

    if not pair_records:
        return []

    activation_threshold = torch_activation_pair_threshold
    if activation_threshold is None:
        activation_threshold = torch_batch_pair_threshold

    scores, _effective_backend = compute_dtw_similarity_scores(
        [record[2] for record in pair_records],
        [record[3] for record in pair_records],
        backend=backend,
        torch_device=torch_device,
        torch_batch_pair_threshold=activation_threshold,
        torch_gpu_chunk_size=torch_gpu_chunk_size,
    )

    edges: list[GraphEdge] = []
    for (left_symbol, right_symbol, _left_values, _right_values, support_points), score in zip(
        pair_records,
        scores,
        strict=True,
    ):
        if score < min_similarity:
            continue
        edges.append(
            GraphEdge(
                graph_layer="dtw_return_similarity_graph",
                edge_type="dtw_return_similarity",
                source_symbol=left_symbol,
                target_symbol=right_symbol,
                snapshot_time=snapshot_time,
                weight=score,
                raw_score=score,
                support_points=support_points,
                edge_confidence=float(window_info["window_confidence"]),
                effective_lookback_minutes=int(window_info["effective_lookback_minutes"]),
                calculation_backend=str(_effective_backend),
            )
        )

    return _keep_top_k_with_exact_scores(
        edges,
        top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    )


def _keep_top_k_with_exact_scores(
    edges: list[GraphEdge],
    top_k_per_symbol: int,
    *,
    reciprocal_top_k: int | None,
    degree_cap: int | None,
) -> list[GraphEdge]:
    if top_k_per_symbol <= 0 and (degree_cap is None or degree_cap <= 0):
        return edges

    from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol

    return keep_top_k_per_symbol(
        edges,
        top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    )
