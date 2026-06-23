from __future__ import annotations

import numpy as np
import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.series_utils import compute_overlap_counts, select_topk_pair_indices
from stocknetv2.domain.graph.torch_graph_backend import (
    compute_pairwise_correlation_metrics_torch,
    resolve_graph_backend,
    resolve_graph_torch_device,
)


def build_return_corr_edges(
    *,
    return_window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    min_correlation: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
    min_overlap_points: int = 2,
    backend: str = "cpu_numpy",
    torch_device: str = "auto",
) -> list[GraphEdge]:
    baseline = return_window.median(axis=1) if return_window.shape[1] >= 10 else return_window.mean(axis=1)
    residual_window = return_window.sub(baseline, axis=0)
    working_window = _prefer_residual_returns(raw_window=return_window, residual_window=residual_window)
    effective_backend = resolve_graph_backend(requested_backend=backend, torch_device=torch_device)
    if effective_backend == "cpu_numpy":
        overlap_counts = compute_overlap_counts(working_window)
        correlation = working_window.corr(min_periods=min_overlap_points).fillna(0.0)
        score_matrix = correlation.to_numpy(dtype=float, copy=True)
        calculation_backend = "cpu_numpy_v1"
    else:
        score_matrix, overlap_counts = compute_pairwise_correlation_metrics_torch(
            working_window,
            min_periods=min_overlap_points,
            min_variance=1e-12,
            device=resolve_graph_torch_device(effective_backend=effective_backend, torch_device=torch_device),
        )
        correlation = pd.DataFrame(score_matrix, index=working_window.columns, columns=working_window.columns)
        calculation_backend = f"{effective_backend}_v1"
    np.fill_diagonal(score_matrix, -np.inf)
    score_matrix = np.where(overlap_counts >= min_overlap_points, score_matrix, -np.inf)
    symbols = correlation.columns.tolist()

    edges: list[GraphEdge] = []
    for left_index, right_index in select_topk_pair_indices(
        score_matrix,
        min_score=min_correlation,
        top_k_per_symbol=top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    ):
        score = float(score_matrix[left_index, right_index])
        edges.append(
            GraphEdge(
                graph_layer="return_corr_graph",
                edge_type="return_correlation",
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


def _prefer_residual_returns(*, raw_window: pd.DataFrame, residual_window: pd.DataFrame) -> pd.DataFrame:
    valid_columns = int((residual_window.std(ddof=0) >= 1e-8).sum())
    if valid_columns >= 2:
        return residual_window
    return raw_window
