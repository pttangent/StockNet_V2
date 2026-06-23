from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from stocknetv2.domain.graph.dtw_backend import compute_dtw_similarity_scores
from stocknetv2.domain.graph.dtw_pair_batch import PairComponentRecord
from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.edge_filter import keep_top_k_per_symbol
from stocknetv2.domain.graph.series_utils import (
    build_pivot_matrix,
    compute_pairwise_correlation_matrix,
    select_topk_pair_indices,
    zscore_frame_columns,
)


def build_dtw_trade_flow_similarity_edges(
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

    minutes = int(window_info["effective_lookback_minutes"])
    effective_min_overlap_points = int(window_info["effective_min_overlap_points"])
    flow_matrix = _build_matrix(
        features_1m,
        value_column="flow_impulse_score",
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    imbalance_matrix = _build_matrix(
        features_1m,
        value_column="imbalance_z",
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    large_trade_matrix = _build_matrix(
        features_1m,
        value_column="large_trade_ratio_z",
        snapshot_time=snapshot_time,
        minutes=minutes,
    )
    matrices = (flow_matrix, imbalance_matrix, large_trade_matrix)
    if all(matrix.empty for matrix in matrices):
        return []

    symbols = sorted({str(symbol) for matrix in matrices for symbol in matrix.columns})
    coarse_matrix = (
        0.50 * _coarse_similarity_matrix(flow_matrix, symbols, effective_min_overlap_points, min_variance)
        + 0.30 * _coarse_similarity_matrix(imbalance_matrix, symbols, effective_min_overlap_points, min_variance)
        + 0.20 * _coarse_similarity_matrix(large_trade_matrix, symbols, effective_min_overlap_points, min_variance)
    )

    pair_indices = list(select_topk_pair_indices(
        coarse_matrix,
        min_score=-1.0,
        top_k_per_symbol=max(top_k_per_symbol * 4, top_k_per_symbol),
        reciprocal_top_k=None,
        degree_cap=None,
    ))

    records = _collect_trade_flow_pair_components(
        symbols=symbols,
        pair_indices=pair_indices,
        flow_matrix=flow_matrix,
        imbalance_matrix=imbalance_matrix,
        large_trade_matrix=large_trade_matrix,
        min_overlap_points=effective_min_overlap_points,
        min_variance=min_variance,
    )

    if not records:
        return []

    activation_threshold = torch_activation_pair_threshold
    if activation_threshold is None:
        activation_threshold = torch_batch_pair_threshold

    scores, effective_backend = compute_dtw_similarity_scores(
        [r.left for r in records],
        [r.right for r in records],
        backend=backend,
        torch_device=torch_device,
        torch_batch_pair_threshold=activation_threshold,
        torch_gpu_chunk_size=torch_gpu_chunk_size,
    )

    pair_weights = defaultdict(float)
    pair_weighted_scores = defaultdict(float)
    pair_min_support = {}

    for r, score in zip(records, scores, strict=True):
        pair_key = r.pair_key
        pair_weights[pair_key] += r.component_weight
        pair_weighted_scores[pair_key] += r.component_weight * score
        if pair_key not in pair_min_support:
            pair_min_support[pair_key] = r.support_points
        else:
            pair_min_support[pair_key] = min(pair_min_support[pair_key], r.support_points)

    edges: list[GraphEdge] = []
    for pair_key, total_weight in pair_weights.items():
        if total_weight <= 0:
            continue
        final_score = pair_weighted_scores[pair_key] / total_weight
        if final_score < min_similarity:
            continue
        left_symbol, right_symbol = pair_key
        edges.append(
            GraphEdge(
                graph_layer="dtw_trade_flow_similarity_graph",
                edge_type="dtw_trade_flow_similarity",
                source_symbol=left_symbol,
                target_symbol=right_symbol,
                snapshot_time=snapshot_time,
                weight=final_score,
                raw_score=final_score,
                support_points=pair_min_support[pair_key],
                edge_confidence=float(window_info["window_confidence"]),
                effective_lookback_minutes=minutes,
                calculation_backend=str(effective_backend),
            )
        )
    return keep_top_k_per_symbol(
        edges,
        top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
    )


def _collect_trade_flow_pair_components(
    symbols: list[str],
    pair_indices: list[tuple[int, int]],
    flow_matrix: pd.DataFrame,
    imbalance_matrix: pd.DataFrame,
    large_trade_matrix: pd.DataFrame,
    min_overlap_points: int,
    min_variance: float,
) -> list[PairComponentRecord]:
    records = []
    for left_index, right_index in pair_indices:
        left_symbol = symbols[left_index]
        right_symbol = symbols[right_index]

        pair_candidates = []
        for component_weight, matrix in (
            (0.50, flow_matrix),
            (0.30, imbalance_matrix),
            (0.20, large_trade_matrix),
        ):
            result = _matrix_series_similarity(
                matrix,
                left_symbol,
                right_symbol,
                min_overlap_points=min_overlap_points,
                min_variance=min_variance,
            )
            if result is None:
                continue
            left_values, right_values, support_points = result
            pair_candidates.append((component_weight, left_values, right_values, support_points))

        if len(pair_candidates) < 2:
            continue
        min_support = min(item[3] for item in pair_candidates)
        if min_support < min_overlap_points:
            continue

        for weight, left, right, support in pair_candidates:
            records.append(
                PairComponentRecord(
                    pair_key=(left_symbol, right_symbol),
                    component_weight=weight,
                    left=left,
                    right=right,
                    support_points=support,
                )
            )
    return records


def _matrix_series_similarity(
    matrix: pd.DataFrame,
    left_symbol: str,
    right_symbol: str,
    *,
    min_overlap_points: int,
    min_variance: float,
) -> tuple[list[float], list[float], int] | None:
    if matrix.empty or left_symbol not in matrix.columns or right_symbol not in matrix.columns:
        return None
    aligned = matrix.loc[:, [left_symbol, right_symbol]].dropna()
    if len(aligned) < min_overlap_points:
        return None

    left_std = float(aligned[left_symbol].std(ddof=0))
    right_std = float(aligned[right_symbol].std(ddof=0))
    if left_std < min_variance or right_std < min_variance:
        return None

    left_values = ((aligned[left_symbol] - aligned[left_symbol].mean()) / left_std).astype(float).tolist()
    right_values = ((aligned[right_symbol] - aligned[right_symbol].mean()) / right_std).astype(float).tolist()
    return left_values, right_values, len(aligned)


def _build_matrix(
    features_1m: pd.DataFrame,
    *,
    value_column: str,
    snapshot_time: pd.Timestamp,
    minutes: int,
) -> pd.DataFrame:
    return build_pivot_matrix(
        features_1m,
        value_column=value_column,
        snapshot_time=snapshot_time,
        minutes=minutes,
    )


def _coarse_similarity_matrix(
    matrix: pd.DataFrame,
    symbols: list[str],
    min_overlap_points: int,
    min_variance: float,
) -> np.ndarray:
    if matrix.empty:
        return np.zeros((len(symbols), len(symbols)), dtype=float)
    aligned = zscore_frame_columns(matrix.reindex(columns=symbols))
    return compute_pairwise_correlation_matrix(
        aligned,
        min_periods=min_overlap_points,
        min_variance=min_variance,
    )


def _backend_label(backend: str, torch_device: str) -> str:
    return f"{backend}:{torch_device}"
