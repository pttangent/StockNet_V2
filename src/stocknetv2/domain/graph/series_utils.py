from __future__ import annotations

import math

import numpy as np
import pandas as pd


def select_time_window(
    frame: pd.DataFrame,
    *,
    snapshot_time: pd.Timestamp,
    minutes: int | None = None,
    availability_lag_minutes: int = 1,
) -> pd.DataFrame:
    """Select observations that were fully available at the decision time.

    One-minute feature and trade-flow timestamps in the legacy data identify the
    bucket start.  Unless an explicit ``available_time`` column is present, the
    observation therefore becomes usable one minute later.  This prevents a
    09:35 snapshot from consuming the unfinished 09:35-09:36 bucket.
    """

    if frame.empty or "timestamp" not in frame.columns:
        return frame.iloc[0:0].copy()

    timestamps = pd.to_datetime(frame["timestamp"])
    if "available_time" in frame.columns:
        available_times = pd.to_datetime(frame["available_time"])
    else:
        available_times = timestamps + pd.Timedelta(minutes=max(availability_lag_minutes, 0))

    eligible = available_times <= snapshot_time
    if minutes is not None:
        window_start = snapshot_time - pd.Timedelta(minutes=minutes)
        eligible &= available_times > window_start
    return frame.loc[eligible].copy()


def zscore_series(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(variance)
    if std_value < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean_value) / std_value for value in values]


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    joined = pd.DataFrame({"left": left, "right": right}).dropna()
    if joined.empty:
        return 0.0

    left_std = float(joined["left"].std(ddof=0))
    right_std = float(joined["right"].std(ddof=0))
    if left_std < 1e-12 or right_std < 1e-12:
        return 0.0

    correlation = float(joined["left"].corr(joined["right"]))
    if pd.isna(correlation):
        return 0.0
    return correlation


def build_symbol_series(
    frame: pd.DataFrame,
    *,
    symbol: str,
    value_column: str,
) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(dtype=float)

    symbol_frame = frame.loc[frame["symbol"] == symbol, ["timestamp", value_column]].dropna()
    if symbol_frame.empty:
        return pd.Series(dtype=float)

    series = pd.Series(symbol_frame[value_column].astype(float).to_numpy(), index=symbol_frame["timestamp"])
    return series.groupby(level=0).mean().sort_index()


def build_pivot_matrix(
    frame: pd.DataFrame,
    *,
    value_column: str,
    snapshot_time: pd.Timestamp,
    minutes: int | None = None,
    availability_lag_minutes: int = 1,
) -> pd.DataFrame:
    window = select_time_window(
        frame,
        snapshot_time=snapshot_time,
        minutes=minutes,
        availability_lag_minutes=availability_lag_minutes,
    )
    if window.empty or value_column not in window.columns:
        return pd.DataFrame()
    return (
        window.pivot_table(index="timestamp", columns="symbol", values=value_column, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )


def zscore_frame_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()

    normalized = matrix.copy().astype(float)
    for column in normalized.columns:
        series = normalized[column].dropna().tolist()
        if not series:
            continue
        zscored = zscore_series(series)
        normalized.loc[normalized[column].notna(), column] = zscored
    return normalized


def compute_overlap_counts(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=int)
    mask = (~matrix.isna()).to_numpy(dtype=np.int32)
    return mask.T @ mask


def compute_same_direction_ratio(matrix: pd.DataFrame) -> np.ndarray:
    return compute_conditional_same_direction_ratio(matrix, epsilon=0.0)


def compute_joint_active_counts(matrix: pd.DataFrame, *, epsilon: float) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=int)

    values = matrix.to_numpy(dtype=float)
    active = (np.abs(values) > epsilon) & ~np.isnan(values)
    active_int = active.astype(np.int32)
    return active_int.T @ active_int


def compute_conditional_same_direction_ratio(matrix: pd.DataFrame, *, epsilon: float) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=float)

    values = matrix.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    positive = ((values > epsilon) & mask).astype(np.int32)
    negative = ((values < -epsilon) & mask).astype(np.int32)
    same_direction = positive.T @ positive + negative.T @ negative
    joint_active = compute_joint_active_counts(matrix, epsilon=epsilon)
    return np.divide(
        same_direction,
        joint_active,
        out=np.zeros_like(same_direction, dtype=float),
        where=joint_active > 0,
    )


def compute_above_threshold_ratio(matrix: pd.DataFrame, threshold: float) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=float)

    values = matrix.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    above_threshold = ((values > threshold) & mask).astype(np.int32)
    overlap = compute_overlap_counts(matrix)
    co_occurrence = above_threshold.T @ above_threshold
    return np.divide(
        co_occurrence,
        overlap,
        out=np.zeros_like(co_occurrence, dtype=float),
        where=overlap > 0,
    )


def compute_pairwise_correlation_matrix(
    matrix: pd.DataFrame,
    *,
    min_periods: int = 2,
    min_variance: float = 1e-12,
) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=float)

    correlation = matrix.corr(min_periods=min_periods).fillna(0.0).to_numpy(dtype=float).copy()
    values = matrix.to_numpy(dtype=float)
    stds = np.nanstd(values, axis=0)
    variance_floor = max(float(min_variance), 1e-12)
    invalid_variance_mask = stds < variance_floor
    if invalid_variance_mask.any():
        invalid_indices = np.where(invalid_variance_mask)[0]
        correlation[invalid_indices, :] = 0.0
        correlation[:, invalid_indices] = 0.0
    return correlation


def select_topk_pair_indices(
    score_matrix: np.ndarray,
    *,
    min_score: float,
    top_k_per_symbol: int,
    reciprocal_top_k: int | None = None,
    degree_cap: int | None = None,
) -> set[tuple[int, int]]:
    pair_indices: set[tuple[int, int]] = set()
    if score_matrix.size == 0:
        return pair_indices

    size = score_matrix.shape[0]
    if top_k_per_symbol <= 0:
        rows, cols = np.where(np.triu(score_matrix, 1) >= min_score)
        pair_indices = {(int(row), int(col)) for row, col in zip(rows, cols, strict=False)}
        return _apply_degree_cap(pair_indices, score_matrix, degree_cap=degree_cap)

    reciprocal_limit = top_k_per_symbol if reciprocal_top_k is None else reciprocal_top_k
    neighbor_lists: list[list[int]] = []
    reciprocal_neighbor_sets: list[set[int]] = []

    for row_index in range(size):
        row = score_matrix[row_index].copy()
        row[row_index] = -np.inf
        candidate_count = min(top_k_per_symbol, max(size - 1, 0))
        if not np.isfinite(row).any():
            neighbor_lists.append([])
            reciprocal_neighbor_sets.append(set())
            continue
        if candidate_count <= 0:
            neighbor_lists.append([])
            reciprocal_neighbor_sets.append(set())
            continue
        top_indices = np.argpartition(row, -candidate_count)[-candidate_count:]
        ranked_indices = sorted(
            (int(index) for index in top_indices if float(row[index]) >= min_score),
            key=lambda index: float(row[index]),
            reverse=True,
        )
        neighbor_lists.append(ranked_indices)
        reciprocal_neighbor_sets.append(set(ranked_indices[: max(reciprocal_limit, 0)]))

    for row_index, ranked_indices in enumerate(neighbor_lists):
        for column_index in ranked_indices:
            score = float(score_matrix[row_index, column_index])
            if score < min_score:
                continue
            if reciprocal_limit > 0 and row_index not in reciprocal_neighbor_sets[column_index]:
                continue
            left = min(row_index, int(column_index))
            right = max(row_index, int(column_index))
            if left != right:
                pair_indices.add((left, right))
    return _apply_degree_cap(pair_indices, score_matrix, degree_cap=degree_cap)


def _apply_degree_cap(
    pair_indices: set[tuple[int, int]],
    score_matrix: np.ndarray,
    *,
    degree_cap: int | None,
) -> set[tuple[int, int]]:
    if degree_cap is None or degree_cap <= 0 or len(pair_indices) <= 1:
        return pair_indices

    sorted_pairs = sorted(
        pair_indices,
        key=lambda pair: float(score_matrix[pair[0], pair[1]]),
        reverse=True,
    )
    degrees = [0] * score_matrix.shape[0]
    kept_pairs: set[tuple[int, int]] = set()
    for left, right in sorted_pairs:
        if degrees[left] >= degree_cap or degrees[right] >= degree_cap:
            continue
        kept_pairs.add((left, right))
        degrees[left] += 1
        degrees[right] += 1
    return kept_pairs
