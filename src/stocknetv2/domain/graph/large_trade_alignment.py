from __future__ import annotations

import pandas as pd

from stocknetv2.domain.graph.series_utils import build_pivot_matrix
from stocknetv2.domain.graph.volume_expansion import ACTIVITY_LAYER_LOOKBACK_MINUTES, _build_activity_edges


def build_large_trade_alignment_edges(
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
):
    value_matrix = build_pivot_matrix(
        feature_frame,
        value_column="large_trade_ratio_z",
        snapshot_time=snapshot_time,
        minutes=ACTIVITY_LAYER_LOOKBACK_MINUTES,
    )
    return _build_activity_edges(
        value_matrix=value_matrix,
        snapshot_time=snapshot_time,
        graph_layer="large_trade_alignment_graph",
        edge_type="large_trade_alignment",
        min_score=min_score,
        threshold=threshold,
        top_k_per_symbol=top_k_per_symbol,
        reciprocal_top_k=reciprocal_top_k,
        degree_cap=degree_cap,
        backend=backend,
        torch_device=torch_device,
    )
