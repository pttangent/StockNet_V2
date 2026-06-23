from __future__ import annotations

import os

for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")

import pandas as pd

from stocknetv2.domain.graph.layer_config import ThemeDiscoverySettings
from stocknetv2.domain.graph.dtw_return_similarity import build_dtw_return_similarity_edges
from stocknetv2.domain.graph.dtw_trade_flow_similarity import build_dtw_trade_flow_similarity_edges
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.domain.graph.flow_alignment import build_flow_alignment_edges
from stocknetv2.domain.graph.large_trade_alignment import build_large_trade_alignment_edges
from stocknetv2.domain.graph.return_corr import build_return_corr_edges
from stocknetv2.domain.graph.volume_expansion import build_volume_expansion_edges


def run_layer_builder(
    layer_name: str,
    feature_frame: pd.DataFrame,
    return_window: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
    settings: ThemeDiscoverySettings | None = None,
) -> list[GraphEdge]:
    settings = settings or ThemeDiscoverySettings()
    if layer_name == "return_corr_graph":
        if return_window.empty:
            return []
        return build_return_corr_edges(
            return_window=return_window,
            snapshot_time=snapshot_time,
            min_correlation=settings.return_corr.min_correlation,
            min_overlap_points=settings.return_corr.min_overlap_points,
            backend=settings.return_corr.backend,
            torch_device=settings.return_corr.torch_device,
            top_k_per_symbol=settings.return_corr.filter.candidate_top_k,
            reciprocal_top_k=settings.return_corr.filter.reciprocal_top_k,
            degree_cap=settings.return_corr.filter.degree_cap,
        )
    if layer_name == "dtw_return_similarity_graph":
        return build_dtw_return_similarity_edges(
            features_1m=feature_frame,
            snapshot_time=snapshot_time,
            session_open=session_open,
            min_similarity=settings.dtw_return.min_similarity,
            min_overlap_points=settings.dtw_return.min_overlap_points,
            min_overlap_floor_points=settings.dtw_return.min_overlap_floor_points,
            min_variance=settings.dtw_return.min_variance,
            warmup_min_minutes=settings.dtw_return.warmup_min_minutes,
            max_lookback_minutes=settings.dtw_return.max_lookback_minutes,
            backend=settings.dtw_return.backend,
            torch_device=settings.dtw_return.torch_device,
            torch_batch_pair_threshold=settings.dtw_return.torch_batch_pair_threshold,
            torch_activation_pair_threshold=settings.dtw_return.torch_activation_pair_threshold,
            torch_gpu_chunk_size=settings.dtw_return.torch_gpu_chunk_size,
            top_k_per_symbol=settings.dtw_return.filter.candidate_top_k,
            reciprocal_top_k=settings.dtw_return.filter.reciprocal_top_k,
            degree_cap=settings.dtw_return.filter.degree_cap,
        )
    if layer_name == "flow_alignment_graph":
        return build_flow_alignment_edges(
            features_1m=feature_frame,
            snapshot_time=snapshot_time,
            min_score=settings.flow_alignment.min_score,
            top_k_per_symbol=settings.flow_alignment.filter.candidate_top_k,
            reciprocal_top_k=settings.flow_alignment.filter.reciprocal_top_k,
            degree_cap=settings.flow_alignment.filter.degree_cap,
            lookback_minutes=settings.flow_alignment.lookback_minutes,
            min_joint_active_points=settings.flow_alignment.min_joint_active_points,
            activity_epsilon=settings.flow_alignment.activity_epsilon,
            min_variance=settings.flow_alignment.min_variance,
            backend=settings.flow_alignment.backend,
            torch_device=settings.flow_alignment.torch_device,
        )
    if layer_name == "dtw_trade_flow_similarity_graph":
        return build_dtw_trade_flow_similarity_edges(
            features_1m=feature_frame,
            snapshot_time=snapshot_time,
            session_open=session_open,
            min_similarity=settings.dtw_trade_flow.min_similarity,
            min_overlap_points=settings.dtw_trade_flow.min_overlap_points,
            min_overlap_floor_points=settings.dtw_trade_flow.min_overlap_floor_points,
            min_variance=settings.dtw_trade_flow.min_variance,
            warmup_min_minutes=settings.dtw_trade_flow.warmup_min_minutes,
            max_lookback_minutes=settings.dtw_trade_flow.max_lookback_minutes,
            backend=settings.dtw_trade_flow.backend,
            torch_device=settings.dtw_trade_flow.torch_device,
            torch_batch_pair_threshold=settings.dtw_trade_flow.torch_batch_pair_threshold,
            torch_activation_pair_threshold=settings.dtw_trade_flow.torch_activation_pair_threshold,
            torch_gpu_chunk_size=settings.dtw_trade_flow.torch_gpu_chunk_size,
            top_k_per_symbol=settings.dtw_trade_flow.filter.candidate_top_k,
            reciprocal_top_k=settings.dtw_trade_flow.filter.reciprocal_top_k,
            degree_cap=settings.dtw_trade_flow.filter.degree_cap,
        )
    if layer_name == "volume_expansion_graph":
        return build_volume_expansion_edges(
            feature_frame=feature_frame,
            snapshot_time=snapshot_time,
            min_score=settings.volume_expansion.min_score,
            threshold=settings.volume_expansion.threshold,
            top_k_per_symbol=settings.volume_expansion.filter.candidate_top_k,
            reciprocal_top_k=settings.volume_expansion.filter.reciprocal_top_k,
            degree_cap=settings.volume_expansion.filter.degree_cap,
            backend=settings.volume_expansion.backend,
            torch_device=settings.volume_expansion.torch_device,
        )
    if layer_name == "large_trade_alignment_graph":
        return build_large_trade_alignment_edges(
            feature_frame=feature_frame,
            snapshot_time=snapshot_time,
            min_score=settings.large_trade_alignment.min_score,
            threshold=settings.large_trade_alignment.threshold,
            top_k_per_symbol=settings.large_trade_alignment.filter.candidate_top_k,
            reciprocal_top_k=settings.large_trade_alignment.filter.reciprocal_top_k,
            degree_cap=settings.large_trade_alignment.filter.degree_cap,
            backend=settings.large_trade_alignment.backend,
            torch_device=settings.large_trade_alignment.torch_device,
        )
    raise ValueError(f"Unsupported layer builder: {layer_name}")
