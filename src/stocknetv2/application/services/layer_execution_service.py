from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from stocknetv2.domain.graph.layer_config import ThemeDiscoverySettings
from stocknetv2.domain.community.community import Community
from stocknetv2.domain.community.detector import detect_communities_from_edges
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.application.services.layer_worker import run_layer_builder
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


@dataclass(frozen=True)
class LayerExecutionResult:
    layer_edges: dict[str, list[GraphEdge]]
    layer_communities: dict[str, list[Community]]


class LayerExecutionService:
    """Build all six T1 graph layers and detect per-layer communities."""

    _LAYER_NAMES = (
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    )

    def __init__(
        self,
        *,
        parallel_workers: int = 1,
        selected_layers: tuple[str, ...] | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
        settings: ThemeDiscoverySettings | None = None,
    ) -> None:
        self._parallel_workers = max(1, parallel_workers)
        self._selected_layers = selected_layers or self._LAYER_NAMES
        self._executor_factory = executor_factory or _build_process_pool_executor
        self._executor: Executor | None = None
        self._settings = settings or ThemeDiscoverySettings()

    def execute_for_snapshot(
        self,
        *,
        inputs: TradeDateInputs,
        snapshot_time: pd.Timestamp,
        session_open: pd.Timestamp,
    ) -> LayerExecutionResult:
        feature_frame = self._build_feature_frame(inputs)
        return_window = self._build_return_window(
            inputs.bars_5m,
            snapshot_time,
            session_open=session_open,
            lookback_bars=self._settings.return_corr.lookback_bars,
        )
        universe_symbol_count = int(feature_frame["symbol"].astype(str).nunique()) if "symbol" in feature_frame.columns else 0

        layer_edges = self._execute_layer_builders(
            feature_frame=feature_frame,
            return_window=return_window,
            snapshot_time=snapshot_time,
            session_open=session_open,
        )

        layer_communities = {
            layer_name: detect_communities_from_edges(
                edges,
                min_members=self._settings.layer_community_detection.min_members,
                algorithm=self._settings.layer_community_detection.algorithm,
                resolution=self._settings.layer_community_detection.resolution,
                universe_symbol_count=universe_symbol_count,
                market_mode_max_member_ratio=self._settings.layer_community_detection.market_mode_max_member_ratio,
                fallback_algorithm=self._settings.layer_community_detection.fallback_algorithm,
            )
            for layer_name, edges in layer_edges.items()
        }
        return LayerExecutionResult(layer_edges=layer_edges, layer_communities=layer_communities)

    def close(self) -> None:
        if self._executor and hasattr(self._executor, "shutdown"):
            self._executor.shutdown(wait=True, cancel_futures=False)
        self._executor = None

    def _execute_layer_builders(
        self,
        *,
        feature_frame: pd.DataFrame,
        return_window: pd.DataFrame,
        snapshot_time: pd.Timestamp,
        session_open: pd.Timestamp,
    ) -> dict[str, list[GraphEdge]]:
        if self._parallel_workers <= 1:
            return {
                layer_name: run_layer_builder(
                    layer_name,
                    feature_frame,
                    return_window,
                    snapshot_time,
                    session_open,
                    self._settings,
                )
                for layer_name in self._selected_layers
            }

        executor = self._get_or_create_executor()
        futures = {
            layer_name: executor.submit(
                run_layer_builder,
                layer_name,
                feature_frame,
                return_window,
                snapshot_time,
                session_open,
                self._settings,
            )
            for layer_name in self._selected_layers
        }
        return {
            layer_name: futures[layer_name].result()
            for layer_name in self._selected_layers
        }

    def _get_or_create_executor(self) -> Executor:
        if self._executor is None:
            self._executor = self._executor_factory(min(self._parallel_workers, len(self._selected_layers)))
        return self._executor

    @staticmethod
    def _build_feature_frame(inputs: TradeDateInputs) -> pd.DataFrame:
        if inputs.features_1m.empty and inputs.trade_flow_1m.empty:
            return pd.DataFrame(columns=["timestamp", "symbol"])
        if inputs.features_1m.empty:
            frame = inputs.trade_flow_1m.copy()
        elif inputs.trade_flow_1m.empty:
            frame = inputs.features_1m.copy()
        else:
            frame = inputs.features_1m.merge(
                inputs.trade_flow_1m,
                on=["timestamp", "symbol"],
                how="left",
                suffixes=("", "_flow"),
            )

        for source_column, target_column in {
            "large_trade_ratio_z_flow": "large_trade_ratio_z",
        }.items():
            if source_column in frame.columns and target_column in frame.columns:
                frame[target_column] = frame[target_column].fillna(frame[source_column])
                frame = frame.drop(columns=[source_column])
        if not inputs.bars_5m.empty and "symbol" in inputs.bars_5m.columns and "symbol" in frame.columns:
            allowed_symbols = set(inputs.bars_5m["symbol"].dropna().astype(str).unique().tolist())
            frame = frame[frame["symbol"].astype(str).isin(allowed_symbols)].copy()
        return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _build_return_window(
        bars_5m: pd.DataFrame,
        snapshot_time: pd.Timestamp,
        *,
        session_open: pd.Timestamp,
        lookback_bars: int,
    ) -> pd.DataFrame:
        """Build a fixed regular-session rolling return window.

        Five-minute bars are right-labelled, so a bar stamped at 09:35 is
        available at the 09:35 snapshot.  Premarket observations are excluded,
        and only the most recent configured number of return observations are
        retained.
        """

        if bars_5m.empty or lookback_bars <= 0:
            return pd.DataFrame()
        timestamps = pd.to_datetime(bars_5m["timestamp"])
        frame = bars_5m.loc[(timestamps > session_open) & (timestamps <= snapshot_time)].copy()
        if frame.empty:
            return pd.DataFrame()
        pivot = frame.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        returns = np.log(pivot / pivot.shift(1)).dropna(how="all")
        return returns.tail(lookback_bars)


def _build_process_pool_executor(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    )
