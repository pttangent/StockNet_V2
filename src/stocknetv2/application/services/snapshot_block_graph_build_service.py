from __future__ import annotations

import os
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.layer_profile_service import LayerProfile
from stocknetv2.domain.graph.layer_config import ThemeDiscoverySettings
from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs
from stocknetv2.infrastructure.repositories.month_pack_read_repository import MonthPackReadRepository, SnapshotBlockInputs
from stocknetv2.infrastructure.repositories.snapshot_artifact_repository import SnapshotArtifactRepository


@dataclass(frozen=True)
class SnapshotComputationResult:
    snapshot_id: str
    trade_date: str
    snapshot_clock: str
    edge_count: int
    layer_count: int
    elapsed_seconds: float
    snapshot_root: Path
    worker_pid: int


class SnapshotBlockGraphBuildService:
    _FULL_GRAPH_LAYERS = set(LayerExecutionService._LAYER_NAMES)

    def __init__(
        self,
        *,
        repository: MonthPackReadRepository,
        artifact_repository: SnapshotArtifactRepository | None = None,
        snapshot_clock: SnapshotClock | None = None,
    ) -> None:
        self._repository = repository
        self._artifact_repository = artifact_repository or SnapshotArtifactRepository()
        self._snapshot_clock = snapshot_clock or SnapshotClock()

    def run_block(
        self,
        *,
        trade_date: str,
        snapshots,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        profile: LayerProfile,
        settings: ThemeDiscoverySettings,
        output_root: Path,
        run_name: str,
    ) -> list[SnapshotComputationResult]:
        inputs = self._repository.read_snapshot_block(
            trade_date=trade_date,
            window_start=window_start,
            window_end=window_end,
            use_graph_features=True,
            include_trade_flow=True,
        )
        session_open = self._snapshot_clock.session_open_timestamp(trade_date)
        service = LayerExecutionService(
            parallel_workers=1,
            selected_layers=profile.selected_layers,
            settings=settings,
        )
        results: list[SnapshotComputationResult] = []
        try:
            for snapshot in snapshots:
                started_at = time.perf_counter()
                snapshot_inputs, input_window_diagnostics = self._prepare_snapshot_inputs(
                    block_inputs=inputs,
                    snapshot_time=snapshot.snapshot_time,
                    profile=profile,
                )
                layer_result = service.execute_for_snapshot(
                    inputs=snapshot_inputs,
                    snapshot_time=snapshot.snapshot_time,
                    session_open=session_open,
                )
                snapshot_root = (
                    output_root
                    / f"month={snapshot.trade_date[:7]}"
                    / "dates"
                    / f"date={snapshot.trade_date}"
                    / "snapshots"
                    / f"snapshot={snapshot.snapshot_clock}"
                )
                completed_layers = list(layer_result.layer_edges.keys())
                missing_layers = sorted(self._FULL_GRAPH_LAYERS.difference(completed_layers))
                edge_count = sum(len(edges) for edges in layer_result.layer_edges.values())
                elapsed_seconds = round(time.perf_counter() - started_at, 4)
                diagnostics = {
                    "run_name": run_name,
                    "trade_date": snapshot.trade_date,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_clock": snapshot.snapshot_clock,
                    "profile": profile.name,
                    "snapshot_time_utc": snapshot.snapshot_time.isoformat(),
                    "bars_5m_timestamp_semantics": "bar_end_utc",
                    "input_windows": input_window_diagnostics,
                    "layer_edge_counts": {
                        layer_name: len(edges) for layer_name, edges in layer_result.layer_edges.items()
                    },
                }
                self._artifact_repository.write_completed_snapshot_artifact(
                    snapshot_root=snapshot_root,
                    payload={
                        "run_name": run_name,
                        "trade_date": snapshot.trade_date,
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_clock": snapshot.snapshot_clock,
                        "profile": profile.name,
                        "profile_status": "complete",
                        "profile_complete": True,
                        "full_graph_complete": not missing_layers,
                        "completed_layers": completed_layers,
                        "missing_layers": missing_layers,
                        "layer_count": len(completed_layers),
                        "edge_count": edge_count,
                        "elapsed_seconds": elapsed_seconds,
                        "worker_pid": os.getpid(),
                        "diagnostics": diagnostics,
                    },
                    layer_edges=layer_result.layer_edges,
                    layer_communities=layer_result.layer_communities,
                )
                results.append(
                    SnapshotComputationResult(
                        snapshot_id=snapshot.snapshot_id,
                        trade_date=snapshot.trade_date,
                        snapshot_clock=snapshot.snapshot_clock,
                        edge_count=edge_count,
                        layer_count=len(completed_layers),
                        elapsed_seconds=elapsed_seconds,
                        snapshot_root=snapshot_root,
                        worker_pid=os.getpid(),
                    )
                )
        finally:
            service.close()
            del inputs
        return results

    def _prepare_snapshot_inputs(
        self,
        *,
        block_inputs: SnapshotBlockInputs,
        snapshot_time: pd.Timestamp,
        profile: LayerProfile,
    ) -> tuple[TradeDateInputs, dict[str, dict[str, object]]]:
        raw_window = self._slice_frame(block_inputs.raw_1m, snapshot_time=snapshot_time)
        trade_flow_window = self._slice_frame(block_inputs.trade_flow_1m, snapshot_time=snapshot_time)
        bars_window = self._slice_frame(block_inputs.bars_5m, snapshot_time=snapshot_time)
        features_window = self._slice_frame(block_inputs.features_1m, snapshot_time=snapshot_time)
        if not features_window.empty and profile.feature_columns:
            available = [column for column in profile.feature_columns if column in features_window.columns]
            features_window = features_window.loc[:, available].copy()

        diagnostics = {
            "raw_1m": self._validate_input_window(
                frame=raw_window,
                requested_start=block_inputs.requested_start,
                requested_end=snapshot_time,
            ),
            "trade_flow_1m": self._validate_input_window(
                frame=trade_flow_window,
                requested_start=block_inputs.requested_start,
                requested_end=snapshot_time,
            ),
            "bars_5m": self._validate_input_window(
                frame=bars_window,
                requested_start=block_inputs.requested_start,
                requested_end=snapshot_time,
            ),
        }
        return (
            TradeDateInputs(
                trade_date=block_inputs.trade_date,
                bars_5m=bars_window,
                trade_flow_1m=trade_flow_window,
                features_1m=features_window,
                data_version=block_inputs.data_version,
            ),
            diagnostics,
        )

    @staticmethod
    def _slice_frame(frame: pd.DataFrame, *, snapshot_time: pd.Timestamp) -> pd.DataFrame:
        if frame.empty or "timestamp" not in frame.columns:
            return frame.copy()
        sliced = frame.loc[frame["timestamp"] <= snapshot_time].copy()
        return sliced.sort_values(["timestamp", "symbol"]).reset_index(drop=True) if "symbol" in sliced.columns else sliced

    @staticmethod
    def _validate_input_window(
        *,
        frame: pd.DataFrame,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
    ) -> dict[str, object]:
        if frame.empty or "timestamp" not in frame.columns:
            return {
                "requested_start": requested_start.isoformat(),
                "requested_end": requested_end.isoformat(),
                "actual_min_timestamp": None,
                "actual_max_timestamp": None,
                "row_count": 0,
                "future_row_count": 0,
            }
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        future_row_count = int((timestamps > requested_end).sum())
        if future_row_count > 0:
            raise RuntimeError(
                f"Future data detected for snapshot {requested_end.isoformat()}: {future_row_count} rows."
            )
        actual_max = timestamps.max()
        if pd.notna(actual_max) and actual_max > requested_end:
            raise RuntimeError(
                f"Input max timestamp {actual_max.isoformat()} exceeded snapshot {requested_end.isoformat()}."
            )
        return {
            "requested_start": requested_start.isoformat(),
            "requested_end": requested_end.isoformat(),
            "actual_min_timestamp": timestamps.min().isoformat() if not timestamps.empty and pd.notna(timestamps.min()) else None,
            "actual_max_timestamp": actual_max.isoformat() if pd.notna(actual_max) else None,
            "row_count": int(len(frame)),
            "future_row_count": future_row_count,
        }
