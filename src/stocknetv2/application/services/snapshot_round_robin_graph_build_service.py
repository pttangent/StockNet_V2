from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.temporal_edge_replay_service import (
    TemporalEdgeReplayService,
    TemporalEdgeState,
)
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


class MarketRepositoryProtocol(Protocol):
    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs: ...


class SnapshotClockProtocol(Protocol):
    def iter_trade_date(self, trade_date: str) -> list[pd.Timestamp]: ...

    def session_open_timestamp(self, trade_date: str) -> pd.Timestamp: ...


@dataclass(frozen=True)
class SnapshotRoundRobinShardResult:
    trade_date: str
    run_id: str
    database_path: Path
    snapshot_count: int
    data_version: str
    elapsed_seconds: float


@dataclass(frozen=True)
class SnapshotRoundRobinRangeSummary:
    processed_dates: list[str]
    shard_results: list[SnapshotRoundRobinShardResult]
    failures: list[object]
    failure_count: int
    elapsed_seconds: float


@dataclass
class _TradeDateState:
    trade_date: str
    run_id: str
    run_name: str
    inputs: TradeDateInputs
    snapshots: list[pd.Timestamp]
    session_open: pd.Timestamp
    completed_snapshot_ids: set[str]
    temporal_edge_states: dict[tuple[str, str, str], TemporalEdgeState]
    started_at: float


class SnapshotRoundRobinGraphBuildService:
    def __init__(
        self,
        *,
        market_repository: MarketRepositoryProtocol,
        audit_repository: AuditRepository,
        snapshot_clock: SnapshotClockProtocol,
        layer_execution_service: LayerExecutionService,
        graph_write_repository: GraphWriteRepository,
        temporal_edge_replay_service: TemporalEdgeReplayService | None = None,
    ) -> None:
        self._market_repository = market_repository
        self._audit_repository = audit_repository
        self._snapshot_clock = snapshot_clock
        self._layer_execution_service = layer_execution_service
        self._graph_write_repository = graph_write_repository
        self._temporal_edge_replay_service = temporal_edge_replay_service

    def run(
        self,
        config: Any,
        *,
        trade_dates: list[str],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> SnapshotRoundRobinRangeSummary:
        started_at = time.perf_counter()
        if not trade_dates:
            return SnapshotRoundRobinRangeSummary(
                processed_dates=[],
                shard_results=[],
                failures=[],
                failure_count=0,
                elapsed_seconds=0.0,
            )

        config_json = {
            "execution_mode": getattr(config, "execution_mode", "snapshot_round_robin"),
            "graph_backend": getattr(config, "graph_backend", "cpu_numpy"),
            "graph_torch_device": getattr(config, "graph_torch_device", "auto"),
            "dtw_backend": getattr(config, "dtw_backend", "cpu_python"),
            "dtw_torch_device": getattr(config, "dtw_torch_device", "auto"),
            "dtw_torch_batch_pair_threshold": getattr(config, "dtw_torch_batch_pair_threshold", 1024),
        }
        self._audit_repository.register_config(
            config_id=config.config_id,
            config_name=config.config_name,
            config_scope="t1",
            config_json=config_json,
            config_version=config.config_version,
        )

        if progress_callback is not None:
            progress_callback(
                {
                    "status": "range_started",
                    "date_start": config.date_start,
                    "date_end": config.date_end,
                    "total_dates": len(trade_dates),
                    "max_workers": 1,
                    "execution_mode": getattr(config, "execution_mode", "snapshot_round_robin"),
                }
            )

        snapshot_rows: list[dict[str, object]] = []
        lineage_records_by_run: dict[str, list[dict[str, object]]] = {}
        trade_date_states: list[_TradeDateState] = []

        for trade_date in trade_dates:
            inputs = self._market_repository.load_trade_date_inputs(trade_date)
            run_id = f"{config.run_prefix}_{trade_date}"
            run_name = f"{config.run_prefix} {trade_date}"
            self._audit_repository.create_run(
                run_id=run_id,
                run_name=run_name,
                date_start=trade_date,
                date_end=trade_date,
                frame_minutes=5,
                config_id=config.config_id,
                config_json=config_json,
                code_commit=config.code_commit,
                data_version=inputs.data_version,
            )
            lineage_records_by_run[run_id] = _build_lineage_records(inputs)
            snapshots = list(self._snapshot_clock.iter_trade_date(trade_date))
            completed_snapshot_ids = self._audit_repository.list_completed_snapshot_ids(
                run_id=run_id,
                trade_date=trade_date,
                expected_layer_count=6,
            )
            trade_date_states.append(
                _TradeDateState(
                    trade_date=trade_date,
                    run_id=run_id,
                    run_name=run_name,
                    inputs=inputs,
                    snapshots=snapshots,
                    session_open=self._snapshot_clock.session_open_timestamp(trade_date),
                    completed_snapshot_ids=completed_snapshot_ids,
                    temporal_edge_states={},
                    started_at=time.perf_counter(),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "trade_date_started",
                        "trade_date": trade_date,
                        "total_snapshots": len(snapshots),
                    }
                )

        max_snapshot_count = max((len(state.snapshots) for state in trade_date_states), default=0)
        shard_results: list[SnapshotRoundRobinShardResult] = []

        try:
            for snapshot_position in range(max_snapshot_count):
                for state in trade_date_states:
                    if snapshot_position >= len(state.snapshots):
                        continue
                    snapshot_time = state.snapshots[snapshot_position]
                    snapshot_index = snapshot_position + 1
                    snapshot_id = f"{state.run_id}_{state.trade_date}_{snapshot_time.strftime('%H%M')}"
                    available_minutes = int((snapshot_time - state.session_open).total_seconds() // 60)
                    snapshot_rows.append(
                        {
                            "snapshot_id": snapshot_id,
                            "run_id": state.run_id,
                            "trade_date": state.trade_date,
                            "timestamp": snapshot_time,
                            "frame_minutes": 5,
                            "market_session": "regular",
                            "graph_status": "pending_layers",
                            "available_minutes_since_open": available_minutes,
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "status": "snapshot_progress",
                                "trade_date": state.trade_date,
                                "snapshot_id": snapshot_id,
                                "snapshot_index": snapshot_index,
                                "total_snapshots": len(state.snapshots),
                                "snapshot_clock_code": snapshot_time.strftime("%H%M"),
                                "available_minutes_since_open": available_minutes,
                                "progress_percent": round(
                                    ((snapshot_index - 1) / max(1, len(state.snapshots))) * 100.0,
                                    4,
                                ),
                                "stage": "snapshot_started",
                            }
                        )
                    if snapshot_id in state.completed_snapshot_ids:
                        continue

                    layer_result = self._layer_execution_service.execute_for_snapshot(
                        inputs=state.inputs,
                        snapshot_time=snapshot_time,
                        session_open=state.session_open,
                    )
                    self._graph_write_repository.save_layer_outputs(
                        run_id=state.run_id,
                        snapshot_id=snapshot_id,
                        trade_date=state.trade_date,
                        snapshot_time=snapshot_time,
                        config_id=config.config_id,
                        layer_edges=layer_result.layer_edges,
                        layer_communities=layer_result.layer_communities,
                        universe_symbol_count=int(state.inputs.bars_5m["symbol"].astype(str).nunique())
                        if "symbol" in state.inputs.bars_5m.columns
                        else None,
                    )
                    if self._temporal_edge_replay_service is not None:
                        temporal_edge_states, state.temporal_edge_states = self._temporal_edge_replay_service.replay(
                            run_id=state.run_id,
                            snapshot_id=snapshot_id,
                            trade_date=state.trade_date,
                            timestamp=snapshot_time,
                            layer_edges=layer_result.layer_edges,
                            previous_states=state.temporal_edge_states,
                        )
                        self._graph_write_repository.save_temporal_edge_states(records=temporal_edge_states)

                    if progress_callback is not None:
                        progress_callback(
                            {
                                "status": "snapshot_progress",
                                "trade_date": state.trade_date,
                                "snapshot_id": snapshot_id,
                                "snapshot_index": snapshot_index,
                                "total_snapshots": len(state.snapshots),
                                "snapshot_clock_code": snapshot_time.strftime("%H%M"),
                                "available_minutes_since_open": available_minutes,
                                "progress_percent": round(
                                    (snapshot_index / max(1, len(state.snapshots))) * 100.0,
                                    4,
                                ),
                                "stage": "snapshot_completed",
                            }
                        )

            for state in trade_date_states:
                self._audit_repository.add_input_lineage(
                    run_id=state.run_id,
                    snapshot_id=None,
                    records=lineage_records_by_run.get(state.run_id, []),
                )
            self._audit_repository.create_snapshots(snapshot_rows)

            for index, state in enumerate(trade_date_states, start=1):
                self._audit_repository.complete_run(run_id=state.run_id, data_version=state.inputs.data_version)
                elapsed_seconds = round(time.perf_counter() - state.started_at, 2)
                shard_results.append(
                    SnapshotRoundRobinShardResult(
                        trade_date=state.trade_date,
                        run_id=state.run_id,
                        database_path=Path(config.output_database_path).expanduser().resolve(),
                        snapshot_count=len(state.snapshots),
                        data_version=state.inputs.data_version,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "shard_completed",
                            "trade_date": state.trade_date,
                            "run_id": state.run_id,
                            "snapshot_count": len(state.snapshots),
                            "elapsed_seconds": elapsed_seconds,
                            "completed_dates": index,
                            "total_dates": len(trade_date_states),
                        }
                    )

            summary = SnapshotRoundRobinRangeSummary(
                processed_dates=[result.trade_date for result in shard_results],
                shard_results=shard_results,
                failures=[],
                failure_count=0,
                elapsed_seconds=round(time.perf_counter() - started_at, 2),
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "range_completed",
                        "processed_dates": list(summary.processed_dates),
                        "failure_count": summary.failure_count,
                        "elapsed_seconds": summary.elapsed_seconds,
                    }
                )
            return summary
        finally:
            if hasattr(self._layer_execution_service, "close"):
                self._layer_execution_service.close()


def _build_lineage_records(inputs: TradeDateInputs) -> list[dict[str, object]]:
    return [
        {
            "source_kind": "dataset",
            "source_name": "bars_5m",
            "source_path": str(Path(f"bars_5m/date={inputs.trade_date}/bars_5m.parquet")),
            "source_version": inputs.trade_date,
            "source_min_timestamp": _min_timestamp(inputs.bars_5m),
            "source_max_timestamp": _max_timestamp(inputs.bars_5m),
        },
        {
            "source_kind": "dataset",
            "source_name": "trade_flow_1m",
            "source_path": str(Path(f"trade_flow_1m/date={inputs.trade_date}/trade_flow_1m.parquet")),
            "source_version": inputs.trade_date,
            "source_min_timestamp": _min_timestamp(inputs.trade_flow_1m),
            "source_max_timestamp": _max_timestamp(inputs.trade_flow_1m),
        },
        {
            "source_kind": "dataset",
            "source_name": "features_1m",
            "source_path": str(Path(f"features_1m/date={inputs.trade_date}/features_1m.parquet")),
            "source_version": inputs.trade_date,
            "source_min_timestamp": _min_timestamp(inputs.features_1m),
            "source_max_timestamp": _max_timestamp(inputs.features_1m),
        },
    ]


def _min_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "timestamp" not in frame.columns:
        return None
    return pd.Timestamp(frame["timestamp"].min())


def _max_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "timestamp" not in frame.columns:
        return None
    return pd.Timestamp(frame["timestamp"].max())
