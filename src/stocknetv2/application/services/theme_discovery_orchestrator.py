from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.application.services.consensus_service import ConsensusService
from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.lifecycle_service import LifecycleRecord, LifecycleService
from stocknetv2.application.services.read_model_service import ReadModelService
from stocknetv2.application.services.semantic_service import SemanticLabelRecord, SemanticService
from stocknetv2.application.services.temporal_edge_replay_service import (
    TemporalEdgeReplayService,
    TemporalEdgeState,
)
from stocknetv2.application.services.theme_flow_service import ThemeFlowService
from stocknetv2.application.services.theme_quality_service import ThemeQualityService
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs
from stocknetv2.infrastructure.repositories.read_model_repository import ReadModelRepository
from stocknetv2.infrastructure.repositories.theme_write_repository import ThemeWriteRepository


class MarketSourceProtocol(Protocol):
    def list_available_trade_dates(self, dataset_name: str) -> list[str]: ...

    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs: ...


@dataclass(frozen=True)
class ThemeDiscoveryRunConfig:
    run_id: str
    run_name: str
    date_start: str
    date_end: str
    config_id: str
    config_name: str
    config_scope: str
    config_version: str
    code_commit: str
    frame_minutes: int = 5
    graph_build_only: bool = False
    discovery_settings: dict[str, object] = field(default_factory=dict)

    def to_config_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ThemeDiscoveryRunSummary:
    run_id: str
    trade_dates_processed: list[str]
    snapshot_count: int
    data_version: str


class ThemeDiscoveryOrchestrator:
    """Drive the first T1 pipeline slice: source dates, run metadata, and snapshots."""

    def __init__(
        self,
        *,
        market_repository: MarketSourceProtocol,
        audit_repository: AuditRepository,
        snapshot_clock: SnapshotClock,
        layer_execution_service: LayerExecutionService | None = None,
        graph_write_repository: GraphWriteRepository | None = None,
        consensus_service: ConsensusService | None = None,
        theme_write_repository: ThemeWriteRepository | None = None,
        semantic_service: SemanticService | None = None,
        lifecycle_service: LifecycleService | None = None,
        temporal_edge_replay_service: TemporalEdgeReplayService | None = None,
        theme_quality_service: ThemeQualityService | None = None,
        theme_flow_service: ThemeFlowService | None = None,
        read_model_service: ReadModelService | None = None,
        read_model_repository: ReadModelRepository | None = None,
    ) -> None:
        self._market_repository = market_repository
        self._audit_repository = audit_repository
        self._snapshot_clock = snapshot_clock
        self._layer_execution_service = layer_execution_service
        self._graph_write_repository = graph_write_repository
        self._consensus_service = consensus_service
        self._theme_write_repository = theme_write_repository
        self._semantic_service = semantic_service
        self._lifecycle_service = lifecycle_service
        self._temporal_edge_replay_service = temporal_edge_replay_service
        self._theme_quality_service = theme_quality_service
        self._theme_flow_service = theme_flow_service
        self._read_model_service = read_model_service
        self._read_model_repository = read_model_repository

    def run(
        self,
        config: ThemeDiscoveryRunConfig,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ThemeDiscoveryRunSummary:
        try:
            trade_dates = self._select_trade_dates(config.date_start, config.date_end)
            if not trade_dates:
                raise RuntimeError("No available trade dates for the configured range.")

            first_inputs = self._market_repository.load_trade_date_inputs(trade_dates[0])
            config_json = config.to_config_json()
            self._audit_repository.register_config(
                config_id=config.config_id,
                config_name=config.config_name,
                config_scope=config.config_scope,
                config_json=config_json,
                config_version=config.config_version,
            )
            self._audit_repository.create_run(
                run_id=config.run_id,
                run_name=config.run_name,
                date_start=config.date_start,
                date_end=config.date_end,
                frame_minutes=config.frame_minutes,
                config_id=config.config_id,
                config_json=config_json,
                code_commit=config.code_commit,
                data_version=first_inputs.data_version,
            )

            snapshot_rows: list[dict[str, object]] = []
            lineage_records: list[dict[str, object]] = []
            last_data_version = first_inputs.data_version
            previous_candidates = []
            previous_lifecycle_records: dict[str, LifecycleRecord] = {}
            previous_temporal_edge_states: dict[tuple[str, str, str], TemporalEdgeState] = {}

            for trade_date in trade_dates:
                inputs = self._market_repository.load_trade_date_inputs(trade_date)
                last_data_version = inputs.data_version
                lineage_records.extend(self._build_lineage_records(inputs))
                session_open = self._snapshot_clock.session_open_timestamp(trade_date)
                snapshots = list(self._snapshot_clock.iter_trade_date(trade_date))
                completed_snapshot_ids = self._audit_repository.list_completed_snapshot_ids(
                    run_id=config.run_id,
                    trade_date=trade_date,
                    expected_layer_count=6,
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "trade_date_started",
                            "trade_date": trade_date,
                            "total_snapshots": len(snapshots),
                        }
                    )

                for snapshot_index, snapshot_time in enumerate(snapshots, start=1):
                    snapshot_id = f"{config.run_id}_{trade_date}_{snapshot_time.strftime('%H%M')}"
                    available_minutes = int((snapshot_time - session_open).total_seconds() // 60)
                    snapshot_rows.append(
                        {
                            "snapshot_id": snapshot_id,
                            "run_id": config.run_id,
                            "trade_date": trade_date,
                            "timestamp": snapshot_time,
                            "frame_minutes": config.frame_minutes,
                            "market_session": "regular",
                            "graph_status": "pending_layers",
                            "available_minutes_since_open": available_minutes,
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "status": "snapshot_progress",
                                "trade_date": trade_date,
                                "snapshot_id": snapshot_id,
                                "snapshot_index": snapshot_index,
                                "total_snapshots": len(snapshots),
                                "snapshot_clock_code": snapshot_time.strftime("%H%M"),
                                "available_minutes_since_open": available_minutes,
                                "progress_percent": round(
                                    ((snapshot_index - 1) / max(1, len(snapshots))) * 100.0,
                                    4,
                                ),
                                "stage": "snapshot_started",
                            }
                        )
                    if snapshot_id in completed_snapshot_ids:
                        continue
                    if self._layer_execution_service and self._graph_write_repository:
                        layer_result = self._layer_execution_service.execute_for_snapshot(
                            inputs=inputs,
                            snapshot_time=snapshot_time,
                            session_open=session_open,
                        )
                        self._graph_write_repository.save_layer_outputs(
                            run_id=config.run_id,
                            snapshot_id=snapshot_id,
                            trade_date=trade_date,
                            snapshot_time=snapshot_time,
                            config_id=config.config_id,
                            layer_edges=layer_result.layer_edges,
                            layer_communities=layer_result.layer_communities,
                            universe_symbol_count=int(inputs.bars_5m["symbol"].astype(str).nunique())
                            if "symbol" in inputs.bars_5m.columns
                            else None,
                        )
                        if self._temporal_edge_replay_service:
                            temporal_edge_states, previous_temporal_edge_states = self._temporal_edge_replay_service.replay(
                                run_id=config.run_id,
                                snapshot_id=snapshot_id,
                                trade_date=trade_date,
                                timestamp=snapshot_time,
                                layer_edges=layer_result.layer_edges,
                                previous_states=previous_temporal_edge_states,
                            )
                            self._graph_write_repository.save_temporal_edge_states(records=temporal_edge_states)
                        if config.graph_build_only:
                            continue
                        if self._consensus_service and self._theme_write_repository:
                            candidates = self._consensus_service.build_consensus_themes(
                                run_id=config.run_id,
                                snapshot_id=snapshot_id,
                                snapshot_time=snapshot_time,
                                layer_communities=layer_result.layer_communities,
                            )
                            lifecycle_records: list[LifecycleRecord] = []
                            if self._lifecycle_service:
                                candidates, lifecycle_records = self._lifecycle_service.assign_paths(
                                    candidates=candidates,
                                    previous_candidates=previous_candidates,
                                    previous_lifecycle_records=previous_lifecycle_records,
                                    timestamp=snapshot_time,
                                    frame_minutes=config.frame_minutes,
                                )
                            semantic_labels: list[SemanticLabelRecord] = []
                            if self._semantic_service:
                                semantic_labels = self._semantic_service.label_themes(candidates)
                            flow_records = []
                            if self._theme_flow_service:
                                flow_records = self._theme_flow_service.build_theme_flow_records(
                                    candidates=candidates,
                                    inputs=inputs,
                                    snapshot_time=snapshot_time,
                                )
                            if self._theme_quality_service:
                                candidates = self._theme_quality_service.score_themes(
                                    candidates,
                                    semantic_labels=semantic_labels,
                                    lifecycle_records=lifecycle_records,
                                )
                            self._theme_write_repository.save_consensus_themes(
                                run_id=config.run_id,
                                snapshot_id=snapshot_id,
                                trade_date=trade_date,
                                snapshot_time=snapshot_time,
                                candidates=candidates,
                            )
                            if semantic_labels:
                                self._theme_write_repository.save_semantic_labels(
                                    run_id=config.run_id,
                                    snapshot_id=snapshot_id,
                                    labels=semantic_labels,
                                )
                            if lifecycle_records:
                                self._theme_write_repository.save_lifecycle_records(
                                    run_id=config.run_id,
                                    snapshot_id=snapshot_id,
                                    records=lifecycle_records,
                                )
                            if flow_records:
                                self._theme_write_repository.save_theme_flow_records(
                                    run_id=config.run_id,
                                    snapshot_id=snapshot_id,
                                    records=flow_records,
                                )
                            if self._read_model_service and self._read_model_repository:
                                snapshot_caches = self._read_model_service.build_snapshot_caches(
                                    run_id=config.run_id,
                                    snapshot_id=snapshot_id,
                                    timestamp=snapshot_time,
                                    candidates=candidates,
                                    semantic_labels=semantic_labels,
                                    lifecycle_records=lifecycle_records,
                                )
                                self._read_model_repository.save_snapshot_caches(snapshot_caches)
                            previous_candidates = candidates
                            previous_lifecycle_records = {
                                record.theme_instance_id: record for record in lifecycle_records
                            } or previous_lifecycle_records
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "status": "snapshot_progress",
                                "trade_date": trade_date,
                                "snapshot_id": snapshot_id,
                                "snapshot_index": snapshot_index,
                                "total_snapshots": len(snapshots),
                                "snapshot_clock_code": snapshot_time.strftime("%H%M"),
                                "available_minutes_since_open": available_minutes,
                                "progress_percent": round(
                                    (snapshot_index / max(1, len(snapshots))) * 100.0,
                                    4,
                                ),
                                "stage": "snapshot_completed",
                            }
                        )

            self._audit_repository.add_input_lineage(run_id=config.run_id, snapshot_id=None, records=lineage_records)
            self._audit_repository.create_snapshots(snapshot_rows)
            self._audit_repository.complete_run(run_id=config.run_id, data_version=last_data_version)

            return ThemeDiscoveryRunSummary(
                run_id=config.run_id,
                trade_dates_processed=trade_dates,
                snapshot_count=len(snapshot_rows),
                data_version=last_data_version,
            )
        finally:
            if self._layer_execution_service and hasattr(self._layer_execution_service, "close"):
                self._layer_execution_service.close()

    def _select_trade_dates(self, date_start: str, date_end: str) -> list[str]:
        available_trade_dates = self._market_repository.list_available_trade_dates("bars_5m")
        return [trade_date for trade_date in available_trade_dates if date_start <= trade_date <= date_end]

    @staticmethod
    def _build_lineage_records(inputs: TradeDateInputs) -> list[dict[str, object]]:
        return [
            {
                "source_kind": "dataset",
                "source_name": "bars_5m",
                "source_path": str(Path(f"bars_5m/date={inputs.trade_date}/bars_5m.parquet")),
                "source_version": inputs.trade_date,
                "source_min_timestamp": ThemeDiscoveryOrchestrator._min_timestamp(inputs.bars_5m),
                "source_max_timestamp": ThemeDiscoveryOrchestrator._max_timestamp(inputs.bars_5m),
            },
            {
                "source_kind": "dataset",
                "source_name": "trade_flow_1m",
                "source_path": str(Path(f"trade_flow_1m/date={inputs.trade_date}/trade_flow_1m.parquet")),
                "source_version": inputs.trade_date,
                "source_min_timestamp": ThemeDiscoveryOrchestrator._min_timestamp(inputs.trade_flow_1m),
                "source_max_timestamp": ThemeDiscoveryOrchestrator._max_timestamp(inputs.trade_flow_1m),
            },
            {
                "source_kind": "dataset",
                "source_name": "features_1m",
                "source_path": str(Path(f"features_1m/date={inputs.trade_date}/features_1m.parquet")),
                "source_version": inputs.trade_date,
                "source_min_timestamp": ThemeDiscoveryOrchestrator._min_timestamp(inputs.features_1m),
                "source_max_timestamp": ThemeDiscoveryOrchestrator._max_timestamp(inputs.features_1m),
            },
        ]

    @staticmethod
    def _min_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
        if frame.empty or "timestamp" not in frame.columns:
            return None
        return pd.Timestamp(frame["timestamp"].min())

    @staticmethod
    def _max_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
        if frame.empty or "timestamp" not in frame.columns:
            return None
        return pd.Timestamp(frame["timestamp"].max())
