from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from hashlib import sha256
from concurrent.futures import Executor, Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Protocol

import duckdb

from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.snapshot_round_robin_graph_build_service import (
    SnapshotRoundRobinGraphBuildService,
)
from stocknetv2.application.services.temporal_edge_replay_service import TemporalEdgeReplayService
from stocknetv2.domain.graph.layer_config import build_theme_discovery_settings
from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import (
    LegacySourceLayout,
    MarketReadRepository,
    MonthPackSourceLayout,
)
from stocknetv2.infrastructure.repositories.snapshot_artifact_repository import SnapshotArtifactRepository
from stocknetv2.interfaces.cli.run_theme_discovery_t1 import run_theme_discovery

for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")


class MarketCalendarProtocol(Protocol):
    def list_available_trade_dates(self, dataset_name: str) -> list[str]: ...


@dataclass(frozen=True)
class GraphBuildRangeConfig:
    data_root: Path | str
    output_database_path: Path | str
    date_start: str
    date_end: str
    run_prefix: str
    config_id: str
    config_name: str
    config_version: str
    code_commit: str
    symbol_limit: int | None = None
    continue_on_error: bool = False
    shard_directory: Path | str | None = None
    keep_shards: bool = False
    layer_workers_per_process: int = 1
    graph_backend: str = "cpu_numpy"
    graph_torch_device: str = "auto"
    dtw_backend: str = "cpu_python"
    dtw_torch_device: str = "auto"
    dtw_torch_batch_pair_threshold: int = 1024
    execution_mode: str = "trade_date_shards"
    data_source_kind: str = "legacy_layout"


@dataclass(frozen=True)
class GraphBuildShardTask:
    trade_date: str
    run_id: str
    run_name: str
    database_path: Path
    data_root: Path
    symbol_limit: int | None
    config_id: str
    config_name: str
    config_version: str
    code_commit: str
    layer_workers: int
    graph_backend: str
    graph_torch_device: str
    dtw_backend: str
    dtw_torch_device: str
    dtw_torch_batch_pair_threshold: int
    data_source_kind: str
    live_progress_dir: Path | None = None


@dataclass(frozen=True)
class GraphBuildShardResult:
    trade_date: str
    run_id: str
    database_path: Path
    snapshot_count: int
    data_version: str
    elapsed_seconds: float


@dataclass(frozen=True)
class GraphBuildShardFailure:
    trade_date: str
    run_id: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class GraphBuildRangeSummary:
    processed_dates: list[str]
    shard_results: list[GraphBuildShardResult]
    failures: list[GraphBuildShardFailure]
    failure_count: int
    elapsed_seconds: float


class GraphBuildRangeService:
    def __init__(
        self,
        *,
        market_calendar: MarketCalendarProtocol,
        max_workers: int = 1,
        shard_runner: Callable[[GraphBuildShardTask], GraphBuildShardResult] | None = None,
        round_robin_runner: Callable[..., GraphBuildRangeSummary] | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
        snapshot_artifact_repository: SnapshotArtifactRepository | None = None,
    ) -> None:
        self._market_calendar = market_calendar
        self._max_workers = max(1, max_workers)
        self._shard_runner = shard_runner or _run_graph_build_shard
        self._round_robin_runner = round_robin_runner or _run_snapshot_round_robin_range
        self._executor_factory = executor_factory or _build_graph_range_executor
        self._snapshot_artifact_repository = snapshot_artifact_repository or SnapshotArtifactRepository()

    def run(
        self,
        config: GraphBuildRangeConfig,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> GraphBuildRangeSummary:
        started_at = time.perf_counter()
        trade_dates = [
            trade_date
            for trade_date in self._market_calendar.list_available_trade_dates("bars_5m")
            if config.date_start <= trade_date <= config.date_end
        ]
        if not trade_dates:
            raise RuntimeError("No available trade dates for the configured range.")

        output_database_path = Path(config.output_database_path).expanduser().resolve()
        output_database_path.parent.mkdir(parents=True, exist_ok=True)
        if config.execution_mode == "snapshot_round_robin":
            return self._round_robin_runner(
                config,
                trade_dates=trade_dates,
                progress_callback=progress_callback,
            )
        shard_directory, should_cleanup_shards = self._resolve_shard_directory(config, output_database_path)
        config_hash = _config_hash(
            {
                "data_root": str(Path(config.data_root).expanduser().resolve()),
                "output_database_path": str(output_database_path),
                "date_start": config.date_start,
                "date_end": config.date_end,
                "run_prefix": config.run_prefix,
                "config_id": config.config_id,
                "config_name": config.config_name,
                "config_version": config.config_version,
                "code_commit": config.code_commit,
                "symbol_limit": config.symbol_limit,
                "continue_on_error": config.continue_on_error,
                "keep_shards": config.keep_shards,
                "layer_workers_per_process": config.layer_workers_per_process,
                "graph_backend": config.graph_backend,
                "graph_torch_device": config.graph_torch_device,
                "dtw_backend": config.dtw_backend,
                "dtw_torch_device": config.dtw_torch_device,
                "dtw_torch_batch_pair_threshold": config.dtw_torch_batch_pair_threshold,
                "execution_mode": config.execution_mode,
                "data_source_kind": config.data_source_kind,
            }
        )
        tasks = [
            GraphBuildShardTask(
                trade_date=trade_date,
                run_id=f"{config.run_prefix}_{trade_date}",
                run_name=f"{config.run_prefix} {trade_date}",
                database_path=shard_directory / trade_date / "raw_graph.duckdb",
                data_root=Path(config.data_root).expanduser().resolve(),
                symbol_limit=config.symbol_limit,
                config_id=config.config_id,
                config_name=config.config_name,
                config_version=config.config_version,
                code_commit=config.code_commit,
                layer_workers=max(1, config.layer_workers_per_process),
                graph_backend=config.graph_backend,
                graph_torch_device=config.graph_torch_device,
                dtw_backend=config.dtw_backend,
                dtw_torch_device=config.dtw_torch_device,
                dtw_torch_batch_pair_threshold=max(1, config.dtw_torch_batch_pair_threshold),
                data_source_kind=config.data_source_kind,
                live_progress_dir=output_database_path.parent / "_live_progress",
            )
            for trade_date in trade_dates
        ]
        reusable_results: list[GraphBuildShardResult] = []
        pending_tasks: list[GraphBuildShardTask] = []
        for task in tasks:
            task.database_path.parent.mkdir(parents=True, exist_ok=True)
            existing_result = self._load_existing_shard_result(task.database_path.parent)
            if existing_result is not None:
                reusable_results.append(existing_result)
            else:
                pending_tasks.append(task)
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "range_started",
                    "date_start": config.date_start,
                    "date_end": config.date_end,
                    "total_dates": len(tasks),
                    "max_workers": self._max_workers,
                }
            )

        shard_results: list[GraphBuildShardResult] = list(reusable_results)
        failures: list[GraphBuildShardFailure] = []
        for completed_dates, result in enumerate(sorted(reusable_results, key=lambda row: row.trade_date), start=1):
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "shard_completed",
                        "trade_date": result.trade_date,
                        "run_id": result.run_id,
                        "snapshot_count": result.snapshot_count,
                        "elapsed_seconds": result.elapsed_seconds,
                        "completed_dates": completed_dates,
                        "total_dates": len(tasks),
                        "reused_existing": True,
                    }
                )

        try:
            if self._max_workers <= 1 or len(pending_tasks) <= 1:
                for task in pending_tasks:
                    try:
                        result = self._shard_runner(task)
                        shard_results.append(result)
                        self._snapshot_artifact_repository.write_completed_date_artifacts(
                            trade_date=task.trade_date,
                            date_root=task.database_path.parent,
                            raw_graph_path=result.database_path,
                            run_id=result.run_id,
                            snapshot_count=result.snapshot_count,
                            data_version=result.data_version,
                            elapsed_seconds=result.elapsed_seconds,
                            config_id=config.config_id,
                            config_version=config.config_version,
                            config_hash=config_hash,
                            code_commit=config.code_commit,
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_completed",
                                    "trade_date": result.trade_date,
                                    "run_id": result.run_id,
                                    "snapshot_count": result.snapshot_count,
                                    "elapsed_seconds": result.elapsed_seconds,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                        self._write_artifact_registry(output_database_path.parent / "date_artifact_registry.csv", shard_results=shard_results, failures=failures)
                    except Exception as exc:
                        failure = GraphBuildShardFailure(
                            trade_date=task.trade_date,
                            run_id=task.run_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        failures.append(failure)
                        self._snapshot_artifact_repository.write_failed_date_artifacts(
                            trade_date=task.trade_date,
                            date_root=task.database_path.parent,
                            run_id=failure.run_id,
                            error_type=failure.error_type,
                            error_message=failure.error_message,
                            config_id=config.config_id,
                            config_version=config.config_version,
                            config_hash=config_hash,
                            code_commit=config.code_commit,
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_failed",
                                    "trade_date": failure.trade_date,
                                    "run_id": failure.run_id,
                                    "error_type": failure.error_type,
                                    "error_message": failure.error_message,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                        self._write_artifact_registry(output_database_path.parent / "date_artifact_registry.csv", shard_results=shard_results, failures=failures)
                        if not config.continue_on_error:
                            raise
            else:
                executor = self._executor_factory(self._max_workers)
                futures: dict[Future, GraphBuildShardTask] = {
                    executor.submit(self._shard_runner, task): task
                    for task in pending_tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        shard_results.append(result)
                        self._snapshot_artifact_repository.write_completed_date_artifacts(
                            trade_date=task.trade_date,
                            date_root=task.database_path.parent,
                            raw_graph_path=result.database_path,
                            run_id=result.run_id,
                            snapshot_count=result.snapshot_count,
                            data_version=result.data_version,
                            elapsed_seconds=result.elapsed_seconds,
                            config_id=config.config_id,
                            config_version=config.config_version,
                            config_hash=config_hash,
                            code_commit=config.code_commit,
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_completed",
                                    "trade_date": result.trade_date,
                                    "run_id": result.run_id,
                                    "snapshot_count": result.snapshot_count,
                                    "elapsed_seconds": result.elapsed_seconds,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                        self._write_artifact_registry(output_database_path.parent / "date_artifact_registry.csv", shard_results=shard_results, failures=failures)
                    except Exception as exc:
                        failure = GraphBuildShardFailure(
                            trade_date=task.trade_date,
                            run_id=task.run_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        failures.append(failure)
                        self._snapshot_artifact_repository.write_failed_date_artifacts(
                            trade_date=task.trade_date,
                            date_root=task.database_path.parent,
                            run_id=failure.run_id,
                            error_type=failure.error_type,
                            error_message=failure.error_message,
                            config_id=config.config_id,
                            config_version=config.config_version,
                            config_hash=config_hash,
                            code_commit=config.code_commit,
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_failed",
                                    "trade_date": failure.trade_date,
                                    "run_id": failure.run_id,
                                    "error_type": failure.error_type,
                                    "error_message": failure.error_message,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                        self._write_artifact_registry(output_database_path.parent / "date_artifact_registry.csv", shard_results=shard_results, failures=failures)
                        if not config.continue_on_error:
                            raise
                if hasattr(executor, "shutdown"):
                    executor.shutdown(wait=True, cancel_futures=False)

            if shard_results:
                _merge_shard_databases(output_database_path, [result.database_path for result in shard_results])
            self._write_artifact_registry(
                output_database_path.parent / "date_artifact_registry.csv",
                shard_results=shard_results,
                failures=failures,
            )

            summary = GraphBuildRangeSummary(
                processed_dates=[result.trade_date for result in shard_results],
                shard_results=sorted(shard_results, key=lambda result: result.trade_date),
                failures=sorted(failures, key=lambda failure: failure.trade_date),
                failure_count=len(failures),
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
            if should_cleanup_shards and shard_directory.exists():
                shutil.rmtree(shard_directory, ignore_errors=True)

    @staticmethod
    def _resolve_shard_directory(
        config: GraphBuildRangeConfig,
        output_database_path: Path,
    ) -> tuple[Path, bool]:
        if config.shard_directory is not None:
            shard_directory = Path(config.shard_directory).expanduser().resolve()
            shard_directory.mkdir(parents=True, exist_ok=True)
            return shard_directory, False
        shard_directory = Path(
            tempfile.mkdtemp(
                prefix=f"{output_database_path.stem}_shards_",
                dir=str(output_database_path.parent),
            )
        )
        return shard_directory, not config.keep_shards

    def _load_existing_shard_result(self, date_root: Path) -> GraphBuildShardResult | None:
        payload = self._snapshot_artifact_repository.load_completed_date_artifact(date_root=date_root)
        if payload is None:
            return None
        return GraphBuildShardResult(
            trade_date=str(payload["trade_date"]),
            run_id=str(payload["run_id"]),
            database_path=Path(payload["database_path"]),
            snapshot_count=int(payload["snapshot_count"]),
            data_version=str(payload["data_version"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
        )

    def _write_artifact_registry(
        self,
        path: Path,
        *,
        shard_results: list[GraphBuildShardResult],
        failures: list[GraphBuildShardFailure],
    ) -> None:
        self._snapshot_artifact_repository.write_date_artifact_registry(
            path,
            completed_rows=[
                {
                    "trade_date": result.trade_date,
                    "run_id": result.run_id,
                    "database_path": result.database_path,
                    "snapshot_count": result.snapshot_count,
                    "data_version": result.data_version,
                    "elapsed_seconds": result.elapsed_seconds,
                }
                for result in shard_results
            ],
            failed_rows=[
                {
                    "trade_date": failure.trade_date,
                    "run_id": failure.run_id,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                }
                for failure in failures
            ],
        )


def _config_hash(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return sha256(normalized.encode("utf-8")).hexdigest()


def _run_graph_build_shard(task: GraphBuildShardTask) -> GraphBuildShardResult:
    task.database_path.parent.mkdir(parents=True, exist_ok=True)
    if task.database_path.exists():
        task.database_path.unlink()
    started_at = time.perf_counter()
    progress_writer = _build_shard_live_progress_writer(task)
    try:
        summary = run_theme_discovery(
            database_path=task.database_path,
            legacy_data_root=task.data_root if task.data_source_kind == "legacy_layout" else None,
            month_pack_root=task.data_root if task.data_source_kind == "month_pack" else None,
            symbol_limit=task.symbol_limit,
            graph_build_only=True,
            run_id=task.run_id,
            run_name=task.run_name,
            date_start=task.trade_date,
            date_end=task.trade_date,
            config_id=task.config_id,
            config_name=task.config_name,
            config_scope="t1",
            config_version=task.config_version,
            code_commit=task.code_commit,
            layer_workers=task.layer_workers,
            graph_backend=task.graph_backend,
            graph_torch_device=task.graph_torch_device,
            dtw_backend=task.dtw_backend,
            dtw_torch_device=task.dtw_torch_device,
            dtw_torch_batch_pair_threshold=task.dtw_torch_batch_pair_threshold,
            progress_callback=progress_writer,
        )
        progress_writer(
            {
                "status": "trade_date_completed",
                "trade_date": task.trade_date,
                "snapshot_index": summary.snapshot_count,
                "total_snapshots": summary.snapshot_count,
                "progress_percent": 100.0,
                "stage": "trade_date_completed",
            }
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=summary.snapshot_count,
            data_version=summary.data_version,
            elapsed_seconds=round(time.perf_counter() - started_at, 2),
        )
    except Exception as exc:
        progress_writer(
            {
                "status": "trade_date_failed",
                "trade_date": task.trade_date,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stage": "trade_date_failed",
            }
        )
        raise


def _merge_shard_databases(output_database_path: Path, shard_paths: list[Path]) -> None:
    if output_database_path.exists():
        output_database_path.unlink()
    wal_path = output_database_path.with_suffix(output_database_path.suffix + ".wal")
    if wal_path.exists():
        wal_path.unlink()

    connection = duckdb.connect(str(output_database_path))
    try:
        SchemaManager(connection).initialize()
        for index, shard_path in enumerate(sorted(shard_paths), start=1):
            alias = f"shard_{index}"
            escaped_path = str(shard_path).replace("\\", "/").replace("'", "''")
            connection.execute(f"ATTACH '{escaped_path}' AS {alias}")
            try:
                for table_name in (
                    "config_registry",
                    "theme_discovery_run",
                    "input_lineage",
                    "graph_snapshot",
                    "graph_edge_summary",
                    "graph_layer_diagnostic",
                    "relation_observation",
                    "graph_edges_thresholded",
                    "temporal_edge_state",
                    "layer_community",
                    "layer_community_membership",
                    "consensus_theme_candidate",
                    "theme_membership",
                    "theme_semantic_label",
                    "theme_path_lifecycle",
                    "theme_level_flow_series",
                    "frontend_snapshot_cache",
                ):
                    insert_prefix = "INSERT OR REPLACE" if table_name == "config_registry" else "INSERT"
                    connection.execute(
                        f"{insert_prefix} INTO {table_name} SELECT * FROM {alias}.{table_name}"
                    )
            finally:
                connection.execute(f"DETACH {alias}")
    finally:
        connection.close()


def _build_graph_range_executor(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    )


def _build_shard_live_progress_writer(task: GraphBuildShardTask) -> Callable[[dict[str, object]], None]:
    if task.live_progress_dir is None:
        return lambda event: None

    live_progress_dir = Path(task.live_progress_dir).expanduser().resolve()
    live_progress_dir.mkdir(parents=True, exist_ok=True)
    output_path = live_progress_dir / f"{task.trade_date}.json"

    def _write(event: dict[str, object]) -> None:
        payload = dict(event)
        payload.setdefault("trade_date", task.trade_date)
        payload.setdefault("run_id", task.run_id)
        payload["updated_at"] = datetime.now(UTC).isoformat()
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return _write


def _run_snapshot_round_robin_range(
    config: GraphBuildRangeConfig,
    *,
    trade_dates: list[str],
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> GraphBuildRangeSummary:
    output_database_path = Path(config.output_database_path).expanduser().resolve()
    output_database_path.parent.mkdir(parents=True, exist_ok=True)
    if output_database_path.exists():
        output_database_path.unlink()
    wal_path = output_database_path.with_suffix(output_database_path.suffix + ".wal")
    if wal_path.exists():
        wal_path.unlink()

    market_repository = MarketReadRepository(
        _build_market_source(
            data_root=Path(config.data_root).expanduser().resolve(),
            data_source_kind=config.data_source_kind,
        ),
        symbol_limit=config.symbol_limit,
    )
    discovery_settings = build_theme_discovery_settings(
        graph_backend=config.graph_backend,
        graph_torch_device=config.graph_torch_device,
        dtw_backend=config.dtw_backend,
        dtw_torch_device=config.dtw_torch_device,
        dtw_torch_batch_pair_threshold=config.dtw_torch_batch_pair_threshold,
    )

    connection = duckdb.connect(str(output_database_path))
    try:
        SchemaManager(connection).initialize()
        summary = SnapshotRoundRobinGraphBuildService(
            market_repository=market_repository,
            audit_repository=AuditRepository(connection),
            snapshot_clock=SnapshotClock(),
            layer_execution_service=LayerExecutionService(
                parallel_workers=max(1, config.layer_workers_per_process),
                settings=discovery_settings,
            ),
            graph_write_repository=GraphWriteRepository(connection),
            temporal_edge_replay_service=TemporalEdgeReplayService(),
        ).run(
            config,
            trade_dates=trade_dates,
            progress_callback=progress_callback,
        )
    finally:
        connection.close()

    return GraphBuildRangeSummary(
        processed_dates=list(summary.processed_dates),
        shard_results=[
            GraphBuildShardResult(
                trade_date=result.trade_date,
                run_id=result.run_id,
                database_path=result.database_path,
                snapshot_count=result.snapshot_count,
                data_version=result.data_version,
                elapsed_seconds=result.elapsed_seconds,
            )
            for result in summary.shard_results
        ],
        failures=[],
        failure_count=0,
        elapsed_seconds=summary.elapsed_seconds,
    )


def _build_market_source(*, data_root: Path, data_source_kind: str):
    if data_source_kind == "month_pack":
        return MonthPackSourceLayout(pack_root=data_root)
    return LegacySourceLayout(data_root=data_root)
