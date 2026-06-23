from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import duckdb

from stocknetv2.application.services.graph_build_range_service import (
    GraphBuildRangeConfig,
    GraphBuildRangeSummary,
    GraphBuildShardFailure,
    GraphBuildRangeService,
    GraphBuildShardResult,
)
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository


class _StubMarketCalendar:
    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        assert dataset_name == "bars_5m"
        return ["2025-01-02", "2025-01-03"]


def _create_shard_database(path: Path, *, run_id: str, trade_date: str, config_id: str) -> None:
    connection = duckdb.connect(str(path))
    SchemaManager(connection).initialize()
    audit_repository = AuditRepository(connection)
    audit_repository.register_config(
        config_id=config_id,
        config_name="Graph build test",
        config_scope="t1",
        config_json={"config_id": config_id},
        config_version="v1",
    )
    audit_repository.create_run(
        run_id=run_id,
        run_name=f"Run {trade_date}",
        date_start=trade_date,
        date_end=trade_date,
        frame_minutes=5,
        config_id=config_id,
        config_json={"config_id": config_id},
        code_commit="abc123",
        data_version=f"bars_5m:{trade_date}",
    )
    audit_repository.create_snapshots(
        [
            {
                "snapshot_id": f"{run_id}_{trade_date}_0930",
                "run_id": run_id,
                "trade_date": trade_date,
                "timestamp": f"{trade_date} 14:30:00",
                "frame_minutes": 5,
                "market_session": "regular",
                "graph_status": "complete",
                "available_minutes_since_open": 0,
            }
        ]
    )
    audit_repository.complete_run(run_id=run_id, data_version=f"bars_5m:{trade_date}")
    connection.execute(
        """
        INSERT INTO graph_layer_diagnostic (
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            active_node_count,
            edge_count,
            average_degree,
            degree_p50,
            degree_p95,
            max_degree,
            edge_score_p50,
            edge_score_p90,
            support_points_p50,
            support_points_p90,
            connected_component_count,
            largest_component_ratio,
            community_count,
            community_size_p50,
            community_size_p95,
            community_size_max,
            market_mode_member_ratio,
            community_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            f"{run_id}_{trade_date}_0930",
            trade_date,
            "return_corr_graph",
            2,
            1,
            1.0,
            1.0,
            1.0,
            1,
            0.8,
            0.8,
            8.0,
            8.0,
            1,
            1.0,
            1,
            2.0,
            2.0,
            2,
            1.0,
            "connected_components",
        ],
    )
    connection.close()


def test_graph_build_range_service_merges_day_shards_into_single_database(tmp_path):
    output_database = tmp_path / "month.duckdb"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()

    def worker(task):
        _create_shard_database(
            task.database_path,
            run_id=task.run_id,
            trade_date=task.trade_date,
            config_id=task.config_id,
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=1,
            data_version=f"bars_5m:{task.trade_date}",
            elapsed_seconds=0.1,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=worker,
        max_workers=1,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=output_database,
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        shard_directory=shard_dir,
        keep_shards=True,
    )

    summary = service.run(config)

    assert summary.processed_dates == ["2025-01-02", "2025-01-03"]
    connection = duckdb.connect(str(output_database))
    assert connection.execute("SELECT COUNT(*) FROM theme_discovery_run").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM graph_snapshot").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM graph_layer_diagnostic").fetchone()[0] == 2
    connection.close()


def test_graph_build_range_service_writes_nested_date_shard_directories(tmp_path):
    output_database = tmp_path / "catalog.duckdb"
    shard_dir = tmp_path / "dates"
    shard_dir.mkdir()

    def worker(task):
        _create_shard_database(
            task.database_path,
            run_id=task.run_id,
            trade_date=task.trade_date,
            config_id=task.config_id,
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=1,
            data_version=f"bars_5m:{task.trade_date}",
            elapsed_seconds=0.1,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=worker,
        max_workers=1,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=output_database,
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        shard_directory=shard_dir,
        keep_shards=True,
    )

    service.run(config)

    for trade_date in ["2025-01-02", "2025-01-03"]:
        date_root = shard_dir / trade_date
        assert (date_root / "raw_graph.duckdb").exists()
        assert (date_root / "status.json").exists()
        assert (date_root / "manifest.json").exists()
        assert (date_root / "diagnostics.json").exists()
        assert (date_root / "_SUCCESS").exists()


def test_graph_build_range_service_dispatches_dates_via_executor(tmp_path):
    submitted_trade_dates: list[str] = []

    class _InlineExecutor:
        def __enter__(self) -> _InlineExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, fn, *args, **kwargs):
            submitted_trade_dates.append(args[0].trade_date)
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    def worker(task):
        _create_shard_database(
            task.database_path,
            run_id=task.run_id,
            trade_date=task.trade_date,
            config_id=task.config_id,
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=1,
            data_version=f"bars_5m:{task.trade_date}",
            elapsed_seconds=0.1,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=worker,
        max_workers=2,
        executor_factory=lambda max_workers: _InlineExecutor(),
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=tmp_path / "month.duckdb",
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        shard_directory=tmp_path / "shards",
        keep_shards=True,
    )

    service.run(config)

    assert submitted_trade_dates == ["2025-01-02", "2025-01-03"]


def test_graph_build_range_service_returns_failure_records_when_shards_fail(tmp_path):
    def failing_worker(task):
        raise RuntimeError(f"boom:{task.trade_date}")

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=failing_worker,
        max_workers=1,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=tmp_path / "month.duckdb",
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        continue_on_error=True,
        keep_shards=True,
    )

    summary = service.run(config)

    assert summary.processed_dates == []
    assert summary.failure_count == 2
    assert summary.failures == [
        GraphBuildShardFailure(
            trade_date="2025-01-02",
            run_id="graph-build_2025-01-02",
            error_type="RuntimeError",
            error_message="boom:2025-01-02",
        ),
        GraphBuildShardFailure(
            trade_date="2025-01-03",
            run_id="graph-build_2025-01-03",
            error_type="RuntimeError",
            error_message="boom:2025-01-03",
        ),
    ]


def test_graph_build_range_service_emits_progress_events_for_completed_shards(tmp_path):
    events: list[dict[str, object]] = []

    def worker(task):
        _create_shard_database(
            task.database_path,
            run_id=task.run_id,
            trade_date=task.trade_date,
            config_id=task.config_id,
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=1,
            data_version=f"bars_5m:{task.trade_date}",
            elapsed_seconds=0.1,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=worker,
        max_workers=1,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=tmp_path / "month.duckdb",
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        shard_directory=tmp_path / "shards",
        keep_shards=True,
    )

    service.run(config, progress_callback=events.append)

    assert [event["status"] for event in events] == [
        "range_started",
        "shard_completed",
        "shard_completed",
        "range_completed",
    ]
    assert events[1]["trade_date"] == "2025-01-02"
    assert events[2]["trade_date"] == "2025-01-03"
    assert events[3]["processed_dates"] == ["2025-01-02", "2025-01-03"]


def test_graph_build_range_service_can_delegate_to_snapshot_round_robin_mode(tmp_path):
    delegated: dict[str, object] = {}

    def round_robin_runner(config, *, trade_dates, progress_callback):
        delegated["execution_mode"] = config.execution_mode
        delegated["trade_dates"] = list(trade_dates)
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "range_started",
                    "total_dates": len(trade_dates),
                    "max_workers": 1,
                }
            )
            progress_callback(
                {
                    "status": "range_completed",
                    "processed_dates": trade_dates,
                    "failure_count": 0,
                    "elapsed_seconds": 0.5,
                }
            )
        return GraphBuildRangeSummary(
            processed_dates=list(trade_dates),
            shard_results=[],
            failures=[],
            failure_count=0,
            elapsed_seconds=0.5,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        max_workers=24,
        round_robin_runner=round_robin_runner,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=tmp_path / "month.duckdb",
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        execution_mode="snapshot_round_robin",
    )

    events: list[dict[str, object]] = []
    summary = service.run(config, progress_callback=events.append)

    assert delegated == {
        "execution_mode": "snapshot_round_robin",
        "trade_dates": ["2025-01-02", "2025-01-03"],
    }
    assert summary.processed_dates == ["2025-01-02", "2025-01-03"]
    assert [event["status"] for event in events] == ["range_started", "range_completed"]


def test_graph_build_range_service_reuses_completed_date_shards_and_skips_worker(tmp_path):
    output_database = tmp_path / "graph.duckdb"
    shard_dir = tmp_path / "dates"
    existing_date_root = shard_dir / "2025-01-02"
    existing_date_root.mkdir(parents=True, exist_ok=True)
    existing_db_path = existing_date_root / "raw_graph.duckdb"
    _create_shard_database(
        existing_db_path,
        run_id="graph-build_2025-01-02",
        trade_date="2025-01-02",
        config_id="graph-build-config",
    )
    (existing_date_root / "status.json").write_text(
        '{"trade_date":"2025-01-02","raw_graph_status":"complete","snapshot_count":1,"data_version":"bars_5m:2025-01-02","elapsed_seconds":0.1,"run_id":"graph-build_2025-01-02"}',
        encoding="utf-8",
    )

    executed_trade_dates: list[str] = []

    def worker(task):
        executed_trade_dates.append(task.trade_date)
        _create_shard_database(
            task.database_path,
            run_id=task.run_id,
            trade_date=task.trade_date,
            config_id=task.config_id,
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=1,
            data_version=f"bars_5m:{task.trade_date}",
            elapsed_seconds=0.1,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=worker,
        max_workers=1,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=output_database,
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        shard_directory=shard_dir,
        keep_shards=True,
    )

    summary = service.run(config)

    assert executed_trade_dates == ["2025-01-03"]
    assert summary.processed_dates == ["2025-01-02", "2025-01-03"]
    connection = duckdb.connect(str(output_database))
    assert connection.execute("SELECT COUNT(*) FROM theme_discovery_run").fetchone()[0] == 2
    connection.close()


def test_graph_build_range_service_writes_registry_csv_for_success_and_failure(tmp_path):
    shard_dir = tmp_path / "dates"
    output_database = tmp_path / "graph.duckdb"

    def worker(task):
        if task.trade_date == "2025-01-02":
            raise RuntimeError("broken date")
        _create_shard_database(
            task.database_path,
            run_id=task.run_id,
            trade_date=task.trade_date,
            config_id=task.config_id,
        )
        return GraphBuildShardResult(
            trade_date=task.trade_date,
            run_id=task.run_id,
            database_path=task.database_path,
            snapshot_count=1,
            data_version=f"bars_5m:{task.trade_date}",
            elapsed_seconds=0.1,
        )

    service = GraphBuildRangeService(
        market_calendar=_StubMarketCalendar(),
        shard_runner=worker,
        max_workers=1,
    )
    config = GraphBuildRangeConfig(
        data_root=tmp_path,
        output_database_path=output_database,
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        shard_directory=shard_dir,
        keep_shards=True,
        continue_on_error=True,
    )

    service.run(config)

    registry_path = output_database.parent / "date_artifact_registry.csv"
    assert registry_path.exists()
    registry = registry_path.read_text(encoding="utf-8")
    assert "2025-01-02,failed,graph-build_2025-01-02" in registry
    assert "2025-01-03,complete,graph-build_2025-01-03" in registry
