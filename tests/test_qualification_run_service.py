from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from stocknetv2.application.services.graph_build_range_service import (
    GraphBuildRangeConfig,
    GraphBuildRangeSummary,
    GraphBuildShardResult,
)
from stocknetv2.application.services.graph_evaluation_pack_service import (
    GraphEvaluationPackConfig,
    GraphEvaluationPackSummary,
)
from stocknetv2.application.services.qualification_run_service import (
    QualificationRunConfig,
    QualificationRunService,
    QualificationWindow,
)
from stocknetv2.infrastructure.db.schema_manager import SchemaManager


class _StubMarketCalendar:
    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        assert dataset_name == "bars_5m"
        return [
            "2025-01-02",
            "2025-01-03",
            "2025-02-03",
            "2025-02-04",
        ]


def _write_alpha_ranking(path: Path, *, score: float, sample_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "graph_layer": "volume_expansion_graph",
                "layer_role": "theme_candidate_layer",
                "factor_name": "community_quality_score",
                "label_horizon": "15m",
                "label_variant": "equal_weight",
                "sample_size": sample_size,
                "rank_ic": 0.08 if score >= 0 else -0.08,
                "top_bottom_spread": 0.02 if score >= 0 else -0.02,
                "top_decile_hit_rate": 0.62,
                "score": score,
                "confidence_bucket": "usable" if sample_size >= 3000 else "watch",
                "research_action": "keep_for_next_round" if score >= 0 else "downgrade",
            },
            {
                "graph_layer": "flow_alignment_graph",
                "layer_role": "event_alignment_layer",
                "factor_name": "community_member_count",
                "label_horizon": "30m",
                "label_variant": "equal_weight",
                "sample_size": sample_size,
                "rank_ic": 0.05 if score >= 0 else -0.05,
                "top_bottom_spread": 0.01 if score >= 0 else -0.01,
                "top_decile_hit_rate": 0.58,
                "score": score / 2.0,
                "confidence_bucket": "usable" if sample_size >= 3000 else "watch",
                "research_action": "keep_for_next_round" if score >= 0 else "downgrade",
            },
        ]
    ).to_csv(path, index=False)


def _write_benchmark_summary(path: Path, *, trade_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "benchmark_label_source": "labels_1m",
                "benchmark_proxy_price_method": "",
                "row_count": 2,
                "symbol_count": 2,
                "excess_ret_1m_coverage_ratio": 1.0,
            }
        ]
    ).to_csv(path, index=False)


def test_qualification_run_service_writes_monthly_outputs_and_progress(tmp_path):
    published_windows: list[str] = []

    def graph_range_runner(
        config: GraphBuildRangeConfig,
        *,
        progress_callback,
    ) -> GraphBuildRangeSummary:
        trade_dates = [
            trade_date
            for trade_date in _StubMarketCalendar().list_available_trade_dates("bars_5m")
            if config.date_start <= trade_date <= config.date_end
        ]
        progress_callback(
            {
                "status": "range_started",
                "date_start": config.date_start,
                "date_end": config.date_end,
                "total_dates": len(trade_dates),
                "max_workers": 2,
            }
        )
        shard_results: list[GraphBuildShardResult] = []
        for trade_date in trade_dates:
            progress_callback(
                {
                    "status": "snapshot_progress",
                    "trade_date": trade_date,
                    "snapshot_id": f"{config.run_prefix}_{trade_date}_{trade_date}_1435",
                    "snapshot_index": 1,
                    "total_snapshots": 78,
                    "snapshot_clock_code": "1435",
                    "available_minutes_since_open": 5,
                    "progress_percent": round(1 / 78 * 100, 4),
                }
            )
            graph_db_path = (
                Path(config.output_database_path).parent
                / f"{trade_date}.duckdb"
            )
            graph_db_path.parent.mkdir(parents=True, exist_ok=True)
            graph_db_path.write_text("stub-db", encoding="utf-8")
            result = GraphBuildShardResult(
                trade_date=trade_date,
                run_id=f"{config.run_prefix}_{trade_date}",
                database_path=graph_db_path,
                snapshot_count=78,
                data_version=f"bars_5m:{trade_date}",
                elapsed_seconds=0.25,
            )
            shard_results.append(result)
            progress_callback(
                {
                    "status": "shard_completed",
                    "trade_date": trade_date,
                    "run_id": result.run_id,
                    "snapshot_count": result.snapshot_count,
                    "elapsed_seconds": result.elapsed_seconds,
                    "completed_dates": len(shard_results),
                    "total_dates": len(trade_dates),
                }
            )
        merged_db = Path(config.output_database_path)
        merged_db.parent.mkdir(parents=True, exist_ok=True)
        merged_db.write_text("merged-db", encoding="utf-8")
        progress_callback(
            {
                "status": "range_completed",
                "processed_dates": trade_dates,
                "failure_count": 0,
                "elapsed_seconds": 1.0,
            }
        )
        return GraphBuildRangeSummary(
            processed_dates=trade_dates,
            shard_results=shard_results,
            failures=[],
            failure_count=0,
            elapsed_seconds=1.0,
        )

    def evaluation_pack_builder(
        config: GraphEvaluationPackConfig,
        *,
        log,
    ) -> GraphEvaluationPackSummary:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        market_dir = output_dir / "market"
        market_dir.mkdir(parents=True, exist_ok=True)
        if config.date_start.startswith("2025-01"):
            _write_alpha_ranking(
                market_dir / "alpha_feature_ranking_by_layer.csv",
                score=0.41,
                sample_size=4200,
            )
        else:
            _write_alpha_ranking(
                market_dir / "alpha_feature_ranking_by_layer.csv",
                score=0.37,
                sample_size=4600,
            )
        _write_benchmark_summary(
            market_dir / "benchmark_label_source_summary.csv",
            trade_date=config.date_start,
        )
        log(f"built pack for {config.date_start}")
        return GraphEvaluationPackSummary(
            output_dir=output_dir,
            artifact_paths={
                "alpha_feature_ranking_by_layer": market_dir / "alpha_feature_ranking_by_layer.csv",
                "benchmark_label_source_summary": market_dir / "benchmark_label_source_summary.csv",
            },
            counts={"edge_rows": 12, "community_rows": 4},
        )

    market_db = tmp_path / "market.duckdb"
    metadata_csv = tmp_path / "symbol_metadata.csv"
    market_db.write_text("stub-market-db", encoding="utf-8")
    metadata_csv.write_text("symbol,sector\nAAA,Technology\n", encoding="utf-8")

    runner = QualificationRunService(
        market_calendar_factory=lambda data_root: _StubMarketCalendar(),
        graph_range_runner=graph_range_runner,
        evaluation_pack_builder=evaluation_pack_builder,
        checkpoint_publisher=lambda context: published_windows.append(context.window.window_id),
    )

    summary = runner.run(
        QualificationRunConfig(
            data_root=tmp_path / "data",
            market_db_path=market_db,
            metadata_csv_path=metadata_csv,
            output_root=tmp_path / "qualification_run",
            run_label="2025 Q1 qualification",
            config_version="v1",
            code_commit="abc123",
            max_date_workers=2,
            layer_workers_per_process=2,
            git_push_enabled=True,
        ),
        windows=[
            QualificationWindow(window_id="2025-01", date_start="2025-01-01", date_end="2025-01-31"),
            QualificationWindow(window_id="2025-02", date_start="2025-02-01", date_end="2025-02-28"),
        ],
    )

    assert published_windows == ["2025-01", "2025-02"]
    assert summary.completed_windows == 2
    assert summary.completed_trade_dates == 4

    progress_payload = json.loads((tmp_path / "qualification_run" / "progress.json").read_text(encoding="utf-8"))
    assert progress_payload["status"] == "complete"
    assert progress_payload["completed_windows"] == 2
    assert progress_payload["completed_trade_dates"] == 4
    assert progress_payload["bars_5m_timestamp_semantics"] == "bar_close_time"
    assert progress_payload["dtw_backend"] == "torch_cuda"
    assert progress_payload["dtw_torch_device"] == "cuda"
    assert progress_payload["dtw_torch_batch_pair_threshold"] == 1024
    assert progress_payload["windows"][0]["trade_dates"][0]["snapshot_index"] == 78
    assert progress_payload["windows"][0]["trade_dates"][0]["snapshot_clock_code"] == "1435"
    assert any(
        artifact["path"].endswith("cross_month_alpha_comparison.csv")
        for artifact in progress_payload["recent_artifacts"]
    )
    assert any(
        artifact["path"].endswith("cross_month_layer_stability.csv")
        for artifact in progress_payload["recent_artifacts"]
    )

    monthly_run_status = pd.read_csv(tmp_path / "qualification_run" / "monthly_run_status.csv")
    assert monthly_run_status["window_id"].tolist() == ["2025-01", "2025-02"]
    assert monthly_run_status["status"].tolist() == ["completed", "completed"]
    assert monthly_run_status["processed_trade_dates"].tolist() == [2, 2]

    monthly_alpha_summary = pd.read_csv(tmp_path / "qualification_run" / "monthly_alpha_summary.csv")
    assert set(monthly_alpha_summary["window_id"].tolist()) == {"2025-01", "2025-02"}
    assert "top_factor_name" in monthly_alpha_summary.columns

    benchmark_summary = pd.read_csv(tmp_path / "qualification_run" / "benchmark_label_source_summary.csv")
    assert set(benchmark_summary["window_id"].tolist()) == {"2025-01", "2025-02"}
    assert set(benchmark_summary["benchmark_label_source"].tolist()) == {"labels_1m"}

    cross_month = pd.read_csv(tmp_path / "qualification_run" / "cross_month_alpha_comparison.csv")
    assert cross_month.loc[0, "first_window_id"] == "2025-01"
    assert cross_month.loc[0, "second_window_id"] == "2025-02"

    layer_stability = pd.read_csv(tmp_path / "qualification_run" / "cross_month_layer_stability.csv")
    assert set(layer_stability["graph_layer"]) == {
        "flow_alignment_graph",
        "volume_expansion_graph",
    }
    assert set(layer_stability["stability_bucket"]) == {"seed_window", "stable_positive"}

    run_context = json.loads((tmp_path / "qualification_run" / "qualification_config.json").read_text(encoding="utf-8"))
    assert run_context["benchmark_symbols"] == ["SPY", "QQQ", "IWM", "DIA"]
    assert run_context["dtw_backend"] == "torch_cuda"


def test_qualification_run_service_writes_temporal_outputs(tmp_path):
    def graph_range_runner(
        config: GraphBuildRangeConfig,
        *,
        progress_callback,
    ) -> GraphBuildRangeSummary:
        trade_dates = ["2025-01-02"]
        progress_callback(
            {
                "status": "range_started",
                "date_start": config.date_start,
                "date_end": config.date_end,
                "total_dates": 1,
                "max_workers": 1,
            }
        )
        trade_date = trade_dates[0]
        date_root = Path(config.output_database_path).parent / "dates" / trade_date
        date_root.mkdir(parents=True, exist_ok=True)
        raw_graph_path = date_root / "raw_graph.duckdb"
        connection = duckdb.connect(str(raw_graph_path))
        SchemaManager(connection).initialize()
        connection.execute(
            """
            INSERT INTO graph_edges_thresholded (
                run_id, snapshot_id, trade_date, timestamp, graph_layer,
                source_symbol, target_symbol, edge_type, weight, raw_score,
                edge_confidence, effective_lookback_minutes, window_start, window_end, support_points, config_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{config.run_prefix}_{trade_date}",
                f"{config.run_prefix}_{trade_date}_0935",
                trade_date,
                pd.Timestamp(f"{trade_date} 14:35:00"),
                "return_corr_graph",
                "AAA",
                "BBB",
                "relation",
                0.5,
                0.5,
                1.0,
                5,
                None,
                pd.Timestamp(f"{trade_date} 14:35:00"),
                8,
                config.config_id,
            ],
        )
        connection.close()
        result = GraphBuildShardResult(
            trade_date=trade_date,
            run_id=f"{config.run_prefix}_{trade_date}",
            database_path=raw_graph_path,
            snapshot_count=78,
            data_version=f"bars_5m:{trade_date}",
            elapsed_seconds=0.25,
        )
        progress_callback(
            {
                "status": "shard_completed",
                "trade_date": trade_date,
                "run_id": result.run_id,
                "snapshot_count": result.snapshot_count,
                "elapsed_seconds": result.elapsed_seconds,
                "completed_dates": 1,
                "total_dates": 1,
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
            processed_dates=trade_dates,
            shard_results=[result],
            failures=[],
            failure_count=0,
            elapsed_seconds=0.5,
        )

    def evaluation_pack_builder(
        config: GraphEvaluationPackConfig,
        *,
        log,
    ) -> GraphEvaluationPackSummary:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return GraphEvaluationPackSummary(output_dir=output_dir, artifact_paths={}, counts={})

    runner = QualificationRunService(
        market_calendar_factory=lambda data_root: _StubMarketCalendar(),
        graph_range_runner=graph_range_runner,
        evaluation_pack_builder=evaluation_pack_builder,
    )

    summary = runner.run(
        QualificationRunConfig(
            data_root=tmp_path / "data",
            market_db_path=tmp_path / "market.duckdb",
            metadata_csv_path=tmp_path / "symbol_metadata.csv",
            output_root=tmp_path / "qualification_run",
            run_label="2025 Q1 qualification",
            config_version="v1",
            code_commit="abc123",
            max_date_workers=1,
            layer_workers_per_process=1,
            keep_shards=True,
        ),
        windows=[
            QualificationWindow(window_id="2025", date_start="2025-01-01", date_end="2025-01-31"),
        ],
    )

    assert summary.completed_windows == 1
    assert (tmp_path / "qualification_run" / "temporal" / "date=2025-01-02" / "temporal_edges.parquet").exists()
    assert (tmp_path / "qualification_run" / "temporal" / "checkpoints" / "2025-01-02_terminal_state.parquet").exists()


def test_qualification_run_service_uses_root_level_outputs_for_single_window(tmp_path):
    captured_graph_paths: list[Path] = []
    captured_pack_dirs: list[Path] = []

    def graph_range_runner(
        config: GraphBuildRangeConfig,
        *,
        progress_callback,
    ) -> GraphBuildRangeSummary:
        captured_graph_paths.append(Path(config.output_database_path))
        date_root = Path(config.shard_directory) / "2025-01-02"
        date_root.mkdir(parents=True, exist_ok=True)
        raw_graph_path = date_root / "raw_graph.duckdb"
        connection = duckdb.connect(str(raw_graph_path))
        SchemaManager(connection).initialize()
        connection.close()
        graph_db_path = Path(config.output_database_path)
        graph_db_path.parent.mkdir(parents=True, exist_ok=True)
        graph_db_path.write_text("merged-db", encoding="utf-8")
        progress_callback({"status": "range_started", "total_dates": 1, "max_workers": 1})
        progress_callback({"status": "shard_completed", "trade_date": "2025-01-02", "completed_dates": 1, "total_dates": 1})
        progress_callback({"status": "range_completed", "processed_dates": ["2025-01-02"], "failure_count": 0, "elapsed_seconds": 0.5})
        return GraphBuildRangeSummary(
            processed_dates=["2025-01-02"],
            shard_results=[
                GraphBuildShardResult(
                    trade_date="2025-01-02",
                    run_id="qualification-graph-build_2025-01-02",
                    database_path=raw_graph_path,
                    snapshot_count=78,
                    data_version="bars_5m:2025-01-02",
                    elapsed_seconds=0.5,
                )
            ],
            failures=[],
            failure_count=0,
            elapsed_seconds=0.5,
        )

    def evaluation_pack_builder(
        config: GraphEvaluationPackConfig,
        *,
        log,
    ) -> GraphEvaluationPackSummary:
        captured_pack_dirs.append(Path(config.output_dir))
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return GraphEvaluationPackSummary(output_dir=output_dir, artifact_paths={}, counts={})

    runner = QualificationRunService(
        market_calendar_factory=lambda data_root: _StubMarketCalendar(),
        graph_range_runner=graph_range_runner,
        evaluation_pack_builder=evaluation_pack_builder,
    )

    runner.run(
        QualificationRunConfig(
            data_root=tmp_path / "data",
            market_db_path=tmp_path / "market.duckdb",
            metadata_csv_path=tmp_path / "symbol_metadata.csv",
            output_root=tmp_path / "qualification_run",
            run_label="single-window qualification",
            config_version="v1",
            code_commit="abc123",
            keep_shards=True,
        ),
        windows=[QualificationWindow(window_id="2025-01-02_to_2025-01-02", date_start="2025-01-02", date_end="2025-01-02")],
    )

    assert captured_graph_paths == [tmp_path / "qualification_run" / "graph.duckdb"]
    assert captured_pack_dirs == [tmp_path / "qualification_run" / "evaluation_pack"]
    assert (tmp_path / "qualification_run" / "dates" / "2025-01-02").exists()
