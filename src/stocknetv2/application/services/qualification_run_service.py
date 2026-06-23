from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stocknetv2.application.services.graph_build_range_service import (
    GraphBuildRangeConfig,
    GraphBuildRangeService,
    GraphBuildRangeSummary,
)
from stocknetv2.application.services.graph_evaluation_pack_service import (
    GraphEvaluationPackConfig,
    GraphEvaluationPackSummary,
    _export_cross_window_alpha_comparison_report,
    build_graph_evaluation_pack,
)
from stocknetv2.application.services.temporal_replay_range_service import TemporalReplayRangeService
from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository


@dataclass(frozen=True)
class QualificationWindow:
    window_id: str
    date_start: str
    date_end: str


@dataclass(frozen=True)
class QualificationRunConfig:
    data_root: Path | str
    market_db_path: Path | str
    metadata_csv_path: Path | str
    output_root: Path | str
    run_label: str
    config_version: str
    code_commit: str
    run_prefix: str = "qualification-graph-build"
    config_id: str = "three-month-qualification"
    config_name: str = "Long-horizon qualification run"
    symbol_limit: int | None = None
    max_date_workers: int = 24
    layer_workers_per_process: int = 1
    keep_shards: bool = False
    continue_on_error: bool = False
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    bars_5m_timestamp_semantics: str = "bar_close_time"
    graph_backend: str = "torch_cuda"
    graph_torch_device: str = "cuda"
    dtw_backend: str = "torch_cuda"
    dtw_torch_device: str = "cuda"
    dtw_torch_batch_pair_threshold: int = 1024
    graph_build_execution_mode: str = "trade_date_shards"
    data_source_kind: str = "legacy_layout"
    git_push_enabled: bool = True
    git_remote: str = "origin"
    git_branch: str | None = None


@dataclass(frozen=True)
class QualificationWindowResult:
    window: QualificationWindow
    status: str
    processed_trade_dates: int
    total_trade_dates: int
    failure_count: int
    graph_db_path: Path
    evaluation_pack_dir: Path | None
    alpha_ranking_path: Path | None
    benchmark_summary_path: Path | None
    elapsed_seconds: float


@dataclass(frozen=True)
class QualificationCheckpointContext:
    config: QualificationRunConfig
    window: QualificationWindow
    window_result: QualificationWindowResult
    output_root: Path
    progress_path: Path
    log_path: Path


@dataclass(frozen=True)
class QualificationRunSummary:
    output_root: Path
    completed_windows: int
    completed_trade_dates: int
    window_results: list[QualificationWindowResult]


class QualificationRunService:
    def __init__(
        self,
        *,
        market_calendar_factory: Callable[[Path], Any] | None = None,
        graph_range_runner: Callable[..., GraphBuildRangeSummary] | None = None,
        evaluation_pack_builder: Callable[..., GraphEvaluationPackSummary] | None = None,
        checkpoint_publisher: Callable[[QualificationCheckpointContext], None] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._market_calendar_factory = market_calendar_factory or (
            lambda data_root: MarketReadRepository(LegacySourceLayout(data_root=data_root))
        )
        self._graph_range_runner = graph_range_runner
        self._evaluation_pack_builder = evaluation_pack_builder or build_graph_evaluation_pack
        self._checkpoint_publisher = checkpoint_publisher or _git_publish_checkpoint
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def run(
        self,
        config: QualificationRunConfig,
        *,
        windows: list[QualificationWindow],
    ) -> QualificationRunSummary:
        data_root = Path(config.data_root).expanduser().resolve()
        market_db_path = Path(config.market_db_path).expanduser().resolve()
        metadata_csv_path = Path(config.metadata_csv_path).expanduser().resolve()
        output_root = Path(config.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        progress_path = output_root / "progress.json"
        log_path = output_root / "run.log"
        qualification_config_path = output_root / "qualification_config.json"

        market_calendar = self._market_calendar_factory(data_root)
        available_trade_dates = market_calendar.list_available_trade_dates("bars_5m")
        window_trade_dates = {
            window.window_id: [
                trade_date
                for trade_date in available_trade_dates
                if window.date_start <= trade_date <= window.date_end
            ]
            for window in windows
        }
        total_trade_dates = sum(len(trade_dates) for trade_dates in window_trade_dates.values())
        window_rows = [
            {
                "window_id": window.window_id,
                "date_start": window.date_start,
                "date_end": window.date_end,
                "status": "pending",
                "total_trade_dates": len(window_trade_dates[window.window_id]),
                "completed_trade_dates": 0,
                "failure_count": 0,
                "graph_db_path": "",
                "evaluation_pack_dir": "",
                "trade_dates": [
                    {
                        "trade_date": trade_date,
                        "status": "pending",
                        "snapshot_index": 0,
                        "total_snapshots": 78,
                        "snapshot_clock_code": None,
                        "available_minutes_since_open": None,
                        "progress_percent": 0.0,
                    }
                    for trade_date in window_trade_dates[window.window_id]
                ],
            }
            for window in windows
        ]
        progress_state: dict[str, Any] = {
            "status": "running",
            "run_label": config.run_label,
            "total_windows": len(windows),
            "completed_windows": 0,
            "total_trade_dates": total_trade_dates,
            "completed_trade_dates": 0,
            "current_window_id": None,
            "current_trade_date": None,
            "current_snapshot_id": None,
            "current_snapshot_clock_code": None,
            "current_stage": "initializing",
            "updated_at": None,
            "bars_5m_timestamp_semantics": config.bars_5m_timestamp_semantics,
            "benchmark_symbols": list(config.benchmark_symbols),
            "graph_backend": config.graph_backend,
            "graph_torch_device": config.graph_torch_device,
            "dtw_backend": config.dtw_backend,
            "dtw_torch_device": config.dtw_torch_device,
            "dtw_torch_batch_pair_threshold": config.dtw_torch_batch_pair_threshold,
            "gpu_name": _detect_gpu_name() if config.dtw_backend != "cpu_python" else None,
            "windows": window_rows,
            "recent_artifacts": [],
        }
        qualification_config_path.write_text(
            json.dumps(
                {
                    "run_label": config.run_label,
                    "config_id": config.config_id,
                    "config_name": config.config_name,
                    "config_version": config.config_version,
                    "code_commit": config.code_commit,
                    "bars_5m_timestamp_semantics": config.bars_5m_timestamp_semantics,
                    "benchmark_symbols": list(config.benchmark_symbols),
                    "graph_backend": config.graph_backend,
                    "graph_torch_device": config.graph_torch_device,
                    "dtw_backend": config.dtw_backend,
                    "dtw_torch_device": config.dtw_torch_device,
                    "dtw_torch_batch_pair_threshold": config.dtw_torch_batch_pair_threshold,
                    "graph_build_execution_mode": config.graph_build_execution_mode,
                    "gpu_name": progress_state["gpu_name"],
                    "windows": [
                        {
                            "window_id": window.window_id,
                            "date_start": window.date_start,
                            "date_end": window.date_end,
                            "trade_dates": window_trade_dates[window.window_id],
                        }
                        for window in windows
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._append_log(
            log_path,
            f"qualification run initialized: {config.run_label} | graph_backend={config.graph_backend} | graph_torch_device={config.graph_torch_device} | dtw_backend={config.dtw_backend} | dtw_torch_device={config.dtw_torch_device} | dtw_torch_batch_pair_threshold={config.dtw_torch_batch_pair_threshold}",
        )
        self._write_progress(progress_path, progress_state)

        window_results: list[QualificationWindowResult] = []
        completed_trade_dates = 0
        single_window_mode = len(windows) == 1

        for window in windows:
            window_state = self._window_state(progress_state, window.window_id)
            window_state["status"] = "running"
            progress_state["current_window_id"] = window.window_id
            progress_state["current_stage"] = "graph_build"
            self._write_progress(progress_path, progress_state)
            self._append_log(log_path, f"window {window.window_id} graph build started")

            trade_dates = window_trade_dates[window.window_id]
            window_root = output_root if single_window_mode else output_root / "windows" / window.window_id
            window_root.mkdir(parents=True, exist_ok=True)
            graph_db_path = window_root / "graph.duckdb"
            pack_output_dir = window_root / "evaluation_pack"

            if not trade_dates:
                window_result = QualificationWindowResult(
                    window=window,
                    status="skipped_no_dates",
                    processed_trade_dates=0,
                    total_trade_dates=0,
                    failure_count=0,
                    graph_db_path=graph_db_path,
                    evaluation_pack_dir=None,
                    alpha_ranking_path=None,
                    benchmark_summary_path=None,
                    elapsed_seconds=0.0,
                )
                window_state["status"] = "skipped_no_dates"
                window_state["graph_db_path"] = str(graph_db_path)
                window_results.append(window_result)
                self._refresh_root_outputs(output_root, window_results)
                progress_state["recent_artifacts"] = self._recent_artifacts(output_root)
                self._write_progress(progress_path, progress_state)
                continue

            base_completed_trade_dates = completed_trade_dates

            def on_graph_progress(event: dict[str, Any]) -> None:
                status = str(event.get("status", ""))
                if status == "range_started":
                    window_state["total_trade_dates"] = int(event.get("total_dates", len(trade_dates)) or 0)
                elif status == "shard_completed":
                    window_state["completed_trade_dates"] = int(event.get("completed_dates", 0) or 0)
                    progress_state["completed_trade_dates"] = base_completed_trade_dates + window_state["completed_trade_dates"]
                    trade_date_state = self._trade_date_state(window_state, str(event.get("trade_date", "")))
                    if trade_date_state is not None:
                        trade_date_state["status"] = "completed"
                        trade_date_state["snapshot_index"] = int(event.get("snapshot_count", 0) or 0)
                        trade_date_state["total_snapshots"] = int(event.get("snapshot_count", 0) or 0)
                        trade_date_state["progress_percent"] = 100.0
                elif status == "shard_failed":
                    window_state["failure_count"] = int(window_state.get("failure_count", 0) or 0) + 1
                    trade_date_state = self._trade_date_state(window_state, str(event.get("trade_date", "")))
                    if trade_date_state is not None:
                        trade_date_state["status"] = "failed"
                elif status == "snapshot_progress":
                    trade_date_state = self._trade_date_state(window_state, str(event.get("trade_date", "")))
                    if trade_date_state is not None:
                        trade_date_state["status"] = "running"
                        trade_date_state["snapshot_index"] = int(event.get("snapshot_index", 0) or 0)
                        trade_date_state["total_snapshots"] = int(event.get("total_snapshots", 0) or 0)
                        trade_date_state["snapshot_clock_code"] = event.get("snapshot_clock_code")
                        trade_date_state["available_minutes_since_open"] = event.get("available_minutes_since_open")
                        trade_date_state["progress_percent"] = float(event.get("progress_percent", 0.0) or 0.0)
                        progress_state["current_trade_date"] = trade_date_state["trade_date"]
                        progress_state["current_snapshot_id"] = event.get("snapshot_id")
                        progress_state["current_snapshot_clock_code"] = event.get("snapshot_clock_code")
                elif status == "range_completed":
                    progress_state["completed_trade_dates"] = base_completed_trade_dates + len(
                        event.get("processed_dates", [])
                    )
                self._write_progress(progress_path, progress_state)

            range_summary = self._run_graph_range(
                config,
                market_calendar,
                window,
                graph_db_path,
                shard_directory=window_root / "dates",
                progress_callback=on_graph_progress,
            )
            completed_trade_dates += len(range_summary.processed_dates)
            progress_state["completed_trade_dates"] = completed_trade_dates
            window_state["graph_db_path"] = str(graph_db_path)
            window_state["completed_trade_dates"] = len(range_summary.processed_dates)
            window_state["failure_count"] = range_summary.failure_count
            self._append_log(
                log_path,
                f"window {window.window_id} graph build completed: {len(range_summary.processed_dates)} trade dates",
            )

            progress_state["current_stage"] = "temporal_replay"
            self._write_progress(progress_path, progress_state)

            temporal_output_dir = output_root / "temporal"
            temporal_summary = TemporalReplayRangeService().run(
                run_id=f"{config.run_prefix}_{window.window_id}",
                date_roots=[result.database_path.parent for result in range_summary.shard_results],
                output_root=temporal_output_dir,
            )
            self._append_log(
                log_path,
                f"window {window.window_id} temporal replay completed: {len(temporal_summary.processed_dates)} trade dates",
            )

            progress_state["current_stage"] = "evaluation_pack"
            self._write_progress(progress_path, progress_state)
            self._append_log(log_path, f"window {window.window_id} evaluation pack started")
            pack_summary = self._evaluation_pack_builder(
                GraphEvaluationPackConfig(
                    graph_database_path=graph_db_path,
                    market_database_path=market_db_path,
                    metadata_csv_path=metadata_csv_path,
                    output_dir=pack_output_dir,
                    date_start=window.date_start,
                    date_end=window.date_end,
                    benchmark_symbols=config.benchmark_symbols,
                    generator_metadata={
                        "qualification_window_id": window.window_id,
                        "bars_5m_timestamp_semantics": config.bars_5m_timestamp_semantics,
                    },
                ),
                log=lambda message, *, _window_id=window.window_id: self._append_log(
                    log_path,
                    f"window {_window_id} pack: {message}",
                ),
            )

            alpha_ranking_path = pack_summary.artifact_paths.get("alpha_feature_ranking_by_layer")
            benchmark_summary_path = pack_summary.artifact_paths.get("benchmark_label_source_summary")
            window_result = QualificationWindowResult(
                window=window,
                status="completed" if range_summary.failure_count == 0 else "completed_with_failures",
                processed_trade_dates=len(range_summary.processed_dates),
                total_trade_dates=len(trade_dates),
                failure_count=range_summary.failure_count,
                graph_db_path=graph_db_path,
                evaluation_pack_dir=pack_summary.output_dir,
                alpha_ranking_path=alpha_ranking_path,
                benchmark_summary_path=benchmark_summary_path,
                elapsed_seconds=range_summary.elapsed_seconds,
            )
            window_results.append(window_result)
            window_state["status"] = window_result.status
            window_state["evaluation_pack_dir"] = str(pack_summary.output_dir)
            progress_state["completed_windows"] = len(
                [result for result in window_results if result.status.startswith("completed")]
            )

            self._refresh_root_outputs(output_root, window_results)
            progress_state["recent_artifacts"] = self._recent_artifacts(output_root)
            progress_state["current_stage"] = "checkpoint_publish"
            self._write_progress(progress_path, progress_state)

            try:
                if config.git_push_enabled:
                    self._checkpoint_publisher(
                        QualificationCheckpointContext(
                            config=config,
                            window=window,
                            window_result=window_result,
                            output_root=output_root,
                            progress_path=progress_path,
                            log_path=log_path,
                        )
                    )
                    self._append_log(log_path, f"window {window.window_id} checkpoint pushed")
            except Exception as exc:
                self._append_log(log_path, f"window {window.window_id} checkpoint push failed: {exc}")

            progress_state["current_stage"] = "window_complete"
            self._write_progress(progress_path, progress_state)

        progress_state["status"] = "complete"
        progress_state["current_window_id"] = None
        progress_state["current_stage"] = "complete"
        progress_state["completed_windows"] = len(
            [result for result in window_results if result.status.startswith("completed")]
        )
        progress_state["recent_artifacts"] = self._recent_artifacts(output_root)
        self._append_log(log_path, "qualification run completed")
        self._write_progress(progress_path, progress_state)

        return QualificationRunSummary(
            output_root=output_root,
            completed_windows=progress_state["completed_windows"],
            completed_trade_dates=completed_trade_dates,
            window_results=window_results,
        )

    def _run_graph_range(
        self,
        config: QualificationRunConfig,
        market_calendar: Any,
        window: QualificationWindow,
        graph_db_path: Path,
        *,
        shard_directory: Path,
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> GraphBuildRangeSummary:
        graph_config = GraphBuildRangeConfig(
            data_root=Path(config.data_root).expanduser().resolve(),
            output_database_path=graph_db_path,
            date_start=window.date_start,
            date_end=window.date_end,
            run_prefix=f"{config.run_prefix}_{window.window_id}",
            config_id=config.config_id,
            config_name=config.config_name,
            config_version=config.config_version,
            code_commit=config.code_commit,
            symbol_limit=config.symbol_limit,
            continue_on_error=config.continue_on_error,
            shard_directory=shard_directory,
            keep_shards=config.keep_shards,
            layer_workers_per_process=max(1, config.layer_workers_per_process),
            graph_backend=config.graph_backend,
            graph_torch_device=config.graph_torch_device,
            dtw_backend=config.dtw_backend,
            dtw_torch_device=config.dtw_torch_device,
            dtw_torch_batch_pair_threshold=max(1, config.dtw_torch_batch_pair_threshold),
            execution_mode=config.graph_build_execution_mode,
            data_source_kind=config.data_source_kind,
        )
        if self._graph_range_runner is not None:
            return self._graph_range_runner(graph_config, progress_callback=progress_callback)
        return GraphBuildRangeService(
            market_calendar=market_calendar,
            max_workers=max(1, config.max_date_workers),
        ).run(
            graph_config,
            progress_callback=progress_callback,
        )

    def _write_progress(self, path: Path, state: dict[str, Any]) -> None:
        prepared = dict(state)
        prepared["updated_at"] = self._now_provider().astimezone(UTC).isoformat()
        path.write_text(json.dumps(prepared, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _append_log(path: Path, message: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).isoformat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    @staticmethod
    def _window_state(progress_state: dict[str, Any], window_id: str) -> dict[str, Any]:
        for row in progress_state["windows"]:
            if row["window_id"] == window_id:
                return row
        raise KeyError(f"Unknown window id: {window_id}")

    @staticmethod
    def _trade_date_state(window_state: dict[str, Any], trade_date: str) -> dict[str, Any] | None:
        for row in window_state.get("trade_dates", []):
            if row.get("trade_date") == trade_date:
                return row
        return None

    @staticmethod
    def _refresh_root_outputs(output_root: Path, window_results: list[QualificationWindowResult]) -> None:
        _write_monthly_run_status(output_root / "monthly_run_status.csv", window_results)
        _write_monthly_alpha_summary(output_root / "monthly_alpha_summary.csv", window_results)
        _write_benchmark_label_source_summary(output_root / "benchmark_label_source_summary.csv", window_results)
        _write_cross_month_alpha_comparison(output_root / "cross_month_alpha_comparison.csv", window_results)
        _write_cross_month_layer_stability(output_root / "cross_month_layer_stability.csv", window_results)
        _write_artifact_inventory(output_root / "artifact_inventory.csv", window_results)

    @staticmethod
    def _recent_artifacts(output_root: Path) -> list[dict[str, str]]:
        artifacts = [
            ("date_artifact_registry", output_root / "date_artifact_registry.csv"),
            ("monthly_run_status", output_root / "monthly_run_status.csv"),
            ("monthly_alpha_summary", output_root / "monthly_alpha_summary.csv"),
            ("cross_month_alpha_comparison", output_root / "cross_month_alpha_comparison.csv"),
            ("cross_month_layer_stability", output_root / "cross_month_layer_stability.csv"),
            ("benchmark_label_source_summary", output_root / "benchmark_label_source_summary.csv"),
            ("artifact_inventory", output_root / "artifact_inventory.csv"),
            ("temporal", output_root / "temporal"),
        ]
        return [
            {"label": label, "path": str(path)}
            for label, path in artifacts
            if path.exists()
        ]


def _write_monthly_run_status(path: Path, window_results: list[QualificationWindowResult]) -> None:
    rows = [
        {
            "window_id": result.window.window_id,
            "date_start": result.window.date_start,
            "date_end": result.window.date_end,
            "status": result.status,
            "processed_trade_dates": result.processed_trade_dates,
            "total_trade_dates": result.total_trade_dates,
            "failure_count": result.failure_count,
            "graph_db_path": str(result.graph_db_path),
            "evaluation_pack_dir": str(result.evaluation_pack_dir) if result.evaluation_pack_dir else "",
            "elapsed_seconds": result.elapsed_seconds,
        }
        for result in window_results
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_monthly_alpha_summary(path: Path, window_results: list[QualificationWindowResult]) -> None:
    rows: list[dict[str, Any]] = []
    for result in window_results:
        ranking_path = result.alpha_ranking_path
        if ranking_path is None or not ranking_path.exists():
            continue
        frame = pd.read_csv(ranking_path)
        if frame.empty:
            continue
        prepared = frame.copy()
        prepared["score"] = pd.to_numeric(prepared["score"], errors="coerce")
        prepared["sample_size"] = pd.to_numeric(prepared["sample_size"], errors="coerce").fillna(0).astype(int)
        prepared = prepared.sort_values(
            ["graph_layer", "score", "sample_size"],
            ascending=[True, False, False],
        )
        for graph_layer, group in prepared.groupby("graph_layer", sort=True):
            top_row = group.iloc[0]
            rows.append(
                {
                    "window_id": result.window.window_id,
                    "graph_layer": graph_layer,
                    "layer_role": top_row.get("layer_role", ""),
                    "top_factor_name": top_row.get("factor_name", ""),
                    "top_label_horizon": top_row.get("label_horizon", ""),
                    "top_label_variant": top_row.get("label_variant", ""),
                    "top_score": top_row.get("score"),
                    "top_sample_size": top_row.get("sample_size"),
                    "top_confidence_bucket": top_row.get("confidence_bucket", ""),
                    "top_research_action": top_row.get("research_action", ""),
                }
            )
    if not rows:
        pd.DataFrame(
            columns=[
                "window_id",
                "graph_layer",
                "layer_role",
                "top_factor_name",
                "top_label_horizon",
                "top_label_variant",
                "top_score",
                "top_sample_size",
                "top_confidence_bucket",
                "top_research_action",
            ]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_benchmark_label_source_summary(path: Path, window_results: list[QualificationWindowResult]) -> None:
    frames: list[pd.DataFrame] = []
    for result in window_results:
        summary_path = result.benchmark_summary_path
        if summary_path is None or not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        frame.insert(0, "window_id", result.window.window_id)
        frames.append(frame)
    if not frames:
        pd.DataFrame(
            columns=[
                "window_id",
                "trade_date",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
                "row_count",
                "symbol_count",
                "excess_ret_1m_coverage_ratio",
            ]
        ).to_csv(path, index=False)
        return
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _write_cross_month_alpha_comparison(path: Path, window_results: list[QualificationWindowResult]) -> None:
    comparable_results = [
        result
        for result in window_results
        if result.alpha_ranking_path is not None and result.alpha_ranking_path.exists()
    ]
    if len(comparable_results) < 2:
        pd.DataFrame(
            columns=[
                "graph_layer",
                "layer_role",
                "factor_name",
                "label_horizon",
                "label_variant",
                "first_window_id",
                "second_window_id",
                "stability_bucket",
                "research_decision",
            ]
        ).to_csv(path, index=False)
        return
    comparison_frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="stocknetv2-cross-month-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for left, right in zip(comparable_results, comparable_results[1:]):
            output_path = tmp_dir_path / f"{left.window.window_id}_{right.window.window_id}.csv"
            _export_cross_window_alpha_comparison_report(
                left.alpha_ranking_path,
                right.alpha_ranking_path,
                output_path,
                first_window_id=left.window.window_id,
                second_window_id=right.window.window_id,
            )
            comparison_frames.append(pd.read_csv(output_path))
    pd.concat(comparison_frames, ignore_index=True).to_csv(path, index=False)


def _write_artifact_inventory(path: Path, window_results: list[QualificationWindowResult]) -> None:
    rows: list[dict[str, Any]] = []
    for result in window_results:
        rows.append(
            {
                "window_id": result.window.window_id,
                "artifact_label": "graph_db",
                "artifact_path": str(result.graph_db_path),
            }
        )
        if result.evaluation_pack_dir is not None:
            rows.append(
                {
                    "window_id": result.window.window_id,
                    "artifact_label": "evaluation_pack_dir",
                    "artifact_path": str(result.evaluation_pack_dir),
                }
            )
        if result.alpha_ranking_path is not None:
            rows.append(
                {
                    "window_id": result.window.window_id,
                    "artifact_label": "alpha_feature_ranking_by_layer",
                    "artifact_path": str(result.alpha_ranking_path),
                }
            )
        if result.benchmark_summary_path is not None:
            rows.append(
                {
                    "window_id": result.window.window_id,
                    "artifact_label": "benchmark_label_source_summary",
                    "artifact_path": str(result.benchmark_summary_path),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_cross_month_layer_stability(path: Path, window_results: list[QualificationWindowResult]) -> None:
    monthly_alpha_summary_path = path.parent / "monthly_alpha_summary.csv"
    if not monthly_alpha_summary_path.exists():
        pd.DataFrame(
            columns=[
                "window_id",
                "graph_layer",
                "layer_role",
                "top_factor_name",
                "top_label_horizon",
                "top_label_variant",
                "top_score",
                "previous_window_id",
                "same_factor_as_previous",
                "same_role_as_previous",
                "score_delta_vs_previous",
                "stability_bucket",
            ]
        ).to_csv(path, index=False)
        return

    try:
        summary = pd.read_csv(monthly_alpha_summary_path)
    except pd.errors.EmptyDataError:
        pd.DataFrame(
            columns=[
                "window_id",
                "graph_layer",
                "layer_role",
                "top_factor_name",
                "top_label_horizon",
                "top_label_variant",
                "top_score",
                "previous_window_id",
                "same_factor_as_previous",
                "same_role_as_previous",
                "score_delta_vs_previous",
                "stability_bucket",
            ]
        ).to_csv(path, index=False)
        return
    if summary.empty:
        summary.to_csv(path, index=False)
        return

    prepared = summary.copy()
    prepared["top_score"] = pd.to_numeric(prepared["top_score"], errors="coerce")
    prepared = prepared.sort_values(["graph_layer", "window_id"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _graph_layer, group in prepared.groupby("graph_layer", sort=True):
        previous_row: pd.Series | None = None
        for _, row in group.iterrows():
            payload = row.to_dict()
            if previous_row is None:
                rows.append(
                    {
                        **payload,
                        "previous_window_id": "",
                        "same_factor_as_previous": False,
                        "same_role_as_previous": False,
                        "score_delta_vs_previous": None,
                        "stability_bucket": "seed_window",
                    }
                )
            else:
                same_factor = row.get("top_factor_name") == previous_row.get("top_factor_name")
                same_role = row.get("layer_role") == previous_row.get("layer_role")
                score_delta = (
                    float(row["top_score"] - previous_row["top_score"])
                    if pd.notna(row.get("top_score")) and pd.notna(previous_row.get("top_score"))
                    else None
                )
                rows.append(
                    {
                        **payload,
                        "previous_window_id": previous_row.get("window_id", ""),
                        "same_factor_as_previous": same_factor,
                        "same_role_as_previous": same_role,
                        "score_delta_vs_previous": score_delta,
                        "stability_bucket": "stable_positive" if same_factor and same_role else "rotating_signal",
                    }
                )
            previous_row = row

    pd.DataFrame(rows).to_csv(path, index=False)


def _git_publish_checkpoint(context: QualificationCheckpointContext) -> None:
    repo_root = _resolve_git_repo_root(context.output_root)
    branch = context.config.git_branch or _git_stdout(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"]
    ).strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "add",
            "--",
            str(context.output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    diff_result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--cached",
            "--quiet",
            "--",
            str(context.output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if diff_result.returncode == 0:
        return
    commit_message = f"checkpoint: {context.config.run_label} {context.window.window_id}"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "commit",
            "-m",
            commit_message,
            "--",
            str(context.output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "push",
            context.config.git_remote,
            branch,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _resolve_git_repo_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    current = resolved
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError(f"Could not locate git repository root for {path}")


def _git_stdout(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _detect_gpu_name() -> str | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))
