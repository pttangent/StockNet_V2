from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.graph_build_chain_service import (  # noqa: E402
    GraphBuildChainService,
    GraphBuildValidationSummary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a monthly graph build and chain into a full-history run.")
    parser.add_argument("--watch-database", required=True)
    parser.add_argument("--watch-process-token", required=True)
    parser.add_argument("--watch-expected-trade-dates", required=True, type=int)
    parser.add_argument("--watch-date-start", required=True)
    parser.add_argument("--watch-date-end", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--full-history-database", required=True)
    parser.add_argument("--full-history-process-token", required=True)
    parser.add_argument("--full-history-stdout", required=True)
    parser.add_argument("--full-history-stderr", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--graph-build-script", default=str(ROOT_DIR / "scripts" / "run_graph_build_range.py"))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-prefix", default="full-history-graph-build")
    parser.add_argument("--config-id", default="full-history-graph-build")
    parser.add_argument("--config-name", default="full_history_graph_build")
    parser.add_argument("--config-version", default="parallel-single-writer-v1")
    parser.add_argument("--max-date-workers", type=int, default=8)
    parser.add_argument("--layer-workers-per-process", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    watch_database = Path(args.watch_database).expanduser().resolve()
    full_history_database = Path(args.full_history_database).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()
    graph_build_script = Path(args.graph_build_script).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_history_database.parent.mkdir(parents=True, exist_ok=True)

    trade_dates = _resolve_full_history_trade_dates(data_root)
    if not trade_dates:
        raise RuntimeError(f"No trade dates found under {data_root}")
    full_history_start = trade_dates[0]
    full_history_end = trade_dates[-1]
    full_history_expected_dates = len(trade_dates)
    code_commit = _git_head_sha(ROOT_DIR.parent)

    def log(message: str) -> None:
        formatted = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(formatted + "\n")

    log("Starting graph-build chain supervisor.")
    log(
        "Full-history target resolved to "
        f"{full_history_start} -> {full_history_end} ({full_history_expected_dates} trade dates)."
    )

    service = GraphBuildChainService(
        is_watched_run_active=lambda: _is_python_command_active(args.watch_process_token),
        validate_watched_run=lambda: _validate_graph_build_database(
            watch_database,
            expected_trade_dates=args.watch_expected_trade_dates,
            expected_date_start=args.watch_date_start,
            expected_date_end=args.watch_date_end,
        ),
        is_full_history_run_active=lambda: _is_python_command_active(args.full_history_process_token),
        validate_full_history_run=lambda: _validate_graph_build_database(
            full_history_database,
            expected_trade_dates=full_history_expected_dates,
            expected_date_start=full_history_start,
            expected_date_end=full_history_end,
        ),
        launch_full_history_run=lambda: _launch_full_history_run(
            python_executable=Path(args.python_exe).expanduser().resolve(),
            graph_build_script=graph_build_script,
            data_root=data_root,
            output_database=full_history_database,
            stdout_path=Path(args.full_history_stdout).expanduser().resolve(),
            stderr_path=Path(args.full_history_stderr).expanduser().resolve(),
            date_start=full_history_start,
            date_end=full_history_end,
            run_prefix=args.run_prefix,
            config_id=args.config_id,
            config_name=args.config_name,
            config_version=args.config_version,
            code_commit=code_commit,
            max_date_workers=args.max_date_workers,
            layer_workers_per_process=args.layer_workers_per_process,
        ),
        sleep=time.sleep,
        log=log,
        poll_seconds=args.poll_seconds,
    )
    summary = service.run()
    log(
        "Chain completed. Monthly dates="
        f"{summary.watched_run.trade_date_count}, full-history dates={summary.full_history_run.trade_date_count}."
    )
    return 0


def _resolve_full_history_trade_dates(data_root: Path) -> list[str]:
    bars_trade_dates = set(_list_trade_dates_for_dataset(data_root, "bars_5m"))
    raw_trade_dates = set(_list_trade_dates_for_dataset(data_root, "raw_1m"))
    trade_flow_trade_dates = set(_list_trade_dates_for_dataset(data_root, "trade_flow_1m"))
    return sorted(bars_trade_dates & raw_trade_dates & trade_flow_trade_dates)


def _list_trade_dates_for_dataset(data_root: Path, dataset_name: str) -> list[str]:
    dataset_root = data_root / dataset_name
    if not dataset_root.exists():
        return []
    return sorted(
        child.name.split("=", 1)[1]
        for child in dataset_root.iterdir()
        if child.is_dir() and child.name.startswith("date=")
    )


def _git_head_sha(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _is_python_command_active(command_token: str) -> bool:
    command = _build_python_process_probe_command(
        command_token=command_token,
        exclude_pid=os.getpid(),
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def _build_python_process_probe_command(*, command_token: str, exclude_pid: int) -> str:
    return (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' "
        f"-and $_.ProcessId -ne {exclude_pid} "
        "-and $_.CommandLine -like '*"
        + command_token.replace("'", "''")
        + "*' } | "
        "Select-Object -First 1 -ExpandProperty ProcessId"
    )


def _validate_graph_build_database(
    database_path: Path,
    *,
    expected_trade_dates: int,
    expected_date_start: str,
    expected_date_end: str,
) -> GraphBuildValidationSummary:
    if not database_path.exists():
        raise FileNotFoundError(str(database_path))
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        run_count, incomplete_run_count = connection.execute(
            """
            SELECT COUNT(*), COUNT(*) FILTER (WHERE status <> 'completed')
            FROM theme_discovery_run
            """
        ).fetchone()
        if incomplete_run_count:
            raise RuntimeError(f"Incomplete runs remain in {database_path}")
        trade_date_count, date_start, date_end = connection.execute(
            """
            SELECT COUNT(DISTINCT trade_date), MIN(trade_date)::VARCHAR, MAX(trade_date)::VARCHAR
            FROM graph_snapshot
            """
        ).fetchone()
        snapshot_count = connection.execute("SELECT COUNT(*) FROM graph_snapshot").fetchone()[0]
        edge_count = connection.execute("SELECT COUNT(*) FROM graph_edges_thresholded").fetchone()[0]
        layer_community_count = connection.execute("SELECT COUNT(*) FROM layer_community").fetchone()[0]
    finally:
        connection.close()

    if trade_date_count != expected_trade_dates:
        raise RuntimeError(
            f"Expected {expected_trade_dates} trade dates in {database_path}, found {trade_date_count}"
        )
    if date_start != expected_date_start or date_end != expected_date_end:
        raise RuntimeError(
            f"Expected {expected_date_start} -> {expected_date_end} in {database_path}, "
            f"found {date_start} -> {date_end}"
        )
    if snapshot_count <= 0 or edge_count <= 0 or layer_community_count <= 0:
        raise RuntimeError(f"Database {database_path} does not contain completed graph outputs.")
    return GraphBuildValidationSummary(
        database_path=str(database_path),
        run_count=run_count,
        trade_date_count=trade_date_count,
        snapshot_count=snapshot_count,
        edge_count=edge_count,
        layer_community_count=layer_community_count,
        date_start=date_start,
        date_end=date_end,
    )


def _launch_full_history_run(
    *,
    python_executable: Path,
    graph_build_script: Path,
    data_root: Path,
    output_database: Path,
    stdout_path: Path,
    stderr_path: Path,
    date_start: str,
    date_end: str,
    run_prefix: str,
    config_id: str,
    config_name: str,
    config_version: str,
    code_commit: str,
    max_date_workers: int,
    layer_workers_per_process: int,
) -> None:
    for path in (output_database, output_database.with_suffix(output_database.suffix + ".wal"), stdout_path, stderr_path):
        if path.exists():
            path.unlink()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [
            str(python_executable),
            str(graph_build_script),
            "--data-root",
            str(data_root),
            "--database",
            str(output_database),
            "--date-start",
            date_start,
            "--date-end",
            date_end,
            "--run-prefix",
            run_prefix,
            "--config-id",
            config_id,
            "--config-name",
            config_name,
            "--config-version",
            config_version,
            "--code-commit",
            code_commit,
            "--max-date-workers",
            str(max_date_workers),
            "--layer-workers-per-process",
            str(layer_workers_per_process),
            "--continue-on-error",
        ],
        cwd=str(ROOT_DIR),
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
