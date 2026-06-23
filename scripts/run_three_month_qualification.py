from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.qualification_run_service import (
    QualificationRunConfig,
    QualificationRunService,
    QualificationWindow,
)
from stocknetv2.infrastructure.project_paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(
        description="Run the three-month StockNetV2 qualification loop with per-window checkpoints."
    )
    parser.add_argument("--data-root", required=True, help="Legacy data root containing bars_5m/raw_1m/trade_flow_1m.")
    parser.add_argument(
        "--market-db",
        default=str(project_paths.market_db_path),
        help="Market DuckDB path. Default: USStock_Proj/data/stocknet_us.duckdb",
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(project_paths.symbol_metadata_csv_path),
        help="Symbol metadata CSV path. Default: USStock_Proj/data/artifacts/input_symbols.csv",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT_DIR / "research_runs" / "qualification_2025_q1"),
        help="Tracked output root for progress, logs, monthly packs, and checkpoint summaries.",
    )
    parser.add_argument("--run-label", default="2025 Q1 qualification")
    parser.add_argument("--run-prefix", default="qualification-graph-build")
    parser.add_argument("--config-id", default="three-month-qualification")
    parser.add_argument("--config-name", default="Three-month qualification run")
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--symbol-limit", type=int)
    parser.add_argument(
        "--max-date-workers",
        type=int,
        default=24,
        help="Per-trade-date worker count. Default: 24 to saturate a 24-core class CPU with day-level multiprocessing.",
    )
    parser.add_argument(
        "--layer-workers-per-process",
        type=int,
        default=1,
        help="Per-day inner layer worker count. Default: 1 to avoid oversubscribing CPU when date-level multiprocessing is already high.",
    )
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--benchmark-symbols", default="SPY,QQQ,IWM,DIA")
    parser.add_argument(
        "--bars-5m-timestamp-semantics",
        default="bar_close_time",
        choices=("bar_close_time", "bar_start_time"),
        help="Explicit 5m bar timestamp semantics recorded into progress/config outputs.",
    )
    parser.add_argument(
        "--dtw-backend",
        default="torch_cuda",
        choices=("cpu_python", "torch_cpu", "torch_cuda", "torch_auto"),
        help="DTW backend for the two DTW graph layers. Default: torch_cuda.",
    )
    parser.add_argument(
        "--dtw-torch-device",
        default="cuda",
        choices=("auto", "cpu", "cuda"),
        help="Torch device preference when using a torch DTW backend. Default: cuda.",
    )
    parser.add_argument(
        "--dtw-torch-batch-pair-threshold",
        type=int,
        default=1024,
        help="Minimum DTW candidate pair count before switching to the torch backend.",
    )
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch")
    parser.add_argument("--skip-git-push", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_symbols = tuple(
        symbol.strip().upper()
        for symbol in args.benchmark_symbols.split(",")
        if symbol.strip()
    )
    service = QualificationRunService()
    summary = service.run(
        QualificationRunConfig(
            data_root=Path(args.data_root).expanduser().resolve(),
            market_db_path=Path(args.market_db).expanduser().resolve(),
            metadata_csv_path=Path(args.metadata_csv).expanduser().resolve(),
            output_root=Path(args.output_root).expanduser().resolve(),
            run_label=args.run_label,
            run_prefix=args.run_prefix,
            config_id=args.config_id,
            config_name=args.config_name,
            config_version=args.config_version,
            code_commit=args.code_commit,
            symbol_limit=args.symbol_limit,
            max_date_workers=max(1, args.max_date_workers),
            layer_workers_per_process=max(1, args.layer_workers_per_process),
            keep_shards=args.keep_shards,
            continue_on_error=args.continue_on_error,
            benchmark_symbols=benchmark_symbols,
            bars_5m_timestamp_semantics=args.bars_5m_timestamp_semantics,
            dtw_backend=args.dtw_backend,
            dtw_torch_device=args.dtw_torch_device,
            dtw_torch_batch_pair_threshold=max(1, args.dtw_torch_batch_pair_threshold),
            git_push_enabled=not args.skip_git_push,
            git_remote=args.git_remote,
            git_branch=args.git_branch,
        ),
        windows=[
            QualificationWindow(window_id="2025-01", date_start="2025-01-01", date_end="2025-01-31"),
            QualificationWindow(window_id="2025-02", date_start="2025-02-01", date_end="2025-02-28"),
            QualificationWindow(window_id="2025-03", date_start="2025-03-01", date_end="2025-03-31"),
        ],
    )
    print(f"Qualification run complete: {summary.output_root}")
    print(f"Completed windows: {summary.completed_windows}")
    print(f"Completed trade dates: {summary.completed_trade_dates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
