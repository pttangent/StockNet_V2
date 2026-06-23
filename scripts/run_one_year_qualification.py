from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Run a one-year StockNetV2 qualification loop with monthly checkpoints.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-db", default=str(project_paths.market_db_path))
    parser.add_argument("--metadata-csv", default=str(project_paths.symbol_metadata_csv_path))
    parser.add_argument("--output-root", default=str(ROOT_DIR / "research_runs" / "qualification_2025_y1"))
    parser.add_argument("--run-label", default="2025 Y1 qualification")
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end")
    parser.add_argument("--max-date-workers", type=int, default=24)
    parser.add_argument("--layer-workers-per-process", type=int, default=1)
    parser.add_argument("--graph-backend", default="torch_cuda", choices=("cpu_numpy", "torch_cpu", "torch_cuda", "torch_auto"))
    parser.add_argument("--graph-torch-device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtw-backend", default="torch_cuda", choices=("cpu_python", "torch_cpu", "torch_cuda", "torch_auto"))
    parser.add_argument("--dtw-torch-device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtw-torch-batch-pair-threshold", type=int, default=1024)
    parser.add_argument(
        "--graph-build-execution-mode",
        default="trade_date_shards",
        choices=("trade_date_shards", "snapshot_round_robin"),
    )
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch")
    parser.add_argument("--skip-git-push", action="store_true")
    return parser.parse_args()


def build_year_window(year: int) -> QualificationWindow:
    return QualificationWindow(
        window_id=str(year),
        date_start=f"{year}-01-01",
        date_end=f"{year}-12-31",
    )


def build_date_window(date_start: str, date_end: str | None = None) -> QualificationWindow:
    start_year = datetime.strptime(date_start, "%Y-%m-%d").year
    resolved_end = date_end or f"{start_year}-12-31"
    return QualificationWindow(
        window_id=f"{date_start}_to_{resolved_end}",
        date_start=date_start,
        date_end=resolved_end,
    )


def main() -> int:
    args = parse_args()
    windows = [build_date_window(args.date_start, args.date_end)]
    summary = QualificationRunService().run(
        QualificationRunConfig(
            data_root=Path(args.data_root).expanduser().resolve(),
            market_db_path=Path(args.market_db).expanduser().resolve(),
            metadata_csv_path=Path(args.metadata_csv).expanduser().resolve(),
            output_root=Path(args.output_root).expanduser().resolve(),
            run_label=args.run_label,
            config_id="one-year-qualification",
            config_name="One-year qualification run",
            config_version=args.config_version,
            code_commit=args.code_commit,
            max_date_workers=max(1, args.max_date_workers),
            layer_workers_per_process=max(1, args.layer_workers_per_process),
            keep_shards=True,
            graph_backend=args.graph_backend,
            graph_torch_device=args.graph_torch_device,
            dtw_backend=args.dtw_backend,
            dtw_torch_device=args.dtw_torch_device,
            dtw_torch_batch_pair_threshold=max(1, args.dtw_torch_batch_pair_threshold),
            graph_build_execution_mode=args.graph_build_execution_mode,
            git_push_enabled=not args.skip_git_push,
            git_remote=args.git_remote,
            git_branch=args.git_branch,
        ),
        windows=windows,
    )
    print(
        json.dumps(
            {
                "output_root": str(summary.output_root),
                "completed_windows": summary.completed_windows,
                "completed_trade_dates": summary.completed_trade_dates,
                "window_ids": [window.window_id for window in windows],
                "date_start": args.date_start,
                "date_end": args.date_end or f"{datetime.strptime(args.date_start, '%Y-%m-%d').year}-12-31",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
