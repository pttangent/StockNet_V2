from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.graph_build_range_service import GraphBuildRangeConfig, GraphBuildRangeService
from stocknetv2.infrastructure.project_paths import ProjectPaths
from stocknetv2.infrastructure.repositories.market_read_repository import MarketReadRepository, MonthPackSourceLayout

PROFILE_DEFAULTS = {
    "cpu_full": {"graph_backend": "cpu_numpy", "dtw_backend": "cpu_python"},
    "cpu_no_dtw": {"graph_backend": "cpu_numpy", "dtw_backend": "cpu_python"},
    "cpu_dtw_only": {"graph_backend": "cpu_numpy", "dtw_backend": "cpu_python"},
    "cuda_dtw_only": {"graph_backend": "cpu_numpy", "dtw_backend": "torch_cuda", "dtw_torch_device": "cuda"},
}


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Run distributed graph build for one month pack.")
    parser.add_argument("--month", required=True)
    parser.add_argument("--month-pack-root")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id", default="distributed-baseline-v1")
    parser.add_argument("--profile", default="cpu_no_dtw", choices=tuple(PROFILE_DEFAULTS))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--snapshot-block-size", type=int, default=8)
    parser.add_argument("--max-tasks-per-child", type=int, default=6)
    parser.add_argument("--system-memory-reserve-gb", type=int, default=10)
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--graph-backend")
    parser.add_argument("--graph-torch-device", default="auto")
    parser.add_argument("--dtw-backend")
    parser.add_argument("--dtw-torch-device", default="auto")
    parser.add_argument("--dtw-pair-batch-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-format", default="parquet_snapshot")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--execution-mode", default="trade_date_shards", choices=("trade_date_shards", "snapshot_round_robin"))
    parser.set_defaults(
        month_pack_root=str(project_paths.distributed_packs_root / "month={month}"),
        output_root=str(project_paths.distributed_runs_root / "run={run_id}" / "month={month}"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    month_pack_root = Path(args.month_pack_root.format(month=args.month, run_id=args.run_id)).expanduser().resolve()
    output_root = Path(args.output_root.format(month=args.month, run_id=args.run_id)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repository = MarketReadRepository(MonthPackSourceLayout(pack_root=month_pack_root))
    trade_dates = repository.list_available_trade_dates("bars_5m")
    if not trade_dates:
        raise RuntimeError(f"No trade dates found in month pack {month_pack_root}")

    defaults = PROFILE_DEFAULTS[args.profile]
    graph_backend = args.graph_backend or defaults.get("graph_backend", "cpu_numpy")
    dtw_backend = args.dtw_backend or defaults.get("dtw_backend", "cpu_python")
    dtw_torch_device = args.dtw_torch_device or defaults.get("dtw_torch_device", "auto")
    date_start = args.date_start or trade_dates[0]
    date_end = args.date_end or trade_dates[-1]

    config = GraphBuildRangeConfig(
        data_root=month_pack_root,
        output_database_path=output_root / "graph.duckdb",
        date_start=date_start,
        date_end=date_end,
        run_prefix=f"{args.run_id}_{args.month}",
        config_id="distributed-month-run",
        config_name="Distributed month graph build",
        config_version="v1",
        code_commit="migrated-without-git-history",
        continue_on_error=args.continue_on_error,
        shard_directory=output_root / "dates",
        keep_shards=True,
        layer_workers_per_process=max(1, args.threads_per_worker),
        graph_backend=graph_backend,
        graph_torch_device=args.graph_torch_device,
        dtw_backend=dtw_backend,
        dtw_torch_device=dtw_torch_device,
        dtw_torch_batch_pair_threshold=max(1, args.dtw_pair_batch_size),
        execution_mode=args.execution_mode,
        data_source_kind="month_pack",
    )
    service = GraphBuildRangeService(
        market_calendar=repository,
        max_workers=max(1, args.max_workers),
    )
    summary = service.run(config, progress_callback=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    print(json.dumps({"status": "complete", "processed_dates": summary.processed_dates, "failure_count": summary.failure_count}, ensure_ascii=False))
    return 0 if summary.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
