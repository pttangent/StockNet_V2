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
from stocknetv2.application.services.graph_evaluation_pack_service import (
    GraphEvaluationPackConfig,
    build_graph_evaluation_pack,
)
from stocknetv2.infrastructure.project_paths import ProjectPaths
from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository

WINDOW_ID = "2025-01-06_2025-01-17"
DATE_START = "2025-01-06"
DATE_END = "2025-01-17"


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Run the two-week causality-safe graph evaluation loop.")
    parser.add_argument("--data-root", required=True, help="Legacy data root containing bars_5m/raw_1m/trade_flow_1m.")
    parser.add_argument("--graph-db", required=True, help="Target graph-build DuckDB path.")
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
    parser.add_argument("--output-dir", required=True, help="Target evaluation-pack output directory.")
    parser.add_argument("--run-prefix", default="two-week-causality-safe")
    parser.add_argument("--config-id", default="two-week-causality-safe")
    parser.add_argument("--config-name", default="Two-week causality-safe evaluation loop")
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--symbol-limit", type=int)
    parser.add_argument("--max-date-workers", type=int, default=4)
    parser.add_argument("--layer-workers-per-process", type=int, default=4)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--benchmark-symbols", default="SPY,QQQ,IWM,DIA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    graph_db = Path(args.graph_db).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    market_db = Path(args.market_db).expanduser().resolve()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    graph_db.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    repository = MarketReadRepository(LegacySourceLayout(data_root=data_root))
    trade_dates = [
        trade_date
        for trade_date in repository.list_available_trade_dates("bars_5m")
        if DATE_START <= trade_date <= DATE_END
    ]
    if not trade_dates:
        print(json.dumps({"status": "no_dates", "window_id": WINDOW_ID, "date_start": DATE_START, "date_end": DATE_END}))
        return 1

    print(json.dumps({"status": "starting", "window_id": WINDOW_ID, "date_start": DATE_START, "date_end": DATE_END, "trade_dates": len(trade_dates)}), flush=True)

    range_service = GraphBuildRangeService(
        market_calendar=repository,
        max_workers=max(1, args.max_date_workers),
    )
    summary = range_service.run(
        GraphBuildRangeConfig(
            data_root=data_root,
            output_database_path=graph_db,
            date_start=DATE_START,
            date_end=DATE_END,
            run_prefix=args.run_prefix,
            config_id=args.config_id,
            config_name=args.config_name,
            config_version=args.config_version,
            code_commit=args.code_commit,
            symbol_limit=args.symbol_limit,
            continue_on_error=args.continue_on_error,
            keep_shards=args.keep_shards,
            layer_workers_per_process=max(1, args.layer_workers_per_process),
        )
    )
    for index, result in enumerate(summary.shard_results, start=1):
        print(
            json.dumps(
                {
                    "status": "graph_build_ok",
                    "window_id": WINDOW_ID,
                    "index": index,
                    "total_dates": len(summary.shard_results),
                    "trade_date": result.trade_date,
                    "run_id": result.run_id,
                    "snapshots": result.snapshot_count,
                    "elapsed_seconds": result.elapsed_seconds,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    benchmark_symbols = tuple(symbol.strip().upper() for symbol in args.benchmark_symbols.split(",") if symbol.strip())
    pack_summary = build_graph_evaluation_pack(
        GraphEvaluationPackConfig(
            graph_database_path=graph_db,
            market_database_path=market_db,
            metadata_csv_path=metadata_csv,
            output_dir=output_dir,
            date_start=DATE_START,
            date_end=DATE_END,
            benchmark_symbols=benchmark_symbols,
        ),
        log=lambda message: print(json.dumps({"status": "pack_progress", "window_id": WINDOW_ID, "message": message}, ensure_ascii=False), flush=True),
    )
    print(
        json.dumps(
            {
                "status": "complete" if summary.failure_count == 0 else "complete_with_failures",
                "window_id": WINDOW_ID,
                "processed_dates": len(summary.processed_dates),
                "failure_count": summary.failure_count,
                "graph_db": str(graph_db),
                "evaluation_pack": str(pack_summary.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
