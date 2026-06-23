from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.graph_evaluation_pack_service import (
    GraphEvaluationPackConfig,
    build_graph_evaluation_pack,
)
from stocknetv2.infrastructure.project_paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Build a graph evaluation pack for manual quality review.")
    parser.add_argument(
        "--graph-db",
        required=True,
        help="Monthly graph-build DuckDB path.",
    )
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
        "--output-dir",
        required=True,
        help="Target evaluation-pack output directory.",
    )
    parser.add_argument("--date-start", help="Optional inclusive trade-date start.")
    parser.add_argument("--date-end", help="Optional inclusive trade-date end.")
    parser.add_argument(
        "--benchmark-symbols",
        default="SPY,QQQ,IWM,DIA",
        help="Comma-separated benchmark symbols. Default: SPY,QQQ,IWM,DIA",
    )
    parser.add_argument(
        "--compare-graph-db",
        help="Optional baseline monthly graph-build DuckDB for old-vs-new comparison exports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_symbols = tuple(
        symbol.strip().upper()
        for symbol in args.benchmark_symbols.split(",")
        if symbol.strip()
    )
    summary = build_graph_evaluation_pack(
        GraphEvaluationPackConfig(
            graph_database_path=args.graph_db,
            market_database_path=args.market_db,
            metadata_csv_path=args.metadata_csv,
            output_dir=args.output_dir,
            date_start=args.date_start,
            date_end=args.date_end,
            benchmark_symbols=benchmark_symbols,
            compare_graph_database_path=args.compare_graph_db,
        ),
        log=print,
    )
    print("Evaluation pack completed.")
    print(f"Output directory: {summary.output_dir}")
    for key, value in sorted(summary.counts.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
