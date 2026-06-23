from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.symbol_metadata_service import (
    SymbolMetadataRefreshConfig,
    refresh_symbol_metadata_cache,
)
from stocknetv2.infrastructure.project_paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Fetch and cache symbol metadata with yfinance.")
    parser.add_argument(
        "--symbols-csv",
        default=str(project_paths.symbol_metadata_csv_path),
        help="CSV containing a Ticker/symbol column. Default: USStock_Proj/data/artifacts/input_symbols.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=str(project_paths.symbol_metadata_csv_path),
        help="Output CSV path. Default overwrites USStock_Proj/data/artifacts/input_symbols.csv",
    )
    parser.add_argument(
        "--output-parquet",
        default=str(ROOT_DIR / "metadata" / "symbol_metadata.parquet"),
        help="Optional parquet cache path. Default: StockNetV2/metadata/symbol_metadata.parquet",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent yfinance workers. Default: 8")
    parser.add_argument("--retries", type=int, default=2, help="Retries per symbol. Default: 2")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refetch symbols already present in the output CSV instead of reusing cached rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = refresh_symbol_metadata_cache(
        SymbolMetadataRefreshConfig(
            symbols_csv_path=args.symbols_csv,
            output_csv_path=args.output_csv,
            output_parquet_path=args.output_parquet,
            max_workers=args.workers,
            retries=args.retries,
            refresh_existing=args.refresh_existing,
        ),
        log=print,
    )
    print("Symbol metadata refresh completed.")
    print(f"Output CSV: {summary.output_csv_path}")
    if summary.output_parquet_path is not None:
        print(f"Output parquet: {summary.output_parquet_path}")
    print(f"Symbols in scope: {summary.symbol_count}")
    print(f"Fetched now: {summary.fetched_count}")
    print(f"Reused cached rows: {summary.reused_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
