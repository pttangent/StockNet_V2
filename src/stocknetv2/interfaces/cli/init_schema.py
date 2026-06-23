from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from stocknetv2.infrastructure.db.schema_manager import SchemaManager


def initialize_schema_database(database_path: Path | str) -> Path:
    resolved_path = Path(database_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(resolved_path))
    try:
        SchemaManager(connection).initialize()
    finally:
        connection.close()

    return resolved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the StockNetV2 DuckDB schema.")
    parser.add_argument("--database", required=True, help="Target DuckDB file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database_path = initialize_schema_database(args.database)
    print(f"Initialized StockNetV2 schema at {database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
