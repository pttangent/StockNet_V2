from __future__ import annotations

import duckdb

from stocknetv2.infrastructure.db.schema_definitions import TABLE_SCHEMAS


class SchemaManager:
    """Create and maintain the base DuckDB schema for StockNetV2."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        for statement in TABLE_SCHEMAS.values():
            self._connection.execute(statement)
