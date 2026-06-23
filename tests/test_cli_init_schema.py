from __future__ import annotations

import duckdb

from stocknetv2.interfaces.cli.init_schema import initialize_schema_database


def test_initialize_schema_database_creates_duckdb_file_and_core_tables(tmp_path):
    database_path = tmp_path / "cli_init.duckdb"

    initialize_schema_database(database_path)

    connection = duckdb.connect(str(database_path))
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert "theme_discovery_run" in tables
    assert "frontend_snapshot_cache" in tables
