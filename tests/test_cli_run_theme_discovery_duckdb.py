from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pandas as pd

from stocknetv2.interfaces.cli.run_theme_discovery_t1 import run_theme_discovery


def test_run_theme_discovery_accepts_legacy_duckdb_source(tmp_path):
    legacy_database_path = tmp_path / "legacy.duckdb"
    output_database_path = tmp_path / "stocknetv2.duckdb"

    connection = duckdb.connect(str(legacy_database_path))
    connection.execute(
        """
        CREATE TABLE bars_5m (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR,
            vwap DOUBLE,
            source VARCHAR,
            date DATE
        );
        CREATE TABLE trade_flow_1m (
            ticker VARCHAR,
            minute TIMESTAMP,
            trade_count DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_dollar_volume DOUBLE,
            large_trade_count DOUBLE,
            off_exchange_volume DOUBLE,
            date DATE
        );
        CREATE TABLE features_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            ret_1m_past DOUBLE,
            volume_z_proxy DOUBLE,
            large_trade_ratio DOUBLE,
            imbalance_proxy DOUBLE,
            date DATE
        );
        """
    )
    bar_rows = []
    for symbol, base_price, price_step in [
        ("AAA", 10.0, 0.10),
        ("BBB", 20.0, 0.20),
        ("CCC", 30.0, 0.15),
    ]:
        for offset, timestamp in enumerate(pd.date_range("2026-01-02T14:35:00Z", periods=5, freq="5min")):
            close_price = base_price + (offset + 1) * price_step
            bar_rows.append(
                (
                    timestamp.to_pydatetime(),
                    close_price - 0.1,
                    close_price + 0.1,
                    close_price - 0.2,
                    close_price,
                    1000.0 + offset * 10.0,
                    symbol,
                    close_price - 0.05,
                    "test",
                    "2026-01-02",
                )
            )
    connection.executemany(
        "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        bar_rows,
    )
    minute_timestamps = list(pd.date_range("2026-01-02T14:31:00Z", periods=20, freq="1min"))
    trade_flow_rows = []
    feature_rows = []
    for symbol, ret_value, volume_z, large_trade_ratio, imbalance in [
        ("AAA", 0.0200, 1.70, 2.20, 0.40),
        ("BBB", 0.0210, 1.80, 2.25, 0.41),
        ("CCC", 0.0205, 1.75, 2.30, 0.39),
    ]:
        for timestamp in minute_timestamps:
            trade_flow_rows.append(
                (
                    symbol,
                    timestamp,
                    15.0,
                    500.0,
                    5050.0,
                    imbalance,
                    1000.0,
                    2.0,
                    25.0,
                    "2026-01-02",
                )
            )
            feature_rows.append(
                (
                    symbol,
                    timestamp,
                    ret_value,
                    volume_z,
                    large_trade_ratio,
                    imbalance,
                    "2026-01-02",
                )
            )
    connection.executemany(
        "INSERT INTO trade_flow_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        trade_flow_rows,
    )
    connection.executemany(
        "INSERT INTO features_1m VALUES (?, ?, ?, ?, ?, ?, ?)",
        feature_rows,
    )
    connection.close()

    summary = run_theme_discovery(
        database_path=output_database_path,
        legacy_database_path=legacy_database_path,
        run_id="duckdb_run_test",
        run_name="DuckDB source test",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="duckdb_config_test",
        config_name="DuckDB baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="duckdb123",
    )

    assert summary.snapshot_count == 78
    output_connection = duckdb.connect(str(output_database_path))
    assert output_connection.execute(
        "SELECT COUNT(*) FROM consensus_theme_candidate WHERE run_id = ?",
        ["duckdb_run_test"],
    ).fetchone()[0] >= 1
    assert output_connection.execute(
        "SELECT COUNT(*) FROM frontend_snapshot_cache WHERE run_id = ?",
        ["duckdb_run_test"],
    ).fetchone()[0] >= 1
    output_connection.close()


def test_run_theme_discovery_can_limit_symbols_for_duckdb_pilot(tmp_path):
    legacy_database_path = tmp_path / "legacy.duckdb"
    output_database_path = tmp_path / "stocknetv2.duckdb"

    connection = duckdb.connect(str(legacy_database_path))
    connection.execute(
        """
        CREATE TABLE bars_5m (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR,
            vwap DOUBLE,
            source VARCHAR,
            date DATE
        );
        CREATE TABLE trade_flow_1m (
            ticker VARCHAR,
            minute TIMESTAMP,
            trade_count DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_dollar_volume DOUBLE,
            large_trade_count DOUBLE,
            off_exchange_volume DOUBLE,
            date DATE
        );
        CREATE TABLE features_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            ret_1m_past DOUBLE,
            volume_z_proxy DOUBLE,
            large_trade_ratio DOUBLE,
            imbalance_proxy DOUBLE,
            date DATE
        );
        """
    )

    minute_timestamps = list(pd.date_range("2026-01-02T14:31:00Z", periods=20, freq="1min"))
    bar_timestamps = list(pd.date_range("2026-01-02T14:35:00Z", periods=4, freq="5min"))
    trade_flow_rows = []
    feature_rows = []
    bar_rows = []
    for symbol, base_price in [("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0)]:
        for offset, timestamp in enumerate(bar_timestamps):
            bar_rows.append(
                (
                    timestamp.to_pydatetime(),
                    base_price + offset,
                    base_price + offset + 0.2,
                    base_price + offset - 0.1,
                    base_price + offset + 0.1,
                    1000.0 + offset,
                    symbol,
                    base_price + offset + 0.05,
                    "test",
                    "2026-01-02",
                )
            )
        for timestamp in minute_timestamps:
            trade_flow_rows.append(
                (
                    symbol,
                    timestamp.to_pydatetime(),
                    15.0,
                    500.0,
                    5050.0,
                    0.4,
                    1000.0,
                    2.0,
                    25.0,
                    "2026-01-02",
                )
            )
            feature_rows.append(
                (
                    symbol,
                    timestamp.to_pydatetime(),
                    0.02,
                    1.7,
                    0.3,
                    0.4,
                    "2026-01-02",
                )
            )
    connection.executemany(
        "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        bar_rows,
    )
    connection.executemany(
        "INSERT INTO trade_flow_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        trade_flow_rows,
    )
    connection.executemany(
        "INSERT INTO features_1m VALUES (?, ?, ?, ?, ?, ?, ?)",
        feature_rows,
    )
    connection.close()

    run_theme_discovery(
        database_path=output_database_path,
        legacy_database_path=legacy_database_path,
        symbol_limit=2,
        run_id="duckdb_symbol_limit_run",
        run_name="DuckDB symbol limit run",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="duckdb_symbol_limit_config",
        config_name="DuckDB symbol limit baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="duckdb-symbol-limit",
    )

    output_connection = duckdb.connect(str(output_database_path))
    symbols = output_connection.execute(
        """
        SELECT DISTINCT symbol
        FROM layer_community_membership
        WHERE run_id = ?
        ORDER BY symbol
        """,
        ["duckdb_symbol_limit_run"],
    ).fetchall()
    output_connection.close()

    assert [row[0] for row in symbols] == ["AAA", "BBB"]


def test_run_theme_discovery_can_run_graph_build_only_mode(tmp_path):
    legacy_database_path = tmp_path / "legacy.duckdb"
    output_database_path = tmp_path / "stocknetv2.duckdb"

    connection = duckdb.connect(str(legacy_database_path))
    connection.execute(
        """
        CREATE TABLE bars_5m (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR,
            vwap DOUBLE,
            source VARCHAR,
            date DATE
        );
        CREATE TABLE trade_flow_1m (
            ticker VARCHAR,
            minute TIMESTAMP,
            trade_count DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_dollar_volume DOUBLE,
            large_trade_count DOUBLE,
            off_exchange_volume DOUBLE,
            date DATE
        );
        CREATE TABLE features_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            ret_1m_past DOUBLE,
            volume_z_proxy DOUBLE,
            large_trade_ratio DOUBLE,
            imbalance_proxy DOUBLE,
            date DATE
        );
        """
    )
    minute_timestamps = list(pd.date_range("2026-01-02T14:31:00Z", periods=20, freq="1min"))
    trade_flow_rows = []
    feature_rows = []
    for symbol, ret_value, volume_z, large_trade_ratio, imbalance in [
        ("AAA", 0.02, 1.7, 0.3, 0.4),
        ("BBB", 0.021, 1.8, 0.31, 0.41),
    ]:
        for timestamp in minute_timestamps:
            trade_flow_rows.append(
                (
                    symbol,
                    timestamp,
                    15.0,
                    500.0,
                    5050.0,
                    imbalance,
                    1000.0,
                    2.0,
                    25.0,
                    "2026-01-02",
                )
            )
            feature_rows.append(
                (
                    symbol,
                    timestamp,
                    ret_value,
                    volume_z,
                    large_trade_ratio,
                    imbalance,
                    "2026-01-02",
                )
            )
    connection.execute(
        """
        INSERT INTO bars_5m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            datetime(2026, 1, 2, 14, 35, tzinfo=UTC), 10.0, 10.2, 9.9, 10.1, 1000.0, "AAA", 10.05, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 40, tzinfo=UTC), 20.0, 20.2, 19.9, 20.1, 2000.0, "BBB", 20.05, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 45, tzinfo=UTC), 10.1, 10.3, 10.0, 10.2, 1100.0, "AAA", 10.15, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 45, tzinfo=UTC), 20.1, 20.3, 20.0, 20.2, 2100.0, "BBB", 20.15, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 50, tzinfo=UTC), 10.2, 10.4, 10.1, 10.3, 1200.0, "AAA", 10.25, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 50, tzinfo=UTC), 20.2, 20.4, 20.1, 20.3, 2200.0, "BBB", 20.25, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 55, tzinfo=UTC), 10.3, 10.5, 10.2, 10.4, 1300.0, "AAA", 10.35, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 55, tzinfo=UTC), 20.3, 20.5, 20.2, 20.4, 2300.0, "BBB", 20.35, "test", "2026-01-02",
        ],
    )
    connection.executemany(
        "INSERT INTO trade_flow_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        trade_flow_rows,
    )
    connection.executemany(
        "INSERT INTO features_1m VALUES (?, ?, ?, ?, ?, ?, ?)",
        feature_rows,
    )
    connection.close()

    run_theme_discovery(
        database_path=output_database_path,
        legacy_database_path=legacy_database_path,
        run_id="duckdb_graph_only_run",
        run_name="DuckDB graph only run",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="duckdb_graph_only_config",
        config_name="DuckDB graph only baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="duckdb-graph-only",
        graph_build_only=True,
    )

    output_connection = duckdb.connect(str(output_database_path))
    assert output_connection.execute(
        "SELECT COUNT(*) FROM graph_edges_thresholded WHERE run_id = ?",
        ["duckdb_graph_only_run"],
    ).fetchone()[0] >= 1
    assert output_connection.execute(
        "SELECT COUNT(*) FROM consensus_theme_candidate WHERE run_id = ?",
        ["duckdb_graph_only_run"],
    ).fetchone()[0] == 0
    output_connection.close()
