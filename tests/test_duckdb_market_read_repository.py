from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pandas as pd

from stocknetv2.infrastructure.repositories.market_read_repository import (
    LegacyDuckDBSource,
    MarketReadRepository,
)


def test_market_read_repository_can_read_and_normalize_legacy_duckdb(tmp_path):
    database_path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(database_path))
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
    connection.execute(
        """
        INSERT INTO bars_5m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            datetime(2026, 1, 2, 14, 35, tzinfo=UTC), 10.0, 10.2, 9.9, 10.1, 1000.0, "AAA", 10.05, "test", "2026-01-02",
            datetime(2026, 1, 2, 14, 40, tzinfo=UTC), 20.0, 20.2, 19.9, 20.1, 2000.0, "BBB", 20.05, "test", "2026-01-02",
        ],
    )
    connection.execute(
        """
        INSERT INTO trade_flow_1m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA",
            datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
            15.0,
            500.0,
            5050.0,
            0.4,
            1000.0,
            2.0,
            25.0,
            "2026-01-02",
        ],
    )
    connection.execute(
        """
        INSERT INTO features_1m VALUES
        (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA",
            datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
            0.02,
            1.7,
            0.3,
            0.4,
            "2026-01-02",
        ],
    )
    connection.close()

    repository = MarketReadRepository(LegacyDuckDBSource(database_path=database_path))

    assert repository.list_available_trade_dates("bars_5m") == ["2026-01-02"]

    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert list(inputs.bars_5m["symbol"]) == ["AAA", "BBB"]
    assert inputs.bars_5m["timestamp"].iloc[0] == pd.Timestamp("2026-01-02T14:35:00Z")
    assert list(inputs.trade_flow_1m["symbol"]) == ["AAA"]
    assert "timestamp" in inputs.trade_flow_1m.columns
    assert "flow_impulse_score" in inputs.trade_flow_1m.columns
    assert "imbalance_z" in inputs.trade_flow_1m.columns
    assert "ret_1m" in inputs.features_1m.columns
    assert "volume_z_12" in inputs.features_1m.columns
    assert "large_trade_ratio_z" in inputs.features_1m.columns
    assert inputs.data_version == f"duckdb:{database_path.name}:2026-01-02"


def test_market_read_repository_can_limit_symbols_for_pilot_runs(tmp_path):
    database_path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(database_path))
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
    for symbol in ["AAA", "BBB", "CCC"]:
        connection.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [datetime(2026, 1, 2, 14, 35, tzinfo=UTC), 10.0, 10.2, 9.9, 10.1, 1000.0, symbol, 10.05, "test", "2026-01-02"],
        )
        connection.execute(
            "INSERT INTO trade_flow_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [symbol, datetime(2026, 1, 2, 14, 35, tzinfo=UTC), 15.0, 500.0, 5050.0, 0.4, 1000.0, 2.0, 25.0, "2026-01-02"],
        )
        connection.execute(
            "INSERT INTO features_1m VALUES (?, ?, ?, ?, ?, ?, ?)",
            [symbol, datetime(2026, 1, 2, 14, 35, tzinfo=UTC), 0.02, 1.7, 0.3, 0.4, "2026-01-02"],
        )
    connection.close()

    repository = MarketReadRepository(LegacyDuckDBSource(database_path=database_path), symbol_limit=2)
    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert sorted(inputs.bars_5m["symbol"].unique().tolist()) == ["AAA", "BBB"]
    assert sorted(inputs.trade_flow_1m["symbol"].unique().tolist()) == ["AAA", "BBB"]
    assert sorted(inputs.features_1m["symbol"].unique().tolist()) == ["AAA", "BBB"]


def test_market_read_repository_auto_detects_utc_duckdb_timestamps(tmp_path):
    database_path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(database_path))
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
    connection.execute(
        """
        INSERT INTO bars_5m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [datetime(2026, 1, 2, 14, 35), 10.0, 10.2, 9.9, 10.1, 1000.0, "AAA", 10.05, "test", "2026-01-02"],
    )
    connection.execute(
        """
        INSERT INTO trade_flow_1m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["AAA", datetime(2026, 1, 2, 14, 35), 15.0, 500.0, 5050.0, 0.4, 1000.0, 2.0, 25.0, "2026-01-02"],
    )
    connection.execute(
        """
        INSERT INTO features_1m VALUES
        (?, ?, ?, ?, ?, ?, ?)
        """,
        ["AAA", datetime(2026, 1, 2, 14, 35), 0.02, 1.7, 0.3, 0.4, "2026-01-02"],
    )
    connection.close()

    repository = MarketReadRepository(LegacyDuckDBSource(database_path=database_path))
    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert inputs.bars_5m["timestamp"].iloc[0] == pd.Timestamp("2026-01-02T14:35:00Z")
    assert inputs.trade_flow_1m["timestamp"].iloc[0] == pd.Timestamp("2026-01-02T14:35:00Z")
    assert inputs.features_1m["timestamp"].iloc[0] == pd.Timestamp("2026-01-02T14:35:00Z")
