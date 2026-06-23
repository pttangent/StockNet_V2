from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stocknetv2.infrastructure.repositories.market_read_repository import (
    LegacySourceLayout,
    MonthPackSourceLayout,
    MarketReadRepository,
)


def _write_partition(root, relative_dir: str, filename: str, frame: pd.DataFrame) -> None:
    target_dir = root / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / filename, index=False)


def test_market_read_repository_lists_available_trade_dates_and_loads_inputs(tmp_path):
    bars_5m = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2026, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "symbol": ["AAA", "BBB"],
            "close": [10.0, 20.0],
        }
    )
    trade_flow_1m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "flow_impulse_score": [1.2],
        }
    )
    features_1m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "ret_1m": [0.01],
        }
    )

    _write_partition(tmp_path, "bars_5m/date=2026-01-02", "bars_5m.parquet", bars_5m)
    _write_partition(tmp_path, "trade_flow_1m/date=2026-01-02", "trade_flow_1m.parquet", trade_flow_1m)
    _write_partition(tmp_path, "features_1m/date=2026-01-02", "features_1m.parquet", features_1m)

    repository = MarketReadRepository(LegacySourceLayout(data_root=tmp_path))

    assert repository.list_available_trade_dates("bars_5m") == ["2026-01-02"]

    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert list(inputs.bars_5m["symbol"]) == ["AAA", "BBB"]
    assert list(inputs.trade_flow_1m["symbol"]) == ["AAA"]
    assert list(inputs.features_1m["symbol"]) == ["AAA"]
    assert "available_time" in inputs.trade_flow_1m.columns
    assert inputs.trade_flow_1m.loc[0, "available_time"] == datetime(2026, 1, 2, 14, 36, tzinfo=UTC)
    assert "available_time" in inputs.features_1m.columns
    assert inputs.features_1m.loc[0, "available_time"] == datetime(2026, 1, 2, 14, 36, tzinfo=UTC)
    assert inputs.data_version == "bars_5m:2026-01-02|trade_flow_1m:2026-01-02|features_1m:2026-01-02"


def test_market_read_repository_returns_empty_optional_frames_when_missing(tmp_path):
    bars_5m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "close": [10.0],
        }
    )
    _write_partition(tmp_path, "bars_5m/date=2026-01-02", "bars_5m.parquet", bars_5m)

    repository = MarketReadRepository(LegacySourceLayout(data_root=tmp_path))
    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert not inputs.bars_5m.empty
    assert inputs.trade_flow_1m.empty
    assert inputs.features_1m.empty


def test_market_read_repository_builds_features_from_raw_bars_when_partition_missing(tmp_path):
    trade_date = "2026-01-02"
    timestamps = [
        datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 32, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 33, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 34, tzinfo=UTC),
    ]
    bars_5m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "close": [10.4],
        }
    )
    bars_1m = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "timestamp": timestamps,
            "bar_end": [timestamp + pd.Timedelta(minutes=1) for timestamp in timestamps],
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [10.1, 10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1, 10.2],
            "close": [10.0, 10.1, 10.2, 10.4],
            "volume": [100.0, 110.0, 125.0, 160.0],
            "dollar_volume": [1000.0, 1111.0, 1275.0, 1664.0],
            "vwap": [10.0, 10.1, 10.2, 10.4],
            "exchange": ["XNYS"] * 4,
            "bar_type": [1] * 4,
            "sequence": [1, 2, 3, 4],
            "source": ["unit-test"] * 4,
        }
    )
    trade_flow_1m = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "minute": timestamps,
            "trade_count": [10.0, 12.0, 14.0, 20.0],
            "volume": [100.0, 110.0, 125.0, 160.0],
            "dollar_volume": [1000.0, 1111.0, 1275.0, 1664.0],
            "imbalance_proxy": [0.10, 0.20, 0.30, 0.60],
            "large_trade_dollar_volume": [100.0, 150.0, 200.0, 500.0],
        }
    )

    _write_partition(tmp_path, f"bars_5m/date={trade_date}", "bars_5m.parquet", bars_5m)
    _write_partition(tmp_path, f"raw_1m/date={trade_date}", "bars_1m.parquet", bars_1m)
    _write_partition(tmp_path, f"trade_flow_1m/date={trade_date}", "trade_flow_1m.parquet", trade_flow_1m)

    repository = MarketReadRepository(LegacySourceLayout(data_root=tmp_path))
    inputs = repository.load_trade_date_inputs(trade_date)

    assert list(inputs.features_1m["symbol"]) == ["AAA"] * 4
    assert "ret_1m" in inputs.features_1m.columns
    assert "volume_z_12" in inputs.features_1m.columns
    assert "imbalance_z" in inputs.features_1m.columns
    assert "large_trade_ratio_z" in inputs.features_1m.columns
    assert "flow_impulse_score" in inputs.features_1m.columns
    assert "available_time" in inputs.features_1m.columns
    assert inputs.features_1m["available_time"].tolist() == [
        timestamp + pd.Timedelta(minutes=1) for timestamp in timestamps
    ]
    assert inputs.trade_flow_1m["symbol"].tolist() == ["AAA"] * 4
    assert inputs.trade_flow_1m["timestamp"].tolist() == timestamps
    assert inputs.trade_flow_1m["available_time"].tolist() == [
        timestamp + pd.Timedelta(minutes=1) for timestamp in timestamps
    ]
    assert "generated_features_1m" in inputs.data_version


def test_market_read_repository_reads_month_pack_dates_and_graph_features(tmp_path):
    pack_root = tmp_path / "month=2026-01"
    trade_date = "2026-01-02"

    bars_5m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "symbol_id": [1],
            "close": [10.0],
        }
    )
    graph_features = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "available_time": [datetime(2026, 1, 2, 14, 36, tzinfo=UTC)],
            "symbol": ["AAA"],
            "symbol_id": [1],
            "ret_1m": [0.01],
            "volume_z_12": [1.2],
            "imbalance_z": [0.3],
            "large_trade_ratio_z": [0.4],
            "flow_impulse_score": [0.5],
        }
    )

    _write_partition(pack_root, f"dates/date={trade_date}", "bars_5m.parquet", bars_5m)
    _write_partition(pack_root, f"dates/date={trade_date}", "graph_features_1m.parquet", graph_features)

    repository = MarketReadRepository(MonthPackSourceLayout(pack_root=pack_root))

    assert repository.list_available_trade_dates("bars_5m") == [trade_date]

    inputs = repository.load_trade_date_inputs(trade_date)

    assert list(inputs.bars_5m["symbol"]) == ["AAA"]
    assert list(inputs.features_1m["symbol"]) == ["AAA"]
    assert inputs.trade_flow_1m.empty
    assert inputs.features_1m.loc[0, "available_time"] == datetime(2026, 1, 2, 14, 36, tzinfo=UTC)
    assert inputs.data_version == "month_pack:2026-01:2026-01-02"
