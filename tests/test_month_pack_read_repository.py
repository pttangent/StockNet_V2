from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from stocknetv2.infrastructure.repositories.month_pack_read_repository import MonthPackReadRepository


def test_month_pack_read_repository_filters_snapshot_window(tmp_path):
    pack_root = tmp_path / "data" / "ready" / "month=2025-01"
    date_root = pack_root / "dates" / "date=2025-01-02"
    date_root.mkdir(parents=True, exist_ok=True)
    (pack_root / "pack_manifest.json").write_text(json.dumps({"month": "2025-01"}), encoding="utf-8")
    pd.DataFrame(
        {
            "trade_date": ["2025-01-02", "2025-01-02"],
            "timestamp": [
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "snapshot_id": ["2025-01-02_0935", "2025-01-02_0940"],
            "snapshot_clock": ["0935", "0940"],
        }
    ).to_parquet(pack_root / "snapshot_schedule.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "symbol": ["AAA", "AAA", "AAA"],
            "symbol_id": [1, 1, 1],
            "close": [10.0, 10.5, 11.0],
        }
    ).to_parquet(date_root / "bars_5m.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 2, 14, 31, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 39, tzinfo=UTC),
            ],
            "available_time": [
                datetime(2025, 1, 2, 14, 32, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 36, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "symbol": ["AAA", "AAA", "AAA"],
            "symbol_id": [1, 1, 1],
            "ret_1m": [0.1, 0.2, 0.3],
            "volume_z_12": [1.0, 1.0, 1.0],
            "imbalance_z": [0.1, 0.2, 0.3],
            "large_trade_ratio_z": [0.2, 0.3, 0.4],
            "flow_impulse_score": [0.5, 0.6, 0.7],
        }
    ).to_parquet(date_root / "graph_features_1m.parquet", index=False)
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "minute": [
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 39, tzinfo=UTC),
            ],
            "dollar_volume": [1000.0, 1200.0],
            "large_trade_dollar_volume": [100.0, 180.0],
            "imbalance_proxy": [0.2, -0.1],
        }
    ).to_parquet(date_root / "trade_flow_1m.parquet", index=False)

    repository = MonthPackReadRepository(pack_root)
    schedule = repository.load_snapshot_schedule("2025-01-02")
    inputs = repository.read_snapshot_block(
        trade_date="2025-01-02",
        window_start=pd.Timestamp("2025-01-02T14:34:00Z"),
        window_end=pd.Timestamp("2025-01-02T14:39:00Z"),
        include_trade_flow=True,
    )

    assert [snapshot.snapshot_id for snapshot in schedule] == ["2025-01-02_0935", "2025-01-02_0940"]
    assert inputs.bars_5m["timestamp"].dt.strftime("%H:%M").tolist() == ["14:35"]
    assert inputs.features_1m["timestamp"].dt.strftime("%H:%M").tolist() == ["14:35", "14:39"]
    assert inputs.trade_flow_1m["timestamp"].dt.strftime("%H:%M").tolist() == ["14:35", "14:39"]
    assert inputs.trade_flow_1m["symbol"].tolist() == ["AAA", "AAA"]
    assert inputs.trade_flow_1m["large_trade_ratio_z"].tolist() == [0.1, 0.15]
    assert len(inputs.trade_flow_1m) == 2
