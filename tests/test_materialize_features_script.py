from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.materialize_features_1m import materialize_features_1m


def _write_partition(root: Path, relative_dir: str, filename: str, frame: pd.DataFrame) -> None:
    target_dir = root / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / filename, index=False)


def test_materialize_features_1m_writes_missing_feature_partition(tmp_path: Path) -> None:
    trade_date = "2026-01-02"
    bars_5m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "close": [10.4],
        }
    )
    bars_1m = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "timestamp": [
                datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
                datetime(2026, 1, 2, 14, 32, tzinfo=UTC),
                datetime(2026, 1, 2, 14, 33, tzinfo=UTC),
            ],
            "bar_end": [
                datetime(2026, 1, 2, 14, 32, tzinfo=UTC),
                datetime(2026, 1, 2, 14, 33, tzinfo=UTC),
                datetime(2026, 1, 2, 14, 34, tzinfo=UTC),
            ],
            "close": [10.0, 10.1, 10.2],
            "volume": [100.0, 110.0, 120.0],
            "dollar_volume": [1000.0, 1111.0, 1224.0],
        }
    )

    _write_partition(tmp_path, f"bars_5m/date={trade_date}", "bars_5m.parquet", bars_5m)
    _write_partition(tmp_path, f"raw_1m/date={trade_date}", "bars_1m.parquet", bars_1m)

    summary = materialize_features_1m(data_root=tmp_path, date_start=trade_date, date_end=trade_date)

    output_path = tmp_path / f"features_1m/date={trade_date}/features_1m.parquet"
    assert output_path.exists()
    output = pd.read_parquet(output_path)
    assert summary["written_dates"] == [trade_date]
    assert "ret_1m" in output.columns
    assert "available_time" in output.columns
