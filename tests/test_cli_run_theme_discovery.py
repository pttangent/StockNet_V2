from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pandas as pd

from stocknetv2.interfaces.cli.run_theme_discovery_t1 import run_theme_discovery


def _write_partition(root, relative_dir: str, filename: str, frame: pd.DataFrame) -> None:
    target_dir = root / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / filename, index=False)


def test_run_theme_discovery_cli_executes_one_trade_day_pipeline(tmp_path):
    legacy_data_root = tmp_path / "legacy_data"
    database_path = tmp_path / "stocknetv2.duckdb"

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

    _write_partition(legacy_data_root, "bars_5m/date=2026-01-02", "bars_5m.parquet", bars_5m)
    _write_partition(legacy_data_root, "trade_flow_1m/date=2026-01-02", "trade_flow_1m.parquet", trade_flow_1m)
    _write_partition(legacy_data_root, "features_1m/date=2026-01-02", "features_1m.parquet", features_1m)

    summary = run_theme_discovery(
        database_path=database_path,
        legacy_data_root=legacy_data_root,
        run_id="cli_run_test",
        run_name="CLI test run",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="cli_config_test",
        config_name="CLI baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="cli123",
    )

    assert summary.snapshot_count == 78

    connection = duckdb.connect(str(database_path))
    run_row = connection.execute(
        "SELECT run_name, status FROM theme_discovery_run WHERE run_id = ?",
        ["cli_run_test"],
    ).fetchone()
    assert run_row == ("CLI test run", "completed")
