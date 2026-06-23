from __future__ import annotations

import duckdb
import pandas as pd

from stocknetv2.application.services.theme_discovery_orchestrator import (
    ThemeDiscoveryOrchestrator,
    ThemeDiscoveryRunConfig,
)
from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


class StubMarketReadRepository:
    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        assert dataset_name == "bars_5m"
        return ["2026-01-02"]

    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs:
        assert trade_date == "2026-01-02"
        frame = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-02T14:35:00Z")],
                "symbol": ["AAA"],
            }
        )
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=frame,
            trade_flow_1m=frame.copy(),
            features_1m=frame.copy(),
            data_version="test-data-version",
        )


def test_theme_discovery_orchestrator_registers_run_config_and_snapshots():
    connection = duckdb.connect(":memory:")
    SchemaManager(connection).initialize()

    orchestrator = ThemeDiscoveryOrchestrator(
        market_repository=StubMarketReadRepository(),
        audit_repository=AuditRepository(connection),
        snapshot_clock=SnapshotClock(),
    )
    config = ThemeDiscoveryRunConfig(
        run_id="run_t1_test",
        run_name="T1 test run",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="config_t1_test",
        config_name="T1 baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="abc123",
    )

    summary = orchestrator.run(config)

    assert summary.run_id == "run_t1_test"
    assert summary.trade_dates_processed == ["2026-01-02"]
    assert summary.snapshot_count == 78

    config_row = connection.execute(
        "SELECT config_name, config_scope, config_version FROM config_registry WHERE config_id = ?",
        ["config_t1_test"],
    ).fetchone()
    assert config_row == ("T1 baseline", "t1", "v1")

    run_row = connection.execute(
        "SELECT run_name, data_version, status FROM theme_discovery_run WHERE run_id = ?",
        ["run_t1_test"],
    ).fetchone()
    assert run_row == ("T1 test run", "test-data-version", "completed")

    snapshot_count = connection.execute(
        "SELECT COUNT(*) FROM graph_snapshot WHERE run_id = ?",
        ["run_t1_test"],
    ).fetchone()[0]
    assert snapshot_count == 78

    lineage_row = connection.execute(
        "SELECT source_kind, source_name, source_version FROM input_lineage WHERE run_id = ? ORDER BY source_name LIMIT 1",
        ["run_t1_test"],
    ).fetchone()
    assert lineage_row == ("dataset", "bars_5m", "2026-01-02")
