from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb
import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusService
from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.theme_discovery_orchestrator import (
    ThemeDiscoveryOrchestrator,
    ThemeDiscoveryRunConfig,
)
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs
from stocknetv2.infrastructure.repositories.read_model_repository import ReadModelRepository
from stocknetv2.infrastructure.repositories.theme_write_repository import ThemeWriteRepository
from stocknetv2.application.services.semantic_service import SemanticService
from stocknetv2.application.services.lifecycle_service import LifecycleService
from stocknetv2.application.services.theme_quality_service import ThemeQualityService
from stocknetv2.application.services.theme_flow_service import ThemeFlowService
from stocknetv2.application.services.read_model_service import ReadModelService


class TwoSnapshotClock:
    def iter_trade_date(self, trade_date: str):
        assert trade_date == "2026-01-02"
        return [
            pd.Timestamp("2026-01-02T14:45:00Z"),
            pd.Timestamp("2026-01-02T14:50:00Z"),
        ]

    def session_open_timestamp(self, trade_date: str):
        assert trade_date == "2026-01-02"
        return pd.Timestamp("2026-01-02T14:30:00Z")


class EnrichedMarketReadRepository:
    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        assert dataset_name == "bars_5m"
        return ["2026-01-02"]

    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs:
        bars_timestamps = [
            datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
            datetime(2026, 1, 2, 14, 40, tzinfo=UTC),
            datetime(2026, 1, 2, 14, 45, tzinfo=UTC),
            datetime(2026, 1, 2, 14, 50, tzinfo=UTC),
        ]
        minute_timestamps = list(pd.date_range("2026-01-02T14:31:00Z", periods=20, freq="1min"))
        bars_5m = pd.DataFrame(
            {
                "timestamp": bars_timestamps * 3,
                "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
                "close": [10.0, 10.1, 10.2, 10.3] + [20.0, 20.2, 20.4, 20.6] + [30.0, 30.3, 30.6, 30.9],
            }
        )
        features_1m = pd.DataFrame(
            {
                "timestamp": minute_timestamps * 3,
                "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
                "ret_1m": [0.01] * 20 + [0.011] * 20 + [0.0105] * 20,
                "volume_z_12": [2.0] * 20 + [2.1] * 20 + [2.05] * 20,
                "large_trade_ratio_z": [2.2] * 20 + [2.25] * 20 + [2.3] * 20,
            }
        )
        trade_flow_1m = pd.DataFrame(
            {
                "timestamp": minute_timestamps * 3,
                "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
                "flow_impulse_score": [1.0] * 20 + [1.01] * 20 + [1.02] * 20,
                "imbalance_z": [0.5] * 20 + [0.49] * 20 + [0.48] * 20,
                "large_trade_ratio_z": [2.2] * 20 + [2.25] * 20 + [2.3] * 20,
            }
        )
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version="enriched-test-version",
        )


def test_orchestrator_persists_semantic_lifecycle_flow_and_cache_outputs():
    connection = duckdb.connect(":memory:")
    SchemaManager(connection).initialize()

    orchestrator = ThemeDiscoveryOrchestrator(
        market_repository=EnrichedMarketReadRepository(),
        audit_repository=AuditRepository(connection),
        snapshot_clock=TwoSnapshotClock(),
        layer_execution_service=LayerExecutionService(),
        graph_write_repository=GraphWriteRepository(connection),
        consensus_service=ConsensusService(),
        theme_write_repository=ThemeWriteRepository(connection),
        semantic_service=SemanticService(),
        lifecycle_service=LifecycleService(),
        theme_quality_service=ThemeQualityService(),
        theme_flow_service=ThemeFlowService(),
        read_model_service=ReadModelService(),
        read_model_repository=ReadModelRepository(connection),
    )
    config = ThemeDiscoveryRunConfig(
        run_id="run_enriched_test",
        run_name="Enriched persistence test",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="config_enriched_test",
        config_name="Enriched baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="enriched123",
    )

    summary = orchestrator.run(config)

    assert summary.snapshot_count == 2

    semantic_count = connection.execute(
        "SELECT COUNT(*) FROM theme_semantic_label WHERE run_id = ?",
        ["run_enriched_test"],
    ).fetchone()[0]
    lifecycle_rows = connection.execute(
        "SELECT event_type, theme_path_id FROM theme_path_lifecycle WHERE run_id = ? ORDER BY timestamp",
        ["run_enriched_test"],
    ).fetchall()
    flow_count = connection.execute(
        "SELECT COUNT(*) FROM theme_level_flow_series WHERE run_id = ?",
        ["run_enriched_test"],
    ).fetchone()[0]
    cache_rows = connection.execute(
        "SELECT cache_type, COUNT(*) FROM frontend_snapshot_cache WHERE run_id = ? GROUP BY cache_type ORDER BY cache_type",
        ["run_enriched_test"],
    ).fetchall()
    quality_rows = connection.execute(
        """
        SELECT
            structure_score,
            cross_layer_consensus_score,
            flow_support_score,
            dtw_flow_support_score,
            volume_support_score,
            large_trade_support_score,
            stability_score,
            semantic_coherence_score,
            theme_quality_score,
            theme_quality_breakdown_json
        FROM consensus_theme_candidate
        WHERE run_id = ?
        ORDER BY timestamp, theme_instance_id
        """,
        ["run_enriched_test"],
    ).fetchall()

    assert semantic_count >= 2
    assert lifecycle_rows[0][0] == "birth"
    assert lifecycle_rows[1][0] == "continuation"
    assert lifecycle_rows[0][1] == lifecycle_rows[1][1]
    assert flow_count >= 2
    assert ("snapshot_summary", 2) in cache_rows
    assert all(row[0] > 0.0 for row in quality_rows)
    assert all(row[1] > 0.0 for row in quality_rows)
    assert any(row[6] > 0.0 for row in quality_rows)
    assert all(row[7] > 0.0 for row in quality_rows)
    assert all(row[8] > 0.0 for row in quality_rows)
    assert all("component_scores" in json.loads(row[9]) for row in quality_rows)
