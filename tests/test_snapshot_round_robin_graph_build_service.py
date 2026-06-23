from __future__ import annotations

import duckdb
import pandas as pd

from stocknetv2.application.services.graph_build_range_service import GraphBuildRangeConfig
from stocknetv2.application.services.layer_execution_service import LayerExecutionResult
from stocknetv2.application.services.snapshot_round_robin_graph_build_service import (
    SnapshotRoundRobinGraphBuildService,
)
from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


class _TwoDateSnapshotClock:
    def iter_trade_date(self, trade_date: str):
        return [pd.Timestamp(f"{trade_date}T14:35:00Z")]

    def session_open_timestamp(self, trade_date: str):
        return pd.Timestamp(f"{trade_date}T14:30:00Z")


class _TwoDateMarketRepository:
    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        assert dataset_name == "bars_5m"
        return ["2025-01-02", "2025-01-03"]

    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs:
        bars_5m = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp(f"{trade_date}T14:35:00Z"),
                    pd.Timestamp(f"{trade_date}T14:35:00Z"),
                ],
                "symbol": ["AAA", "BBB"],
                "close": [10.0, 10.5],
            }
        )
        features_1m = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp(f"{trade_date}T14:31:00Z"),
                    pd.Timestamp(f"{trade_date}T14:31:00Z"),
                ],
                "available_time": [
                    pd.Timestamp(f"{trade_date}T14:32:00Z"),
                    pd.Timestamp(f"{trade_date}T14:32:00Z"),
                ],
                "symbol": ["AAA", "BBB"],
                "ret_1m": [0.01, 0.012],
                "volume_z_12": [1.5, 1.6],
                "imbalance_z": [0.2, 0.21],
                "large_trade_ratio_z": [0.4, 0.41],
                "flow_impulse_score": [0.6, 0.61],
            }
        )
        trade_flow_1m = features_1m[
            ["timestamp", "available_time", "symbol", "imbalance_z", "large_trade_ratio_z", "flow_impulse_score"]
        ].copy()
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version=f"bars_5m:{trade_date}",
        )


class _StaticLayerExecutionService:
    def __init__(self) -> None:
        self.snapshot_calls: list[tuple[str, str]] = []

    def execute_for_snapshot(self, *, inputs, snapshot_time, session_open):
        self.snapshot_calls.append((inputs.trade_date, snapshot_time.strftime("%H%M")))
        edge = GraphEdge(
            graph_layer="return_corr_graph",
            edge_type="return_correlation",
            source_symbol="AAA",
            target_symbol="BBB",
            snapshot_time=snapshot_time,
            weight=0.82,
            raw_score=0.82,
            support_points=8,
        )
        return LayerExecutionResult(
            layer_edges={
                "return_corr_graph": [edge],
                "dtw_return_similarity_graph": [],
                "flow_alignment_graph": [],
                "dtw_trade_flow_similarity_graph": [],
                "volume_expansion_graph": [],
                "large_trade_alignment_graph": [],
            },
            layer_communities={
                "return_corr_graph": [Community(members=["AAA", "BBB"])],
                "dtw_return_similarity_graph": [],
                "flow_alignment_graph": [],
                "dtw_trade_flow_similarity_graph": [],
                "volume_expansion_graph": [],
                "large_trade_alignment_graph": [],
            },
        )

    def close(self):
        return None


def test_snapshot_round_robin_graph_build_service_persists_multiple_dates_in_one_database():
    connection = duckdb.connect(":memory:")
    SchemaManager(connection).initialize()
    layer_service = _StaticLayerExecutionService()
    service = SnapshotRoundRobinGraphBuildService(
        market_repository=_TwoDateMarketRepository(),
        audit_repository=AuditRepository(connection),
        snapshot_clock=_TwoDateSnapshotClock(),
        layer_execution_service=layer_service,
        graph_write_repository=GraphWriteRepository(connection),
    )
    config = GraphBuildRangeConfig(
        data_root="unused",
        output_database_path="unused.duckdb",
        date_start="2025-01-02",
        date_end="2025-01-03",
        run_prefix="graph-build",
        config_id="graph-build-config",
        config_name="Graph build config",
        config_version="v1",
        code_commit="abc123",
        execution_mode="snapshot_round_robin",
    )

    events: list[dict[str, object]] = []
    summary = service.run(
        config,
        trade_dates=["2025-01-02", "2025-01-03"],
        progress_callback=events.append,
    )

    assert summary.processed_dates == ["2025-01-02", "2025-01-03"]
    assert layer_service.snapshot_calls == [("2025-01-02", "1435"), ("2025-01-03", "1435")]
    assert connection.execute("SELECT COUNT(*) FROM theme_discovery_run").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM graph_snapshot").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM graph_layer_diagnostic").fetchone()[0] == 12
    assert connection.execute("SELECT COUNT(*) FROM graph_edges_thresholded").fetchone()[0] == 2
    assert [event["status"] for event in events[:4]] == [
        "range_started",
        "trade_date_started",
        "trade_date_started",
        "snapshot_progress",
    ]
