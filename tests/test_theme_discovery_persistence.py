from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pandas as pd

from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.layer_execution_service import LayerExecutionResult
from stocknetv2.application.services.temporal_edge_replay_service import TemporalEdgeReplayService
from stocknetv2.application.services.theme_discovery_orchestrator import (
    ThemeDiscoveryOrchestrator,
    ThemeDiscoveryRunConfig,
)
from stocknetv2.application.services.consensus_service import ConsensusService
from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.theme_write_repository import ThemeWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


class OneSnapshotClock:
    def iter_trade_date(self, trade_date: str):
        assert trade_date == "2026-01-02"
        return [pd.Timestamp("2026-01-02T14:50:00Z")]

    def session_open_timestamp(self, trade_date: str):
        assert trade_date == "2026-01-02"
        return pd.Timestamp("2026-01-02T14:30:00Z")


class LayeredMarketReadRepository:
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
            data_version="layered-test-version",
        )


class StaticLayerExecutionService:
    def execute_for_snapshot(self, *, inputs, snapshot_time, session_open):
        shared_members = ["AAA", "BBB", "CCC"]
        layer_edges = {
            "return_corr_graph": [
                GraphEdge(
                    graph_layer="return_corr_graph",
                    edge_type="return_correlation",
                    source_symbol="AAA",
                    target_symbol="BBB",
                    snapshot_time=snapshot_time,
                    weight=0.81,
                    raw_score=0.81,
                    support_points=8,
                )
            ],
            "flow_alignment_graph": [
                GraphEdge(
                    graph_layer="flow_alignment_graph",
                    edge_type="flow_alignment",
                    source_symbol="AAA",
                    target_symbol="CCC",
                    snapshot_time=snapshot_time,
                    weight=0.79,
                    raw_score=0.79,
                    support_points=12,
                )
            ],
            "volume_expansion_graph": [
                GraphEdge(
                    graph_layer="volume_expansion_graph",
                    edge_type="volume_expansion",
                    source_symbol="BBB",
                    target_symbol="CCC",
                    snapshot_time=snapshot_time,
                    weight=0.77,
                    raw_score=0.77,
                    support_points=10,
                )
            ],
            "dtw_return_similarity_graph": [],
            "dtw_trade_flow_similarity_graph": [],
            "large_trade_alignment_graph": [],
        }
        layer_communities = {
            "return_corr_graph": [Community(members=shared_members)],
            "flow_alignment_graph": [Community(members=shared_members)],
            "volume_expansion_graph": [Community(members=shared_members)],
            "dtw_return_similarity_graph": [],
            "dtw_trade_flow_similarity_graph": [],
            "large_trade_alignment_graph": [],
        }
        return LayerExecutionResult(layer_edges=layer_edges, layer_communities=layer_communities)

    def close(self):
        return None


def test_orchestrator_persists_layer_community_and_consensus_outputs():
    connection = duckdb.connect(":memory:")
    SchemaManager(connection).initialize()

    orchestrator = ThemeDiscoveryOrchestrator(
        market_repository=LayeredMarketReadRepository(),
        audit_repository=AuditRepository(connection),
        snapshot_clock=OneSnapshotClock(),
            layer_execution_service=StaticLayerExecutionService(),
            graph_write_repository=GraphWriteRepository(connection),
            consensus_service=ConsensusService(),
            temporal_edge_replay_service=TemporalEdgeReplayService(),
            theme_write_repository=ThemeWriteRepository(connection),
    )
    config = ThemeDiscoveryRunConfig(
        run_id="run_layers_test",
        run_name="Layer persistence test",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="config_layers_test",
        config_name="Layer baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="layers123",
    )

    summary = orchestrator.run(config)

    assert summary.snapshot_count == 1

    edge_count = connection.execute("SELECT COUNT(*) FROM graph_edges_thresholded WHERE run_id = ?", ["run_layers_test"]).fetchone()[0]
    summary_count = connection.execute("SELECT COUNT(*) FROM graph_edge_summary WHERE run_id = ?", ["run_layers_test"]).fetchone()[0]
    diagnostic_count = connection.execute(
        "SELECT COUNT(*) FROM graph_layer_diagnostic WHERE run_id = ?",
        ["run_layers_test"],
    ).fetchone()[0]
    community_count = connection.execute("SELECT COUNT(*) FROM layer_community WHERE run_id = ?", ["run_layers_test"]).fetchone()[0]
    relation_observation_count = connection.execute(
        "SELECT COUNT(*) FROM relation_observation WHERE run_id = ?",
        ["run_layers_test"],
    ).fetchone()[0]
    temporal_edge_state_count = connection.execute(
        "SELECT COUNT(*) FROM temporal_edge_state WHERE run_id = ?",
        ["run_layers_test"],
    ).fetchone()[0]
    membership_count = connection.execute(
        "SELECT COUNT(*) FROM layer_community_membership WHERE run_id = ?",
        ["run_layers_test"],
    ).fetchone()[0]
    theme_count = connection.execute(
        "SELECT COUNT(*) FROM consensus_theme_candidate WHERE run_id = ?",
        ["run_layers_test"],
    ).fetchone()[0]
    theme_membership_count = connection.execute(
        "SELECT COUNT(*) FROM theme_membership WHERE run_id = ?",
        ["run_layers_test"],
    ).fetchone()[0]

    assert edge_count == 3
    assert summary_count == 6
    assert diagnostic_count == 6
    assert community_count == 3
    assert relation_observation_count >= edge_count
    assert temporal_edge_state_count >= edge_count
    assert membership_count == 9
    assert theme_count >= 1
    assert theme_membership_count >= 3


def test_orchestrator_can_run_graph_build_only_mode():
    connection = duckdb.connect(":memory:")
    SchemaManager(connection).initialize()

    orchestrator = ThemeDiscoveryOrchestrator(
        market_repository=LayeredMarketReadRepository(),
        audit_repository=AuditRepository(connection),
        snapshot_clock=OneSnapshotClock(),
            layer_execution_service=StaticLayerExecutionService(),
            graph_write_repository=GraphWriteRepository(connection),
            consensus_service=ConsensusService(),
            temporal_edge_replay_service=TemporalEdgeReplayService(),
            theme_write_repository=ThemeWriteRepository(connection),
    )
    config = ThemeDiscoveryRunConfig(
        run_id="run_graph_only_test",
        run_name="Graph only persistence test",
        date_start="2026-01-02",
        date_end="2026-01-02",
        config_id="config_graph_only_test",
        config_name="Graph only baseline",
        config_scope="t1",
        config_version="v1",
        code_commit="graphonly123",
        graph_build_only=True,
    )

    summary = orchestrator.run(config)

    assert summary.snapshot_count == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM graph_edges_thresholded WHERE run_id = ?",
        ["run_graph_only_test"],
    ).fetchone()[0] == 3
    assert connection.execute(
        "SELECT COUNT(*) FROM layer_community WHERE run_id = ?",
        ["run_graph_only_test"],
    ).fetchone()[0] == 3
    assert connection.execute(
        "SELECT COUNT(*) FROM relation_observation WHERE run_id = ?",
        ["run_graph_only_test"],
    ).fetchone()[0] == 3
    assert connection.execute(
        "SELECT COUNT(*) FROM temporal_edge_state WHERE run_id = ?",
        ["run_graph_only_test"],
    ).fetchone()[0] == 3
    assert connection.execute(
        "SELECT COUNT(*) FROM consensus_theme_candidate WHERE run_id = ?",
        ["run_graph_only_test"],
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM frontend_snapshot_cache WHERE run_id = ?",
        ["run_graph_only_test"],
    ).fetchone()[0] == 0
