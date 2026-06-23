from __future__ import annotations

import duckdb

from stocknetv2.infrastructure.db.schema_manager import SchemaManager


def _column_names(connection: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = connection.execute(f"DESCRIBE {table_name}").fetchall()
    return {row[0] for row in rows}


def test_schema_manager_creates_required_t1_tables(tmp_path):
    database_path = tmp_path / "stocknetv2.duckdb"
    connection = duckdb.connect(str(database_path))

    manager = SchemaManager(connection)
    manager.initialize()

    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert {
        "config_registry",
        "theme_discovery_run",
        "input_lineage",
        "graph_snapshot",
        "graph_edge_summary",
        "graph_layer_diagnostic",
        "relation_observation",
        "graph_edges_thresholded",
        "temporal_edge_state",
        "layer_community",
        "layer_community_membership",
        "consensus_theme_candidate",
        "theme_membership",
        "theme_semantic_label",
        "theme_path_lifecycle",
        "theme_level_flow_series",
        "frontend_snapshot_cache",
    }.issubset(tables)


def test_schema_manager_includes_requested_schema_extensions(tmp_path):
    database_path = tmp_path / "stocknetv2.duckdb"
    connection = duckdb.connect(str(database_path))

    manager = SchemaManager(connection)
    manager.initialize()

    edge_columns = _column_names(connection, "graph_edges_thresholded")
    assert {"run_id", "trade_date"}.issubset(edge_columns)

    diagnostic_columns = _column_names(connection, "graph_layer_diagnostic")
    assert {"average_degree", "largest_component_ratio", "market_mode_member_ratio"}.issubset(diagnostic_columns)

    theme_columns = _column_names(connection, "consensus_theme_candidate")
    assert "theme_quality_breakdown_json" in theme_columns

    membership_columns = _column_names(connection, "theme_membership")
    assert {"snapshot_id", "theme_path_id"}.issubset(membership_columns)

    semantic_columns = _column_names(connection, "theme_semantic_label")
    assert {"semantic_metadata_json", "semantic_prompt_text", "dictionary_version"}.issubset(semantic_columns)

    lifecycle_columns = _column_names(connection, "theme_path_lifecycle")
    assert {"transition_parent_path_id", "transition_child_path_id", "transition_kind"}.issubset(lifecycle_columns)

    cache_columns = _column_names(connection, "frontend_snapshot_cache")
    assert "cache_type" in cache_columns


def test_schema_manager_creates_membership_and_lineage_support_tables(tmp_path):
    database_path = tmp_path / "stocknetv2.duckdb"
    connection = duckdb.connect(str(database_path))

    manager = SchemaManager(connection)
    manager.initialize()

    layer_membership_columns = _column_names(connection, "layer_community_membership")
    assert {"layer_community_id", "symbol", "member_rank", "member_weight"}.issubset(layer_membership_columns)

    lineage_columns = _column_names(connection, "input_lineage")
    assert {"run_id", "snapshot_id", "source_kind", "source_version"}.issubset(lineage_columns)

    config_columns = _column_names(connection, "config_registry")
    assert {"config_id", "config_name", "config_json", "config_version"}.issubset(config_columns)

    relation_columns = _column_names(connection, "relation_observation")
    assert {
        "relation_observation_id",
        "raw_score",
        "support_points",
        "temporal_policy_id",
        "calculation_backend",
    }.issubset(relation_columns)

    temporal_columns = _column_names(connection, "temporal_edge_state")
    assert {"temporal_edge_state_id", "temporal_score", "presence_count", "missing_frames", "state"}.issubset(
        temporal_columns
    )
