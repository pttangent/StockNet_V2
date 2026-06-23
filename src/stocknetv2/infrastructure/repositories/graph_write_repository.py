from __future__ import annotations

import json

import duckdb
import pandas as pd

from stocknetv2.application.services.temporal_edge_replay_service import TemporalEdgeState
from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge


class GraphWriteRepository:
    """Persist graph-layer edges, summaries, communities, and memberships."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def save_layer_outputs(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        snapshot_time: pd.Timestamp,
        config_id: str,
        layer_edges: dict[str, list[GraphEdge]],
        layer_communities: dict[str, list[Community]],
        universe_symbol_count: int | None = None,
    ) -> None:
        for layer_name, edges in layer_edges.items():
            self._write_edge_summary(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                layer_name=layer_name,
                edges=edges,
            )
            self._write_relation_observations(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                snapshot_time=snapshot_time,
                edges=edges,
            )
            self._write_edges(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                snapshot_time=snapshot_time,
                config_id=config_id,
                edges=edges,
            )
            self._write_layer_communities(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                layer_name=layer_name,
                communities=layer_communities.get(layer_name, []),
                edges=edges,
            )
            self._write_graph_diagnostic(
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                layer_name=layer_name,
                communities=layer_communities.get(layer_name, []),
                edges=edges,
                universe_symbol_count=universe_symbol_count,
            )

    def save_temporal_edge_states(self, *, records: list[TemporalEdgeState]) -> None:
        for record in records:
            self._connection.execute(
                """
                INSERT INTO temporal_edge_state (
                    temporal_edge_state_id, relation_observation_id, run_id, snapshot_id, trade_date, timestamp,
                    graph_layer, source_symbol, target_symbol, raw_score, temporal_score, support_points,
                    effective_lookback_minutes, presence_count, age_frames, missing_frames, entered_at,
                    last_seen_at, state, temporal_policy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.temporal_edge_state_id,
                    record.relation_observation_id,
                    record.run_id,
                    record.snapshot_id,
                    record.trade_date,
                    record.timestamp,
                    record.graph_layer,
                    record.source_symbol,
                    record.target_symbol,
                    record.raw_score,
                    record.temporal_score,
                    record.support_points,
                    record.effective_lookback_minutes,
                    record.presence_count,
                    record.age_frames,
                    record.missing_frames,
                    record.entered_at,
                    record.last_seen_at,
                    record.state,
                    record.temporal_policy_id,
                ],
            )

    def _write_edge_summary(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        layer_name: str,
        edges: list[GraphEdge],
    ) -> None:
        weights = [edge.weight for edge in edges]
        node_count = len({symbol for edge in edges for symbol in (edge.source_symbol, edge.target_symbol)})
        self._connection.execute(
            """
            INSERT INTO graph_edge_summary (
                run_id, snapshot_id, trade_date, graph_layer, edge_count, node_count,
                avg_weight, median_weight, p90_weight, threshold, top_k_per_symbol, effective_lookback_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                snapshot_id,
                trade_date,
                layer_name,
                len(edges),
                node_count,
                (sum(weights) / len(weights)) if weights else 0.0,
                _median(weights),
                _percentile(weights, 0.9),
                None,
                None,
                next((edge.effective_lookback_minutes for edge in edges if edge.effective_lookback_minutes is not None), None),
            ],
        )

    def _write_edges(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        snapshot_time: pd.Timestamp,
        config_id: str,
        edges: list[GraphEdge],
    ) -> None:
        for edge in edges:
            self._connection.execute(
                """
                INSERT INTO graph_edges_thresholded (
                    run_id, snapshot_id, trade_date, timestamp, graph_layer,
                    source_symbol, target_symbol, edge_type, weight, raw_score,
                    edge_confidence, effective_lookback_minutes, window_start, window_end, support_points, config_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    snapshot_id,
                    trade_date,
                    snapshot_time,
                    edge.graph_layer,
                    edge.source_symbol,
                    edge.target_symbol,
                    edge.edge_type,
                    edge.weight,
                    edge.raw_score,
                    edge.edge_confidence,
                    edge.effective_lookback_minutes,
                    None,
                    snapshot_time,
                    edge.support_points,
                    config_id,
                ],
            )

    def _write_relation_observations(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        snapshot_time: pd.Timestamp,
        edges: list[GraphEdge],
    ) -> None:
        for edge in edges:
            self._connection.execute(
                """
                INSERT INTO relation_observation (
                    relation_observation_id, run_id, snapshot_id, trade_date, timestamp, graph_layer,
                    relation_type, source_symbol, target_symbol, raw_score, edge_weight, edge_confidence,
                    calculation_backend, support_points, effective_lookback_minutes, window_start, window_end, temporal_policy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    _relation_observation_id(snapshot_id, edge.graph_layer, edge.source_symbol, edge.target_symbol),
                    run_id,
                    snapshot_id,
                    trade_date,
                    snapshot_time,
                    edge.graph_layer,
                    edge.edge_type,
                    edge.source_symbol,
                    edge.target_symbol,
                    edge.raw_score,
                    edge.weight,
                    edge.edge_confidence,
                    edge.calculation_backend,
                    edge.support_points,
                    edge.effective_lookback_minutes,
                    None,
                    snapshot_time,
                    "raw_snapshot_v1",
                ],
            )

    def _write_layer_communities(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        layer_name: str,
        communities: list[Community],
        edges: list[GraphEdge],
    ) -> None:
        for index, community in enumerate(communities, start=1):
            layer_community_id = f"{snapshot_id}_{layer_name}_community_{index:03d}"
            member_set = set(community.members)
            community_edges = [
                edge
                for edge in edges
                if edge.source_symbol in member_set and edge.target_symbol in member_set
            ]
            weights = [edge.weight for edge in community_edges]
            possible_edges = max(len(community.members) * (len(community.members) - 1) / 2, 1)
            self._connection.execute(
                """
                INSERT INTO layer_community (
                    layer_community_id, run_id, snapshot_id, trade_date, graph_layer, community_local_id,
                    members_json, member_count, edge_count, edge_density, avg_weight, min_weight, max_weight, community_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    layer_community_id,
                    run_id,
                    snapshot_id,
                    trade_date,
                    layer_name,
                    f"C{index:03d}",
                    json.dumps(community.members),
                    len(community.members),
                    len(community_edges),
                    len(community_edges) / possible_edges,
                    (sum(weights) / len(weights)) if weights else 0.0,
                    min(weights) if weights else 0.0,
                    max(weights) if weights else 0.0,
                    community.method,
                ],
            )
            for member_rank, symbol in enumerate(community.members, start=1):
                self._connection.execute(
                    """
                    INSERT INTO layer_community_membership (
                        layer_community_id, run_id, snapshot_id, trade_date, graph_layer,
                        community_local_id, symbol, member_rank, member_weight
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        layer_community_id,
                        run_id,
                        snapshot_id,
                        trade_date,
                        layer_name,
                        f"C{index:03d}",
                        symbol,
                        member_rank,
                        1.0,
                    ],
                )

    def _write_graph_diagnostic(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        layer_name: str,
        communities: list[Community],
        edges: list[GraphEdge],
        universe_symbol_count: int | None,
    ) -> None:
        active_symbols = sorted({symbol for edge in edges for symbol in (edge.source_symbol, edge.target_symbol)})
        degrees = _build_degree_counts(edges)
        degree_values = list(degrees.values())
        edge_weights = [edge.weight for edge in edges]
        support_points = [float(edge.support_points) for edge in edges]
        community_sizes = [len(community.members) for community in communities]
        market_mode_members = sum(len(community.members) for community in communities if community.is_market_mode)
        denominator = float(universe_symbol_count or len(active_symbols) or 1)
        largest_component_ratio = (
            max((len(community.members) for community in communities), default=0) / float(len(active_symbols) or 1)
        )
        self._connection.execute(
            """
            INSERT INTO graph_layer_diagnostic (
                run_id, snapshot_id, trade_date, graph_layer, active_node_count, edge_count,
                average_degree, degree_p50, degree_p95, max_degree, edge_score_p50, edge_score_p90,
                support_points_p50, support_points_p90, connected_component_count, largest_component_ratio,
                community_count, community_size_p50, community_size_p95, community_size_max,
                market_mode_member_ratio, community_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                snapshot_id,
                trade_date,
                layer_name,
                len(active_symbols),
                len(edges),
                (sum(degree_values) / len(degree_values)) if degree_values else 0.0,
                _percentile(degree_values, 0.5),
                _percentile(degree_values, 0.95),
                max(degree_values) if degree_values else 0,
                _percentile(edge_weights, 0.5),
                _percentile(edge_weights, 0.9),
                _percentile(support_points, 0.5),
                _percentile(support_points, 0.9),
                len(communities),
                largest_component_ratio,
                len(communities),
                _percentile(community_sizes, 0.5),
                _percentile(community_sizes, 0.95),
                max(community_sizes) if community_sizes else 0,
                market_mode_members / denominator,
                communities[0].method if communities else None,
            ],
        )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1)
    return ordered[index]


def _build_degree_counts(edges: list[GraphEdge]) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for edge in edges:
        degrees[edge.source_symbol] = degrees.get(edge.source_symbol, 0) + 1
        degrees[edge.target_symbol] = degrees.get(edge.target_symbol, 0) + 1
    return degrees


def _relation_observation_id(snapshot_id: str, graph_layer: str, source_symbol: str, target_symbol: str) -> str:
    return f"{snapshot_id}_{graph_layer}_{source_symbol}_{target_symbol}"
