from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import duckdb
import pandas as pd

from stocknetv2.application.services.symbol_metadata_service import (
    empty_symbol_metadata_frame,
    read_symbol_metadata_csv,
)


_BENCHMARK_PROXY_PRICE_METHOD = "dollar_volume_over_volume"
_ALPHA_FACTOR_COLUMNS_BY_LAYER = {
    "volume_expansion_graph": [
        "edge_density_feature",
        "community_avg_weight_feature",
        "feature_coverage_ratio",
        "community_quality_score",
        "community_mean_volume_z_12",
    ],
    "flow_alignment_graph": [
        "community_member_count",
        "flow_member_count_z",
        "flow_layer_participation_ratio",
        "flow_breadth_expansion",
        "community_mean_flow_impulse_score",
        "community_quality_score",
    ],
    "dtw_trade_flow_similarity_graph": [
        "community_mean_volume_z_12",
        "community_avg_weight_feature",
        "edge_density_feature",
        "community_quality_score",
    ],
    "dtw_return_similarity_graph": [
        "edge_density_feature",
        "community_avg_weight_feature",
        "community_quality_score",
    ],
    "return_corr_graph": [
        "community_member_count",
        "edge_density_feature",
        "community_quality_score",
    ],
    "large_trade_alignment_graph": [
        "community_avg_weight_feature",
        "positive_large_trade_breadth",
        "community_quality_score",
    ],
}
_ALPHA_LABEL_VARIANTS = [
    ("equal_weight", "community_equal_weight_excess_future_ret"),
    ("member_weight", "community_member_weight_excess_future_ret"),
    ("top5_member", "community_top5_member_excess_future_ret"),
    ("top10_member", "community_top10_member_excess_future_ret"),
    ("core_weighted", "community_core_weighted_excess_future_ret"),
]
_LAYER_RESEARCH_ROLES = {
    "volume_expansion_graph": "theme_candidate_layer",
    "flow_alignment_graph": "event_alignment_layer",
    "return_corr_graph": "beta_context_layer",
    "dtw_trade_flow_similarity_graph": "pair_flow_leadlag_candidate",
    "dtw_return_similarity_graph": "weak_pair_candidate",
    "large_trade_alignment_graph": "sparse_event_flag",
}


@dataclass(frozen=True)
class GraphEvaluationPackConfig:
    graph_database_path: Path | str
    market_database_path: Path | str
    output_dir: Path | str
    metadata_csv_path: Path | str | None = None
    date_start: str | None = None
    date_end: str | None = None
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    compare_graph_database_path: Path | str | None = None
    generator_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GraphEvaluationPackSummary:
    output_dir: Path
    artifact_paths: dict[str, Path]
    counts: dict[str, int]


def build_graph_evaluation_pack(
    config: GraphEvaluationPackConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> GraphEvaluationPackSummary:
    logger = log or (lambda message: None)
    graph_database_path = Path(config.graph_database_path).expanduser().resolve()
    market_database_path = Path(config.market_database_path).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    metadata_csv_path = (
        Path(config.metadata_csv_path).expanduser().resolve()
        if config.metadata_csv_path is not None
        else None
    )
    compare_graph_database_path = (
        Path(config.compare_graph_database_path).expanduser().resolve()
        if config.compare_graph_database_path is not None
        else None
    )

    if not graph_database_path.exists():
        raise FileNotFoundError(f"Graph database not found: {graph_database_path}")
    if not market_database_path.exists():
        raise FileNotFoundError(f"Market database not found: {market_database_path}")
    if metadata_csv_path is not None and not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")
    if compare_graph_database_path is not None and not compare_graph_database_path.exists():
        raise FileNotFoundError(f"Comparison graph database not found: {compare_graph_database_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    graph_output_dir = output_dir / "graph"
    market_output_dir = output_dir / "market"
    compare_output_dir = output_dir / "compare_old_vs_new"
    temp_output_dir = output_dir / ".duckdb_tmp"
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    market_output_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    if compare_graph_database_path is not None:
        compare_output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}

    connection: duckdb.DuckDBPyConnection | None = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=8")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET memory_limit='110GB'")
        connection.execute(f"SET temp_directory='{_escape_sql_literal(str(temp_output_dir))}'")
        connection.execute("SET max_temp_directory_size='280GiB'")
        _attach_database(connection, "graph_db", graph_database_path)
        _attach_database(connection, "market_db", market_database_path)
        if compare_graph_database_path is not None:
            _attach_database(connection, "compare_db", compare_graph_database_path)

        resolved_date_start, resolved_date_end = _resolve_date_range(
            connection=connection,
            configured_start=config.date_start,
            configured_end=config.date_end,
        )
        benchmark_symbols = tuple(symbol.upper() for symbol in config.benchmark_symbols if symbol)
        benchmark_list_sql = _sql_string_list(benchmark_symbols)
        primary_benchmark = benchmark_symbols[0] if benchmark_symbols else "SPY"

        logger(f"[1/8] Preparing evaluation views for {resolved_date_start} to {resolved_date_end}.")
        _create_pack_views(
            connection=connection,
            date_start=resolved_date_start,
            date_end=resolved_date_end,
            benchmark_list_sql=benchmark_list_sql,
            primary_benchmark=primary_benchmark,
            metadata_csv_path=metadata_csv_path,
        )
        trade_dates = _trade_dates(connection)
        symbol_master_frame = connection.execute(
            """
            SELECT
                symbol,
                source_symbol,
                company_name,
                sector_code,
                industry_code,
                last_price,
                rank,
                market_cap,
                exchange,
                country,
                quote_type,
                shares_outstanding,
                enterprise_value,
                currency,
                security_type,
                is_etf,
                fetch_status,
                fetch_error
            FROM pack_symbol_master
            """
        ).fetchdf()

        logger("[2/8] Exporting graph edge and diagnostic artifacts.")
        artifact_paths["all_edges"] = graph_output_dir / "all_edges"
        _copy_query_to_partitioned_parquet(
            connection,
            """
            SELECT
                e.run_id,
                e.snapshot_id,
                e.trade_date,
                e.snapshot_timestamp,
                e.snapshot_clock_code,
                e.available_minutes_since_open,
                e.graph_layer,
                e.source_symbol,
                e.target_symbol,
                e.edge_type,
                e.weight,
                e.raw_score,
                e.edge_confidence,
                e.effective_lookback_minutes,
                e.window_start,
                e.window_end,
                e.support_points,
                e.config_id
            FROM pack_edges e
            """,
            artifact_paths["all_edges"],
            trade_date_column="trade_date",
        )
        artifact_paths["snapshot_layer_diagnostics"] = graph_output_dir / "snapshot_layer_diagnostics.csv"
        _copy_query_to_csv(
            connection,
            """
            WITH node_rollup AS (
                SELECT
                    snapshot_id,
                    graph_layer,
                    COUNT(*) AS active_node_count,
                    AVG(degree) AS average_degree,
                    quantile_cont(degree, 0.50) AS degree_p50,
                    quantile_cont(degree, 0.95) AS degree_p95,
                    MAX(degree) AS max_degree
                FROM pack_node_metrics_base
                GROUP BY 1, 2
            ),
            edge_rollup AS (
                SELECT
                    snapshot_id,
                    graph_layer,
                    COUNT(*) AS edge_count,
                    AVG(weight) AS average_edge_weight,
                    quantile_cont(weight, 0.50) AS edge_weight_p50,
                    quantile_cont(weight, 0.90) AS edge_weight_p90,
                    quantile_cont(COALESCE(support_points, 0), 0.50) AS support_points_p50,
                    quantile_cont(COALESCE(support_points, 0), 0.90) AS support_points_p90
                FROM pack_edges
                GROUP BY 1, 2
            ),
            community_rollup AS (
                SELECT
                    c.snapshot_id,
                    c.graph_layer,
                    COUNT(*) AS community_count,
                    quantile_cont(c.member_count, 0.50) AS community_size_p50,
                    quantile_cont(c.member_count, 0.95) AS community_size_p95,
                    MAX(c.member_count) AS community_size_max,
                    SUM(CASE WHEN c.member_count = 1 THEN 1 ELSE 0 END) AS singleton_community_count
                FROM pack_communities c
                GROUP BY 1, 2
            )
            SELECT
                s.run_id,
                s.snapshot_id,
                ctx.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                s.graph_layer,
                s.edge_count AS summary_edge_count,
                s.node_count AS summary_node_count,
                s.avg_weight AS summary_avg_weight,
                s.median_weight AS summary_median_weight,
                s.p90_weight AS summary_p90_weight,
                s.threshold,
                s.top_k_per_symbol,
                s.effective_lookback_minutes,
                COALESCE(n.active_node_count, 0) AS active_node_count,
                COALESCE(n.average_degree, 0) AS average_degree,
                COALESCE(n.degree_p50, 0) AS degree_p50,
                COALESCE(n.degree_p95, 0) AS degree_p95,
                COALESCE(n.max_degree, 0) AS max_degree,
                COALESCE(e.average_edge_weight, 0) AS average_edge_weight,
                COALESCE(e.edge_weight_p50, 0) AS edge_weight_p50,
                COALESCE(e.edge_weight_p90, 0) AS edge_weight_p90,
                COALESCE(e.support_points_p50, 0) AS support_points_p50,
                COALESCE(e.support_points_p90, 0) AS support_points_p90,
                COALESCE(c.community_count, 0) AS community_count,
                COALESCE(c.community_size_p50, 0) AS community_size_p50,
                COALESCE(c.community_size_p95, 0) AS community_size_p95,
                COALESCE(c.community_size_max, 0) AS community_size_max,
                COALESCE(c.singleton_community_count, 0) AS singleton_community_count,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN 0
                    ELSE COALESCE(c.community_size_max, 0) * 1.0 / n.active_node_count
                END AS largest_community_ratio,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN FALSE
                    ELSE COALESCE(c.community_size_max, 0) * 1.0 / n.active_node_count >= 0.15
                END AS has_market_mode_cluster
            FROM graph_db.graph_edge_summary s
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = s.snapshot_id
            LEFT JOIN node_rollup n
                ON n.snapshot_id = s.snapshot_id
               AND n.graph_layer = s.graph_layer
            LEFT JOIN edge_rollup e
                ON e.snapshot_id = s.snapshot_id
               AND e.graph_layer = s.graph_layer
            LEFT JOIN community_rollup c
                ON c.snapshot_id = s.snapshot_id
               AND c.graph_layer = s.graph_layer
            ORDER BY ctx.trade_date, s.snapshot_id, s.graph_layer
            """,
            artifact_paths["snapshot_layer_diagnostics"],
        )

        logger("[3/8] Exporting node and community artifacts.")
        artifact_paths["node_layer_metrics"] = graph_output_dir / "node_layer_metrics"
        _copy_query_to_partitioned_parquet(
            connection,
            """
            SELECT
                n.run_id,
                n.snapshot_id,
                n.trade_date,
                n.snapshot_timestamp,
                n.snapshot_clock_code,
                n.available_minutes_since_open,
                n.graph_layer,
                n.symbol,
                n.degree,
                n.weighted_degree,
                n.avg_incident_weight,
                n.max_incident_weight,
                n.support_points_total,
                n.support_points_avg,
                n.raw_score_avg,
                n.edge_confidence_avg,
                COALESCE(m.layer_community_id, '') AS layer_community_id,
                COALESCE(m.community_local_id, '') AS community_local_id,
                COALESCE(m.community_assignment_count, 0) AS community_assignment_count,
                COALESCE(m.member_rank, 0) AS member_rank,
                COALESCE(m.member_weight, 0) AS member_weight,
                COALESCE(m.community_member_count, 0) AS community_member_count,
                sm.company_name,
                sm.sector_code,
                sm.industry_code,
                sm.market_cap
            FROM pack_node_metrics_base n
            LEFT JOIN pack_membership_lookup m
                ON m.snapshot_id = n.snapshot_id
               AND m.graph_layer = n.graph_layer
               AND m.symbol = n.symbol
            LEFT JOIN pack_symbol_master sm
                ON sm.symbol = n.symbol
            """,
            artifact_paths["node_layer_metrics"],
            trade_date_column="trade_date",
        )
        artifact_paths["community_metrics"] = graph_output_dir / "community_metrics.parquet"
        _copy_query_to_parquet(
            connection,
            """
            WITH layer_active_nodes AS (
                SELECT
                    snapshot_id,
                    graph_layer,
                    COUNT(*) AS active_node_count
                FROM pack_node_metrics_base
                GROUP BY 1, 2
            ),
            member_market_cap AS (
                SELECT
                    m.layer_community_id,
                    AVG(sm.market_cap) AS avg_market_cap,
                    quantile_cont(sm.market_cap, 0.50) AS market_cap_p50
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                GROUP BY 1
            ),
            member_degree AS (
                SELECT
                    m.layer_community_id,
                    AVG(n.degree) AS avg_member_degree,
                    quantile_cont(n.degree, 0.50) AS degree_p50,
                    quantile_cont(n.degree, 0.95) AS degree_p95,
                    SUM(n.weighted_degree) AS community_weighted_degree
                FROM pack_memberships m
                LEFT JOIN pack_node_metrics_base n
                    ON n.snapshot_id = m.snapshot_id
                   AND n.graph_layer = m.graph_layer
                   AND n.symbol = m.symbol
                GROUP BY 1
            ),
            member_metadata_coverage AS (
                SELECT
                    m.layer_community_id,
                    SUM(
                        CASE
                            WHEN sm.sector_code IS NOT NULL AND UPPER(TRIM(sm.sector_code)) <> 'UNKNOWN' THEN 1
                            ELSE 0
                        END
                    ) AS known_sector_member_count,
                    SUM(
                        CASE
                            WHEN sm.industry_code IS NOT NULL AND UPPER(TRIM(sm.industry_code)) <> 'UNKNOWN' THEN 1
                            ELSE 0
                        END
                    ) AS known_industry_member_count,
                    SUM(
                        CASE
                            WHEN sm.market_cap IS NOT NULL AND sm.market_cap > 0 THEN 1
                            ELSE 0
                        END
                    ) AS known_market_cap_member_count
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                GROUP BY 1
            ),
            sector_rank AS (
                SELECT
                    m.layer_community_id,
                    sm.sector_code AS sector_code,
                    COUNT(*) AS sector_member_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.layer_community_id
                        ORDER BY COUNT(*) DESC, sm.sector_code
                    ) AS sector_rank
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                WHERE sm.sector_code IS NOT NULL
                  AND UPPER(TRIM(sm.sector_code)) <> 'UNKNOWN'
                GROUP BY 1, 2
            ),
            industry_rank AS (
                SELECT
                    m.layer_community_id,
                    sm.industry_code AS industry_code,
                    COUNT(*) AS industry_member_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.layer_community_id
                        ORDER BY COUNT(*) DESC, sm.industry_code
                    ) AS industry_rank
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                WHERE sm.industry_code IS NOT NULL
                  AND UPPER(TRIM(sm.industry_code)) <> 'UNKNOWN'
                GROUP BY 1, 2
            )
            SELECT
                c.layer_community_id,
                c.run_id,
                c.snapshot_id,
                c.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                c.graph_layer,
                c.community_local_id,
                c.member_count,
                c.edge_count,
                c.edge_density,
                c.avg_weight,
                c.min_weight,
                c.max_weight,
                c.community_method,
                COALESCE(n.active_node_count, 0) AS active_node_count,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN 0
                    ELSE c.member_count * 1.0 / n.active_node_count
                END AS layer_member_ratio,
                COALESCE(d.avg_member_degree, 0) AS avg_member_degree,
                COALESCE(d.degree_p50, 0) AS degree_p50,
                COALESCE(d.degree_p95, 0) AS degree_p95,
                COALESCE(d.community_weighted_degree, 0) AS community_weighted_degree,
                COALESCE(mc.avg_market_cap, 0) AS avg_market_cap,
                COALESCE(mc.market_cap_p50, 0) AS market_cap_p50,
                COALESCE(mm.known_sector_member_count, 0) AS known_sector_member_count,
                CASE
                    WHEN c.member_count = 0 THEN 0
                    ELSE COALESCE(mm.known_sector_member_count, 0) * 1.0 / c.member_count
                END AS known_sector_ratio,
                COALESCE(sr.sector_code, 'UNKNOWN') AS top_sector,
                CASE
                    WHEN COALESCE(mm.known_sector_member_count, 0) = 0 THEN 0
                    ELSE COALESCE(sr.sector_member_count, 0) * 1.0 / mm.known_sector_member_count
                END AS top_sector_ratio,
                COALESCE(mm.known_industry_member_count, 0) AS known_industry_member_count,
                CASE
                    WHEN c.member_count = 0 THEN 0
                    ELSE COALESCE(mm.known_industry_member_count, 0) * 1.0 / c.member_count
                END AS known_industry_ratio,
                COALESCE(ir.industry_code, 'UNKNOWN') AS top_industry,
                CASE
                    WHEN COALESCE(mm.known_industry_member_count, 0) = 0 THEN 0
                    ELSE COALESCE(ir.industry_member_count, 0) * 1.0 / mm.known_industry_member_count
                END AS top_industry_ratio,
                COALESCE(mm.known_market_cap_member_count, 0) AS known_market_cap_member_count,
                CASE
                    WHEN c.member_count = 0 THEN 0
                    ELSE COALESCE(mm.known_market_cap_member_count, 0) * 1.0 / c.member_count
                END AS known_market_cap_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY c.snapshot_id, c.graph_layer
                    ORDER BY c.member_count DESC, COALESCE(c.avg_weight, 0) DESC, c.layer_community_id
                ) AS size_rank_in_layer,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN FALSE
                    ELSE c.member_count * 1.0 / n.active_node_count >= 0.15
                END AS is_market_mode_community
            FROM pack_communities c
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = c.snapshot_id
            LEFT JOIN layer_active_nodes n
                ON n.snapshot_id = c.snapshot_id
               AND n.graph_layer = c.graph_layer
            LEFT JOIN member_market_cap mc
                ON mc.layer_community_id = c.layer_community_id
            LEFT JOIN member_degree d
                ON d.layer_community_id = c.layer_community_id
            LEFT JOIN member_metadata_coverage mm
                ON mm.layer_community_id = c.layer_community_id
            LEFT JOIN sector_rank sr
                ON sr.layer_community_id = c.layer_community_id
               AND sr.sector_rank = 1
            LEFT JOIN industry_rank ir
                ON ir.layer_community_id = c.layer_community_id
               AND ir.industry_rank = 1
            """,
            artifact_paths["community_metrics"],
        )
        artifact_paths["community_membership"] = graph_output_dir / "community_membership.parquet"
        _copy_query_to_parquet(
            connection,
            """
            WITH member_metrics AS (
                SELECT
                    m.layer_community_id,
                    m.run_id,
                    m.snapshot_id,
                    m.trade_date,
                    m.graph_layer,
                    m.community_local_id,
                    m.symbol,
                    m.member_rank,
                    m.member_weight,
                    COALESCE(n.weighted_degree, 0.0) AS weighted_degree,
                    COALESCE(n.avg_incident_weight, 0.0) AS avg_incident_weight,
                    COALESCE(n.support_points_avg, 0.0) AS support_points_avg,
                    COALESCE(n.edge_confidence_avg, 0.0) AS edge_confidence_avg
                FROM pack_memberships m
                LEFT JOIN pack_node_metrics_base n
                    ON n.snapshot_id = m.snapshot_id
                   AND n.graph_layer = m.graph_layer
                   AND n.symbol = m.symbol
            ),
            community_metric_max AS (
                SELECT
                    layer_community_id,
                    MAX(weighted_degree) AS max_weighted_degree,
                    MAX(avg_incident_weight) AS max_avg_incident_weight,
                    MAX(support_points_avg) AS max_support_points_avg,
                    MAX(edge_confidence_avg) AS max_edge_confidence_avg
                FROM member_metrics
                GROUP BY 1
            )
            SELECT
                mm.layer_community_id,
                mm.run_id,
                mm.snapshot_id,
                mm.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                mm.graph_layer,
                mm.community_local_id,
                mm.symbol,
                mm.member_rank,
                mm.member_weight,
                c.member_count AS community_member_count,
                c.edge_count AS community_edge_count,
                c.edge_density,
                c.avg_weight AS community_avg_weight,
                mm.weighted_degree,
                mm.avg_incident_weight,
                mm.support_points_avg,
                mm.edge_confidence_avg,
                CASE
                    WHEN COALESCE(c.member_count, 0) <= 0 THEN 0.0
                    ELSE (COALESCE(c.member_count, 0) - COALESCE(mm.member_rank, 0) + 1) * 1.0 / c.member_count
                END AS member_rank_score,
                (
                    CASE
                        WHEN COALESCE(c.member_count, 0) <= 0 THEN 0.0
                        ELSE (COALESCE(c.member_count, 0) - COALESCE(mm.member_rank, 0) + 1) * 1.0 / c.member_count
                    END
                    + CASE
                        WHEN COALESCE(cm.max_weighted_degree, 0) <= 0 THEN 0.0
                        ELSE mm.weighted_degree / cm.max_weighted_degree
                    END
                    + CASE
                        WHEN COALESCE(cm.max_avg_incident_weight, 0) <= 0 THEN 0.0
                        ELSE mm.avg_incident_weight / cm.max_avg_incident_weight
                    END
                    + CASE
                        WHEN COALESCE(cm.max_support_points_avg, 0) <= 0 THEN 0.0
                        ELSE mm.support_points_avg / cm.max_support_points_avg
                    END
                    + CASE
                        WHEN COALESCE(cm.max_edge_confidence_avg, 0) <= 0 THEN 0.0
                        ELSE mm.edge_confidence_avg / cm.max_edge_confidence_avg
                    END
                ) / 5.0 AS member_core_score,
                sm.company_name,
                sm.sector_code,
                sm.industry_code,
                sm.market_cap,
                sm.exchange,
                sm.country,
                sm.quote_type
            FROM member_metrics mm
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = mm.snapshot_id
            LEFT JOIN pack_communities c
                ON c.layer_community_id = mm.layer_community_id
            LEFT JOIN community_metric_max cm
                ON cm.layer_community_id = mm.layer_community_id
            LEFT JOIN pack_symbol_master sm
                ON sm.symbol = mm.symbol
            """,
            artifact_paths["community_membership"],
        )
        artifact_paths["community_member_symbols"] = graph_output_dir / "community_member_symbols.csv"
        _copy_query_to_csv(
            connection,
            """
            SELECT
                m.layer_community_id,
                m.run_id,
                m.snapshot_id,
                m.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                m.graph_layer,
                m.community_local_id,
                MAX(c.member_count) AS member_count,
                string_agg(
                    m.symbol,
                    ',' ORDER BY COALESCE(m.member_rank, 999999), m.symbol
                ) AS member_symbols
            FROM pack_memberships m
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = m.snapshot_id
            LEFT JOIN pack_communities c
                ON c.layer_community_id = m.layer_community_id
            GROUP BY
                m.layer_community_id,
                m.run_id,
                m.snapshot_id,
                m.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                m.graph_layer,
                m.community_local_id
            ORDER BY
                m.trade_date,
                ctx.snapshot_clock_code,
                m.graph_layer,
                member_count DESC,
                m.layer_community_id
            """,
            artifact_paths["community_member_symbols"],
        )
        artifact_paths["metadata_coverage_report"] = graph_output_dir / "metadata_coverage_report.csv"
        _copy_query_to_csv(
            connection,
            """
            WITH membership_metadata AS (
                SELECT
                    m.symbol,
                    sm.sector_code,
                    sm.industry_code,
                    sm.market_cap,
                    sm.exchange,
                    sm.country,
                    sm.quote_type
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
            )
            SELECT
                COUNT(*) AS membership_rows,
                COUNT(DISTINCT symbol) AS active_symbol_count,
                AVG(CASE WHEN sector_code IS NOT NULL AND UPPER(TRIM(sector_code)) <> 'UNKNOWN' THEN 1.0 ELSE 0.0 END) AS sector_coverage_ratio,
                AVG(CASE WHEN industry_code IS NOT NULL AND UPPER(TRIM(industry_code)) <> 'UNKNOWN' THEN 1.0 ELSE 0.0 END) AS industry_coverage_ratio,
                AVG(CASE WHEN market_cap IS NOT NULL AND market_cap > 0 THEN 1.0 ELSE 0.0 END) AS market_cap_coverage_ratio,
                AVG(CASE WHEN exchange IS NOT NULL AND TRIM(exchange) <> '' THEN 1.0 ELSE 0.0 END) AS exchange_coverage_ratio,
                AVG(CASE WHEN country IS NOT NULL AND TRIM(country) <> '' THEN 1.0 ELSE 0.0 END) AS country_coverage_ratio,
                AVG(CASE WHEN quote_type IS NOT NULL AND TRIM(quote_type) <> '' THEN 1.0 ELSE 0.0 END) AS quote_type_coverage_ratio
            FROM membership_metadata
            """,
            artifact_paths["metadata_coverage_report"],
        )
        artifact_paths["layer_review_candidates"] = graph_output_dir / "layer_review_candidates.csv"
        _copy_query_to_csv(
            connection,
            """
            WITH community_metrics AS (
                SELECT * FROM read_parquet($path)
            )
            SELECT
                layer_community_id,
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                graph_layer,
                community_local_id,
                member_count,
                edge_count,
                edge_density,
                avg_weight,
                layer_member_ratio,
                avg_member_degree,
                known_sector_ratio,
                top_sector,
                top_sector_ratio,
                known_industry_ratio,
                top_industry,
                top_industry_ratio,
                is_market_mode_community,
                (
                    COALESCE(avg_weight, 0)
                    * LN(member_count + 1.0)
                    * CASE
                        WHEN layer_member_ratio >= 0.50 THEN 0.20
                        WHEN layer_member_ratio >= 0.15 THEN 0.50
                        ELSE 1.00
                    END
                ) AS review_priority_score,
                CASE
                    WHEN is_market_mode_community THEN 'market_mode'
                    WHEN member_count >= 50 THEN 'large_cluster'
                    WHEN known_sector_ratio >= 0.80 AND top_sector_ratio >= 0.60 THEN 'sector_concentrated'
                    ELSE 'balanced_cluster'
                END AS review_reason
            FROM community_metrics
            ORDER BY review_priority_score DESC, member_count DESC, snapshot_id, graph_layer, community_local_id
            """,
            artifact_paths["layer_review_candidates"],
            parameters={"path": str(artifact_paths["community_metrics"])},
        )

        logger("[4/8] Exporting optional comparison artifacts.")
        if compare_graph_database_path is not None:
            artifact_paths["compare_layer_edge_summary"] = compare_output_dir / "layer_edge_summary_compare.csv"
            _copy_query_to_csv(
                connection,
                f"""
                WITH current_edges AS (
                    SELECT graph_layer, COUNT(*) AS edge_count
                    FROM pack_edges
                    GROUP BY 1
                ),
                baseline_edges AS (
                    SELECT graph_layer, COUNT(*) AS edge_count
                    FROM compare_db.graph_edges_thresholded
                    WHERE trade_date BETWEEN DATE '{resolved_date_start}' AND DATE '{resolved_date_end}'
                    GROUP BY 1
                )
                SELECT
                    COALESCE(b.graph_layer, c.graph_layer) AS graph_layer,
                    COALESCE(b.edge_count, 0) AS baseline_edge_count,
                    COALESCE(c.edge_count, 0) AS current_edge_count,
                    COALESCE(c.edge_count, 0) - COALESCE(b.edge_count, 0) AS edge_count_delta,
                    CASE
                        WHEN COALESCE(b.edge_count, 0) = 0 THEN NULL
                        ELSE COALESCE(c.edge_count, 0) * 1.0 / b.edge_count - 1.0
                    END AS edge_count_delta_ratio
                FROM baseline_edges b
                FULL OUTER JOIN current_edges c
                    ON c.graph_layer = b.graph_layer
                ORDER BY graph_layer
                """,
                artifact_paths["compare_layer_edge_summary"],
            )
            artifact_paths["compare_layer_community_distribution"] = compare_output_dir / "layer_community_distribution_compare.csv"
            _copy_query_to_csv(
                connection,
                f"""
                WITH baseline AS (
                    SELECT
                        graph_layer,
                        COUNT(*) AS community_count,
                        AVG(member_count) AS avg_member_count,
                        quantile_cont(member_count, 0.50) AS member_count_p50,
                        quantile_cont(member_count, 0.95) AS member_count_p95,
                        MAX(member_count) AS member_count_max
                    FROM compare_db.layer_community
                    WHERE trade_date BETWEEN DATE '{resolved_date_start}' AND DATE '{resolved_date_end}'
                    GROUP BY 1
                ),
                current AS (
                    SELECT
                        graph_layer,
                        COUNT(*) AS community_count,
                        AVG(member_count) AS avg_member_count,
                        quantile_cont(member_count, 0.50) AS member_count_p50,
                        quantile_cont(member_count, 0.95) AS member_count_p95,
                        MAX(member_count) AS member_count_max
                    FROM pack_communities
                    GROUP BY 1
                )
                SELECT
                    COALESCE(b.graph_layer, c.graph_layer) AS graph_layer,
                    COALESCE(b.community_count, 0) AS baseline_community_count,
                    COALESCE(c.community_count, 0) AS current_community_count,
                    b.avg_member_count AS baseline_avg_member_count,
                    c.avg_member_count AS current_avg_member_count,
                    b.member_count_p50 AS baseline_member_count_p50,
                    c.member_count_p50 AS current_member_count_p50,
                    b.member_count_p95 AS baseline_member_count_p95,
                    c.member_count_p95 AS current_member_count_p95,
                    b.member_count_max AS baseline_member_count_max,
                    c.member_count_max AS current_member_count_max
                FROM baseline b
                FULL OUTER JOIN current c
                    ON c.graph_layer = b.graph_layer
                ORDER BY graph_layer
                """,
                artifact_paths["compare_layer_community_distribution"],
            )

        active_snapshot_key_dir = temp_output_dir / "active_symbol_snapshots"
        _copy_query_to_partitioned_parquet(
            connection,
            """
            SELECT
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                symbol
            FROM pack_active_symbol_snapshots
            """,
            active_snapshot_key_dir,
            trade_date_column="trade_date",
        )
        active_symbols = {
            row[0]
            for row in connection.execute(
                "SELECT symbol FROM pack_active_symbols"
            ).fetchall()
        }
        counts = {
            "run_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_runs"),
            "snapshot_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_snapshot_context"),
            "edge_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_edges"),
            "community_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_communities"),
            "community_membership_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_memberships"),
            "active_symbol_snapshot_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_active_symbol_snapshots"),
            "active_symbol_count": _scalar_int(connection, "SELECT COUNT(*) FROM pack_active_symbols"),
        }
        code_commits = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT code_commit
                FROM pack_runs
                WHERE code_commit IS NOT NULL
                ORDER BY code_commit
                """
            ).fetchall()
        ]
        layers = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT graph_layer
                FROM pack_edges
                ORDER BY graph_layer
                """
            ).fetchall()
        ]

        connection.close()
        connection = None

        logger("[5/8] Exporting symbol features, labels, and benchmark context.")
        market_data_root = market_database_path.parent
        artifact_paths["symbol_snapshot_features"] = market_output_dir / "symbol_snapshot_features"
        _export_symbol_snapshot_feature_shards(
            trade_dates,
            active_snapshot_key_dir,
            artifact_paths["symbol_snapshot_features"],
            symbol_master_frame,
            market_data_root,
        )
        artifact_paths["symbol_forward_labels"] = market_output_dir / "symbol_forward_labels"
        _export_symbol_forward_label_shards(
            trade_dates,
            active_snapshot_key_dir,
            artifact_paths["symbol_forward_labels"],
            primary_benchmark,
            market_data_root,
        )
        artifact_paths["benchmark_label_source_summary"] = market_output_dir / "benchmark_label_source_summary.csv"
        _export_benchmark_label_source_summary(
            artifact_paths["symbol_forward_labels"],
            artifact_paths["benchmark_label_source_summary"],
        )
        artifact_paths["community_snapshot_features"] = market_output_dir / "community_snapshot_features.parquet"
        _export_community_snapshot_features(
            artifact_paths["community_membership"],
            artifact_paths["symbol_snapshot_features"],
            artifact_paths["community_snapshot_features"],
        )
        artifact_paths["community_forward_labels"] = market_output_dir / "community_forward_labels.parquet"
        _export_community_forward_labels(
            artifact_paths["community_membership"],
            artifact_paths["symbol_forward_labels"],
            artifact_paths["community_forward_labels"],
        )
        artifact_paths["alpha_sanity_report"] = market_output_dir / "alpha_sanity_report.csv"
        _export_alpha_sanity_report(
            artifact_paths["community_snapshot_features"],
            artifact_paths["community_forward_labels"],
            artifact_paths["alpha_sanity_report"],
        )
        artifact_paths["alpha_feature_ranking_by_layer"] = market_output_dir / "alpha_feature_ranking_by_layer.csv"
        _export_alpha_feature_ranking_report(
            artifact_paths["alpha_sanity_report"],
            artifact_paths["alpha_feature_ranking_by_layer"],
        )
        artifact_paths["benchmark_series"] = market_output_dir / "benchmark_series"
        _export_benchmark_series_shards(
            trade_dates,
            artifact_paths["benchmark_series"],
            benchmark_symbols,
            market_data_root,
        )
        artifact_paths["metadata_trust_policy"] = market_output_dir / "metadata_trust_policy.json"
        _write_metadata_trust_policy(artifact_paths["metadata_trust_policy"])

        artifact_paths["symbol_master"] = market_output_dir / "symbol_master.csv"
        active_symbol_master = symbol_master_frame[symbol_master_frame["symbol"].isin(active_symbols)].copy()
        active_symbol_master.sort_values("symbol").to_csv(artifact_paths["symbol_master"], index=False)

        logger("[6/8] Writing README and manifest.")
        artifact_paths["README"] = output_dir / "README.md"
        _write_readme(
            artifact_paths["README"],
            date_start=resolved_date_start,
            date_end=resolved_date_end,
            primary_benchmark=primary_benchmark,
            compare_included=compare_graph_database_path is not None,
        )
        artifact_paths["run_manifest"] = output_dir / "run_manifest.json"
        generator_metadata = _resolve_generator_metadata(
            provided_metadata=config.generator_metadata,
            output_dir=output_dir,
        )
        _write_manifest(
            artifact_paths["run_manifest"],
            graph_database_path=graph_database_path,
            market_database_path=market_database_path,
            metadata_csv_path=metadata_csv_path,
            compare_graph_database_path=compare_graph_database_path,
            output_dir=output_dir,
            date_start=resolved_date_start,
            date_end=resolved_date_end,
            primary_benchmark=primary_benchmark,
            benchmark_symbols=benchmark_symbols,
            counts=counts,
            artifact_paths=artifact_paths,
            code_commits=code_commits,
            layers=layers,
            generator_metadata=generator_metadata,
        )
        logger("[7/8] Verifying artifact files.")
        for path in artifact_paths.values():
            if not path.exists():
                raise RuntimeError(f"Expected artifact missing after export: {path}")

        logger("[8/8] Evaluation pack is ready.")
        return GraphEvaluationPackSummary(
            output_dir=output_dir,
            artifact_paths=artifact_paths,
            counts=counts,
        )
    finally:
        if connection is not None:
            connection.close()


def _resolve_date_range(
    *,
    connection: duckdb.DuckDBPyConnection,
    configured_start: str | None,
    configured_end: str | None,
) -> tuple[str, str]:
    default_start, default_end = connection.execute(
        """
        SELECT
            CAST(MIN(trade_date) AS VARCHAR),
            CAST(MAX(trade_date) AS VARCHAR)
        FROM graph_db.graph_snapshot
        """
    ).fetchone()
    if default_start is None or default_end is None:
        raise RuntimeError("Graph database does not contain any snapshots.")
    return configured_start or default_start, configured_end or default_end


def _create_pack_views(
    *,
    connection: duckdb.DuckDBPyConnection,
    date_start: str,
    date_end: str,
    benchmark_list_sql: str,
    primary_benchmark: str,
    metadata_csv_path: Path | None,
) -> None:
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_runs AS
        SELECT *
        FROM graph_db.theme_discovery_run
        WHERE date_start <= DATE '{date_end}'
          AND date_end >= DATE '{date_start}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_snapshot_context AS
        SELECT
            snapshot_id,
            run_id,
            trade_date,
            timestamp AS snapshot_timestamp,
            frame_minutes,
            market_session,
            graph_status,
            available_minutes_since_open,
            RIGHT(snapshot_id, 4) AS snapshot_clock_code
        FROM graph_db.graph_snapshot
        WHERE trade_date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_edges AS
        SELECT
            e.run_id,
            e.snapshot_id,
            ctx.trade_date,
            ctx.snapshot_timestamp,
            ctx.snapshot_clock_code,
            ctx.available_minutes_since_open,
            e.graph_layer,
            e.source_symbol,
            e.target_symbol,
            e.edge_type,
            e.weight,
            e.raw_score,
            e.edge_confidence,
            e.effective_lookback_minutes,
            e.window_start,
            e.window_end,
            e.support_points,
            e.config_id
        FROM graph_db.graph_edges_thresholded e
        JOIN pack_snapshot_context ctx
            ON ctx.snapshot_id = e.snapshot_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_communities AS
        SELECT c.*
        FROM graph_db.layer_community c
        JOIN pack_snapshot_context ctx
            ON ctx.snapshot_id = c.snapshot_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_memberships AS
        SELECT m.*
        FROM graph_db.layer_community_membership m
        JOIN pack_snapshot_context ctx
            ON ctx.snapshot_id = m.snapshot_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_node_metrics_base AS
        WITH incident_edges AS (
            SELECT
                run_id,
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                graph_layer,
                source_symbol AS symbol,
                weight,
                raw_score,
                edge_confidence,
                support_points
            FROM pack_edges
            UNION ALL
            SELECT
                run_id,
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                graph_layer,
                target_symbol AS symbol,
                weight,
                raw_score,
                edge_confidence,
                support_points
            FROM pack_edges
        )
        SELECT
            run_id,
            snapshot_id,
            trade_date,
            snapshot_timestamp,
            snapshot_clock_code,
            available_minutes_since_open,
            graph_layer,
            symbol,
            COUNT(*) AS degree,
            SUM(weight) AS weighted_degree,
            AVG(weight) AS avg_incident_weight,
            MAX(weight) AS max_incident_weight,
            SUM(COALESCE(support_points, 0)) AS support_points_total,
            AVG(COALESCE(support_points, 0)) AS support_points_avg,
            AVG(COALESCE(raw_score, weight)) AS raw_score_avg,
            AVG(COALESCE(edge_confidence, 1.0)) AS edge_confidence_avg
        FROM incident_edges
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_membership_lookup AS
        SELECT
            m.snapshot_id,
            m.graph_layer,
            m.symbol,
            ANY_VALUE(m.layer_community_id) AS layer_community_id,
            ANY_VALUE(m.community_local_id) AS community_local_id,
            COUNT(*) AS community_assignment_count,
            MIN(COALESCE(m.member_rank, 0)) AS member_rank,
            AVG(COALESCE(m.member_weight, 0)) AS member_weight,
            MAX(COALESCE(c.member_count, 0)) AS community_member_count
        FROM pack_memberships m
        LEFT JOIN pack_communities c
            ON c.layer_community_id = m.layer_community_id
        GROUP BY 1, 2, 3
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_active_symbol_snapshots AS
        SELECT DISTINCT
            snapshot_id,
            trade_date,
            snapshot_timestamp,
            snapshot_clock_code,
            available_minutes_since_open,
            symbol
        FROM pack_node_metrics_base
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_active_symbols AS
        SELECT DISTINCT symbol
        FROM pack_active_symbol_snapshots
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_features AS
        SELECT *
        FROM market_db.features_1m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_labels AS
        SELECT *
        FROM market_db.labels_1m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_trade_flow AS
        SELECT *
        FROM market_db.trade_flow_1m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_bars_5m AS
        SELECT *
        FROM market_db.bars_5m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )

    if metadata_csv_path is None:
        symbol_master_frame = empty_symbol_metadata_frame()
    else:
        symbol_master_frame = read_symbol_metadata_csv(metadata_csv_path)
    connection.register("pack_symbol_master_frame", symbol_master_frame)
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_symbol_master AS
        SELECT *
        FROM pack_symbol_master_frame
        WHERE symbol IS NOT NULL
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_benchmarks_in_scope AS
        SELECT DISTINCT symbol
        FROM pack_market_bars_5m
        WHERE symbol IN ({benchmark_list_sql})
        """
    )


def _write_readme(
    path: Path,
    *,
    date_start: str,
    date_end: str,
    primary_benchmark: str,
    compare_included: bool,
) -> None:
    compare_note = (
        "- `compare_old_vs_new/`: baseline-vs-current structural comparison for the same month.\n"
        if compare_included
        else ""
    )
    content = (
        "# Graph Evaluation Pack\n\n"
        f"Date range: `{date_start}` to `{date_end}`\n\n"
        "Status: `Graph evaluation artifact ready for manual review`\n"
        "Quality gate: `Not sufficient by itself to approve theme discovery, lifecycle analysis, or backtesting`\n\n"
        "See `ASSESSMENT.md` if a month-specific research conclusion has been written.\n\n"
        "This pack is designed for manual graph-quality and financial-meaning review of the first month.\n"
        "It does not depend on rerunning the full T1 theme pipeline; instead it reconstructs evaluation context from the monthly graph-build database plus the market database.\n\n"
        "## Start Here\n\n"
        "1. Open `graph/layer_review_candidates.csv`.\n"
        "2. Use `graph/community_member_symbols.csv` for a fast CSV roster of each community, then `graph/community_metrics.parquet` and `graph/community_membership.parquet` to inspect whether large communities are real themes, sector baskets, or market-mode clusters.\n"
        "3. Use `market/symbol_snapshot_features/` to inspect the causality-safe state of each member at the snapshot.\n"
        f"4. Use `market/symbol_forward_labels/` to check whether members outperformed `{primary_benchmark}` over the next 1m/5m/15m/30m windows.\n"
        "5. Use `market/community_snapshot_features.parquet`, `market/community_forward_labels.parquet`, `market/alpha_sanity_report.csv`, `market/alpha_feature_ranking_by_layer.csv`, and `market/benchmark_label_source_summary.csv` for the first community-level alpha sanity pass.\n"
        "6. Use `graph/snapshot_layer_diagnostics.csv` to find pathological layers, giant clusters, or snapshots where one layer dominates the universe.\n\n"
        "## Time Notes\n\n"
        "- `snapshot_clock_code` is the canonical market-clock label from the snapshot id suffix.\n"
        "- `snapshot_timestamp` is the stored timestamp value from the graph database.\n"
        "- `available_minutes_since_open` is the safest field for intraday sequencing if timezone display looks inconsistent.\n"
        "- Symbol features now carry `graph_input_available_time` and trade-flow `flow_available_time`; they are only joined into a snapshot when `available_time <= snapshot_timestamp`.\n"
        "- Forward labels now carry `label_available_time` and are aligned from the last completed 1m bucket available at the snapshot, not from unfinished 1m bars.\n"
        "- Benchmark-relative labels now carry provenance: `benchmark_label_source` tells you whether they came from `labels_1m` or a `trade_flow_1m` proxy fallback.\n"
        "- Metadata is exported for post-hoc validation only; see `market/metadata_trust_policy.json` before using any field in modeling.\n\n"
        "## Files\n\n"
        "- `graph/all_edges/`: thresholded graph edges, sharded by trade date as parquet.\n"
        "- `graph/snapshot_layer_diagnostics.csv`: per-snapshot, per-layer structure diagnostics.\n"
        "- `graph/node_layer_metrics/`: per-symbol, per-layer node metrics, sharded by trade date as parquet.\n"
        "- `graph/community_metrics.parquet`: community-level structure and concentration metrics.\n"
        "- `graph/community_membership.parquet`: member roster for each community.\n"
        "- `graph/community_member_symbols.csv`: one CSV row per community with the ordered member-symbol list for quick theme review.\n"
        "- `graph/layer_review_candidates.csv`: ranked shortlist for manual review.\n"
        "- `market/symbol_snapshot_features/`: snapshot-aligned symbol state features and actual graph inputs, sharded by trade date as parquet.\n"
        "- `market/symbol_forward_labels/`: causality-safe forward returns and benchmark-relative labels, sharded by trade date as parquet.\n"
        "- `market/community_snapshot_features.parquet`: community-level feature aggregates built only from snapshot-time-available symbol inputs.\n"
        "- `market/community_forward_labels.parquet`: community-level forward labels kept physically separate from features.\n"
        "- `market/alpha_sanity_report.csv`: first-pass RankIC / decile / hit-rate summary for community-level evaluation.\n"
        "- `market/alpha_feature_ranking_by_layer.csv`: per-layer factor ranking with sample-size-aware confidence buckets and research actions.\n"
        "- `market/benchmark_label_source_summary.csv`: benchmark label provenance coverage for `labels_1m` vs `trade_flow_1m` proxy fallback.\n"
        "- `market/metadata_trust_policy.json`: allowed post-hoc validation use vs modeling restrictions for metadata fields.\n"
        "- `market/symbol_master.csv`: symbol metadata used for joins.\n"
        "- `market/benchmark_series/`: benchmark bar series for context, sharded by trade date as parquet.\n"
        f"{compare_note}"
        "\n## Suggested Evaluation Questions\n\n"
        "- Do top-ranked communities have reasonable member counts, or are they still market-mode clusters?\n"
        "- Are the members concentrated in one sector or industry for an interpretable reason?\n"
        "- Do symbols inside a community share similar flow, volume, and short-horizon forward return behavior?\n"
        "- Which layers create the most false giant clusters, and at what time of day?\n"
        "- Are review-worthy communities associated with positive benchmark-relative forward returns, or only with generic market beta?\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    graph_database_path: Path,
    market_database_path: Path,
    metadata_csv_path: Path | None,
    compare_graph_database_path: Path | None,
    output_dir: Path,
    date_start: str,
    date_end: str,
    primary_benchmark: str,
    benchmark_symbols: tuple[str, ...],
    counts: dict[str, int],
    artifact_paths: dict[str, Path],
    code_commits: list[str],
    layers: list[str],
    generator_metadata: dict[str, Any],
) -> None:
    manifest = _build_manifest_payload(
        graph_database_path=graph_database_path,
        market_database_path=market_database_path,
        metadata_csv_path=metadata_csv_path,
        compare_graph_database_path=compare_graph_database_path,
        output_dir=output_dir,
        date_start=date_start,
        date_end=date_end,
        primary_benchmark=primary_benchmark,
        benchmark_symbols=benchmark_symbols,
        counts=counts,
        artifact_paths=artifact_paths,
        code_commits=code_commits,
        layers=layers,
        generator_metadata=generator_metadata,
    )
    manifest["artifacts"]["run_manifest"]["size_bytes"] = 0
    for _ in range(3):
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        actual_size = path.stat().st_size
        if manifest["artifacts"]["run_manifest"]["size_bytes"] == actual_size:
            break
        manifest["artifacts"]["run_manifest"]["size_bytes"] = actual_size
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _attach_database(connection: duckdb.DuckDBPyConnection, alias: str, database_path: Path) -> None:
    connection.execute(f"ATTACH '{_escape_sql_literal(str(database_path))}' AS {alias} (READ_ONLY)")


def _build_manifest_payload(
    *,
    graph_database_path: Path,
    market_database_path: Path,
    metadata_csv_path: Path | None,
    compare_graph_database_path: Path | None,
    output_dir: Path,
    date_start: str,
    date_end: str,
    primary_benchmark: str,
    benchmark_symbols: tuple[str, ...],
    counts: dict[str, int],
    artifact_paths: dict[str, Path],
    code_commits: list[str],
    layers: list[str],
    generator_metadata: dict[str, Any],
) -> dict[str, Any]:
    config_payload = {
        "date_start": date_start,
        "date_end": date_end,
        "primary_benchmark": primary_benchmark,
        "benchmark_symbols": list(benchmark_symbols),
        "layers": layers,
        "compare_graph_database_path": str(compare_graph_database_path) if compare_graph_database_path is not None else None,
    }
    return {
        "date_start": date_start,
        "date_end": date_end,
        "primary_benchmark": primary_benchmark,
        "benchmark_symbols": list(benchmark_symbols),
        "counts": counts,
        "code_commits": code_commits,
        "layers": layers,
        "generator": generator_metadata,
        "provenance": {
            "graph_build_commits": code_commits,
            "evaluation_pack_generator": generator_metadata,
            "config": {
                **config_payload,
                "sha256": _sha256_json(config_payload),
            },
            "inputs": {
                "graph_database": _file_provenance(graph_database_path),
                "market_database": _file_provenance(market_database_path),
                "metadata_csv": _file_provenance(metadata_csv_path),
                "compare_graph_database": _file_provenance(compare_graph_database_path),
            },
            "dependency_versions": _dependency_versions(),
        },
        "sources": {
            "graph_database_path": str(graph_database_path),
            "market_database_path": str(market_database_path),
            "metadata_csv_path": str(metadata_csv_path) if metadata_csv_path is not None else None,
            "compare_graph_database_path": str(compare_graph_database_path) if compare_graph_database_path is not None else None,
        },
        "artifacts": {
            name: {
                "path": str(path_obj.relative_to(output_dir)),
                "size_bytes": _artifact_size_bytes(path_obj),
            }
            for name, path_obj in sorted(artifact_paths.items())
        },
    }


def _resolve_generator_metadata(
    *,
    provided_metadata: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    if provided_metadata is not None:
        return dict(provided_metadata)

    repo_root = _git_output(output_dir, ["rev-parse", "--show-toplevel"])
    generated_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if repo_root is None:
        return {
            "git_head": None,
            "git_branch": None,
            "repo_root": None,
            "repo_worktree_dirty": None,
            "relevant_worktree_dirty": None,
            "dirty_paths": [],
            "relevant_dirty_paths": [],
            "generated_at_utc": generated_at_utc,
        }

    repo_root_path = Path(repo_root)
    git_head = _git_output(repo_root_path, ["rev-parse", "HEAD"])
    git_branch = _git_output(repo_root_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_output = _git_output(repo_root_path, ["status", "--porcelain=v1", "--untracked-files=all"]) or ""
    dirty_paths = _parse_git_status_paths(status_output)
    output_prefix: str | None = None
    try:
        relative_output_dir = output_dir.resolve().relative_to(repo_root_path.resolve())
    except ValueError:
        relative_output_dir = None
    if relative_output_dir is not None:
        output_prefix = relative_output_dir.as_posix().rstrip("/") + "/"
    excluded_prefixes = [
        "data/",
        "docs/superpowers/plans/",
    ]
    if output_prefix is not None:
        excluded_prefixes.append(output_prefix)
    relevant_dirty_paths = [
        path
        for path in dirty_paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in excluded_prefixes)
    ]
    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "repo_root": str(repo_root_path),
        "repo_worktree_dirty": bool(dirty_paths),
        "relevant_worktree_dirty": bool(relevant_dirty_paths),
        "dirty_paths": dirty_paths,
        "relevant_dirty_paths": relevant_dirty_paths,
        "generated_at_utc": generated_at_utc,
    }


def _git_output(cwd: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return None
    output = completed.stdout.rstrip("\r\n")
    return output or None


def _parse_git_status_paths(status_output: str) -> list[str]:
    paths: list[str] = []
    for line in status_output.splitlines():
        if not line:
            continue
        path_text = line[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", maxsplit=1)[1]
        paths.append(path_text.replace("\\", "/"))
    return paths


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _export_symbol_snapshot_feature_shards(
    trade_dates: list[str],
    active_snapshot_key_dir: Path,
    output_dir: Path,
    symbol_master_frame: pd.DataFrame,
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        active_day = pd.read_parquet(active_snapshot_key_dir / f"{trade_date}.parquet")
        if active_day.empty:
            _write_parquet_dataframe(active_day, output_dir / f"{trade_date}.parquet")
            continue
        active_day = _prepare_active_snapshot_frame(active_day)
        bars_full_day = _read_partition_parquet(
            market_data_root,
            "bars_5m",
            trade_date,
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "date"],
        )
        if not bars_full_day.empty:
            bars_full_day = bars_full_day.sort_values(["symbol", "timestamp"]).copy()
            bars_full_day["bar_ret_5m_past"] = bars_full_day.groupby("symbol")["close"].pct_change(1)
            bars_full_day["bar_ret_15m_past"] = bars_full_day.groupby("symbol")["close"].pct_change(3)
            bars_full_day["bar_dollar_volume"] = bars_full_day["close"] * bars_full_day["volume"]
            bars_full_day["bar_volume_cs_z"] = bars_full_day.groupby("timestamp")["volume"].transform(_safe_zscore)
            bars_full_day["bar_dollar_volume_cs_z"] = bars_full_day.groupby("timestamp")["bar_dollar_volume"].transform(_safe_zscore)
            bars_day = bars_full_day.rename(
                columns={
                    "open": "bar_open",
                    "high": "bar_high",
                    "low": "bar_low",
                    "close": "bar_close",
                    "volume": "bar_volume",
                    "vwap": "bar_vwap",
                }
            )
        else:
            bars_day = bars_full_day
        if not bars_day.empty:
            bars_day["bar_timestamp"] = bars_day["timestamp"]
            bars_day["bar_available_time"] = bars_day["timestamp"]

        features_day = _read_partition_parquet(
            market_data_root,
            "features_1m",
            trade_date,
            columns=[
                "symbol",
                "timestamp",
                "available_time",
                "bar_end",
                "date",
                "close",
                "volume",
                "dollar_volume",
                "trade_count",
                "imbalance_proxy",
                "large_trade_count",
                "large_trade_dollar_volume",
                "ret_1m",
                "ret_1m_past",
                "ret_3m_past",
                "ret_5m_past",
                "ret_15m_past",
                "large_trade_ratio",
                "large_trade_ratio_z",
                "volume_z_12",
                "volume_z_proxy",
                "flow_impulse_score",
            ],
        )
        features_day = _prepare_feature_review_frame(features_day)

        trade_flow_day = _read_partition_parquet(
            market_data_root,
            "trade_flow_1m",
            trade_date,
            columns=[
                "ticker",
                "minute",
                "trade_count",
                "volume",
                "dollar_volume",
                "imbalance_proxy",
                "large_trade_count",
                "large_trade_dollar_volume",
                "off_exchange_volume",
            ],
        ).rename(
            columns={
                "trade_count": "flow_trade_count",
                "volume": "flow_volume",
                "dollar_volume": "flow_dollar_volume",
                "imbalance_proxy": "flow_imbalance_proxy",
                "large_trade_count": "flow_large_trade_count",
                "large_trade_dollar_volume": "flow_large_trade_dollar_volume",
            }
        )
        trade_flow_day = _prepare_trade_flow_review_frame(trade_flow_day)

        merged = _merge_latest_available(
            active_day,
            bars_day,
            right_time_column="bar_available_time",
        )
        merged = _merge_latest_available(
            merged,
            features_day,
            right_time_column="graph_input_available_time",
        )
        merged = _merge_latest_available(
            merged,
            trade_flow_day,
            right_time_column="flow_available_time",
        )
        merged = merged.drop(columns=[column for column in ("ticker", "minute") if column in merged.columns])
        merged = merged.merge(symbol_master_frame, how="left", on="symbol")
        merged["feature_source"] = "causality_safe_graph_inputs_plus_context"
        _write_parquet_dataframe(merged, output_dir / f"{trade_date}.parquet")


def _export_symbol_forward_label_shards(
    trade_dates: list[str],
    active_snapshot_key_dir: Path,
    output_dir: Path,
    primary_benchmark: str,
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_columns = [
        "label_source_timestamp",
        "label_available_time",
        "future_ret_1m",
        "future_ret_5m",
        "future_ret_15m",
        "future_ret_30m",
    ]
    benchmark_columns = [
        "benchmark_label_source_timestamp",
        "benchmark_label_available_time",
        "benchmark_future_ret_1m",
        "benchmark_future_ret_5m",
        "benchmark_future_ret_15m",
        "benchmark_future_ret_30m",
        "benchmark_label_source",
        "benchmark_proxy_price_method",
    ]
    for trade_date in trade_dates:
        active_day = pd.read_parquet(active_snapshot_key_dir / f"{trade_date}.parquet")
        if active_day.empty:
            _write_parquet_dataframe(active_day, output_dir / f"{trade_date}.parquet")
            continue
        active_day = _prepare_active_snapshot_frame(active_day)
        labels_day = _read_partition_parquet(
            market_data_root,
            "labels_1m",
            trade_date,
            columns=["symbol", "timestamp", "future_ret_1m", "future_ret_5m", "future_ret_15m", "future_ret_30m"],
        )
        labels_day = _prepare_label_review_frame(labels_day)
        benchmark_day = _build_benchmark_label_frame(
            trade_date=trade_date,
            primary_benchmark=primary_benchmark,
            labels_day=labels_day,
            market_data_root=market_data_root,
        )

        merged = _merge_latest_available(
            active_day,
            labels_day,
            right_time_column="label_available_time",
        )
        _ensure_columns(merged, label_columns)
        merged.insert(5, "benchmark_symbol", primary_benchmark)
        merged = _merge_latest_available(
            merged,
            benchmark_day,
            by_column="benchmark_symbol",
            right_time_column="benchmark_label_available_time",
        )
        _ensure_columns(merged, benchmark_columns)
        merged["benchmark_label_source"] = merged["benchmark_label_source"].fillna("missing")
        merged["excess_future_ret_1m"] = merged["future_ret_1m"] - merged["benchmark_future_ret_1m"]
        merged["excess_future_ret_5m"] = merged["future_ret_5m"] - merged["benchmark_future_ret_5m"]
        merged["excess_future_ret_15m"] = merged["future_ret_15m"] - merged["benchmark_future_ret_15m"]
        merged["excess_future_ret_30m"] = merged["future_ret_30m"] - merged["benchmark_future_ret_30m"]
        _write_parquet_dataframe(merged, output_dir / f"{trade_date}.parquet")


def _export_benchmark_series_shards(
    trade_dates: list[str],
    output_dir: Path,
    benchmark_symbols: tuple[str, ...],
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        frame = _read_partition_parquet(
            market_data_root,
            "bars_5m",
            trade_date,
            columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "vwap", "source", "date"],
        )
        frame = frame.loc[frame["symbol"].isin(benchmark_symbols)].copy()
        if not frame.empty:
            frame = frame.sort_values(["symbol", "timestamp"])
            frame["ret_5m"] = frame.groupby("symbol")["close"].pct_change(1)
        _write_parquet_dataframe(frame, output_dir / f"{trade_date}.parquet")


def _export_benchmark_label_source_summary(
    symbol_forward_label_dir: Path,
    output_path: Path,
) -> None:
    connection = duckdb.connect()
    try:
        query = f"""
        SELECT
            CAST(trade_date AS VARCHAR) AS trade_date,
            COALESCE(CAST(benchmark_label_source AS VARCHAR), 'missing') AS benchmark_label_source,
            COALESCE(CAST(benchmark_proxy_price_method AS VARCHAR), '') AS benchmark_proxy_price_method,
            COUNT(*) AS row_count,
            COUNT(DISTINCT symbol) AS symbol_count,
            AVG(
                CASE
                    WHEN excess_future_ret_1m IS NOT NULL THEN 1.0
                    ELSE 0.0
                END
            ) AS excess_ret_1m_coverage_ratio
        FROM read_parquet('{_escape_sql_literal(str(symbol_forward_label_dir / "*.parquet"))}')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
        _copy_query_to_csv(connection, query, output_path)
    finally:
        connection.close()


def _build_benchmark_label_frame(
    *,
    trade_date: str,
    primary_benchmark: str,
    labels_day: pd.DataFrame,
    market_data_root: Path,
) -> pd.DataFrame:
    benchmark_day = labels_day.loc[labels_day["symbol"] == primary_benchmark, [
        "symbol",
        "label_source_timestamp",
        "label_available_time",
        "future_ret_1m",
        "future_ret_5m",
        "future_ret_15m",
        "future_ret_30m",
    ]].copy()
    if benchmark_day.empty:
        benchmark_day = _synthesize_benchmark_labels_from_trade_flow(
            trade_date=trade_date,
            benchmark_symbol=primary_benchmark,
            market_data_root=market_data_root,
        )
    else:
        benchmark_day["benchmark_label_source"] = "labels_1m"
        benchmark_day["benchmark_proxy_price_method"] = pd.NA
    return benchmark_day.rename(
        columns={
            "symbol": "benchmark_symbol",
            "label_source_timestamp": "benchmark_label_source_timestamp",
            "label_available_time": "benchmark_label_available_time",
            "future_ret_1m": "benchmark_future_ret_1m",
            "future_ret_5m": "benchmark_future_ret_5m",
            "future_ret_15m": "benchmark_future_ret_15m",
            "future_ret_30m": "benchmark_future_ret_30m",
        }
    )


def _synthesize_benchmark_labels_from_trade_flow(
    *,
    trade_date: str,
    benchmark_symbol: str,
    market_data_root: Path,
) -> pd.DataFrame:
    trade_flow_day = _read_partition_parquet(
        market_data_root,
        "trade_flow_1m",
        trade_date,
        columns=["ticker", "minute", "volume", "dollar_volume", "date"],
    )
    if trade_flow_day.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
            ]
        )
    benchmark_flow = trade_flow_day.loc[trade_flow_day["ticker"] == benchmark_symbol].copy()
    if benchmark_flow.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
            ]
        )
    benchmark_flow["volume"] = pd.to_numeric(benchmark_flow["volume"], errors="coerce")
    benchmark_flow["dollar_volume"] = pd.to_numeric(benchmark_flow["dollar_volume"], errors="coerce")
    benchmark_flow["proxy_price"] = (
        benchmark_flow["dollar_volume"] / benchmark_flow["volume"].replace(0.0, pd.NA)
    )
    benchmark_flow["timestamp"] = pd.to_datetime(benchmark_flow["minute"])
    benchmark_flow = (
        benchmark_flow.dropna(subset=["timestamp", "proxy_price"])
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )
    if benchmark_flow.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
            ]
        )
    labels = pd.DataFrame(
        {
            "symbol": benchmark_symbol,
            "timestamp": benchmark_flow["timestamp"].to_numpy(),
        }
    )
    price_series = benchmark_flow["proxy_price"].astype(float).reset_index(drop=True)
    for horizon in (1, 5, 15, 30):
        labels[f"future_ret_{horizon}m"] = (price_series.shift(-horizon) / price_series) - 1.0
    labels = _prepare_label_review_frame(labels)
    labels["benchmark_label_source"] = "trade_flow_proxy"
    labels["benchmark_proxy_price_method"] = _BENCHMARK_PROXY_PRICE_METHOD
    return labels


def _export_community_snapshot_features(
    community_membership_path: Path,
    symbol_snapshot_feature_dir: Path,
    output_path: Path,
) -> None:
    available_columns = _parquet_dataset_columns(symbol_snapshot_feature_dir)
    feature_timestamp_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["graph_input_feature_timestamp", "flow_feature_timestamp", "bar_timestamp", "timestamp"],
        "graph_input_feature_timestamp",
    )
    feature_available_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["graph_input_available_time", "flow_available_time", "bar_available_time"],
        "graph_input_available_time",
    )
    ret_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["ret_1m", "bar_ret_5m_past"],
        "ret_1m",
    )
    volume_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["volume_z_12", "bar_volume_cs_z"],
        "volume_z_12",
    )
    imbalance_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["imbalance_z", "flow_imbalance_proxy"],
        "imbalance_z",
    )
    large_trade_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["large_trade_ratio_z"],
        "large_trade_ratio_z",
    )
    flow_impulse_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["flow_impulse_score"],
        "flow_impulse_score",
    )
    connection = duckdb.connect()
    try:
        query = f"""
        WITH member_features AS (
            SELECT
                m.trade_date,
                m.snapshot_id,
                m.snapshot_timestamp,
                m.snapshot_clock_code,
                m.graph_layer,
                m.layer_community_id,
                m.community_local_id,
                m.community_member_count,
                m.community_edge_count,
                m.edge_density,
                m.community_avg_weight,
                m.symbol,
                m.member_rank,
                m.member_weight,
                {feature_timestamp_expr},
                {feature_available_expr},
                {ret_expr},
                {volume_expr},
                {imbalance_expr},
                {large_trade_expr},
                {flow_impulse_expr},
                sf.bar_ret_5m_past,
                sf.bar_ret_15m_past,
                sf.market_cap,
                sf.sector_code,
                sf.industry_code
            FROM read_parquet('{_escape_sql_literal(str(community_membership_path))}') m
            LEFT JOIN read_parquet('{_escape_sql_literal(str(symbol_snapshot_feature_dir / "*.parquet"))}') sf
                ON sf.snapshot_id = m.snapshot_id
               AND sf.symbol = m.symbol
        ),
        layer_active_counts AS (
            SELECT
                snapshot_id,
                graph_layer,
                COUNT(DISTINCT symbol) AS layer_active_node_count
            FROM member_features
            GROUP BY 1, 2
        ),
        snapshot_active_counts AS (
            SELECT
                snapshot_id,
                COUNT(DISTINCT symbol) AS snapshot_active_symbol_count
            FROM member_features
            GROUP BY 1
        )
        SELECT
            mf.trade_date,
            mf.snapshot_id,
            mf.snapshot_timestamp,
            mf.snapshot_clock_code,
            mf.graph_layer,
            mf.layer_community_id,
            mf.community_local_id,
            MAX(mf.community_member_count) AS community_member_count,
            MAX(mf.community_edge_count) AS community_edge_count,
            MAX(mf.edge_density) AS edge_density,
            MAX(mf.community_avg_weight) AS community_avg_weight,
            COUNT(*) AS membership_rows,
            AVG(CASE WHEN ret_1m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS feature_coverage_ratio,
            AVG(ret_1m) AS community_mean_ret_1m,
            AVG(volume_z_12) AS community_mean_volume_z_12,
            AVG(imbalance_z) AS community_mean_imbalance_z,
            AVG(large_trade_ratio_z) AS community_mean_large_trade_ratio_z,
            AVG(flow_impulse_score) AS community_mean_flow_impulse_score,
            AVG(bar_ret_5m_past) AS community_mean_bar_ret_5m_past,
            AVG(bar_ret_15m_past) AS community_mean_bar_ret_15m_past,
            AVG(CASE WHEN ret_1m > 0 THEN 1.0 ELSE 0.0 END) AS positive_ret_1m_breadth,
            AVG(CASE WHEN flow_impulse_score > 0 THEN 1.0 ELSE 0.0 END) AS positive_flow_breadth,
            AVG(CASE WHEN large_trade_ratio_z > 0 THEN 1.0 ELSE 0.0 END) AS positive_large_trade_breadth,
            AVG(CASE WHEN market_cap IS NOT NULL AND market_cap > 0 THEN 1.0 ELSE 0.0 END) AS market_cap_coverage_ratio,
            AVG(CASE WHEN sector_code IS NOT NULL AND UPPER(TRIM(sector_code)) <> 'UNKNOWN' THEN 1.0 ELSE 0.0 END) AS sector_coverage_ratio,
            AVG(CASE WHEN industry_code IS NOT NULL AND UPPER(TRIM(industry_code)) <> 'UNKNOWN' THEN 1.0 ELSE 0.0 END) AS industry_coverage_ratio,
            MAX(lac.layer_active_node_count) AS layer_active_node_count,
            MAX(sac.snapshot_active_symbol_count) AS snapshot_active_symbol_count,
            MIN(graph_input_feature_timestamp) AS earliest_graph_input_feature_timestamp,
            MAX(graph_input_feature_timestamp) AS latest_graph_input_feature_timestamp,
            MAX(graph_input_available_time) AS latest_graph_input_available_time
        FROM member_features mf
        LEFT JOIN layer_active_counts lac
            ON lac.snapshot_id = mf.snapshot_id
           AND lac.graph_layer = mf.graph_layer
        LEFT JOIN snapshot_active_counts sac
            ON sac.snapshot_id = mf.snapshot_id
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
        frame = connection.execute(query).fetchdf()
        _write_parquet_dataframe(_augment_community_snapshot_features(frame), output_path)
    finally:
        connection.close()


def _export_community_forward_labels(
    community_membership_path: Path,
    symbol_forward_label_dir: Path,
    output_path: Path,
) -> None:
    connection = duckdb.connect()
    try:
        query = f"""
        WITH member_labels AS (
            SELECT
                m.trade_date,
                m.snapshot_id,
                m.snapshot_timestamp,
                m.snapshot_clock_code,
                m.graph_layer,
                m.layer_community_id,
                m.community_local_id,
                m.community_member_count,
                m.community_edge_count,
                m.edge_density,
                m.community_avg_weight,
                m.symbol,
                m.member_rank,
                m.member_weight,
                m.member_core_score,
                sf.label_source_timestamp,
                sf.label_available_time,
                sf.future_ret_1m,
                sf.future_ret_5m,
                sf.future_ret_15m,
                sf.future_ret_30m,
                sf.excess_future_ret_1m,
                sf.excess_future_ret_5m,
                sf.excess_future_ret_15m,
                sf.excess_future_ret_30m,
                CAST(sf.benchmark_label_source AS VARCHAR) AS benchmark_label_source,
                CAST(sf.benchmark_proxy_price_method AS VARCHAR) AS benchmark_proxy_price_method
            FROM read_parquet('{_escape_sql_literal(str(community_membership_path))}') m
            LEFT JOIN read_parquet('{_escape_sql_literal(str(symbol_forward_label_dir / "*.parquet"))}') sf
                ON sf.snapshot_id = m.snapshot_id
               AND sf.symbol = m.symbol
        )
        SELECT
            trade_date,
            snapshot_id,
            snapshot_timestamp,
            snapshot_clock_code,
            graph_layer,
            layer_community_id,
            community_local_id,
            MAX(community_member_count) AS community_member_count,
            MAX(community_edge_count) AS community_edge_count,
            MAX(edge_density) AS edge_density,
            MAX(community_avg_weight) AS community_avg_weight,
            COUNT(*) AS membership_rows,
            AVG(CASE WHEN future_ret_1m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS label_coverage_ratio,
            AVG(future_ret_1m) AS community_mean_future_ret_1m,
            AVG(future_ret_5m) AS community_mean_future_ret_5m,
            AVG(future_ret_15m) AS community_mean_future_ret_15m,
            AVG(future_ret_30m) AS community_mean_future_ret_30m,
            AVG(excess_future_ret_1m) AS community_equal_weight_excess_future_ret_1m,
            AVG(excess_future_ret_5m) AS community_equal_weight_excess_future_ret_5m,
            AVG(excess_future_ret_15m) AS community_equal_weight_excess_future_ret_15m,
            AVG(excess_future_ret_30m) AS community_equal_weight_excess_future_ret_30m,
            AVG(excess_future_ret_1m) AS community_mean_excess_future_ret_1m,
            AVG(excess_future_ret_5m) AS community_mean_excess_future_ret_5m,
            AVG(excess_future_ret_15m) AS community_mean_excess_future_ret_15m,
            AVG(excess_future_ret_30m) AS community_mean_excess_future_ret_30m,
            SUM(
                CASE
                    WHEN excess_future_ret_1m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_1m * member_weight
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_1m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight
                        ELSE 0
                    END
                ),
                0
            ) AS community_member_weight_excess_future_ret_1m,
            SUM(
                CASE
                    WHEN excess_future_ret_5m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_5m * member_weight
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_5m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight
                        ELSE 0
                    END
                ),
                0
            ) AS community_member_weight_excess_future_ret_5m,
            SUM(
                CASE
                    WHEN excess_future_ret_15m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_15m * member_weight
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_15m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight
                        ELSE 0
                    END
                ),
                0
            ) AS community_member_weight_excess_future_ret_15m,
            SUM(
                CASE
                    WHEN excess_future_ret_30m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_30m * member_weight
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_30m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight
                        ELSE 0
                    END
                ),
                0
            ) AS community_member_weight_excess_future_ret_30m,
            SUM(
                CASE
                    WHEN excess_future_ret_1m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_1m * member_core_score
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_1m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score
                        ELSE 0
                    END
                ),
                0
            ) AS community_core_weighted_excess_future_ret_1m,
            SUM(
                CASE
                    WHEN excess_future_ret_5m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_5m * member_core_score
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_5m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score
                        ELSE 0
                    END
                ),
                0
            ) AS community_core_weighted_excess_future_ret_5m,
            SUM(
                CASE
                    WHEN excess_future_ret_15m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_15m * member_core_score
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_15m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score
                        ELSE 0
                    END
                ),
                0
            ) AS community_core_weighted_excess_future_ret_15m,
            SUM(
                CASE
                    WHEN excess_future_ret_30m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_30m * member_core_score
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN excess_future_ret_30m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score
                        ELSE 0
                    END
                ),
                0
            ) AS community_core_weighted_excess_future_ret_30m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_1m END) AS community_top5_member_excess_future_ret_1m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_5m END) AS community_top5_member_excess_future_ret_5m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_15m END) AS community_top5_member_excess_future_ret_15m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_30m END) AS community_top5_member_excess_future_ret_30m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_1m END) AS community_top10_member_excess_future_ret_1m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_5m END) AS community_top10_member_excess_future_ret_5m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_15m END) AS community_top10_member_excess_future_ret_15m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_30m END) AS community_top10_member_excess_future_ret_30m,
            AVG(CASE WHEN future_ret_1m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_1m_breadth,
            AVG(CASE WHEN future_ret_5m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_5m_breadth,
            AVG(CASE WHEN future_ret_15m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_15m_breadth,
            AVG(CASE WHEN future_ret_30m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_30m_breadth,
            CASE
                WHEN COUNT(DISTINCT CASE WHEN benchmark_label_source IS NOT NULL THEN benchmark_label_source END) = 0 THEN 'missing'
                WHEN COUNT(DISTINCT CASE WHEN benchmark_label_source IS NOT NULL THEN benchmark_label_source END) = 1 THEN MAX(benchmark_label_source)
                ELSE 'mixed'
            END AS benchmark_label_source,
            CASE
                WHEN COUNT(DISTINCT CASE WHEN benchmark_proxy_price_method IS NOT NULL THEN benchmark_proxy_price_method END) = 0 THEN CAST(NULL AS VARCHAR)
                WHEN COUNT(DISTINCT CASE WHEN benchmark_proxy_price_method IS NOT NULL THEN benchmark_proxy_price_method END) = 1 THEN MAX(benchmark_proxy_price_method)
                ELSE 'mixed'
            END AS benchmark_proxy_price_method,
            MIN(label_source_timestamp) AS earliest_label_source_timestamp,
            MAX(label_source_timestamp) AS latest_label_source_timestamp,
            MAX(label_available_time) AS latest_label_available_time
        FROM member_labels
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
        _copy_query_to_parquet(connection, query, output_path)
    finally:
        connection.close()


def _export_alpha_sanity_report(
    community_snapshot_feature_path: Path,
    community_forward_label_path: Path,
    output_path: Path,
) -> None:
    features = pd.read_parquet(community_snapshot_feature_path)
    labels = pd.read_parquet(community_forward_label_path)
    labels = labels.drop(
        columns=[
            column
            for column in (
                "trade_date",
                "snapshot_timestamp",
                "snapshot_clock_code",
                "community_local_id",
                "community_member_count",
                "community_edge_count",
                "edge_density",
                "community_avg_weight",
            )
            if column in labels.columns
        ]
    )
    merged = features.merge(
        labels,
        how="inner",
        on=["snapshot_id", "graph_layer", "layer_community_id"],
        suffixes=("_feature", "_label"),
    )
    if "community_member_count_feature" in merged.columns:
        merged["community_member_count"] = merged["community_member_count_feature"]
    if "edge_density_feature" in merged.columns:
        merged["edge_density_feature"] = merged["edge_density_feature"]
    elif "edge_density" in merged.columns:
        merged["edge_density_feature"] = merged["edge_density"]
    target_variants = [
        (label_variant, horizon_name, f"{column_prefix}_{horizon_name}")
        for label_variant, column_prefix in _ALPHA_LABEL_VARIANTS
        for horizon_name in ("1m", "5m", "15m", "30m")
    ]
    rows: list[dict[str, Any]] = []
    for graph_layer, layer_frame in merged.groupby("graph_layer", dropna=False):
        for factor_name in _alpha_factors_for_layer(graph_layer):
            if factor_name not in layer_frame.columns:
                continue
            for label_variant, horizon_name, target_name in target_variants:
                if target_name not in layer_frame.columns:
                    continue
                sample = layer_frame[[factor_name, target_name]].dropna()
                if sample.empty:
                    rows.append(
                        {
                            "graph_layer": graph_layer,
                            "factor_name": factor_name,
                            "label_horizon": horizon_name,
                            "label_variant": label_variant,
                            "sample_size": 0,
                            "rank_ic": None,
                            "top_decile_mean": None,
                            "bottom_decile_mean": None,
                            "top_bottom_spread": None,
                            "top_decile_hit_rate": None,
                        }
                    )
                    continue
                rank_ic = sample[factor_name].rank().corr(sample[target_name].rank())
                decile_size = max(1, len(sample) // 10)
                sorted_sample = sample.sort_values(factor_name)
                bottom = sorted_sample.head(decile_size)[target_name]
                top = sorted_sample.tail(decile_size)[target_name]
                rows.append(
                        {
                            "graph_layer": graph_layer,
                            "factor_name": factor_name,
                            "label_horizon": horizon_name,
                            "label_variant": label_variant,
                            "sample_size": int(len(sample)),
                            "rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
                            "top_decile_mean": float(top.mean()),
                        "bottom_decile_mean": float(bottom.mean()),
                        "top_bottom_spread": float(top.mean() - bottom.mean()),
                        "top_decile_hit_rate": float((top > 0).mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _export_alpha_feature_ranking_report(
    alpha_sanity_report_path: Path,
    output_path: Path,
) -> None:
    report = pd.read_csv(alpha_sanity_report_path)
    if report.empty:
        pd.DataFrame(
            columns=[
                "graph_layer",
                "layer_role",
                "factor_name",
                "label_horizon",
                "label_variant",
                "sample_size",
                "rank_ic",
                "top_bottom_spread",
                "top_decile_hit_rate",
                "score",
                "confidence_bucket",
                "research_action",
            ]
        ).to_csv(output_path, index=False)
        return
    ranking = report.copy()
    ranking["sample_size"] = pd.to_numeric(ranking["sample_size"], errors="coerce").fillna(0).astype(int)
    ranking["rank_ic"] = pd.to_numeric(ranking["rank_ic"], errors="coerce")
    ranking["top_bottom_spread"] = pd.to_numeric(ranking["top_bottom_spread"], errors="coerce")
    ranking["top_decile_hit_rate"] = pd.to_numeric(ranking["top_decile_hit_rate"], errors="coerce")
    ranking["layer_role"] = ranking["graph_layer"].map(_LAYER_RESEARCH_ROLES).fillna("unclassified_layer")
    ranking["score"] = ranking.apply(_alpha_ranking_score, axis=1)
    ranking["confidence_bucket"] = ranking["sample_size"].apply(_alpha_confidence_bucket)
    ranking["research_action"] = ranking.apply(_alpha_research_action, axis=1)
    ranking = ranking.sort_values(
        ["score", "sample_size", "graph_layer", "factor_name", "label_variant", "label_horizon"],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    ranking.to_csv(output_path, index=False)


def _export_cross_window_alpha_comparison_report(
    first_window_ranking_path: Path,
    second_window_ranking_path: Path,
    output_path: Path,
    *,
    first_window_id: str,
    second_window_id: str,
) -> None:
    key_columns = [
        "graph_layer",
        "layer_role",
        "factor_name",
        "label_horizon",
        "label_variant",
    ]
    metric_columns = [
        "sample_size",
        "rank_ic",
        "top_bottom_spread",
        "top_decile_hit_rate",
        "score",
        "confidence_bucket",
        "research_action",
    ]
    first = _prepare_cross_window_ranking_frame(first_window_ranking_path, key_columns, metric_columns, "first")
    second = _prepare_cross_window_ranking_frame(second_window_ranking_path, key_columns, metric_columns, "second")
    comparison = first.merge(
        second,
        how="outer",
        on=key_columns,
    )
    if comparison.empty:
        pd.DataFrame(
            columns=key_columns
            + [
                "first_window_id",
                "second_window_id",
                "first_score_sign",
                "second_score_sign",
                "score_direction_consistent",
                "rank_ic_direction_consistent",
                "sample_qualified_both",
                "research_action_consistent",
                "score_delta",
                "sample_size_delta",
                "stability_bucket",
                "research_decision",
            ]
        ).to_csv(output_path, index=False)
        return
    comparison["first_window_id"] = first_window_id
    comparison["second_window_id"] = second_window_id
    for prefix in ("first", "second"):
        comparison[f"{prefix}_sample_size"] = pd.to_numeric(
            comparison[f"{prefix}_sample_size"],
            errors="coerce",
        ).fillna(0).astype(int)
        comparison[f"{prefix}_rank_ic"] = pd.to_numeric(comparison[f"{prefix}_rank_ic"], errors="coerce")
        comparison[f"{prefix}_top_bottom_spread"] = pd.to_numeric(
            comparison[f"{prefix}_top_bottom_spread"],
            errors="coerce",
        )
        comparison[f"{prefix}_top_decile_hit_rate"] = pd.to_numeric(
            comparison[f"{prefix}_top_decile_hit_rate"],
            errors="coerce",
        )
        comparison[f"{prefix}_score"] = pd.to_numeric(comparison[f"{prefix}_score"], errors="coerce")
        comparison[f"{prefix}_score_sign"] = comparison[f"{prefix}_score"].apply(_alpha_sign)
    comparison["score_direction_consistent"] = comparison.apply(
        lambda row: row["first_score_sign"] != 0
        and row["second_score_sign"] != 0
        and row["first_score_sign"] == row["second_score_sign"],
        axis=1,
    )
    comparison["rank_ic_direction_consistent"] = comparison.apply(
        lambda row: _alpha_sign(row["first_rank_ic"]) != 0
        and _alpha_sign(row["second_rank_ic"]) != 0
        and _alpha_sign(row["first_rank_ic"]) == _alpha_sign(row["second_rank_ic"]),
        axis=1,
    )
    comparison["sample_qualified_both"] = comparison.apply(
        lambda row: row["first_sample_size"] >= 3000 and row["second_sample_size"] >= 3000,
        axis=1,
    )
    comparison["research_action_consistent"] = (
        comparison["first_research_action"].fillna("") == comparison["second_research_action"].fillna("")
    )
    comparison["score_delta"] = comparison["second_score"] - comparison["first_score"]
    comparison["sample_size_delta"] = comparison["second_sample_size"] - comparison["first_sample_size"]
    comparison["stability_bucket"] = comparison.apply(_cross_window_stability_bucket, axis=1)
    comparison["research_decision"] = comparison["stability_bucket"].map(
        {
            "stable_positive": "confirm_layer_role",
            "stable_negative": "deprioritize",
            "insufficient_sample": "needs_more_sample",
            "missing_in_one_window": "rebuild_missing_window",
        }
    ).fillna("review_manually")
    comparison = comparison.sort_values(
        [
            "stability_bucket",
            "graph_layer",
            "factor_name",
            "label_variant",
            "label_horizon",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    comparison.to_csv(output_path, index=False)


def _prepare_cross_window_ranking_frame(
    ranking_path: Path,
    key_columns: list[str],
    metric_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    frame = pd.read_csv(ranking_path)
    available_key_columns = [column for column in key_columns if column in frame.columns]
    available_metric_columns = [column for column in metric_columns if column in frame.columns]
    prepared = frame.loc[:, available_key_columns + available_metric_columns].copy()
    for column in key_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    for column in metric_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    rename_map = {column: f"{prefix}_{column}" for column in metric_columns}
    return prepared.loc[:, key_columns + metric_columns].rename(columns=rename_map)


def _alpha_ranking_score(row: pd.Series) -> float:
    rank_ic = row.get("rank_ic")
    spread = row.get("top_bottom_spread")
    sample_size = int(row.get("sample_size", 0) or 0)
    if pd.isna(rank_ic) or pd.isna(spread) or sample_size <= 0:
        return 0.0
    spread_sign = 1.0 if spread > 0 else -1.0 if spread < 0 else 0.0
    return float(abs(rank_ic) * math.log10(sample_size + 1) * spread_sign)


def _alpha_confidence_bucket(sample_size: int) -> str:
    if sample_size < 500:
        return "ignore"
    if sample_size < 3000:
        return "watch"
    if sample_size <= 10000:
        return "usable"
    return "strong_sample"


def _alpha_research_action(row: pd.Series) -> str:
    sample_size = int(row.get("sample_size", 0) or 0)
    score = float(row.get("score", 0.0) or 0.0)
    layer_role = row.get("layer_role")
    if sample_size < 500:
        return "ignore_sparse"
    if pd.isna(row.get("rank_ic")) or pd.isna(row.get("top_bottom_spread")):
        return "insufficient_signal"
    if score > 0:
        if sample_size > 10000 and layer_role == "theme_candidate_layer":
            return "prioritize_for_next_round"
        if sample_size >= 3000:
            return "keep_for_next_round"
        return "watch"
    if sample_size >= 3000:
        return "downgrade"
    return "watch"


def _alpha_sign(value: Any) -> int:
    if pd.isna(value):
        return 0
    numeric_value = float(value)
    if numeric_value > 0:
        return 1
    if numeric_value < 0:
        return -1
    return 0


def _cross_window_stability_bucket(row: pd.Series) -> str:
    if row["first_sample_size"] == 0 or row["second_sample_size"] == 0:
        return "missing_in_one_window"
    if not row["sample_qualified_both"]:
        return "insufficient_sample"
    if row["score_direction_consistent"]:
        if row["first_score_sign"] > 0:
            return "stable_positive"
        if row["first_score_sign"] < 0:
            return "stable_negative"
    return "unstable_direction"


def _alpha_factors_for_layer(graph_layer: Any) -> list[str]:
    return list(_ALPHA_FACTOR_COLUMNS_BY_LAYER.get(str(graph_layer), ["community_quality_score"]))


def _augment_community_snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        for column in (
            "edge_density_feature",
            "community_avg_weight_feature",
            "layer_member_ratio",
            "flow_member_count_z",
            "flow_layer_participation_ratio",
            "flow_breadth_expansion",
            "community_quality_score",
        ):
            frame[column] = pd.Series(dtype="float64")
        return frame
    augmented = frame.copy()
    augmented["snapshot_timestamp"] = pd.to_datetime(augmented["snapshot_timestamp"])
    augmented["edge_density_feature"] = pd.to_numeric(augmented["edge_density"], errors="coerce")
    augmented["community_avg_weight_feature"] = pd.to_numeric(augmented["community_avg_weight"], errors="coerce")
    augmented["community_member_count"] = pd.to_numeric(augmented["community_member_count"], errors="coerce")
    augmented["feature_coverage_ratio"] = pd.to_numeric(augmented["feature_coverage_ratio"], errors="coerce")
    augmented["layer_active_node_count"] = pd.to_numeric(augmented["layer_active_node_count"], errors="coerce")
    augmented["snapshot_active_symbol_count"] = pd.to_numeric(augmented["snapshot_active_symbol_count"], errors="coerce")
    augmented["layer_member_ratio"] = (
        augmented["community_member_count"] / augmented["layer_active_node_count"].replace(0, pd.NA)
    )
    augmented["flow_layer_participation_ratio"] = (
        augmented["layer_active_node_count"] / augmented["snapshot_active_symbol_count"].replace(0, pd.NA)
    )
    augmented["flow_member_count_z"] = (
        augmented.groupby("graph_layer", dropna=False)["community_member_count"].transform(_safe_zscore)
    )
    quality_components = (
        augmented.groupby("graph_layer", dropna=False)["edge_density_feature"].transform(_safe_zscore)
        + augmented.groupby("graph_layer", dropna=False)["community_avg_weight_feature"].transform(_safe_zscore)
        + augmented.groupby("graph_layer", dropna=False)["feature_coverage_ratio"].transform(_safe_zscore)
        - augmented.groupby("graph_layer", dropna=False)["layer_member_ratio"].transform(_safe_zscore)
    )
    augmented["community_quality_score"] = quality_components
    breadth_base = (
        augmented.loc[:, ["snapshot_id", "graph_layer", "snapshot_timestamp", "flow_layer_participation_ratio"]]
        .drop_duplicates()
        .sort_values(["graph_layer", "snapshot_timestamp", "snapshot_id"])
        .reset_index(drop=True)
    )
    breadth_base["flow_breadth_expansion"] = (
        breadth_base.groupby("graph_layer", dropna=False)["flow_layer_participation_ratio"].diff().fillna(0.0)
    )
    augmented = augmented.merge(
        breadth_base[["snapshot_id", "graph_layer", "flow_breadth_expansion"]],
        how="left",
        on=["snapshot_id", "graph_layer"],
    )
    return augmented


def _write_metadata_trust_policy(output_path: Path) -> None:
    policy_path = Path(__file__).resolve().parents[4] / "metadata_trust_policy.json"
    if policy_path.exists():
        output_path.write_text(policy_path.read_text(encoding="utf-8"), encoding="utf-8")
        return
    output_path.write_text(
        json.dumps(_default_metadata_trust_policy(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _default_metadata_trust_policy() -> dict[str, Any]:
    return {
        "policy_version": "2026-06-20",
        "purpose": "Metadata is exported for post-hoc validation and review. It must not introduce future bias into model training or graph construction.",
        "safe_model_features": ["sector", "industry", "exchange", "country"],
        "time_dependent_features": ["market_cap_bucket_at_t", "price_at_t", "shares_outstanding_at_t"],
        "unsafe_interpretation_only": ["supplier", "customer", "theme", "narrative", "moat"],
        "notes": [
            "Current metadata exports are intended for ex-post evaluation only.",
            "Do not backfill modern narrative labels into historical model features.",
            "If market cap is required in modeling, reconstruct it at time t from contemporaneous price and historically valid shares outstanding.",
        ],
    }


def _prepare_active_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["snapshot_timestamp"] = pd.to_datetime(prepared["snapshot_timestamp"])
    return prepared.sort_values(["symbol", "snapshot_timestamp"]).reset_index(drop=True)


def _prepare_feature_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "graph_input_feature_timestamp",
                "graph_input_available_time",
                "ret_1m",
                "volume_z_12",
                "imbalance_z",
                "large_trade_ratio_z",
                "flow_impulse_score",
            ]
        )
    prepared = frame.copy()
    if "ret_1m" not in prepared.columns and "ret_1m_past" in prepared.columns:
        prepared["ret_1m"] = prepared["ret_1m_past"]
    if "volume_z_12" not in prepared.columns and "volume_z_proxy" in prepared.columns:
        prepared["volume_z_12"] = prepared["volume_z_proxy"]
    if "large_trade_ratio_z" not in prepared.columns and "large_trade_ratio" in prepared.columns:
        prepared["large_trade_ratio_z"] = prepared["large_trade_ratio"]
    if "imbalance_z" not in prepared.columns and "imbalance_proxy" in prepared.columns:
        prepared["imbalance_z"] = prepared["imbalance_proxy"]
    if "flow_impulse_score" not in prepared.columns:
        if "imbalance_z" in prepared.columns:
            prepared["flow_impulse_score"] = pd.to_numeric(prepared["imbalance_z"], errors="coerce").fillna(0.0)
        else:
            prepared["flow_impulse_score"] = 0.0
    if "graph_input_available_time" not in prepared.columns:
        if "available_time" in prepared.columns:
            prepared["graph_input_available_time"] = pd.to_datetime(prepared["available_time"])
        elif "bar_end" in prepared.columns:
            prepared["graph_input_available_time"] = pd.to_datetime(prepared["bar_end"])
        else:
            prepared["graph_input_available_time"] = pd.to_datetime(prepared["timestamp"]) + pd.Timedelta(minutes=1)
    prepared["graph_input_feature_timestamp"] = pd.to_datetime(prepared["timestamp"])
    return prepared.sort_values(["symbol", "graph_input_available_time"]).reset_index(drop=True)


def _prepare_trade_flow_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "flow_feature_timestamp",
                "flow_available_time",
                "flow_trade_count",
                "flow_volume",
                "flow_dollar_volume",
                "flow_imbalance_proxy",
                "flow_large_trade_count",
                "flow_large_trade_dollar_volume",
            ]
        )
    prepared = frame.copy()
    if "ticker" in prepared.columns:
        prepared = prepared.rename(columns={"ticker": "symbol"})
    if "minute" in prepared.columns:
        prepared["flow_feature_timestamp"] = pd.to_datetime(prepared["minute"])
    else:
        prepared["flow_feature_timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["flow_available_time"] = prepared["flow_feature_timestamp"] + pd.Timedelta(minutes=1)
    return prepared.sort_values(["symbol", "flow_available_time"]).reset_index(drop=True)


def _prepare_label_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
            ]
        )
    prepared = frame.copy()
    prepared["label_source_timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["label_available_time"] = prepared["label_source_timestamp"] + pd.Timedelta(minutes=1)
    return prepared.sort_values(["symbol", "label_available_time"]).reset_index(drop=True)


def _merge_latest_available(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    right_time_column: str,
    by_column: str = "symbol",
) -> pd.DataFrame:
    if left.empty or right.empty:
        return left.copy()
    left_working = left.copy()
    left_working["_merge_row_order"] = range(len(left_working))
    right_working = right.copy()
    if by_column in left_working.columns:
        left_working[by_column] = pd.Series(left_working[by_column], dtype="string[python]")
    if by_column in right_working.columns:
        right_working[by_column] = pd.Series(right_working[by_column], dtype="string[python]")
    left_sorted = left_working.sort_values(["snapshot_timestamp", by_column]).reset_index(drop=True)
    right_sorted = right_working.sort_values([right_time_column, by_column]).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        by=by_column,
        left_on="snapshot_timestamp",
        right_on=right_time_column,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("_merge_row_order").drop(columns=["_merge_row_order"]).reset_index(drop=True)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA


def _parquet_dataset_columns(path: Path) -> set[str]:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    existing_files = [file_path for file_path in files if file_path.exists()]
    if not existing_files:
        return set()
    sample = pd.read_parquet(existing_files[0])
    return set(str(column) for column in sample.columns)


def _select_parquet_column_expr(
    available_columns: set[str],
    table_alias: str,
    candidates: list[str],
    output_alias: str,
) -> str:
    for candidate in candidates:
        if candidate in available_columns:
            return f"{table_alias}.{candidate} AS {output_alias}"
    return f"NULL AS {output_alias}"


def _copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    connection.execute(
        f"COPY ({query}) TO '{_escape_sql_literal(str(output_path))}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def _copy_query_to_partitioned_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_dir: Path,
    *,
    trade_date_column: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_dates = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT CAST(trade_date AS VARCHAR) FROM pack_snapshot_context ORDER BY 1"
        ).fetchall()
    ]
    for trade_date in trade_dates:
        output_path = output_dir / f"{trade_date}.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM ({query}) AS shard_source
                WHERE {trade_date_column} = DATE '{trade_date}'
            ) TO '{_escape_sql_literal(str(output_path))}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )


def _copy_trade_date_queries_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    trade_dates: list[str],
    output_dir: Path,
    query_builder: Callable[[str], str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        output_path = output_dir / f"{trade_date}.parquet"
        _copy_query_to_parquet(connection, query_builder(trade_date), output_path)


def _copy_query_to_csv(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
    *,
    parameters: dict[str, str] | None = None,
) -> None:
    if parameters:
        relation = connection.sql(query, params=parameters)
        relation.write_csv(str(output_path))
        return
    connection.execute(
        f"COPY ({query}) TO '{_escape_sql_literal(str(output_path))}' (HEADER, DELIMITER ',')"
    )


def _scalar_int(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    value = connection.execute(query).fetchone()[0]
    return int(value or 0)


def _trade_dates(connection: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT CAST(trade_date AS VARCHAR) FROM pack_snapshot_context ORDER BY 1"
        ).fetchall()
    ]


def _artifact_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _read_partition_parquet(
    market_data_root: Path,
    dataset_name: str,
    trade_date: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    partition_path = market_data_root / dataset_name / f"date={trade_date}" / f"{dataset_name}.parquet"
    if not partition_path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        frame = pd.read_parquet(partition_path, columns=columns)
    except Exception:
        frame = pd.read_parquet(partition_path)
        if columns is not None:
            existing_columns = [column for column in columns if column in frame.columns]
            frame = frame.loc[:, existing_columns]
    for column in ("timestamp", "minute", "bar_end", "available_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
            if getattr(frame[column].dt, "tz", None) is not None:
                frame[column] = frame[column].dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    mean = series.mean()
    return (series - mean) / std


def _write_parquet_dataframe(frame: pd.DataFrame, output_path: Path) -> None:
    try:
        frame.to_parquet(output_path, index=False, compression="zstd")
    except Exception:
        connection = duckdb.connect()
        try:
            connection.register("frame_df", frame)
            connection.execute(
                f"COPY frame_df TO '{_escape_sql_literal(str(output_path))}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            connection.close()


def _escape_sql_literal(value: str) -> str:
    return value.replace("\\", "/").replace("'", "''")


def _sql_string_list(values: tuple[str, ...]) -> str:
    if not values:
        return "'SPY'"
    return ", ".join(f"'{_escape_sql_literal(value)}'" for value in values)
