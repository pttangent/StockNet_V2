from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from stocknetv2.application.services import graph_evaluation_pack_service as graph_pack_service
from stocknetv2.application.services.graph_evaluation_pack_service import (
    GraphEvaluationPackConfig,
    build_graph_evaluation_pack,
)
from stocknetv2.infrastructure.db.schema_manager import SchemaManager


def _create_graph_database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    SchemaManager(connection).initialize()
    connection.execute(
        """
        INSERT INTO theme_discovery_run (
            run_id,
            run_name,
            date_start,
            date_end,
            frame_minutes,
            config_id,
            config_json,
            code_commit,
            data_version,
            status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02",
            "Eval run",
            "2025-01-02",
            "2025-01-02",
            5,
            "config_eval",
            '{"config_id":"config_eval"}',
            "abc123",
            "bars_5m:2025-01-02|trade_flow_1m:2025-01-02|features_1m:2025-01-02",
            "completed",
        ],
    )
    connection.execute(
        """
        INSERT INTO graph_snapshot (
            snapshot_id,
            run_id,
            trade_date,
            timestamp,
            frame_minutes,
            market_session,
            graph_status,
            available_minutes_since_open
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02_2025-01-02_1435",
            "run_eval_2025-01-02",
            "2025-01-02",
            "2025-01-02 22:35:00",
            5,
            "regular",
            "complete",
            5,
        ],
    )
    connection.execute(
        """
        INSERT INTO graph_edge_summary (
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            edge_count,
            node_count,
            avg_weight,
            median_weight,
            p90_weight,
            threshold,
            top_k_per_symbol,
            effective_lookback_minutes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            1,
            2,
            0.82,
            0.82,
            0.82,
            0.65,
            3,
            60,
        ],
    )
    connection.execute(
        """
        INSERT INTO graph_edges_thresholded (
            run_id,
            snapshot_id,
            trade_date,
            timestamp,
            graph_layer,
            source_symbol,
            target_symbol,
            edge_type,
            weight,
            raw_score,
            edge_confidence,
            effective_lookback_minutes,
            window_start,
            window_end,
            support_points,
            config_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "2025-01-02 22:35:00",
            "return_corr_graph",
            "AAA",
            "BBB",
            "correlation",
            0.82,
            0.82,
            0.9,
            60,
            "2025-01-02 21:35:00",
            "2025-01-02 22:35:00",
            8,
            "config_eval",
        ],
    )
    connection.execute(
        """
        INSERT INTO layer_community (
            layer_community_id,
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            community_local_id,
            members_json,
            member_count,
            edge_count,
            edge_density,
            avg_weight,
            min_weight,
            max_weight,
            community_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "lc_001",
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            "C001",
            '["AAA","BBB"]',
            2,
            1,
            1.0,
            0.82,
            0.82,
            0.82,
            "connected_components",
        ],
    )
    connection.execute(
        """
        INSERT INTO layer_community_membership (
            layer_community_id,
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            community_local_id,
            symbol,
            member_rank,
            member_weight
        ) VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "lc_001",
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            "C001",
            "AAA",
            1,
            0.9,
            "lc_001",
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            "C001",
            "BBB",
            2,
            0.8,
        ],
    )
    connection.close()


def _create_market_database(path: Path) -> None:
    root = path.parent
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE features_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            date DATE,
            close DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            trade_count DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_count DOUBLE,
            large_trade_dollar_volume DOUBLE,
            ret_1m_past DOUBLE,
            ret_3m_past DOUBLE,
            ret_5m_past DOUBLE,
            ret_15m_past DOUBLE,
            large_trade_ratio DOUBLE,
            volume_z_proxy DOUBLE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE labels_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            future_ret_1m DOUBLE,
            future_ret_5m DOUBLE,
            future_ret_15m DOUBLE,
            future_ret_30m DOUBLE,
            date DATE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE trade_flow_1m (
            ticker VARCHAR,
            minute TIMESTAMP,
            trade_count DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_count DOUBLE,
            large_trade_dollar_volume DOUBLE,
            off_exchange_volume DOUBLE,
            date DATE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE bars_5m (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR,
            vwap DOUBLE,
            source VARCHAR,
            date DATE
        )
        """
    )
    connection.execute(
        """
        INSERT INTO features_1m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA", "2025-01-02 22:35:00", "2025-01-02", 10.1, 1000.0, 10100.0, 15.0, 0.4, 2.0, 1000.0, 0.01, 0.02, 0.03, 0.04, 0.1, 1.2,
            "BBB", "2025-01-02 22:35:00", "2025-01-02", 20.2, 2000.0, 40400.0, 18.0, 0.5, 3.0, 1500.0, 0.011, 0.021, 0.031, 0.041, 0.2, 1.5,
            "SPY", "2025-01-02 22:35:00", "2025-01-02", 500.0, 5000.0, 2500000.0, 40.0, 0.1, 1.0, 5000.0, 0.005, 0.006, 0.007, 0.008, 0.05, 0.8,
        ],
    )
    connection.execute(
        """
        INSERT INTO labels_1m VALUES
        (?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA", "2025-01-02 22:35:00", 0.002, 0.006, 0.010, 0.015, "2025-01-02",
            "BBB", "2025-01-02 22:35:00", 0.001, 0.005, 0.009, 0.014, "2025-01-02",
            "SPY", "2025-01-02 22:35:00", 0.0005, 0.0025, 0.0040, 0.0060, "2025-01-02",
        ],
    )
    connection.execute(
        """
        INSERT INTO trade_flow_1m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "SPY", "2025-01-02 22:34:00", 39.0, 4998.0, 2497251.4994, 0.1, 1.0, 5000.0, 240.0, "2025-01-02",
            "AAA", "2025-01-02 22:35:00", 15.0, 1000.0, 10100.0, 0.4, 2.0, 1000.0, 100.0, "2025-01-02",
            "BBB", "2025-01-02 22:35:00", 18.0, 2000.0, 40400.0, 0.5, 3.0, 1500.0, 200.0, "2025-01-02",
            "SPY", "2025-01-02 22:35:00", 40.0, 5000.0, 2500000.0, 0.1, 1.0, 5000.0, 250.0, "2025-01-02",
        ],
    )
    connection.execute(
        """
        INSERT INTO bars_5m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "2025-01-02 22:35:00", 10.0, 10.2, 9.9, 10.1, 1000.0, "AAA", 10.05, "test", "2025-01-02",
            "2025-01-02 22:35:00", 20.0, 20.3, 19.9, 20.2, 2000.0, "BBB", 20.1, "test", "2025-01-02",
            "2025-01-02 22:35:00", 499.0, 501.0, 498.0, 500.0, 5000.0, "SPY", 499.8, "test", "2025-01-02",
        ],
    )
    connection.close()

    features_partition = root / "features_1m" / "date=2025-01-02"
    labels_partition = root / "labels_1m" / "date=2025-01-02"
    trade_flow_partition = root / "trade_flow_1m" / "date=2025-01-02"
    bars_partition = root / "bars_5m" / "date=2025-01-02"
    for partition in (features_partition, labels_partition, trade_flow_partition, bars_partition):
        partition.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": "2025-01-02 22:34:00",
                "date": "2025-01-02",
                "close": 10.0,
                "volume": 950.0,
                "dollar_volume": 9500.0,
                "trade_count": 14.0,
                "imbalance_proxy": 0.3,
                "large_trade_count": 1.0,
                "large_trade_dollar_volume": 900.0,
                "ret_1m_past": 0.009,
                "ret_3m_past": 0.019,
                "ret_5m_past": 0.029,
                "ret_15m_past": 0.039,
                "large_trade_ratio": 0.09,
                "volume_z_proxy": 1.1,
            },
            {
                "symbol": "AAA",
                "timestamp": "2025-01-02 22:35:00",
                "date": "2025-01-02",
                "close": 10.1,
                "volume": 1000.0,
                "dollar_volume": 10100.0,
                "trade_count": 15.0,
                "imbalance_proxy": 0.4,
                "large_trade_count": 2.0,
                "large_trade_dollar_volume": 1000.0,
                "ret_1m_past": 0.01,
                "ret_3m_past": 0.02,
                "ret_5m_past": 0.03,
                "ret_15m_past": 0.04,
                "large_trade_ratio": 0.1,
                "volume_z_proxy": 1.2,
            },
            {
                "symbol": "BBB",
                "timestamp": "2025-01-02 22:34:00",
                "date": "2025-01-02",
                "close": 20.1,
                "volume": 1950.0,
                "dollar_volume": 39195.0,
                "trade_count": 17.0,
                "imbalance_proxy": 0.4,
                "large_trade_count": 2.0,
                "large_trade_dollar_volume": 1400.0,
                "ret_1m_past": 0.010,
                "ret_3m_past": 0.020,
                "ret_5m_past": 0.030,
                "ret_15m_past": 0.040,
                "large_trade_ratio": 0.18,
                "volume_z_proxy": 1.4,
            },
            {
                "symbol": "BBB",
                "timestamp": "2025-01-02 22:35:00",
                "date": "2025-01-02",
                "close": 20.2,
                "volume": 2000.0,
                "dollar_volume": 40400.0,
                "trade_count": 18.0,
                "imbalance_proxy": 0.5,
                "large_trade_count": 3.0,
                "large_trade_dollar_volume": 1500.0,
                "ret_1m_past": 0.011,
                "ret_3m_past": 0.021,
                "ret_5m_past": 0.031,
                "ret_15m_past": 0.041,
                "large_trade_ratio": 0.2,
                "volume_z_proxy": 1.5,
            },
        ]
    ).to_parquet(features_partition / "features_1m.parquet", index=False)
    pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2025-01-02 22:34:00", "future_ret_1m": 0.003, "future_ret_5m": 0.007, "future_ret_15m": 0.011, "future_ret_30m": 0.016, "date": "2025-01-02"},
            {"symbol": "AAA", "timestamp": "2025-01-02 22:35:00", "future_ret_1m": 0.002, "future_ret_5m": 0.006, "future_ret_15m": 0.010, "future_ret_30m": 0.015, "date": "2025-01-02"},
            {"symbol": "BBB", "timestamp": "2025-01-02 22:34:00", "future_ret_1m": 0.002, "future_ret_5m": 0.006, "future_ret_15m": 0.010, "future_ret_30m": 0.015, "date": "2025-01-02"},
            {"symbol": "BBB", "timestamp": "2025-01-02 22:35:00", "future_ret_1m": 0.001, "future_ret_5m": 0.005, "future_ret_15m": 0.009, "future_ret_30m": 0.014, "date": "2025-01-02"},
            {"symbol": "SPY", "timestamp": "2025-01-02 22:34:00", "future_ret_1m": 0.0006, "future_ret_5m": 0.0026, "future_ret_15m": 0.0041, "future_ret_30m": 0.0061, "date": "2025-01-02"},
            {"symbol": "SPY", "timestamp": "2025-01-02 22:35:00", "future_ret_1m": 0.0005, "future_ret_5m": 0.0025, "future_ret_15m": 0.0040, "future_ret_30m": 0.0060, "date": "2025-01-02"},
        ]
    ).to_parquet(labels_partition / "labels_1m.parquet", index=False)
    pd.DataFrame(
        [
            {"ticker": "AAA", "minute": "2025-01-02 22:34:00", "trade_count": 14.0, "volume": 950.0, "dollar_volume": 9500.0, "imbalance_proxy": 0.3, "large_trade_count": 1.0, "large_trade_dollar_volume": 900.0, "off_exchange_volume": 90.0, "date": "2025-01-02"},
            {"ticker": "AAA", "minute": "2025-01-02 22:35:00", "trade_count": 15.0, "volume": 1000.0, "dollar_volume": 10100.0, "imbalance_proxy": 0.4, "large_trade_count": 2.0, "large_trade_dollar_volume": 1000.0, "off_exchange_volume": 100.0, "date": "2025-01-02"},
            {"ticker": "BBB", "minute": "2025-01-02 22:34:00", "trade_count": 17.0, "volume": 1950.0, "dollar_volume": 39195.0, "imbalance_proxy": 0.4, "large_trade_count": 2.0, "large_trade_dollar_volume": 1400.0, "off_exchange_volume": 180.0, "date": "2025-01-02"},
            {"ticker": "BBB", "minute": "2025-01-02 22:35:00", "trade_count": 18.0, "volume": 2000.0, "dollar_volume": 40400.0, "imbalance_proxy": 0.5, "large_trade_count": 3.0, "large_trade_dollar_volume": 1500.0, "off_exchange_volume": 200.0, "date": "2025-01-02"},
            {"ticker": "SPY", "minute": "2025-01-02 22:34:00", "trade_count": 39.0, "volume": 4998.0, "dollar_volume": 2497251.4994, "imbalance_proxy": 0.1, "large_trade_count": 1.0, "large_trade_dollar_volume": 5000.0, "off_exchange_volume": 240.0, "date": "2025-01-02"},
            {"ticker": "SPY", "minute": "2025-01-02 22:35:00", "trade_count": 40.0, "volume": 5000.0, "dollar_volume": 2500000.0, "imbalance_proxy": 0.1, "large_trade_count": 1.0, "large_trade_dollar_volume": 5000.0, "off_exchange_volume": 250.0, "date": "2025-01-02"},
        ]
    ).to_parquet(trade_flow_partition / "trade_flow_1m.parquet", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2025-01-02 22:30:00", "open": 9.9, "high": 10.0, "low": 9.8, "close": 10.0, "volume": 900.0, "symbol": "AAA", "vwap": 9.95, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:35:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000.0, "symbol": "AAA", "vwap": 10.05, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:30:00", "open": 19.9, "high": 20.0, "low": 19.8, "close": 20.0, "volume": 1900.0, "symbol": "BBB", "vwap": 19.95, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:35:00", "open": 20.0, "high": 20.3, "low": 19.9, "close": 20.2, "volume": 2000.0, "symbol": "BBB", "vwap": 20.10, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:35:00", "open": 499.0, "high": 501.0, "low": 498.0, "close": 500.0, "volume": 5000.0, "symbol": "SPY", "vwap": 499.8, "source": "test", "date": "2025-01-02"},
        ]
    ).to_parquet(bars_partition / "bars_5m.parquet", index=False)


def test_build_graph_evaluation_pack_exports_review_artifacts(tmp_path):
    graph_database_path = tmp_path / "graph.duckdb"
    market_database_path = tmp_path / "market.duckdb"
    metadata_csv_path = tmp_path / "input_symbols.csv"
    output_dir = tmp_path / "evaluation_pack"

    _create_graph_database(graph_database_path)
    _create_market_database(market_database_path)
    metadata_csv_path.write_text(
        "\n".join(
            [
                "symbol,source_symbol,company_name,sector_code,industry_code,last_price,rank,market_cap,exchange,country,quote_type",
                "AAA,AAA,Alpha,TECH,SOFT,10.1,1,100000000,NMS,United States,EQUITY",
                "BBB,BBB,Beta,,,20.2,2,200000000,NYQ,United States,EQUITY",
                "SPY,SPY,SPDR S&P 500 ETF Trust,ETF,INDEX,500,0,0,PCX,United States,ETF",
            ]
        ),
        encoding="utf-8",
    )

    summary = build_graph_evaluation_pack(
        GraphEvaluationPackConfig(
            graph_database_path=graph_database_path,
            market_database_path=market_database_path,
            metadata_csv_path=metadata_csv_path,
            output_dir=output_dir,
            date_start="2025-01-02",
            date_end="2025-01-02",
            benchmark_symbols=("SPY",),
            generator_metadata={
                "git_head": "pack123",
                "git_branch": "test-branch",
                "repo_worktree_dirty": False,
                "relevant_worktree_dirty": False,
                "dirty_paths": [],
                "relevant_dirty_paths": [],
                "generated_at_utc": "2026-06-19T00:00:00Z",
            },
        )
    )

    assert summary.output_dir == output_dir.resolve()

    expected_files = [
        output_dir / "run_manifest.json",
        output_dir / "README.md",
        output_dir / "graph" / "all_edges",
        output_dir / "graph" / "snapshot_layer_diagnostics.csv",
        output_dir / "graph" / "node_layer_metrics",
        output_dir / "graph" / "community_metrics.parquet",
        output_dir / "graph" / "community_membership.parquet",
        output_dir / "graph" / "community_member_symbols.csv",
        output_dir / "graph" / "layer_review_candidates.csv",
        output_dir / "graph" / "metadata_coverage_report.csv",
        output_dir / "market" / "symbol_snapshot_features",
        output_dir / "market" / "symbol_forward_labels",
        output_dir / "market" / "community_snapshot_features.parquet",
        output_dir / "market" / "community_forward_labels.parquet",
        output_dir / "market" / "alpha_sanity_report.csv",
        output_dir / "market" / "alpha_feature_ranking_by_layer.csv",
        output_dir / "market" / "benchmark_label_source_summary.csv",
        output_dir / "market" / "metadata_trust_policy.json",
        output_dir / "market" / "symbol_master.csv",
        output_dir / "market" / "benchmark_series",
    ]
    for path in expected_files:
        assert path.exists(), path

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert manifest["date_start"] == "2025-01-02"
    assert manifest["date_end"] == "2025-01-02"
    assert manifest["counts"]["edge_rows"] == 1
    assert manifest["counts"]["community_rows"] == 1
    assert "Status: `Graph evaluation artifact ready for manual review`" in readme_text
    assert "See `ASSESSMENT.md` if a month-specific research conclusion has been written." in readme_text
    assert manifest["generator"]["git_head"] == "pack123"
    assert manifest["generator"]["git_branch"] == "test-branch"
    assert manifest["generator"]["relevant_worktree_dirty"] is False
    assert manifest["provenance"]["graph_build_commits"] == ["abc123"]
    assert manifest["provenance"]["evaluation_pack_generator"]["git_head"] == "pack123"
    assert manifest["provenance"]["config"]["sha256"]
    assert manifest["provenance"]["inputs"]["graph_database"]["sha256"]
    assert manifest["provenance"]["dependency_versions"]["duckdb"] == duckdb.__version__
    assert manifest["artifacts"]["run_manifest"]["size_bytes"] > 0

    connection = duckdb.connect()
    assert connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(output_dir / "graph" / "community_membership.parquet")],
    ).fetchone()[0] == 2
    community_membership = connection.execute(
        """
        SELECT
            symbol,
            member_rank,
            member_weight,
            member_core_score
        FROM read_parquet(?)
        ORDER BY symbol
        """,
        [str(output_dir / "graph" / "community_membership.parquet")],
    ).fetchdf()
    assert community_membership.loc[0, "member_core_score"] > community_membership.loc[1, "member_core_score"]
    community_symbol_lists = connection.execute(
        """
        SELECT
            graph_layer,
            member_count,
            member_symbols
        FROM read_csv_auto(?)
        """,
        [str(output_dir / "graph" / "community_member_symbols.csv")],
    ).fetchdf()
    assert len(community_symbol_lists) == 1
    assert community_symbol_lists.loc[0, "graph_layer"] == "return_corr_graph"
    assert community_symbol_lists.loc[0, "member_count"] == 2
    assert community_symbol_lists.loc[0, "member_symbols"] == "AAA,BBB"
    community_metrics = connection.execute(
        """
        SELECT
            top_sector,
            top_sector_ratio,
            known_sector_ratio,
            top_industry,
            top_industry_ratio,
            known_industry_ratio,
            known_market_cap_ratio
        FROM read_parquet(?)
        """,
        [str(output_dir / "graph" / "community_metrics.parquet")],
    ).fetchdf()
    assert community_metrics.loc[0, "top_sector"] == "TECH"
    assert community_metrics.loc[0, "top_sector_ratio"] == 1.0
    assert community_metrics.loc[0, "known_sector_ratio"] == 0.5
    assert community_metrics.loc[0, "top_industry"] == "SOFT"
    assert community_metrics.loc[0, "top_industry_ratio"] == 1.0
    assert community_metrics.loc[0, "known_industry_ratio"] == 0.5
    assert community_metrics.loc[0, "known_market_cap_ratio"] == 1.0
    review_candidates = connection.execute(
        "SELECT review_reason FROM read_csv_auto(?)",
        [str(output_dir / "graph" / "layer_review_candidates.csv")],
    ).fetchdf()
    assert review_candidates.loc[0, "review_reason"] != "sector_concentrated"
    metadata_coverage = connection.execute(
        "SELECT * FROM read_csv_auto(?)",
        [str(output_dir / "graph" / "metadata_coverage_report.csv")],
    ).fetchdf()
    assert metadata_coverage.loc[0, "sector_coverage_ratio"] == 0.5
    assert metadata_coverage.loc[0, "industry_coverage_ratio"] == 0.5
    assert metadata_coverage.loc[0, "market_cap_coverage_ratio"] == 1.0
    symbol_master = connection.execute(
        "SELECT symbol, exchange, country, quote_type FROM read_csv_auto(?) ORDER BY symbol",
        [str(output_dir / "market" / "symbol_master.csv")],
    ).fetchdf()
    assert symbol_master.to_dict(orient="records")[0] == {
        "symbol": "AAA",
        "exchange": "NMS",
        "country": "United States",
        "quote_type": "EQUITY",
    }
    symbol_features = connection.execute(
        """
        SELECT
            symbol,
            graph_input_feature_timestamp,
            graph_input_available_time,
            ret_1m,
            flow_feature_timestamp,
            flow_available_time,
            flow_trade_count
        FROM read_parquet(?)
        ORDER BY symbol
        """,
        [str(output_dir / "market" / "symbol_snapshot_features" / "*.parquet")],
    ).fetchdf()
    assert len(symbol_features) == 2
    assert symbol_features.loc[0, "symbol"] == "AAA"
    assert str(symbol_features.loc[0, "graph_input_feature_timestamp"]) == "2025-01-02 22:34:00"
    assert str(symbol_features.loc[0, "graph_input_available_time"]) == "2025-01-02 22:35:00"
    assert symbol_features.loc[0, "ret_1m"] == 0.009
    assert str(symbol_features.loc[0, "flow_feature_timestamp"]) == "2025-01-02 22:34:00"
    assert str(symbol_features.loc[0, "flow_available_time"]) == "2025-01-02 22:35:00"
    assert symbol_features.loc[0, "flow_trade_count"] == 14.0

    symbol_labels = connection.execute(
        """
        SELECT
            symbol,
            label_source_timestamp,
            label_available_time,
            future_ret_1m,
            excess_future_ret_1m,
            benchmark_label_source,
            benchmark_proxy_price_method
        FROM read_parquet(?)
        ORDER BY symbol
        """,
        [str(output_dir / "market" / "symbol_forward_labels" / "*.parquet")],
    ).fetchdf()
    assert len(symbol_labels) == 2
    assert str(symbol_labels.loc[0, "label_source_timestamp"]) == "2025-01-02 22:34:00"
    assert str(symbol_labels.loc[0, "label_available_time"]) == "2025-01-02 22:35:00"
    assert symbol_labels.loc[0, "future_ret_1m"] == 0.003
    assert round(symbol_labels.loc[0, "excess_future_ret_1m"], 6) == round(0.003 - 0.0006, 6)
    assert symbol_labels.loc[0, "benchmark_label_source"] == "labels_1m"
    assert pd.isna(symbol_labels.loc[0, "benchmark_proxy_price_method"])

    community_snapshot_features = connection.execute(
        "SELECT * FROM read_parquet(?)",
        [str(output_dir / "market" / "community_snapshot_features.parquet")],
    ).fetchdf()
    assert len(community_snapshot_features) == 1
    assert community_snapshot_features.loc[0, "community_member_count"] == 2
    assert community_snapshot_features.loc[0, "edge_density"] == 1.0
    assert community_snapshot_features.loc[0, "feature_coverage_ratio"] == 1.0
    assert pd.notna(community_snapshot_features.loc[0, "community_mean_bar_ret_5m_past"])
    assert "community_mean_bar_ret_15m_past" in community_snapshot_features.columns
    assert community_snapshot_features.loc[0, "positive_large_trade_breadth"] == 1.0

    community_forward_labels = connection.execute(
        "SELECT * FROM read_parquet(?)",
        [str(output_dir / "market" / "community_forward_labels.parquet")],
    ).fetchdf()
    assert len(community_forward_labels) == 1
    assert community_forward_labels.loc[0, "benchmark_label_source"] == "labels_1m"
    assert pd.isna(community_forward_labels.loc[0, "benchmark_proxy_price_method"])
    assert round(community_forward_labels.loc[0, "community_equal_weight_excess_future_ret_1m"], 6) == round((0.0024 + 0.0014) / 2, 6)
    assert round(community_forward_labels.loc[0, "community_mean_excess_future_ret_1m"], 6) == round((0.0024 + 0.0014) / 2, 6)
    assert round(community_forward_labels.loc[0, "community_member_weight_excess_future_ret_1m"], 6) == round(((0.0024 * 0.9) + (0.0014 * 0.8)) / (0.9 + 0.8), 6)
    assert round(community_forward_labels.loc[0, "community_top5_member_excess_future_ret_1m"], 6) == round((0.0024 + 0.0014) / 2, 6)
    assert round(community_forward_labels.loc[0, "community_top10_member_excess_future_ret_1m"], 6) == round((0.0024 + 0.0014) / 2, 6)
    expected_core_weighted_excess_1m = ((0.0024 * 1.0) + (0.0014 * 0.9)) / (1.0 + 0.9)
    assert round(community_forward_labels.loc[0, "community_core_weighted_excess_future_ret_1m"], 6) == round(expected_core_weighted_excess_1m, 6)
    assert community_forward_labels.loc[0, "community_core_weighted_excess_future_ret_1m"] > community_forward_labels.loc[0, "community_equal_weight_excess_future_ret_1m"]

    alpha_report = connection.execute(
        "SELECT * FROM read_csv_auto(?)",
        [str(output_dir / "market" / "alpha_sanity_report.csv")],
    ).fetchdf()
    assert len(alpha_report) > 0
    assert {
        "community_member_count",
        "edge_density_feature",
        "community_quality_score",
    }.issubset(set(alpha_report["factor_name"]))
    assert {
        "equal_weight",
        "member_weight",
        "top5_member",
        "top10_member",
        "core_weighted",
    }.issubset(set(alpha_report["label_variant"]))
    assert "flow_member_count_z" not in set(alpha_report["factor_name"])
    alpha_ranking = connection.execute(
        "SELECT * FROM read_csv_auto(?)",
        [str(output_dir / "market" / "alpha_feature_ranking_by_layer.csv")],
    ).fetchdf()
    assert len(alpha_ranking) == len(alpha_report)
    assert {"score", "confidence_bucket", "research_action", "layer_role"}.issubset(alpha_ranking.columns)
    benchmark_label_source_summary = connection.execute(
        "SELECT * FROM read_csv_auto(?)",
        [str(output_dir / "market" / "benchmark_label_source_summary.csv")],
    ).fetchdf()
    assert len(benchmark_label_source_summary) > 0
    assert "benchmark_label_source" in benchmark_label_source_summary.columns
    assert benchmark_label_source_summary.loc[0, "row_count"] >= 1
    metadata_policy = json.loads((output_dir / "market" / "metadata_trust_policy.json").read_text(encoding="utf-8"))
    assert "safe_model_features" in metadata_policy
    connection.close()


def test_build_graph_evaluation_pack_synthesizes_benchmark_labels_from_trade_flow_when_missing(tmp_path):
    graph_database_path = tmp_path / "graph.duckdb"
    market_database_path = tmp_path / "market.duckdb"
    metadata_csv_path = tmp_path / "input_symbols.csv"
    output_dir = tmp_path / "evaluation_pack"

    _create_graph_database(graph_database_path)
    _create_market_database(market_database_path)
    metadata_csv_path.write_text(
        "\n".join(
            [
                "symbol,source_symbol,company_name,sector_code,industry_code,last_price,rank,market_cap,exchange,country,quote_type",
                "AAA,AAA,Alpha,TECH,SOFT,10.1,1,100000000,NMS,United States,EQUITY",
                "BBB,BBB,Beta,,,20.2,2,200000000,NYQ,United States,EQUITY",
                "SPY,SPY,SPDR S&P 500 ETF Trust,ETF,INDEX,500,0,0,PCX,United States,ETF",
            ]
        ),
        encoding="utf-8",
    )

    connection = duckdb.connect(str(market_database_path))
    connection.execute("DELETE FROM labels_1m WHERE symbol = 'SPY'")
    connection.close()

    labels_partition = tmp_path / "labels_1m" / "date=2025-01-02" / "labels_1m.parquet"
    labels_frame = pd.read_parquet(labels_partition)
    labels_frame = labels_frame.loc[labels_frame["symbol"] != "SPY"].copy()
    labels_frame.to_parquet(labels_partition, index=False)

    build_graph_evaluation_pack(
        GraphEvaluationPackConfig(
            graph_database_path=graph_database_path,
            market_database_path=market_database_path,
            metadata_csv_path=metadata_csv_path,
            output_dir=output_dir,
            date_start="2025-01-02",
            date_end="2025-01-02",
            benchmark_symbols=("SPY",),
        )
    )

    connection = duckdb.connect()
    symbol_labels = connection.execute(
        """
        SELECT
            symbol,
            benchmark_future_ret_1m,
            excess_future_ret_1m,
            benchmark_label_source,
            benchmark_proxy_price_method
        FROM read_parquet(?)
        ORDER BY symbol
        """,
        [str(output_dir / "market" / "symbol_forward_labels" / "*.parquet")],
    ).fetchdf()
    assert symbol_labels["benchmark_future_ret_1m"].notna().all()
    expected_trade_flow_benchmark_ret = (2500000.0 / 5000.0) / (2497251.4994 / 4998.0) - 1.0
    assert round(float(symbol_labels.loc[0, "benchmark_future_ret_1m"]), 6) == round(expected_trade_flow_benchmark_ret, 6)
    assert symbol_labels["benchmark_label_source"].eq("trade_flow_proxy").all()
    assert symbol_labels["benchmark_proxy_price_method"].eq("dollar_volume_over_volume").all()
    alpha_report = connection.execute(
        """
        SELECT MAX(sample_size)
        FROM read_csv_auto(?)
        """,
        [str(output_dir / "market" / "alpha_sanity_report.csv")],
    ).fetchone()[0]
    assert alpha_report > 0
    connection.close()


def test_export_alpha_feature_ranking_report_scores_confidence_and_actions(tmp_path):
    alpha_report_path = tmp_path / "alpha_sanity_report.csv"
    ranking_output_path = tmp_path / "alpha_feature_ranking_by_layer.csv"
    pd.DataFrame(
        [
            {
                "graph_layer": "volume_expansion_graph",
                "factor_name": "community_mean_volume_z_12",
                "label_horizon": "15m",
                "label_variant": "equal_weight",
                "sample_size": 12000,
                "rank_ic": 0.05,
                "top_decile_mean": 0.0020,
                "bottom_decile_mean": 0.0010,
                "top_bottom_spread": 0.0010,
                "top_decile_hit_rate": 0.60,
            },
            {
                "graph_layer": "flow_alignment_graph",
                "factor_name": "positive_flow_breadth",
                "label_horizon": "5m",
                "label_variant": "equal_weight",
                "sample_size": 2200,
                "rank_ic": 0.03,
                "top_decile_mean": 0.0007,
                "bottom_decile_mean": 0.0002,
                "top_bottom_spread": 0.0005,
                "top_decile_hit_rate": 0.55,
            },
                {
                    "graph_layer": "return_corr_graph",
                    "factor_name": "edge_density_feature",
                    "label_horizon": "15m",
                    "label_variant": "equal_weight",
                    "sample_size": 15000,
                "rank_ic": 0.02,
                "top_decile_mean": -0.0003,
                "bottom_decile_mean": 0.0001,
                "top_bottom_spread": -0.0004,
                "top_decile_hit_rate": 0.47,
            },
            {
                "graph_layer": "large_trade_alignment_graph",
                "factor_name": "community_avg_weight_feature",
                "label_horizon": "30m",
                "label_variant": "top5_member",
                "sample_size": 120,
                "rank_ic": 0.40,
                "top_decile_mean": 0.0100,
                "bottom_decile_mean": 0.0010,
                "top_bottom_spread": 0.0090,
                "top_decile_hit_rate": 0.75,
            },
        ]
    ).to_csv(alpha_report_path, index=False)

    graph_pack_service._export_alpha_feature_ranking_report(alpha_report_path, ranking_output_path)

    ranking = pd.read_csv(ranking_output_path)
    volume_row = ranking.loc[ranking["graph_layer"] == "volume_expansion_graph"].iloc[0]
    assert volume_row["confidence_bucket"] == "strong_sample"
    assert volume_row["layer_role"] == "theme_candidate_layer"
    assert volume_row["research_action"] == "prioritize_for_next_round"
    assert volume_row["score"] > 0

    flow_row = ranking.loc[ranking["graph_layer"] == "flow_alignment_graph"].iloc[0]
    assert flow_row["confidence_bucket"] == "watch"
    assert flow_row["layer_role"] == "event_alignment_layer"

    return_corr_row = ranking.loc[ranking["graph_layer"] == "return_corr_graph"].iloc[0]
    assert return_corr_row["confidence_bucket"] == "strong_sample"
    assert return_corr_row["score"] < 0
    assert return_corr_row["research_action"] == "downgrade"

    large_trade_row = ranking.loc[ranking["graph_layer"] == "large_trade_alignment_graph"].iloc[0]
    assert large_trade_row["confidence_bucket"] == "ignore"
    assert large_trade_row["research_action"] == "ignore_sparse"


def test_export_cross_window_alpha_comparison_report_flags_stability_and_sample_quality(tmp_path):
    first_window_path = tmp_path / "first_window_alpha_feature_ranking.csv"
    second_window_path = tmp_path / "second_window_alpha_feature_ranking.csv"
    comparison_output_path = tmp_path / "cross_window_alpha_comparison.csv"

    pd.DataFrame(
        [
            {
                "graph_layer": "volume_expansion_graph",
                "layer_role": "theme_candidate_layer",
                "factor_name": "community_quality_score",
                "label_horizon": "30m",
                "label_variant": "equal_weight",
                "sample_size": 12000,
                "rank_ic": 0.05,
                "top_bottom_spread": 0.0020,
                "top_decile_hit_rate": 0.61,
                "score": 0.20,
                "confidence_bucket": "strong_sample",
                "research_action": "prioritize_for_next_round",
            },
            {
                "graph_layer": "flow_alignment_graph",
                "layer_role": "event_alignment_layer",
                "factor_name": "flow_layer_participation_ratio",
                "label_horizon": "15m",
                "label_variant": "equal_weight",
                "sample_size": 4200,
                "rank_ic": 0.06,
                "top_bottom_spread": 0.0015,
                "top_decile_hit_rate": 0.59,
                "score": 0.18,
                "confidence_bucket": "usable",
                "research_action": "keep_for_next_round",
            },
            {
                "graph_layer": "return_corr_graph",
                "layer_role": "beta_context_layer",
                "factor_name": "edge_density_feature",
                "label_horizon": "15m",
                "label_variant": "equal_weight",
                "sample_size": 16000,
                "rank_ic": -0.03,
                "top_bottom_spread": -0.0011,
                "top_decile_hit_rate": 0.45,
                "score": -0.13,
                "confidence_bucket": "strong_sample",
                "research_action": "downgrade",
            },
            {
                "graph_layer": "large_trade_alignment_graph",
                "layer_role": "sparse_event_flag",
                "factor_name": "community_quality_score",
                "label_horizon": "30m",
                "label_variant": "equal_weight",
                "sample_size": 220,
                "rank_ic": 0.20,
                "top_bottom_spread": 0.0040,
                "top_decile_hit_rate": 0.62,
                "score": 0.11,
                "confidence_bucket": "ignore",
                "research_action": "ignore_sparse",
            },
        ]
    ).to_csv(first_window_path, index=False)

    pd.DataFrame(
        [
            {
                "graph_layer": "volume_expansion_graph",
                "layer_role": "theme_candidate_layer",
                "factor_name": "community_quality_score",
                "label_horizon": "30m",
                "label_variant": "equal_weight",
                "sample_size": 11800,
                "rank_ic": 0.04,
                "top_bottom_spread": 0.0017,
                "top_decile_hit_rate": 0.58,
                "score": 0.16,
                "confidence_bucket": "strong_sample",
                "research_action": "prioritize_for_next_round",
            },
            {
                "graph_layer": "flow_alignment_graph",
                "layer_role": "event_alignment_layer",
                "factor_name": "flow_layer_participation_ratio",
                "label_horizon": "15m",
                "label_variant": "equal_weight",
                "sample_size": 4100,
                "rank_ic": -0.02,
                "top_bottom_spread": -0.0006,
                "top_decile_hit_rate": 0.48,
                "score": -0.07,
                "confidence_bucket": "usable",
                "research_action": "downgrade",
            },
            {
                "graph_layer": "return_corr_graph",
                "layer_role": "beta_context_layer",
                "factor_name": "edge_density_feature",
                "label_horizon": "15m",
                "label_variant": "equal_weight",
                "sample_size": 15200,
                "rank_ic": -0.02,
                "top_bottom_spread": -0.0008,
                "top_decile_hit_rate": 0.46,
                "score": -0.09,
                "confidence_bucket": "strong_sample",
                "research_action": "downgrade",
            },
            {
                "graph_layer": "large_trade_alignment_graph",
                "layer_role": "sparse_event_flag",
                "factor_name": "community_quality_score",
                "label_horizon": "30m",
                "label_variant": "equal_weight",
                "sample_size": 240,
                "rank_ic": 0.12,
                "top_bottom_spread": 0.0020,
                "top_decile_hit_rate": 0.57,
                "score": 0.08,
                "confidence_bucket": "ignore",
                "research_action": "ignore_sparse",
            },
        ]
    ).to_csv(second_window_path, index=False)

    graph_pack_service._export_cross_window_alpha_comparison_report(
        first_window_path,
        second_window_path,
        comparison_output_path,
        first_window_id="2025-01-06_2025-01-17",
        second_window_id="2025-01-21_2025-01-31",
    )

    comparison = pd.read_csv(comparison_output_path)

    volume_row = comparison.loc[comparison["graph_layer"] == "volume_expansion_graph"].iloc[0]
    assert bool(volume_row["score_direction_consistent"]) is True
    assert bool(volume_row["sample_qualified_both"]) is True
    assert volume_row["stability_bucket"] == "stable_positive"
    assert volume_row["research_decision"] == "confirm_layer_role"

    flow_row = comparison.loc[comparison["graph_layer"] == "flow_alignment_graph"].iloc[0]
    assert bool(flow_row["score_direction_consistent"]) is False
    assert flow_row["stability_bucket"] == "unstable_direction"
    assert flow_row["research_decision"] == "review_manually"

    return_corr_row = comparison.loc[comparison["graph_layer"] == "return_corr_graph"].iloc[0]
    assert bool(return_corr_row["score_direction_consistent"]) is True
    assert return_corr_row["stability_bucket"] == "stable_negative"
    assert return_corr_row["research_decision"] == "deprioritize"

    large_trade_row = comparison.loc[comparison["graph_layer"] == "large_trade_alignment_graph"].iloc[0]
    assert bool(large_trade_row["sample_qualified_both"]) is False
    assert large_trade_row["stability_bucket"] == "insufficient_sample"
    assert large_trade_row["research_decision"] == "needs_more_sample"


def test_alpha_factors_are_layer_aware():
    assert graph_pack_service._alpha_factors_for_layer("volume_expansion_graph") == [
        "edge_density_feature",
        "community_avg_weight_feature",
        "feature_coverage_ratio",
        "community_quality_score",
        "community_mean_volume_z_12",
    ]
    assert graph_pack_service._alpha_factors_for_layer("flow_alignment_graph") == [
        "community_member_count",
        "flow_member_count_z",
        "flow_layer_participation_ratio",
        "flow_breadth_expansion",
        "community_mean_flow_impulse_score",
        "community_quality_score",
    ]
    assert graph_pack_service._alpha_factors_for_layer("return_corr_graph") == [
        "community_member_count",
        "edge_density_feature",
        "community_quality_score",
    ]


def test_resolve_generator_metadata_tolerates_output_dir_outside_repo(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    output_dir = tmp_path / "outside-pack"
    repo_root.mkdir()
    output_dir.mkdir()

    def fake_git_output(cwd: Path, args: list[str]) -> str | None:
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return str(repo_root)
        if command == ("rev-parse", "HEAD"):
            return "head123"
        if command == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "branch-x"
        if command == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M data/generated.parquet\n M src/real_code.py\n"
        raise AssertionError(f"Unexpected git args: {args}")

    monkeypatch.setattr(graph_pack_service, "_git_output", fake_git_output)

    metadata = graph_pack_service._resolve_generator_metadata(
        provided_metadata=None,
        output_dir=output_dir,
    )

    assert metadata["git_head"] == "head123"
    assert metadata["git_branch"] == "branch-x"
    assert metadata["repo_worktree_dirty"] is True
    assert metadata["relevant_worktree_dirty"] is True
    assert metadata["dirty_paths"] == ["data/generated.parquet", "src/real_code.py"]
    assert metadata["relevant_dirty_paths"] == ["src/real_code.py"]


def test_parse_git_status_paths_handles_leading_space_status_codes():
    status_output = " M data/bars_15m/date=2026-06-09/bars_15m.parquet\n?? data/stocknet_us.duckdb\n"

    paths = graph_pack_service._parse_git_status_paths(status_output)

    assert paths == [
        "data/bars_15m/date=2026-06-09/bars_15m.parquet",
        "data/stocknet_us.duckdb",
    ]


def test_git_output_preserves_leading_spaces(monkeypatch):
    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Completed(" M data/example.parquet\n")

    monkeypatch.setattr(graph_pack_service.subprocess, "run", fake_run)

    output = graph_pack_service._git_output(Path("D:/DEV/stocknetwork/StockNet"), ["status"])

    assert output == " M data/example.parquet"
