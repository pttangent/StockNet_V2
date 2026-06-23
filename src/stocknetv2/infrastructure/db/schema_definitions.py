from __future__ import annotations


TABLE_SCHEMAS: dict[str, str] = {
    "config_registry": """
        CREATE TABLE IF NOT EXISTS config_registry (
            config_id TEXT PRIMARY KEY,
            config_name TEXT NOT NULL,
            config_scope TEXT NOT NULL,
            config_json TEXT NOT NULL,
            config_version TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "theme_discovery_run": """
        CREATE TABLE IF NOT EXISTS theme_discovery_run (
            run_id TEXT PRIMARY KEY,
            run_name TEXT,
            date_start DATE NOT NULL,
            date_end DATE NOT NULL,
            frame_minutes INTEGER NOT NULL,
            config_id TEXT NOT NULL,
            config_json TEXT NOT NULL,
            code_commit TEXT,
            data_version TEXT,
            status TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "input_lineage": """
        CREATE TABLE IF NOT EXISTS input_lineage (
            lineage_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            snapshot_id TEXT,
            source_kind TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_path TEXT,
            source_version TEXT,
            source_min_timestamp TIMESTAMP,
            source_max_timestamp TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "graph_snapshot": """
        CREATE TABLE IF NOT EXISTS graph_snapshot (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            frame_minutes INTEGER NOT NULL,
            market_session TEXT,
            graph_status TEXT NOT NULL,
            available_minutes_since_open INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "graph_edge_summary": """
        CREATE TABLE IF NOT EXISTS graph_edge_summary (
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            graph_layer TEXT NOT NULL,
            edge_count INTEGER NOT NULL,
            node_count INTEGER NOT NULL,
            avg_weight DOUBLE,
            median_weight DOUBLE,
            p90_weight DOUBLE,
            threshold DOUBLE,
            top_k_per_symbol INTEGER,
            effective_lookback_minutes INTEGER,
            PRIMARY KEY (snapshot_id, graph_layer)
        )
    """,
    "graph_layer_diagnostic": """
        CREATE TABLE IF NOT EXISTS graph_layer_diagnostic (
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            graph_layer TEXT NOT NULL,
            active_node_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            average_degree DOUBLE,
            degree_p50 DOUBLE,
            degree_p95 DOUBLE,
            max_degree INTEGER,
            edge_score_p50 DOUBLE,
            edge_score_p90 DOUBLE,
            support_points_p50 DOUBLE,
            support_points_p90 DOUBLE,
            connected_component_count INTEGER,
            largest_component_ratio DOUBLE,
            community_count INTEGER,
            community_size_p50 DOUBLE,
            community_size_p95 DOUBLE,
            community_size_max INTEGER,
            market_mode_member_ratio DOUBLE,
            community_method TEXT,
            PRIMARY KEY (snapshot_id, graph_layer)
        )
    """,
    "relation_observation": """
        CREATE TABLE IF NOT EXISTS relation_observation (
            relation_observation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            graph_layer TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            source_symbol TEXT NOT NULL,
            target_symbol TEXT NOT NULL,
            raw_score DOUBLE,
            edge_weight DOUBLE,
            edge_confidence DOUBLE,
            calculation_backend TEXT,
            support_points INTEGER,
            effective_lookback_minutes INTEGER,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            temporal_policy_id TEXT
        )
    """,
    "graph_edges_thresholded": """
        CREATE TABLE IF NOT EXISTS graph_edges_thresholded (
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            graph_layer TEXT NOT NULL,
            source_symbol TEXT NOT NULL,
            target_symbol TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight DOUBLE NOT NULL,
            raw_score DOUBLE,
            edge_confidence DOUBLE,
            effective_lookback_minutes INTEGER,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            support_points INTEGER,
            config_id TEXT,
            PRIMARY KEY (snapshot_id, graph_layer, source_symbol, target_symbol)
        )
    """,
    "temporal_edge_state": """
        CREATE TABLE IF NOT EXISTS temporal_edge_state (
            temporal_edge_state_id TEXT PRIMARY KEY,
            relation_observation_id TEXT,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            graph_layer TEXT NOT NULL,
            source_symbol TEXT NOT NULL,
            target_symbol TEXT NOT NULL,
            raw_score DOUBLE,
            temporal_score DOUBLE,
            support_points INTEGER,
            effective_lookback_minutes INTEGER,
            presence_count INTEGER,
            age_frames INTEGER,
            missing_frames INTEGER,
            entered_at TIMESTAMP,
            last_seen_at TIMESTAMP,
            state TEXT NOT NULL,
            temporal_policy_id TEXT NOT NULL
        )
    """,
    "layer_community": """
        CREATE TABLE IF NOT EXISTS layer_community (
            layer_community_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            graph_layer TEXT NOT NULL,
            community_local_id TEXT NOT NULL,
            members_json TEXT NOT NULL,
            member_count INTEGER NOT NULL,
            edge_count INTEGER,
            edge_density DOUBLE,
            avg_weight DOUBLE,
            min_weight DOUBLE,
            max_weight DOUBLE,
            community_method TEXT NOT NULL
        )
    """,
    "layer_community_membership": """
        CREATE TABLE IF NOT EXISTS layer_community_membership (
            layer_community_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            graph_layer TEXT NOT NULL,
            community_local_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            member_rank INTEGER,
            member_weight DOUBLE,
            PRIMARY KEY (layer_community_id, symbol)
        )
    """,
    "consensus_theme_candidate": """
        CREATE TABLE IF NOT EXISTS consensus_theme_candidate (
            theme_instance_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            trade_date DATE NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            theme_path_id TEXT,
            members_json TEXT NOT NULL,
            member_count INTEGER NOT NULL,
            source_layers_json TEXT NOT NULL,
            consensus_score DOUBLE,
            structure_score DOUBLE,
            cross_layer_consensus_score DOUBLE,
            flow_support_score DOUBLE,
            dtw_flow_support_score DOUBLE,
            volume_support_score DOUBLE,
            large_trade_support_score DOUBLE,
            stability_score DOUBLE,
            semantic_coherence_score DOUBLE,
            theme_quality_score DOUBLE,
            theme_quality_breakdown_json TEXT,
            keep_status TEXT,
            reject_reason TEXT
        )
    """,
    "theme_membership": """
        CREATE TABLE IF NOT EXISTS theme_membership (
            theme_instance_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            theme_path_id TEXT,
            trade_date DATE NOT NULL,
            symbol TEXT NOT NULL,
            member_rank INTEGER,
            contribution_score DOUBLE,
            return_contribution DOUBLE,
            flow_contribution DOUBLE,
            dtw_flow_contribution DOUBLE,
            large_trade_contribution DOUBLE,
            PRIMARY KEY (theme_instance_id, symbol)
        )
    """,
    "theme_semantic_label": """
        CREATE TABLE IF NOT EXISTS theme_semantic_label (
            theme_instance_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            label_short TEXT,
            label_long TEXT,
            sector_summary TEXT,
            industry_summary TEXT,
            bucket_tags_json TEXT,
            top_companies_json TEXT,
            semantic_coherence_score DOUBLE,
            explanation TEXT,
            semantic_method TEXT,
            semantic_metadata_json TEXT,
            semantic_prompt_text TEXT,
            dictionary_version TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "theme_path_lifecycle": """
        CREATE TABLE IF NOT EXISTS theme_path_lifecycle (
            theme_path_id TEXT NOT NULL,
            theme_instance_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            event_type TEXT NOT NULL,
            age_frames INTEGER,
            duration_minutes INTEGER,
            match_score DOUBLE,
            previous_theme_instance_id TEXT,
            member_retention DOUBLE,
            status TEXT,
            transition_parent_path_id TEXT,
            transition_child_path_id TEXT,
            transition_kind TEXT,
            PRIMARY KEY (theme_path_id, theme_instance_id)
        )
    """,
    "theme_level_flow_series": """
        CREATE TABLE IF NOT EXISTS theme_level_flow_series (
            theme_instance_id TEXT NOT NULL,
            theme_path_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            theme_net_flow DOUBLE,
            theme_inflow DOUBLE,
            theme_outflow DOUBLE,
            flow_breadth DOUBLE,
            price_breadth DOUBLE,
            dtw_flow_coherence DOUBLE,
            large_trade_breadth DOUBLE,
            member_count INTEGER,
            PRIMARY KEY (theme_instance_id, timestamp)
        )
    """,
    "frontend_snapshot_cache": """
        CREATE TABLE IF NOT EXISTS frontend_snapshot_cache (
            snapshot_cache_id TEXT PRIMARY KEY,
            snapshot_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            cache_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_version TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """,
}
