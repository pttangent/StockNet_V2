from __future__ import annotations

import json

import duckdb
import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate
from stocknetv2.application.services.lifecycle_service import LifecycleRecord
from stocknetv2.application.services.semantic_service import SemanticLabelRecord
from stocknetv2.application.services.theme_flow_service import ThemeFlowRecord


class ThemeWriteRepository:
    """Persist consensus theme candidates and theme memberships."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def save_consensus_themes(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        snapshot_time: pd.Timestamp,
        candidates: list[ConsensusThemeCandidate],
    ) -> None:
        for candidate in candidates:
            self._connection.execute(
                """
                INSERT INTO consensus_theme_candidate (
                    theme_instance_id, run_id, snapshot_id, trade_date, timestamp, theme_path_id,
                    members_json, member_count, source_layers_json, consensus_score,
                    structure_score, cross_layer_consensus_score, flow_support_score, dtw_flow_support_score,
                    volume_support_score, large_trade_support_score, stability_score,
                    semantic_coherence_score, theme_quality_score, theme_quality_breakdown_json,
                    keep_status, reject_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    candidate.theme_instance_id,
                    run_id,
                    snapshot_id,
                    trade_date,
                    snapshot_time,
                    candidate.theme_path_id,
                    json.dumps(candidate.members),
                    len(candidate.members),
                    json.dumps(candidate.source_layers),
                    candidate.consensus_score,
                    candidate.structure_score,
                    candidate.cross_layer_consensus_score,
                    candidate.flow_support_score,
                    candidate.dtw_flow_support_score,
                    candidate.volume_support_score,
                    candidate.large_trade_support_score,
                    candidate.stability_score,
                    candidate.semantic_coherence_score,
                    candidate.theme_quality_score,
                    candidate.theme_quality_breakdown_json,
                    "keep",
                    "",
                ],
            )
            for member_rank, symbol in enumerate(candidate.members, start=1):
                self._connection.execute(
                    """
                    INSERT INTO theme_membership (
                        theme_instance_id, run_id, snapshot_id, theme_path_id, trade_date, symbol,
                        member_rank, contribution_score, return_contribution, flow_contribution,
                        dtw_flow_contribution, large_trade_contribution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        candidate.theme_instance_id,
                        run_id,
                        snapshot_id,
                        candidate.theme_path_id,
                        trade_date,
                        symbol,
                        member_rank,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                )

    def save_semantic_labels(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        labels: list[SemanticLabelRecord],
    ) -> None:
        for label in labels:
            self._connection.execute(
                """
                INSERT INTO theme_semantic_label (
                    theme_instance_id, run_id, snapshot_id, label_short, label_long, sector_summary,
                    industry_summary, bucket_tags_json, top_companies_json, semantic_coherence_score,
                    explanation, semantic_method, semantic_metadata_json, semantic_prompt_text,
                    dictionary_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    label.theme_instance_id,
                    run_id,
                    snapshot_id,
                    label.label_short,
                    label.label_long,
                    label.sector_summary,
                    label.industry_summary,
                    label.bucket_tags_json,
                    label.top_companies_json,
                    label.semantic_coherence_score,
                    label.explanation,
                    label.semantic_method,
                    label.semantic_metadata_json,
                    label.semantic_prompt_text,
                    label.dictionary_version,
                ],
            )

    def save_lifecycle_records(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        records: list[LifecycleRecord],
    ) -> None:
        for record in records:
            self._connection.execute(
                """
                INSERT INTO theme_path_lifecycle (
                    theme_path_id, theme_instance_id, run_id, snapshot_id, timestamp, event_type,
                    age_frames, duration_minutes, match_score, previous_theme_instance_id, member_retention,
                    status, transition_parent_path_id, transition_child_path_id, transition_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.theme_path_id,
                    record.theme_instance_id,
                    run_id,
                    snapshot_id,
                    record.timestamp,
                    record.event_type,
                    record.age_frames,
                    record.duration_minutes,
                    record.match_score,
                    record.previous_theme_instance_id,
                    record.member_retention,
                    record.status,
                    record.transition_parent_path_id,
                    record.transition_child_path_id,
                    record.transition_kind,
                ],
            )

    def save_theme_flow_records(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        records: list[ThemeFlowRecord],
    ) -> None:
        for record in records:
            self._connection.execute(
                """
                INSERT INTO theme_level_flow_series (
                    theme_instance_id, theme_path_id, run_id, snapshot_id, timestamp,
                    theme_net_flow, theme_inflow, theme_outflow, flow_breadth, price_breadth,
                    dtw_flow_coherence, large_trade_breadth, member_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.theme_instance_id,
                    record.theme_path_id,
                    run_id,
                    snapshot_id,
                    record.timestamp,
                    record.theme_net_flow,
                    record.theme_inflow,
                    record.theme_outflow,
                    record.flow_breadth,
                    record.price_breadth,
                    record.dtw_flow_coherence,
                    record.large_trade_breadth,
                    record.member_count,
                ],
            )
