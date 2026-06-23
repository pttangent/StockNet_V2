from __future__ import annotations

import json
from dataclasses import replace

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate
from stocknetv2.application.services.lifecycle_service import LifecycleRecord
from stocknetv2.application.services.semantic_service import SemanticLabelRecord


QUALITY_COMPONENT_WEIGHTS: dict[str, float] = {
    "structure_score": 0.20,
    "cross_layer_consensus_score": 0.20,
    "flow_support_score": 0.15,
    "dtw_flow_support_score": 0.15,
    "volume_support_score": 0.10,
    "large_trade_support_score": 0.10,
    "stability_score": 0.05,
    "semantic_coherence_score": 0.05,
}


class ThemeQualityService:
    """Compute a persistent, explainable theme-quality breakdown after enrichment."""

    def score_themes(
        self,
        candidates: list[ConsensusThemeCandidate],
        *,
        semantic_labels: list[SemanticLabelRecord],
        lifecycle_records: list[LifecycleRecord],
    ) -> list[ConsensusThemeCandidate]:
        semantic_by_id = {label.theme_instance_id: label for label in semantic_labels}
        lifecycle_by_id = {record.theme_instance_id: record for record in lifecycle_records}

        scored_candidates: list[ConsensusThemeCandidate] = []
        for candidate in candidates:
            semantic_label = semantic_by_id.get(candidate.theme_instance_id)
            lifecycle_record = lifecycle_by_id.get(candidate.theme_instance_id)

            size_penalty = min(1.0, 8.0 / max(len(candidate.members), 1))
            family_bonus = min(1.0, candidate.distinct_family_count / 3.0)
            market_mode_penalty = 0.25 if candidate.is_market_mode else 1.0
            structure_score = min(
                1.0,
                candidate.structure_score * (0.85 + 0.15 * family_bonus) * market_mode_penalty,
            )
            cross_layer_consensus_score = candidate.consensus_score
            flow_support_score = 1.0 if "flow_alignment_graph" in candidate.source_layers else 0.0
            dtw_flow_support_score = 1.0 if "dtw_trade_flow_similarity_graph" in candidate.source_layers else 0.0
            volume_support_score = 1.0 if "volume_expansion_graph" in candidate.source_layers else 0.0
            large_trade_support_score = 1.0 if "large_trade_alignment_graph" in candidate.source_layers else 0.0
            stability_score = min(1.0, lifecycle_record.age_frames / 12.0) if lifecycle_record else 0.0
            semantic_coherence_score = semantic_label.semantic_coherence_score if semantic_label else 0.0

            component_scores = {
                "structure_score": structure_score,
                "cross_layer_consensus_score": cross_layer_consensus_score,
                "flow_support_score": flow_support_score,
                "dtw_flow_support_score": dtw_flow_support_score,
                "volume_support_score": volume_support_score,
                "large_trade_support_score": large_trade_support_score,
                "stability_score": stability_score,
                "semantic_coherence_score": semantic_coherence_score,
            }
            theme_quality_score = sum(
                component_scores[name] * QUALITY_COMPONENT_WEIGHTS[name] for name in QUALITY_COMPONENT_WEIGHTS
            )
            theme_quality_breakdown_json = json.dumps(
                {
                    "version": "quality_v1",
                    "weights": QUALITY_COMPONENT_WEIGHTS,
                    "component_scores": component_scores,
                    "member_count": len(candidate.members),
                    "member_ratio": candidate.member_ratio,
                    "distinct_family_count": candidate.distinct_family_count,
                    "size_penalty": size_penalty,
                    "source_layers": candidate.source_layers,
                    "theme_quality_score": theme_quality_score,
                }
            )
            scored_candidates.append(
                replace(
                    candidate,
                    structure_score=structure_score,
                    cross_layer_consensus_score=cross_layer_consensus_score,
                    flow_support_score=flow_support_score,
                    dtw_flow_support_score=dtw_flow_support_score,
                    volume_support_score=volume_support_score,
                    large_trade_support_score=large_trade_support_score,
                    stability_score=stability_score,
                    semantic_coherence_score=semantic_coherence_score,
                    theme_quality_score=theme_quality_score,
                    theme_quality_breakdown_json=theme_quality_breakdown_json,
                )
            )
        return scored_candidates
