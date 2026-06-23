from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from stocknetv2.domain.graph.layer_config import ConsensusConfig
from stocknetv2.domain.community.community import Community
from stocknetv2.domain.community.consensus_matrix import build_consensus_matrix
from stocknetv2.domain.community.detector import detect_communities_from_edges
from stocknetv2.domain.graph.edge import GraphEdge


@dataclass(frozen=True)
class ConsensusThemeCandidate:
    theme_instance_id: str
    theme_path_id: str
    members: list[str]
    source_layers: list[str]
    consensus_score: float
    structure_score: float
    cross_layer_consensus_score: float
    flow_support_score: float
    dtw_flow_support_score: float
    volume_support_score: float
    large_trade_support_score: float
    stability_score: float
    semantic_coherence_score: float
    theme_quality_score: float
    theme_quality_breakdown_json: str
    distinct_family_count: int = 0
    member_ratio: float = 0.0
    is_market_mode: bool = False


class ConsensusService:
    """Aggregate per-layer communities into minimal consensus theme candidates."""

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self._config = config or ConsensusConfig()

    def build_consensus_themes(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        snapshot_time: pd.Timestamp,
        layer_communities: dict[str, list[Community]],
    ) -> list[ConsensusThemeCandidate]:
        communities_payload = {
            layer_name: [community.members for community in communities if not community.is_market_mode]
            for layer_name, communities in layer_communities.items()
            if communities
        }
        if not communities_payload:
            return []

        matrix = build_consensus_matrix(communities_payload, self._config.layer_weights)
        threshold_edges: list[GraphEdge] = []
        for left_symbol, right_symbol in combinations(sorted(matrix.index.tolist()), 2):
            score = float(matrix.loc[left_symbol, right_symbol])
            if score >= self._config.min_consensus_score:
                threshold_edges.append(
                    GraphEdge(
                        graph_layer="consensus_graph",
                        edge_type="consensus_coassignment",
                        source_symbol=left_symbol,
                        target_symbol=right_symbol,
                        snapshot_time=snapshot_time,
                        weight=score,
                        raw_score=score,
                        support_points=1,
                    )
                )

        communities = detect_communities_from_edges(
            threshold_edges,
            min_members=self._config.min_members,
            algorithm=self._config.community_detection.algorithm,
            resolution=self._config.community_detection.resolution,
            universe_symbol_count=len(matrix.index),
            market_mode_max_member_ratio=self._config.community_detection.market_mode_max_member_ratio,
            fallback_algorithm=self._config.community_detection.fallback_algorithm,
        )
        candidates: list[ConsensusThemeCandidate] = []
        for index, community in enumerate(communities, start=1):
            source_layers = sorted(
                layer_name
                for layer_name, members_list in communities_payload.items()
                if any(len(set(members) & set(community.members)) >= 2 for members in members_list)
            )
            distinct_families = sorted(
                {self._config.family_map.get(layer_name, layer_name) for layer_name in source_layers}
            )
            if len(community.members) < self._config.min_members:
                continue
            if len(distinct_families) < self._config.min_distinct_families:
                continue
            if community.is_market_mode:
                continue
            pair_scores = []
            for left_symbol, right_symbol in combinations(community.members, 2):
                pair_scores.append(float(matrix.loc[left_symbol, right_symbol]))
            consensus_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
            size_penalty = min(1.0, 8.0 / max(len(community.members), 1))
            structure_score = min(1.0, consensus_score * (0.6 + 0.4 * size_penalty))
            quality_breakdown = json.dumps(
                {
                    "version": "consensus_v2",
                    "weights": self._config.layer_weights,
                    "consensus_score": consensus_score,
                    "source_layers": source_layers,
                    "distinct_families": distinct_families,
                    "size_penalty": size_penalty,
                    "member_ratio": community.universe_ratio or 0.0,
                    "market_mode_filtered": community.is_market_mode,
                }
            )
            candidates.append(
                ConsensusThemeCandidate(
                    theme_instance_id=f"{snapshot_id}_theme_{index:03d}",
                    theme_path_id=f"{run_id}_path_{index:03d}",
                    members=community.members,
                    source_layers=source_layers,
                    consensus_score=consensus_score,
                    structure_score=structure_score,
                    cross_layer_consensus_score=consensus_score,
                    flow_support_score=1.0 if "flow_alignment_graph" in source_layers else 0.0,
                    dtw_flow_support_score=1.0 if "dtw_trade_flow_similarity_graph" in source_layers else 0.0,
                    volume_support_score=1.0 if "volume_expansion_graph" in source_layers else 0.0,
                    large_trade_support_score=1.0 if "large_trade_alignment_graph" in source_layers else 0.0,
                    stability_score=0.0,
                    semantic_coherence_score=0.0,
                    theme_quality_score=structure_score,
                    theme_quality_breakdown_json=quality_breakdown,
                    distinct_family_count=len(distinct_families),
                    member_ratio=community.universe_ratio or 0.0,
                    is_market_mode=community.is_market_mode,
                )
            )
        return candidates
