from __future__ import annotations

import json
from dataclasses import dataclass

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate


@dataclass(frozen=True)
class SemanticLabelRecord:
    theme_instance_id: str
    label_short: str
    label_long: str
    sector_summary: str
    industry_summary: str
    bucket_tags_json: str
    top_companies_json: str
    semantic_coherence_score: float
    explanation: str
    semantic_method: str
    semantic_metadata_json: str
    semantic_prompt_text: str
    dictionary_version: str


class SemanticService:
    """Build deterministic dictionary-first semantic labels for theme candidates."""

    def label_themes(self, candidates: list[ConsensusThemeCandidate]) -> list[SemanticLabelRecord]:
        labels: list[SemanticLabelRecord] = []
        for candidate in candidates:
            top_members = candidate.members[:3]
            label_short = f"Theme: {', '.join(top_members)}"
            labels.append(
                SemanticLabelRecord(
                    theme_instance_id=candidate.theme_instance_id,
                    label_short=label_short,
                    label_long=label_short,
                    sector_summary="Mixed",
                    industry_summary="Mixed",
                    bucket_tags_json=json.dumps(candidate.source_layers),
                    top_companies_json=json.dumps(candidate.members[:5]),
                    semantic_coherence_score=min(1.0, 0.5 + 0.1 * len(candidate.source_layers)),
                    explanation="Dictionary-first label generated from leading members and supporting layers.",
                    semantic_method="dictionary_v1",
                    semantic_metadata_json=json.dumps(
                        {
                            "source_layers": candidate.source_layers,
                            "member_count": len(candidate.members),
                        }
                    ),
                    semantic_prompt_text="",
                    dictionary_version="builtin-v1",
                )
            )
        return labels
