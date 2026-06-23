from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate
from stocknetv2.application.services.lifecycle_service import LifecycleRecord
from stocknetv2.application.services.semantic_service import SemanticLabelRecord


@dataclass(frozen=True)
class SnapshotCacheRecord:
    snapshot_id: str
    run_id: str
    timestamp: pd.Timestamp
    cache_type: str
    payload_json: str
    payload_version: str


class ReadModelService:
    """Build frontend/read-only cache payloads from persisted theme outputs."""

    def build_snapshot_caches(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        timestamp: pd.Timestamp,
        candidates: list[ConsensusThemeCandidate],
        semantic_labels: list[SemanticLabelRecord],
        lifecycle_records: list[LifecycleRecord],
    ) -> list[SnapshotCacheRecord]:
        semantic_by_id = {record.theme_instance_id: record for record in semantic_labels}
        lifecycle_by_id = {record.theme_instance_id: record for record in lifecycle_records}
        payload = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            "themes": [
                {
                    "theme_instance_id": candidate.theme_instance_id,
                    "theme_path_id": candidate.theme_path_id,
                    "members": candidate.members,
                    "source_layers": candidate.source_layers,
                    "consensus_score": candidate.consensus_score,
                    "theme_quality_score": candidate.theme_quality_score,
                    "label_short": semantic_by_id[candidate.theme_instance_id].label_short
                    if candidate.theme_instance_id in semantic_by_id
                    else "",
                    "event_type": lifecycle_by_id[candidate.theme_instance_id].event_type
                    if candidate.theme_instance_id in lifecycle_by_id
                    else "",
                }
                for candidate in candidates
            ],
        }
        return [
            SnapshotCacheRecord(
                snapshot_id=snapshot_id,
                run_id=run_id,
                timestamp=timestamp,
                cache_type="snapshot_summary",
                payload_json=json.dumps(payload),
                payload_version="v1",
            )
        ]
