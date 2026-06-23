from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate


@dataclass(frozen=True)
class LifecycleRecord:
    theme_path_id: str
    theme_instance_id: str
    timestamp: pd.Timestamp
    event_type: str
    age_frames: int
    duration_minutes: int
    match_score: float
    previous_theme_instance_id: str | None
    member_retention: float
    status: str
    transition_parent_path_id: str | None = None
    transition_child_path_id: str | None = None
    transition_kind: str | None = None


class LifecycleService:
    """Assign causal theme-path continuity across snapshots."""

    def assign_paths(
        self,
        *,
        candidates: list[ConsensusThemeCandidate],
        previous_candidates: list[ConsensusThemeCandidate],
        previous_lifecycle_records: dict[str, LifecycleRecord],
        timestamp: pd.Timestamp,
        frame_minutes: int,
        min_overlap: float = 0.5,
    ) -> tuple[list[ConsensusThemeCandidate], list[LifecycleRecord]]:
        assigned_candidates: list[ConsensusThemeCandidate] = []
        records: list[LifecycleRecord] = []
        used_previous_theme_ids: set[str] = set()
        current_to_previous_matches = {
            candidate.theme_instance_id: _rank_previous_matches(
                candidate_members=set(candidate.members),
                previous_candidates=previous_candidates,
                min_overlap=min_overlap,
            )
            for candidate in candidates
        }
        previous_to_current_matches = _build_previous_to_current_matches(
            candidates=candidates,
            previous_candidates=previous_candidates,
            min_overlap=min_overlap,
        )

        for candidate in candidates:
            matches = current_to_previous_matches[candidate.theme_instance_id]
            best_previous = matches[0][0] if matches else None
            best_score = matches[0][1] if matches else 0.0
            merge_matches = matches[1:] if len(matches) > 1 else []
            split_match_count = (
                len(previous_to_current_matches.get(best_previous.theme_instance_id, []))
                if best_previous
                else 0
            )

            if best_previous and best_previous.theme_instance_id not in used_previous_theme_ids:
                used_previous_theme_ids.add(best_previous.theme_instance_id)
                previous_record = previous_lifecycle_records[best_previous.theme_instance_id]
                assigned_candidate = replace(candidate, theme_path_id=best_previous.theme_path_id)
                event_type = "merge" if merge_matches else "continuation"
                record = LifecycleRecord(
                    theme_path_id=best_previous.theme_path_id,
                    theme_instance_id=assigned_candidate.theme_instance_id,
                    timestamp=timestamp,
                    event_type=event_type,
                    age_frames=previous_record.age_frames + 1,
                    duration_minutes=(previous_record.age_frames + 1) * frame_minutes,
                    match_score=best_score,
                    previous_theme_instance_id=best_previous.theme_instance_id,
                    member_retention=best_score,
                    status="active",
                    transition_parent_path_id=merge_matches[0][0].theme_path_id if merge_matches else None,
                    transition_child_path_id=best_previous.theme_path_id if merge_matches else None,
                    transition_kind="merge" if merge_matches else None,
                )
            elif best_previous and split_match_count > 1:
                assigned_candidate = candidate
                record = LifecycleRecord(
                    theme_path_id=assigned_candidate.theme_path_id,
                    theme_instance_id=assigned_candidate.theme_instance_id,
                    timestamp=timestamp,
                    event_type="split",
                    age_frames=1,
                    duration_minutes=frame_minutes,
                    match_score=best_score,
                    previous_theme_instance_id=best_previous.theme_instance_id,
                    member_retention=best_score,
                    status="active",
                    transition_parent_path_id=best_previous.theme_path_id,
                    transition_child_path_id=assigned_candidate.theme_path_id,
                    transition_kind="split",
                )
            else:
                assigned_candidate = candidate
                record = LifecycleRecord(
                    theme_path_id=assigned_candidate.theme_path_id,
                    theme_instance_id=assigned_candidate.theme_instance_id,
                    timestamp=timestamp,
                    event_type="birth",
                    age_frames=1,
                    duration_minutes=frame_minutes,
                    match_score=1.0,
                    previous_theme_instance_id=None,
                    member_retention=1.0,
                    status="active",
                )
            assigned_candidates.append(assigned_candidate)
            records.append(record)

        return assigned_candidates, records


def _overlap_small(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _rank_previous_matches(
    *,
    candidate_members: set[str],
    previous_candidates: list[ConsensusThemeCandidate],
    min_overlap: float,
) -> list[tuple[ConsensusThemeCandidate, float]]:
    matches = []
    for previous in previous_candidates:
        score = _overlap_small(candidate_members, set(previous.members))
        if score >= min_overlap:
            matches.append((previous, score))
    return sorted(matches, key=lambda item: (-item[1], item[0].theme_instance_id))


def _build_previous_to_current_matches(
    *,
    candidates: list[ConsensusThemeCandidate],
    previous_candidates: list[ConsensusThemeCandidate],
    min_overlap: float,
) -> dict[str, list[tuple[ConsensusThemeCandidate, float]]]:
    previous_to_current: dict[str, list[tuple[ConsensusThemeCandidate, float]]] = {}
    for previous in previous_candidates:
        matches = []
        previous_members = set(previous.members)
        for candidate in candidates:
            score = _overlap_small(set(candidate.members), previous_members)
            if score >= min_overlap:
                matches.append((candidate, score))
        previous_to_current[previous.theme_instance_id] = sorted(
            matches,
            key=lambda item: (-item[1], item[0].theme_instance_id),
        )
    return previous_to_current
