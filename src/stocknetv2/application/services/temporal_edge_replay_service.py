from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocknetv2.domain.graph.edge import GraphEdge


@dataclass(frozen=True)
class TemporalEdgeState:
    temporal_edge_state_id: str
    relation_observation_id: str
    run_id: str
    snapshot_id: str
    trade_date: str
    timestamp: pd.Timestamp
    graph_layer: str
    source_symbol: str
    target_symbol: str
    raw_score: float
    temporal_score: float
    support_points: int
    effective_lookback_minutes: int | None
    presence_count: int
    age_frames: int
    missing_frames: int
    entered_at: pd.Timestamp
    last_seen_at: pd.Timestamp
    state: str
    temporal_policy_id: str


class TemporalEdgeReplayService:
    """Replay instantaneous relation observations into finite-memory temporal edge states."""

    def __init__(self, *, alpha: float = 0.6, ttl_frames: int = 2, policy_id: str = "ewma_ttl_v1") -> None:
        self._alpha = alpha
        self._ttl_frames = max(1, ttl_frames)
        self._policy_id = policy_id

    def replay(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        trade_date: str,
        timestamp: pd.Timestamp,
        layer_edges: dict[str, list[GraphEdge]],
        previous_states: dict[tuple[str, str, str], TemporalEdgeState],
    ) -> tuple[list[TemporalEdgeState], dict[tuple[str, str, str], TemporalEdgeState]]:
        next_states: dict[tuple[str, str, str], TemporalEdgeState] = {}
        emitted_states: list[TemporalEdgeState] = []
        current_keys: set[tuple[str, str, str]] = set()

        for graph_layer, edges in layer_edges.items():
            for edge in edges:
                key = (graph_layer, edge.source_symbol, edge.target_symbol)
                current_keys.add(key)
                previous_state = previous_states.get(key)
                if previous_state is None:
                    temporal_score = float(edge.raw_score)
                    presence_count = 1
                    age_frames = 1
                    entered_at = timestamp
                else:
                    temporal_score = (self._alpha * float(edge.raw_score)) + (
                        (1.0 - self._alpha) * float(previous_state.temporal_score)
                    )
                    presence_count = previous_state.presence_count + 1
                    age_frames = previous_state.age_frames + 1
                    entered_at = previous_state.entered_at
                state = TemporalEdgeState(
                    temporal_edge_state_id=_temporal_state_id(snapshot_id, graph_layer, edge.source_symbol, edge.target_symbol),
                    relation_observation_id=_relation_observation_id(snapshot_id, graph_layer, edge.source_symbol, edge.target_symbol),
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    trade_date=trade_date,
                    timestamp=timestamp,
                    graph_layer=graph_layer,
                    source_symbol=edge.source_symbol,
                    target_symbol=edge.target_symbol,
                    raw_score=float(edge.raw_score),
                    temporal_score=round(temporal_score, 6),
                    support_points=int(edge.support_points),
                    effective_lookback_minutes=edge.effective_lookback_minutes,
                    presence_count=presence_count,
                    age_frames=age_frames,
                    missing_frames=0,
                    entered_at=entered_at,
                    last_seen_at=timestamp,
                    state="active",
                    temporal_policy_id=self._policy_id,
                )
                next_states[key] = state
                emitted_states.append(state)

        for key, previous_state in previous_states.items():
            if key in current_keys:
                continue
            missing_frames = previous_state.missing_frames + 1
            if missing_frames > self._ttl_frames:
                continue
            temporal_score = round((1.0 - self._alpha) * float(previous_state.temporal_score), 6)
            decayed_state = TemporalEdgeState(
                temporal_edge_state_id=_temporal_state_id(snapshot_id, key[0], key[1], key[2]),
                relation_observation_id=previous_state.relation_observation_id,
                run_id=run_id,
                snapshot_id=snapshot_id,
                trade_date=trade_date,
                timestamp=timestamp,
                graph_layer=previous_state.graph_layer,
                source_symbol=previous_state.source_symbol,
                target_symbol=previous_state.target_symbol,
                raw_score=previous_state.raw_score,
                temporal_score=temporal_score,
                support_points=previous_state.support_points,
                effective_lookback_minutes=previous_state.effective_lookback_minutes,
                presence_count=previous_state.presence_count,
                age_frames=previous_state.age_frames,
                missing_frames=missing_frames,
                entered_at=previous_state.entered_at,
                last_seen_at=previous_state.last_seen_at,
                state="cooldown",
                temporal_policy_id=self._policy_id,
            )
            next_states[key] = decayed_state
            emitted_states.append(decayed_state)

        emitted_states.sort(key=lambda row: (row.graph_layer, row.source_symbol, row.target_symbol))
        return emitted_states, next_states


def _relation_observation_id(snapshot_id: str, graph_layer: str, source_symbol: str, target_symbol: str) -> str:
    return f"{snapshot_id}_{graph_layer}_{source_symbol}_{target_symbol}"


def _temporal_state_id(snapshot_id: str, graph_layer: str, source_symbol: str, target_symbol: str) -> str:
    return f"{snapshot_id}_{graph_layer}_{source_symbol}_{target_symbol}_temporal"
