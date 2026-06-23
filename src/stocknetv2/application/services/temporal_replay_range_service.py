from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from stocknetv2.application.services.temporal_edge_replay_service import (
    TemporalEdgeReplayService,
    TemporalEdgeState,
)
from stocknetv2.domain.graph.edge import GraphEdge


@dataclass(frozen=True)
class TemporalReplayRangeSummary:
    processed_dates: list[str]
    output_root: Path


class TemporalReplayRangeService:
    def __init__(self, *, temporal_edge_replay_service: TemporalEdgeReplayService | None = None) -> None:
        self._temporal_edge_replay_service = temporal_edge_replay_service or TemporalEdgeReplayService()

    def run(
        self,
        *,
        run_id: str,
        date_roots: Iterable[Path | str],
        output_root: Path | str,
    ) -> TemporalReplayRangeSummary:
        output_root = Path(output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoints_root = output_root / "checkpoints"
        checkpoints_root.mkdir(parents=True, exist_ok=True)

        previous_states: dict[tuple[str, str, str], TemporalEdgeState] = {}
        processed_dates: list[str] = []

        for date_root in sorted((Path(path).expanduser().resolve() for path in date_roots), key=lambda path: path.name):
            trade_date = date_root.name
            raw_db_path = date_root / "raw_graph.duckdb"
            if not raw_db_path.exists():
                continue
            existing_status_path = output_root / f"date={trade_date}" / "status.json"
            existing_checkpoint_path = checkpoints_root / f"{trade_date}_terminal_state.parquet"
            if existing_status_path.exists() and existing_checkpoint_path.exists():
                try:
                    status_payload = json.loads(existing_status_path.read_text(encoding="utf-8"))
                except Exception:
                    status_payload = {}
                if status_payload.get("temporal_status") == "complete":
                    previous_states = _load_terminal_states(existing_checkpoint_path)
                    processed_dates.append(trade_date)
                    continue

            connection = duckdb.connect(str(raw_db_path))
            try:
                frame = connection.execute(
                    """
                    SELECT
                        run_id, snapshot_id, trade_date, timestamp, graph_layer,
                        source_symbol, target_symbol, edge_type, weight, raw_score,
                        edge_confidence, effective_lookback_minutes, support_points
                    FROM graph_edges_thresholded
                    ORDER BY timestamp, snapshot_id, graph_layer, source_symbol, target_symbol
                    """
                ).df()
            finally:
                connection.close()

            if frame.empty:
                _write_empty_temporal_artifacts(output_root, trade_date)
                processed_dates.append(trade_date)
                continue

            emitted_rows: list[TemporalEdgeState] = []
            for (snapshot_id, snapshot_timestamp), snapshot_frame in frame.groupby(["snapshot_id", "timestamp"], sort=True):
                layer_edges: dict[str, list[GraphEdge]] = {}
                for graph_layer, layer_frame in snapshot_frame.groupby("graph_layer", sort=True):
                    layer_edges[graph_layer] = [
                        GraphEdge(
                            graph_layer=str(row.graph_layer),
                            edge_type=str(row.edge_type),
                            source_symbol=str(row.source_symbol),
                            target_symbol=str(row.target_symbol),
                            snapshot_time=pd.Timestamp(row.timestamp),
                            weight=float(row.weight),
                            raw_score=float(row.raw_score) if pd.notna(row.raw_score) else float(row.weight),
                            support_points=int(row.support_points) if pd.notna(row.support_points) else 0,
                            edge_confidence=float(row.edge_confidence) if pd.notna(row.edge_confidence) else 1.0,
                            effective_lookback_minutes=int(row.effective_lookback_minutes)
                            if pd.notna(row.effective_lookback_minutes)
                            else None,
                            calculation_backend="duckdb",
                        )
                        for row in layer_frame.itertuples(index=False)
                    ]

                emitted_snapshot_rows, previous_states = self._temporal_edge_replay_service.replay(
                    run_id=run_id,
                    snapshot_id=str(snapshot_id),
                    trade_date=trade_date,
                    timestamp=pd.Timestamp(snapshot_timestamp),
                    layer_edges=layer_edges,
                    previous_states=previous_states,
                )
                emitted_rows.extend(emitted_snapshot_rows)

            _write_temporal_date_outputs(
                output_root=output_root,
                trade_date=trade_date,
                emitted_rows=emitted_rows,
                terminal_states=previous_states.values(),
                run_id=run_id,
            )
            processed_dates.append(trade_date)

        return TemporalReplayRangeSummary(processed_dates=processed_dates, output_root=output_root)


def _write_temporal_date_outputs(
    *,
    output_root: Path,
    trade_date: str,
    emitted_rows: list[TemporalEdgeState],
    terminal_states: Iterable[TemporalEdgeState],
    run_id: str,
) -> None:
    date_root = output_root / f"date={trade_date}"
    date_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    emitted_frame = pd.DataFrame([asdict(row) for row in emitted_rows])
    emitted_path = date_root / "temporal_edges.parquet"
    emitted_frame.to_parquet(emitted_path, index=False)

    terminal_frame = pd.DataFrame([asdict(row) for row in terminal_states])
    terminal_path = checkpoint_root / f"{trade_date}_terminal_state.parquet"
    terminal_frame.to_parquet(terminal_path, index=False)

    status_payload = {
        "trade_date": trade_date,
        "run_id": run_id,
        "temporal_status": "complete",
        "temporal_edge_count": int(len(emitted_frame)),
        "terminal_state_count": int(len(terminal_frame)),
    }
    (date_root / "status.json").write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_empty_temporal_artifacts(output_root: Path, trade_date: str) -> None:
    date_root = output_root / f"date={trade_date}"
    date_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_parquet(date_root / "temporal_edges.parquet", index=False)
    pd.DataFrame().to_parquet(checkpoint_root / f"{trade_date}_terminal_state.parquet", index=False)


def _load_terminal_states(path: Path) -> dict[tuple[str, str, str], TemporalEdgeState]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    states: dict[tuple[str, str, str], TemporalEdgeState] = {}
    for row in frame.to_dict(orient="records"):
        state = TemporalEdgeState(
            temporal_edge_state_id=str(row["temporal_edge_state_id"]),
            relation_observation_id=str(row["relation_observation_id"]),
            run_id=str(row["run_id"]),
            snapshot_id=str(row["snapshot_id"]),
            trade_date=str(row["trade_date"]),
            timestamp=pd.Timestamp(row["timestamp"]),
            graph_layer=str(row["graph_layer"]),
            source_symbol=str(row["source_symbol"]),
            target_symbol=str(row["target_symbol"]),
            raw_score=float(row["raw_score"]),
            temporal_score=float(row["temporal_score"]),
            support_points=int(row["support_points"]),
            effective_lookback_minutes=int(row["effective_lookback_minutes"]) if pd.notna(row["effective_lookback_minutes"]) else None,
            presence_count=int(row["presence_count"]),
            age_frames=int(row["age_frames"]),
            missing_frames=int(row["missing_frames"]),
            entered_at=pd.Timestamp(row["entered_at"]),
            last_seen_at=pd.Timestamp(row["last_seen_at"]),
            state=str(row["state"]),
            temporal_policy_id=str(row["temporal_policy_id"]),
        )
        states[(state.graph_layer, state.source_symbol, state.target_symbol)] = state
    return states
