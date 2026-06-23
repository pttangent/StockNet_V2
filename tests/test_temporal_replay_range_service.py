from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from stocknetv2.application.services.temporal_replay_range_service import TemporalReplayRangeService
from stocknetv2.infrastructure.db.schema_manager import SchemaManager


def _create_raw_day_database(path: Path, *, trade_date: str, snapshot_id: str, weight: float) -> None:
    connection = duckdb.connect(str(path))
    SchemaManager(connection).initialize()
    connection.execute(
        """
        INSERT INTO graph_edges_thresholded (
            run_id, snapshot_id, trade_date, timestamp, graph_layer,
            source_symbol, target_symbol, edge_type, weight, raw_score,
            edge_confidence, effective_lookback_minutes, window_start, window_end, support_points, config_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            f"run_{trade_date}",
            snapshot_id,
            trade_date,
            pd.Timestamp(f"{trade_date} 14:35:00"),
            "return_corr_graph",
            "AAA",
            "BBB",
            "relation",
            weight,
            weight,
            1.0,
            5,
            None,
            pd.Timestamp(f"{trade_date} 14:35:00"),
            8,
            "cfg",
        ],
    )
    connection.close()


def test_temporal_replay_range_service_advances_state_across_dates(tmp_path: Path):
    first_day_root = tmp_path / "dates" / "2025-01-02"
    second_day_root = tmp_path / "dates" / "2025-01-03"
    first_day_root.mkdir(parents=True)
    second_day_root.mkdir(parents=True)
    _create_raw_day_database(first_day_root / "raw_graph.duckdb", trade_date="2025-01-02", snapshot_id="snap_1", weight=0.5)
    _create_raw_day_database(second_day_root / "raw_graph.duckdb", trade_date="2025-01-03", snapshot_id="snap_2", weight=0.7)

    service = TemporalReplayRangeService()
    summary = service.run(
        run_id="run_2025",
        date_roots=[first_day_root, second_day_root],
        output_root=tmp_path / "temporal",
    )

    assert summary.processed_dates == ["2025-01-02", "2025-01-03"]
    assert (tmp_path / "temporal" / "date=2025-01-02" / "temporal_edges.parquet").exists()
    assert (tmp_path / "temporal" / "date=2025-01-03" / "temporal_edges.parquet").exists()
    assert (tmp_path / "temporal" / "checkpoints" / "2025-01-02_terminal_state.parquet").exists()
    assert (tmp_path / "temporal" / "checkpoints" / "2025-01-03_terminal_state.parquet").exists()

    checkpoint = pd.read_parquet(tmp_path / "temporal" / "checkpoints" / "2025-01-03_terminal_state.parquet")
    assert int(checkpoint.loc[0, "presence_count"]) == 2
