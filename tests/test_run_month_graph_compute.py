from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.infrastructure.repositories.snapshot_artifact_repository import SnapshotArtifactRepository


_SPEC = spec_from_file_location(
    "run_month_graph_compute",
    Path(__file__).resolve().parents[1] / "scripts" / "run_month_graph_compute.py",
)
assert _SPEC and _SPEC.loader
run_month_graph_compute = module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_month_graph_compute)


def test_run_month_graph_compute_skips_logged_snapshot_and_computes_remaining(tmp_path, monkeypatch):
    pack_root = tmp_path / "data" / "ready" / "month=2025-01"
    date_root = pack_root / "dates" / "date=2025-01-02"
    output_root = tmp_path / "research_runs" / "demo"
    date_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (pack_root / "pack_manifest.json").write_text(json.dumps({"month": "2025-01"}), encoding="utf-8")
    pd.DataFrame(
        {
            "trade_date": ["2025-01-02", "2025-01-02"],
            "timestamp": [
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "snapshot_id": ["2025-01-02_0935", "2025-01-02_0940"],
            "snapshot_clock": ["0935", "0940"],
        }
    ).to_parquet(pack_root / "snapshot_schedule.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "symbol": ["AAA", "AAA", "AAA"],
            "symbol_id": [1, 1, 1],
            "close": [10.0, 10.5, 11.0],
        }
    ).to_parquet(date_root / "bars_5m.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "available_time": [
                datetime(2025, 1, 2, 14, 36, tzinfo=UTC),
                datetime(2025, 1, 2, 14, 41, tzinfo=UTC),
            ],
            "symbol": ["AAA", "AAA"],
            "symbol_id": [1, 1],
            "ret_1m": [0.1, 0.2],
            "volume_z_12": [1.0, 1.0],
            "imbalance_z": [0.1, 0.2],
            "large_trade_ratio_z": [0.2, 0.3],
            "flow_impulse_score": [0.5, 0.6],
        }
    ).to_parquet(date_root / "graph_features_1m.parquet", index=False)

    first_snapshot_root = output_root / "month=2025-01" / "dates" / "date=2025-01-02" / "snapshots" / "snapshot=0935"
    SnapshotArtifactRepository().write_completed_snapshot_artifact(
        snapshot_root=first_snapshot_root,
        payload={
            "run_name": "demo",
            "trade_date": "2025-01-02",
            "snapshot_id": "2025-01-02_0935",
            "snapshot_clock": "0935",
            "profile": "cpu_no_dtw",
            "profile_status": "complete",
            "profile_complete": True,
            "full_graph_complete": False,
            "completed_layers": ["flow_alignment_graph"],
            "missing_layers": ["dtw_return_similarity_graph"],
            "layer_count": 1,
            "edge_count": 1,
            "elapsed_seconds": 0.1,
            "worker_pid": 1,
            "diagnostics": {"ok": True},
        },
        layer_edges={
            "flow_alignment_graph": [
                GraphEdge(
                    graph_layer="flow_alignment_graph",
                    edge_type="flow_alignment",
                    source_symbol="AAA",
                    target_symbol="AAA",
                    snapshot_time=datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                    weight=1.0,
                    raw_score=1.0,
                    support_points=1,
                )
            ]
        },
        layer_communities={"flow_alignment_graph": [Community(members=["AAA"])]},
    )
    (output_root / "run.log").write_text(
        json.dumps({"status": "snapshot_complete", "snapshot_id": "2025-01-02_0935"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_month_graph_compute.py",
            "--month",
            "2025-01",
            "--month-pack-root",
            str(pack_root),
            "--output-root",
            str(output_root),
            "--run-name",
            "demo",
            "--snapshot-block-size",
            "1",
        ],
    )

    assert run_month_graph_compute.main() == 0
    second_snapshot_root = output_root / "month=2025-01" / "dates" / "date=2025-01-02" / "snapshots" / "snapshot=0940"
    log_text = (output_root / "run.log").read_text(encoding="utf-8")
    progress_text = (output_root / "progress.jsonl").read_text(encoding="utf-8")

    assert second_snapshot_root.exists()
    assert log_text.count("2025-01-02_0935") == 1
    assert "2025-01-02_0940" in log_text
    assert '"status": "snapshot_started"' in progress_text
    assert '"status": "snapshot_complete"' in progress_text
    diagnostics = json.loads((second_snapshot_root / "diagnostics.json").read_text(encoding="utf-8"))
    status_payload = json.loads((second_snapshot_root / "status.json").read_text(encoding="utf-8"))
    assert diagnostics["bars_5m_timestamp_semantics"] == "bar_end_utc"
    assert diagnostics["input_windows"]["bars_5m"]["future_row_count"] == 0
    assert status_payload["profile_complete"] is True
    assert status_payload["full_graph_complete"] is False
    assert "dtw_return_similarity_graph" in status_payload["missing_layers"]


def test_run_month_graph_compute_collects_multiple_month_packs_from_date_range(tmp_path, monkeypatch):
    ready_root = tmp_path / "data" / "ready"
    output_root = tmp_path / "research_runs" / "range_demo"
    output_root.mkdir(parents=True, exist_ok=True)

    for month, trade_date, clock in [
        ("2025-01", "2025-01-31", "0935"),
        ("2025-02", "2025-02-03", "0940"),
    ]:
        pack_root = ready_root / f"month={month}"
        date_root = pack_root / "dates" / f"date={trade_date}"
        date_root.mkdir(parents=True, exist_ok=True)
        (pack_root / "pack_manifest.json").write_text(json.dumps({"month": month}), encoding="utf-8")
        timestamp = pd.Timestamp(f"{trade_date}T14:{clock[2:]}:00Z")
        pd.DataFrame(
            {
                "trade_date": [trade_date],
                "timestamp": [timestamp],
                "snapshot_id": [f"{trade_date}_{clock}"],
                "snapshot_clock": [clock],
            }
        ).to_parquet(pack_root / "snapshot_schedule.parquet", index=False)
        pd.DataFrame(
            {
                "timestamp": [timestamp],
                "symbol": ["AAA"],
                "symbol_id": [1],
                "close": [10.0],
            }
        ).to_parquet(date_root / "bars_5m.parquet", index=False)
        pd.DataFrame(
            {
                "timestamp": [timestamp],
                "available_time": [timestamp + pd.Timedelta(minutes=1)],
                "symbol": ["AAA"],
                "symbol_id": [1],
                "ret_1m": [0.1],
                "volume_z_12": [1.0],
                "imbalance_z": [0.1],
                "large_trade_ratio_z": [0.2],
                "flow_impulse_score": [0.5],
            }
        ).to_parquet(date_root / "graph_features_1m.parquet", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_month_graph_compute.py",
            "--pack-root",
            str(ready_root),
            "--output-root",
            str(output_root),
            "--run-name",
            "range_demo",
            "--date-start",
            "2025-01-31",
            "--date-end",
            "2025-02-03",
            "--snapshot-block-size",
            "1",
        ],
    )

    assert run_month_graph_compute.main() == 0
    assert (output_root / "month=2025-01" / "dates" / "date=2025-01-31" / "snapshots" / "snapshot=0935").exists()
    assert (output_root / "month=2025-02" / "dates" / "date=2025-02-03" / "snapshots" / "snapshot=0940").exists()
