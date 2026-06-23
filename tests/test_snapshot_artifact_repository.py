from __future__ import annotations

from datetime import UTC, datetime

import json
import pandas as pd

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge
from stocknetv2.infrastructure.repositories.snapshot_artifact_repository import SnapshotArtifactRepository


def test_snapshot_artifact_repository_writes_snapshot_payload(tmp_path):
    repository = SnapshotArtifactRepository()
    snapshot_root = tmp_path / "month=2025-01" / "dates" / "date=2025-01-02" / "snapshots" / "snapshot=0935"
    repository.write_completed_snapshot_artifact(
        snapshot_root=snapshot_root,
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
            "worker_pid": 123,
            "diagnostics": {"ok": True},
        },
        layer_edges={
            "flow_alignment_graph": [
                GraphEdge(
                    graph_layer="flow_alignment_graph",
                    edge_type="flow_alignment",
                    source_symbol="AAA",
                    target_symbol="BBB",
                    snapshot_time=datetime(2025, 1, 2, 14, 35, tzinfo=UTC),
                    weight=0.9,
                    raw_score=0.9,
                    support_points=8,
                )
            ]
        },
        layer_communities={"flow_alignment_graph": [Community(members=["AAA", "BBB"])]},
    )

    assert (snapshot_root / "_SUCCESS").exists()
    assert (snapshot_root / "_PROFILE_SUCCESS").exists()
    assert not (snapshot_root / "_FULL_GRAPH_SUCCESS").exists()
    assert (snapshot_root / "edges.parquet").exists()
    assert (snapshot_root / "community_membership.parquet").exists()
    status_payload = json.loads((snapshot_root / "status.json").read_text(encoding="utf-8"))
    assert status_payload["snapshot_id"] == "2025-01-02_0935"
    membership = pd.read_parquet(snapshot_root / "community_membership.parquet")
    assert "local_community_id" in membership.columns
    assert "persistent_community_id" in membership.columns
    assert pd.isna(membership.loc[0, "persistent_community_id"])
    assert repository.snapshot_success_exists(snapshot_root)
