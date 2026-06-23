from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from stocknetv2.domain.community.community import Community
from stocknetv2.domain.graph.edge import GraphEdge


class SnapshotArtifactRepository:
    """Manage distributed date-level artifact directories for graph-build runs."""

    def write_completed_snapshot_artifact(
        self,
        *,
        snapshot_root: Path,
        payload: dict[str, object],
        layer_edges: dict[str, list[GraphEdge]],
        layer_communities: dict[str, list[Community]],
    ) -> None:
        temp_root = snapshot_root.with_name(f"{snapshot_root.name}.__tmp__.{payload['worker_pid']}")
        if temp_root.exists():
            self._remove_tree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)

        self._build_edge_frame(layer_edges).to_parquet(temp_root / "edges.parquet", index=False)
        self._build_membership_frame(
            snapshot_id=str(payload["snapshot_id"]),
            layer_communities=layer_communities,
        ).to_parquet(
            temp_root / "community_membership.parquet",
            index=False,
        )
        self._build_community_metrics_frame(
            snapshot_id=str(payload["snapshot_id"]),
            layer_communities=layer_communities,
        ).to_parquet(
            temp_root / "community_metrics.parquet",
            index=False,
        )
        (temp_root / "diagnostics.json").write_text(
            json.dumps(payload["diagnostics"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        status_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"diagnostics", "worker_pid"}
        }
        (temp_root / "status.json").write_text(
            json.dumps(status_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if snapshot_root.exists():
            self._remove_tree(snapshot_root)
        temp_root.replace(snapshot_root)
        (snapshot_root / "_PROFILE_SUCCESS").write_text("", encoding="utf-8")
        if bool(status_payload.get("full_graph_complete")):
            (snapshot_root / "_FULL_GRAPH_SUCCESS").write_text("", encoding="utf-8")
        (snapshot_root / "_SUCCESS").write_text("", encoding="utf-8")

    def snapshot_success_exists(self, snapshot_root: Path) -> bool:
        return (snapshot_root / "_SUCCESS").exists() and (snapshot_root / "status.json").exists()

    @staticmethod
    def _build_edge_frame(layer_edges: dict[str, list[GraphEdge]]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for layer_name, edges in layer_edges.items():
            for edge in edges:
                rows.append(
                    {
                        "graph_layer": layer_name,
                        "edge_type": edge.edge_type,
                        "source_symbol": edge.source_symbol,
                        "target_symbol": edge.target_symbol,
                        "timestamp": edge.snapshot_time,
                        "weight": edge.weight,
                        "raw_score": edge.raw_score,
                        "support_points": edge.support_points,
                        "edge_confidence": edge.edge_confidence,
                        "effective_lookback_minutes": edge.effective_lookback_minutes,
                        "calculation_backend": edge.calculation_backend,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_membership_frame(
        *,
        snapshot_id: str,
        layer_communities: dict[str, list[Community]],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for layer_name, communities in layer_communities.items():
            for community_index, community in enumerate(communities, start=1):
                local_community_id = f"{snapshot_id}_{layer_name}_community_{community_index:03d}"
                for member_rank, symbol in enumerate(community.members, start=1):
                    rows.append(
                        {
                            "graph_layer": layer_name,
                            "local_community_id": local_community_id,
                            "persistent_community_id": None,
                            "symbol": symbol,
                            "member_rank": member_rank,
                            "method": community.method,
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_community_metrics_frame(
        *,
        snapshot_id: str,
        layer_communities: dict[str, list[Community]],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for layer_name, communities in layer_communities.items():
            for community_index, community in enumerate(communities, start=1):
                rows.append(
                    {
                        "graph_layer": layer_name,
                        "local_community_id": f"{snapshot_id}_{layer_name}_community_{community_index:03d}",
                        "persistent_community_id": None,
                        "member_count": len(community.members),
                        "method": community.method,
                        "is_market_mode": community.is_market_mode,
                        "resolution": community.resolution,
                        "universe_ratio": community.universe_ratio,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists():
            return
        for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()

    def write_completed_date_artifacts(
        self,
        *,
        trade_date: str,
        date_root: Path,
        raw_graph_path: Path,
        run_id: str,
        snapshot_count: int,
        data_version: str,
        elapsed_seconds: float,
        config_id: str,
        config_version: str,
        config_hash: str,
        code_commit: str,
    ) -> None:
        date_root.mkdir(parents=True, exist_ok=True)
        status_payload = {
            "trade_date": trade_date,
            "raw_graph_status": "complete",
            "snapshot_count": snapshot_count,
            "config_hash": config_hash,
            "code_commit": code_commit,
            "elapsed_seconds": elapsed_seconds,
            "data_version": data_version,
            "run_id": run_id,
        }
        manifest_payload = {
            "trade_date": trade_date,
            "run_id": run_id,
            "artifacts": {
                "raw_graph": str(raw_graph_path),
                "status": str(date_root / "status.json"),
                "manifest": str(date_root / "manifest.json"),
                "diagnostics": str(date_root / "diagnostics.json"),
                "success_marker": str(date_root / "_SUCCESS"),
            },
        }
        diagnostics_payload = {
            "trade_date": trade_date,
            "run_id": run_id,
            "snapshot_count": snapshot_count,
            "data_version": data_version,
            "config_id": config_id,
            "config_version": config_version,
            "config_hash": config_hash,
            "code_commit": code_commit,
            "elapsed_seconds": elapsed_seconds,
        }
        (date_root / "status.json").write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (date_root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (date_root / "diagnostics.json").write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (date_root / "_SUCCESS").write_text("", encoding="utf-8")

    def write_failed_date_artifacts(
        self,
        *,
        trade_date: str,
        date_root: Path,
        run_id: str,
        error_type: str,
        error_message: str,
        config_id: str,
        config_version: str,
        config_hash: str,
        code_commit: str,
    ) -> None:
        date_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "trade_date": trade_date,
            "raw_graph_status": "failed",
            "error_type": error_type,
            "error_message": error_message,
            "config_hash": config_hash,
            "code_commit": code_commit,
            "config_id": config_id,
            "config_version": config_version,
            "run_id": run_id,
        }
        (date_root / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        success_marker = date_root / "_SUCCESS"
        if success_marker.exists():
            success_marker.unlink()

    def load_completed_date_artifact(self, *, date_root: Path) -> dict[str, object] | None:
        status_path = date_root / "status.json"
        database_path = date_root / "raw_graph.duckdb"
        if not status_path.exists() or not database_path.exists():
            return None
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if payload.get("raw_graph_status") != "complete":
            return None
        trade_date = str(payload.get("trade_date", "")).strip()
        run_id = str(payload.get("run_id", "")).strip()
        if not trade_date or not run_id:
            return None
        return {
            "trade_date": trade_date,
            "run_id": run_id,
            "database_path": database_path,
            "snapshot_count": int(payload.get("snapshot_count", 0) or 0),
            "data_version": str(payload.get("data_version", "") or ""),
            "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
        }

    def write_date_artifact_registry(
        self,
        path: Path,
        *,
        completed_rows: list[dict[str, object]],
        failed_rows: list[dict[str, object]],
    ) -> None:
        rows: list[dict[str, object]] = [
            {
                "trade_date": row["trade_date"],
                "raw_graph_status": "complete",
                "run_id": row["run_id"],
                "raw_graph_path": str(row["database_path"]),
                "snapshot_count": row["snapshot_count"],
                "data_version": row["data_version"],
                "elapsed_seconds": row["elapsed_seconds"],
                "error_type": "",
                "error_message": "",
            }
            for row in sorted(completed_rows, key=lambda item: str(item["trade_date"]))
        ]
        rows.extend(
            {
                "trade_date": row["trade_date"],
                "raw_graph_status": "failed",
                "run_id": row["run_id"],
                "raw_graph_path": "",
                "snapshot_count": 0,
                "data_version": "",
                "elapsed_seconds": None,
                "error_type": row["error_type"],
                "error_message": row["error_message"],
            }
            for row in sorted(failed_rows, key=lambda item: str(item["trade_date"]))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "trade_date",
                    "raw_graph_status",
                    "run_id",
                    "raw_graph_path",
                    "snapshot_count",
                    "data_version",
                    "elapsed_seconds",
                    "error_type",
                    "error_message",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
