from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from uuid import uuid4

import duckdb


class AuditRepository:
    """Persist run, lineage, and snapshot audit rows for T1."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def register_config(
        self,
        *,
        config_id: str,
        config_name: str,
        config_scope: str,
        config_json: dict[str, Any],
        config_version: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO config_registry (
                config_id, config_name, config_scope, config_json, config_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [config_id, config_name, config_scope, json.dumps(config_json), config_version],
        )

    def create_run(
        self,
        *,
        run_id: str,
        run_name: str,
        date_start: str,
        date_end: str,
        frame_minutes: int,
        config_id: str,
        config_json: dict[str, Any],
        code_commit: str,
        data_version: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO theme_discovery_run (
                run_id, run_name, date_start, date_end, frame_minutes,
                config_id, config_json, code_commit, data_version, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                run_name,
                date_start,
                date_end,
                frame_minutes,
                config_id,
                json.dumps(config_json),
                code_commit,
                data_version,
                "running",
            ],
        )

    def list_completed_snapshot_ids(
        self,
        *,
        run_id: str,
        trade_date: str,
        expected_layer_count: int,
    ) -> set[str]:
        rows = self._connection.execute(
            """
            SELECT snapshot_id
            FROM graph_layer_diagnostic
            WHERE run_id = ? AND trade_date = ?
            GROUP BY snapshot_id
            HAVING COUNT(DISTINCT graph_layer) >= ?
            """,
            [run_id, trade_date, expected_layer_count],
        ).fetchall()
        return {str(row[0]) for row in rows}

    def add_input_lineage(self, *, run_id: str, snapshot_id: str | None, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._connection.execute(
                """
                INSERT INTO input_lineage (
                    lineage_id, run_id, snapshot_id, source_kind, source_name, source_path,
                    source_version, source_min_timestamp, source_max_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    uuid4().hex,
                    run_id,
                    snapshot_id,
                    record["source_kind"],
                    record["source_name"],
                    record.get("source_path"),
                    record.get("source_version"),
                    record.get("source_min_timestamp"),
                    record.get("source_max_timestamp"),
                ],
            )

    def create_snapshots(self, snapshot_rows: list[dict[str, Any]]) -> None:
        for row in snapshot_rows:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO graph_snapshot (
                    snapshot_id, run_id, trade_date, timestamp, frame_minutes,
                    market_session, graph_status, available_minutes_since_open
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row["snapshot_id"],
                    row["run_id"],
                    row["trade_date"],
                    row["timestamp"],
                    row["frame_minutes"],
                    row["market_session"],
                    row["graph_status"],
                    row["available_minutes_since_open"],
                ],
            )

    def complete_run(self, *, run_id: str, data_version: str) -> None:
        self._connection.execute(
            "UPDATE theme_discovery_run SET status = ?, data_version = ? WHERE run_id = ?",
            ["completed", data_version, run_id],
        )
