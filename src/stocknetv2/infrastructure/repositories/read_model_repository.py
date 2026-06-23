from __future__ import annotations

from uuid import uuid4

import duckdb

from stocknetv2.application.services.read_model_service import SnapshotCacheRecord


class ReadModelRepository:
    """Persist read-only snapshot cache rows."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def save_snapshot_caches(self, caches: list[SnapshotCacheRecord]) -> None:
        for cache in caches:
            self._connection.execute(
                """
                INSERT INTO frontend_snapshot_cache (
                    snapshot_cache_id, snapshot_id, run_id, timestamp, cache_type, payload_json, payload_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    uuid4().hex,
                    cache.snapshot_id,
                    cache.run_id,
                    cache.timestamp,
                    cache.cache_type,
                    cache.payload_json,
                    cache.payload_version,
                ],
            )
