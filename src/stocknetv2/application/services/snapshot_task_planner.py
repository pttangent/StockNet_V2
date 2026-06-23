from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocknetv2.application.services.layer_profile_service import LayerProfile
from stocknetv2.infrastructure.repositories.month_pack_read_repository import SnapshotSpec


@dataclass(frozen=True)
class SnapshotBlockTask:
    trade_date: str
    snapshots: tuple[SnapshotSpec, ...]
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    block_id: str


class SnapshotTaskPlanner:
    def plan_blocks(
        self,
        *,
        snapshots: list[SnapshotSpec],
        profile: LayerProfile,
        snapshot_block_size: int,
    ) -> list[SnapshotBlockTask]:
        tasks: list[SnapshotBlockTask] = []
        grouped: dict[str, list[SnapshotSpec]] = {}
        for snapshot in snapshots:
            grouped.setdefault(snapshot.trade_date, []).append(snapshot)
        for trade_date, day_snapshots in sorted(grouped.items()):
            ordered = sorted(day_snapshots, key=lambda item: item.snapshot_time)
            for block_index in range(0, len(ordered), max(1, snapshot_block_size)):
                block_snapshots = tuple(ordered[block_index : block_index + max(1, snapshot_block_size)])
                window_end = block_snapshots[-1].snapshot_time
                lookback_minutes = max(profile.max_feature_lookback_minutes, profile.max_return_lookback_minutes)
                window_start = block_snapshots[0].snapshot_time - pd.Timedelta(minutes=lookback_minutes)
                tasks.append(
                    SnapshotBlockTask(
                        trade_date=trade_date,
                        snapshots=block_snapshots,
                        window_start=window_start,
                        window_end=window_end,
                        block_id=f"{trade_date}_{block_snapshots[0].snapshot_clock}_{block_snapshots[-1].snapshot_clock}",
                    )
                )
        return tasks
