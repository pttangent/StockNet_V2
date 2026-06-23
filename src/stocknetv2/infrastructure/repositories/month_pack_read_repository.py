from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


@dataclass(frozen=True)
class SnapshotSpec:
    trade_date: str
    snapshot_time: pd.Timestamp
    snapshot_id: str
    snapshot_clock: str
    month_pack_root: str


@dataclass(frozen=True)
class SnapshotBlockInputs:
    trade_date: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    raw_1m: pd.DataFrame
    bars_5m: pd.DataFrame
    trade_flow_1m: pd.DataFrame
    features_1m: pd.DataFrame
    data_version: str

    def to_trade_date_inputs(self) -> TradeDateInputs:
        return TradeDateInputs(
            trade_date=self.trade_date,
            bars_5m=self.bars_5m,
            trade_flow_1m=self.trade_flow_1m,
            features_1m=self.features_1m,
            data_version=self.data_version,
        )


class MonthPackReadRepository:
    def __init__(self, pack_root: Path | str) -> None:
        self._pack_root = Path(pack_root).expanduser().resolve()
        self._dates_root = self._pack_root / "dates"

    @property
    def pack_root(self) -> Path:
        return self._pack_root

    def load_manifest(self) -> dict[str, object]:
        manifest_path = self._pack_root / "pack_manifest.json"
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def list_trade_dates(self) -> list[str]:
        if not self._dates_root.exists():
            return []
        trade_dates = [
            child.name.split("=", 1)[1]
            for child in self._dates_root.iterdir()
            if child.is_dir() and child.name.startswith("date=")
        ]
        return sorted(trade_dates)

    def load_snapshot_schedule(self, trade_date: str | None = None) -> list[SnapshotSpec]:
        schedule_path = self._pack_root / "snapshot_schedule.parquet"
        if not schedule_path.exists():
            return []
        frame = pd.read_parquet(schedule_path)
        if frame.empty:
            return []
        if "timestamp" not in frame.columns:
            raise RuntimeError("snapshot_schedule.parquet is missing timestamp column.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if "trade_date" not in frame.columns:
            frame["trade_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        if trade_date is not None:
            frame = frame.loc[frame["trade_date"].astype(str) == trade_date].copy()
        if "snapshot_clock" not in frame.columns:
            frame["snapshot_clock"] = frame["timestamp"].dt.strftime("%H%M")
        if "snapshot_id" not in frame.columns:
            frame["snapshot_id"] = frame["trade_date"].astype(str) + "_" + frame["snapshot_clock"].astype(str)
        frame = frame.sort_values(["trade_date", "timestamp"]).reset_index(drop=True)
        return [
            SnapshotSpec(
                trade_date=str(row.trade_date),
                snapshot_time=pd.Timestamp(row.timestamp),
                snapshot_id=str(row.snapshot_id),
                snapshot_clock=str(row.snapshot_clock),
                month_pack_root=str(self._pack_root),
            )
            for row in frame.itertuples(index=False)
        ]

    def read_snapshot_block(
        self,
        *,
        trade_date: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        use_graph_features: bool = True,
        include_trade_flow: bool = False,
    ) -> SnapshotBlockInputs:
        date_root = self._dates_root / f"date={trade_date}"
        raw_1m = self._read_filtered_parquet(
            date_root / "raw_1m.parquet",
            window_start=window_start,
            window_end=window_end,
        )
        bars_5m = self._read_filtered_parquet(
            date_root / "bars_5m.parquet",
            columns=["timestamp", "symbol", "symbol_id", "close"],
            window_start=window_start,
            window_end=window_end,
        )
        features_path = date_root / ("graph_features_1m.parquet" if use_graph_features else "features_1m.parquet")
        features_1m = self._read_filtered_parquet(
            features_path,
            window_start=window_start,
            window_end=window_end,
        )
        trade_flow_1m = pd.DataFrame()
        if include_trade_flow:
            trade_flow_1m = self._normalize_trade_flow_1m(
                self._read_filtered_parquet(
                    date_root / "trade_flow_1m.parquet",
                    timestamp_column="minute",
                    window_start=window_start,
                    window_end=window_end,
                )
            )
        return SnapshotBlockInputs(
            trade_date=trade_date,
            requested_start=window_start,
            requested_end=window_end,
            raw_1m=raw_1m.sort_values(["timestamp", "symbol"]).reset_index(drop=True) if not raw_1m.empty else raw_1m,
            bars_5m=bars_5m.sort_values(["timestamp", "symbol"]).reset_index(drop=True),
            trade_flow_1m=trade_flow_1m.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
            if not trade_flow_1m.empty
            else trade_flow_1m,
            features_1m=features_1m.sort_values(["timestamp", "symbol"]).reset_index(drop=True),
            data_version=f"month_pack:{self._pack_root.name}:{trade_date}:{window_start.isoformat()}:{window_end.isoformat()}",
        )

    @staticmethod
    def _read_filtered_parquet(
        path: Path,
        *,
        columns: list[str] | None = None,
        timestamp_column: str = "timestamp",
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        available_columns = set(pq.ParquetFile(path).schema_arrow.names)
        resolved_columns = columns
        if columns is not None:
            resolved_columns = [column for column in columns if column in available_columns]
            if not resolved_columns:
                resolved_columns = None
        filters = []
        if timestamp_column in available_columns:
            filters = [
                (timestamp_column, ">=", window_start.to_pydatetime()),
                (timestamp_column, "<=", window_end.to_pydatetime()),
            ]
        try:
            frame = pd.read_parquet(path, columns=resolved_columns, filters=filters or None)
        except Exception:
            frame = pd.read_parquet(path, columns=resolved_columns)
        if timestamp_column in frame.columns:
            frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
            frame = frame.loc[(frame[timestamp_column] >= window_start) & (frame[timestamp_column] <= window_end)].copy()
        if "available_time" in frame.columns:
            frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True, errors="coerce")
        if "bar_end" in frame.columns:
            frame["bar_end"] = pd.to_datetime(frame["bar_end"], utc=True, errors="coerce")
        return frame

    @staticmethod
    def _normalize_trade_flow_1m(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "timestamp"])
        normalized = frame.copy()
        if "ticker" in normalized.columns:
            normalized = normalized.rename(columns={"ticker": "symbol"})
        if "minute" in normalized.columns:
            normalized = normalized.rename(columns={"minute": "timestamp"})
        if "timestamp" in normalized.columns:
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
        if "available_time" in normalized.columns:
            normalized["available_time"] = pd.to_datetime(normalized["available_time"], utc=True, errors="coerce")
        else:
            normalized["available_time"] = normalized["timestamp"] if "timestamp" in normalized.columns else pd.NaT
        if "imbalance_proxy" in normalized.columns and "imbalance_z" not in normalized.columns:
            normalized = normalized.rename(columns={"imbalance_proxy": "imbalance_z"})
        if "flow_impulse_score" not in normalized.columns:
            if "imbalance_z" in normalized.columns:
                normalized["flow_impulse_score"] = pd.to_numeric(normalized["imbalance_z"], errors="coerce").fillna(0.0)
            else:
                normalized["flow_impulse_score"] = 0.0
        if "large_trade_ratio_z" not in normalized.columns:
            if "large_trade_dollar_volume" in normalized.columns and "dollar_volume" in normalized.columns:
                dollar_volume = pd.to_numeric(normalized["dollar_volume"], errors="coerce").replace(0.0, np.nan)
                normalized["large_trade_ratio_z"] = (
                    pd.to_numeric(normalized["large_trade_dollar_volume"], errors="coerce").fillna(0.0)
                    / dollar_volume.astype(float)
                ).fillna(0.0)
            else:
                normalized["large_trade_ratio_z"] = 0.0
        return normalized
