from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - exercised only in limited local runtimes
    duckdb = None

from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock


@dataclass(frozen=True)
class LegacySourceLayout:
    data_root: Path | str

    def __post_init__(self) -> None:
        data_root = Path(self.data_root).expanduser().resolve()
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "raw_1m_root", data_root / "raw_1m")
        object.__setattr__(self, "bars_5m_root", data_root / "bars_5m")
        object.__setattr__(self, "trade_flow_1m_root", data_root / "trade_flow_1m")
        object.__setattr__(self, "features_1m_root", data_root / "features_1m")


@dataclass(frozen=True)
class LegacyDuckDBSource:
    database_path: Path | str
    source_timezone: str = "auto"

    def __post_init__(self) -> None:
        database_path = Path(self.database_path).expanduser().resolve()
        object.__setattr__(self, "database_path", database_path)


@dataclass(frozen=True)
class MonthPackSourceLayout:
    pack_root: Path | str

    def __post_init__(self) -> None:
        pack_root = Path(self.pack_root).expanduser().resolve()
        object.__setattr__(self, "pack_root", pack_root)
        object.__setattr__(self, "dates_root", pack_root / "dates")
        object.__setattr__(self, "month", pack_root.name.split("=", 1)[1] if "=" in pack_root.name else pack_root.name)


@dataclass(frozen=True)
class TradeDateInputs:
    trade_date: str
    bars_5m: pd.DataFrame
    trade_flow_1m: pd.DataFrame
    features_1m: pd.DataFrame
    data_version: str


class SourceProtocol(Protocol):
    pass


class MarketReadRepository:
    """Read legacy source partitions without leaking legacy business logic."""

    def __init__(
        self,
        source: LegacySourceLayout | LegacyDuckDBSource | MonthPackSourceLayout,
        symbol_limit: int | None = None,
    ) -> None:
        self._source = source
        self._symbol_limit = symbol_limit
        self._resolved_duckdb_source_timezone: str | None = None

    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        if isinstance(self._source, LegacyDuckDBSource):
            self._require_duckdb()
            return self._list_available_trade_dates_from_duckdb(dataset_name)
        if isinstance(self._source, MonthPackSourceLayout):
            return self._list_available_trade_dates_from_month_pack()
        root = self._dataset_root(dataset_name)
        if not root.exists():
            return []

        trade_dates: list[str] = []
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("date="):
                trade_dates.append(child.name.split("=", 1)[1])
        return sorted(trade_dates)

    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs:
        if isinstance(self._source, LegacyDuckDBSource):
            self._require_duckdb()
            return self._load_trade_date_inputs_from_duckdb(trade_date)
        if isinstance(self._source, MonthPackSourceLayout):
            return self._load_trade_date_inputs_from_month_pack(trade_date)

        bars_5m = self._read_optional_parquet(
            self._source.bars_5m_root / f"date={trade_date}" / "bars_5m.parquet",
            columns=["timestamp", "close", "symbol"],
        )
        raw_1m = self._read_optional_parquet(
            self._source.raw_1m_root / f"date={trade_date}" / "bars_1m.parquet",
            columns=["symbol", "timestamp", "close", "volume", "dollar_volume"],
        )
        trade_flow_raw = self._read_optional_parquet(
            self._source.trade_flow_1m_root / f"date={trade_date}" / "trade_flow_1m.parquet",
            columns=[
                "ticker",
                "minute",
                "trade_count",
                "dollar_volume",
                "imbalance_proxy",
                "large_trade_dollar_volume",
            ],
        )
        features_raw = self._read_optional_parquet(
            self._source.features_1m_root / f"date={trade_date}" / "features_1m.parquet",
            columns=[
                "symbol",
                "timestamp",
                "bar_end",
                "ret_1m_past",
                "volume_z_proxy",
                "large_trade_ratio",
                "imbalance_proxy",
                "flow_impulse_score",
                "ret_1m",
                "volume_z_12",
                "large_trade_ratio_z",
                "imbalance_z",
            ],
        )
        trade_flow_1m = self._sort_frame(
            self._normalize_trade_flow_1m(trade_flow_raw),
            ["timestamp", "symbol"],
        )
        features_1m = self._sort_frame(
            self._normalize_layout_features_1m(features_raw),
            ["timestamp", "symbol"],
        )
        generated_features = False
        if features_1m.empty and not raw_1m.empty:
            features_1m = self._build_generated_features_1m(raw_1m=raw_1m, trade_flow_1m=trade_flow_raw)
            generated_features = not features_1m.empty

        bars_5m, trade_flow_1m, features_1m = self._apply_symbol_limit(
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
        )

        data_version = f"bars_5m:{trade_date}|trade_flow_1m:{trade_date}|features_1m:{trade_date}"
        if generated_features:
            data_version = f"{data_version}|generated_features_1m:raw_1m+trade_flow_1m"
        if self._symbol_limit is not None:
            data_version = f"{data_version}|symbol_limit:{self._symbol_limit}"
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version=data_version,
        )

    def _list_available_trade_dates_from_month_pack(self) -> list[str]:
        if not self._source.dates_root.exists():
            return []
        trade_dates: list[str] = []
        for child in self._source.dates_root.iterdir():
            if (
                child.is_dir()
                and child.name.startswith("date=")
                and any(
                    (child / filename).exists()
                    for filename in ("bars_5m.parquet", "features_1m.parquet", "graph_features_1m.parquet")
                )
            ):
                trade_dates.append(child.name.split("=", 1)[1])
        return sorted(trade_dates)

    def _load_trade_date_inputs_from_month_pack(self, trade_date: str) -> TradeDateInputs:
        date_root = self._source.dates_root / f"date={trade_date}"
        bars_5m = self._read_optional_parquet(
            date_root / "bars_5m.parquet",
            columns=["timestamp", "symbol", "symbol_id", "close"],
        )
        graph_features = self._read_optional_parquet(date_root / "graph_features_1m.parquet")
        features_source = graph_features
        if features_source.empty:
            features_source = self._read_optional_parquet(date_root / "features_1m.parquet")
        features_1m = self._sort_frame(
            self._normalize_layout_features_1m(features_source),
            ["timestamp", "symbol"],
        )
        trade_flow_1m = self._sort_frame(
            self._normalize_trade_flow_1m(
                self._read_optional_parquet(date_root / "trade_flow_1m.parquet")
            ),
            ["timestamp", "symbol"],
        )
        bars_5m, trade_flow_1m, features_1m = self._apply_symbol_limit(
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
        )
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=self._sort_frame(bars_5m, ["timestamp", "symbol"]),
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version=f"month_pack:{self._source.month}:{trade_date}",
        )

    def _dataset_root(self, dataset_name: str) -> Path:
        if dataset_name == "bars_5m":
            return self._source.bars_5m_root
        if dataset_name == "trade_flow_1m":
            return self._source.trade_flow_1m_root
        if dataset_name == "features_1m":
            return self._source.features_1m_root
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @staticmethod
    def _read_optional_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        resolved_columns = columns
        if columns is not None:
            available_columns = set(pq.ParquetFile(path).schema_arrow.names)
            resolved_columns = [column for column in columns if column in available_columns]
            if not resolved_columns:
                resolved_columns = None
        frame = pd.read_parquet(path, columns=resolved_columns)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def _list_available_trade_dates_from_duckdb(self, dataset_name: str) -> list[str]:
        table_name = self._table_name_for_dataset(dataset_name)
        with duckdb.connect(str(self._source.database_path), read_only=True) as connection:
            rows = connection.execute(
                f"SELECT DISTINCT CAST(date AS VARCHAR) AS trade_date FROM {table_name} ORDER BY trade_date"
            ).fetchall()
        return [row[0] for row in rows]

    def _load_trade_date_inputs_from_duckdb(self, trade_date: str) -> TradeDateInputs:
        with duckdb.connect(str(self._source.database_path), read_only=True) as connection:
            self._resolved_duckdb_source_timezone = self._resolve_duckdb_source_timezone(connection, trade_date)
            limited_symbols = self._load_limited_symbols(connection, trade_date)
            filter_sql, filter_params = self._build_symbol_filter_sql(limited_symbols)
            bars_5m = connection.execute(
                f"""
                SELECT timestamp, close, symbol, date
                FROM bars_5m
                WHERE date = ?
                {filter_sql}
                """,
                [trade_date, *filter_params],
            ).df()
            trade_flow_raw = connection.execute(
                f"""
                SELECT ticker, minute, dollar_volume, imbalance_proxy, large_trade_dollar_volume, date
                FROM trade_flow_1m
                WHERE date = ?
                {filter_sql.replace('symbol', 'ticker')}
                """,
                [trade_date, *filter_params],
            ).df()
            features_raw = connection.execute(
                f"""
                SELECT symbol, timestamp, ret_1m_past, volume_z_proxy, large_trade_ratio, imbalance_proxy, date
                FROM features_1m
                WHERE date = ?
                {filter_sql}
                """,
                [trade_date, *filter_params],
            ).df()

        bars_5m = self._normalize_duckdb_timestamps(bars_5m, ["timestamp"]).sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)
        trade_flow_1m = self._normalize_trade_flow_1m(trade_flow_raw).sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)
        features_1m = self._normalize_features_1m(features_raw).sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)
        data_version = f"duckdb:{self._source.database_path.name}:{trade_date}"
        if self._symbol_limit is not None:
            data_version = f"{data_version}:symbol_limit:{self._symbol_limit}"
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version=data_version,
        )

    @staticmethod
    def _table_name_for_dataset(dataset_name: str) -> str:
        if dataset_name == "bars_5m":
            return "bars_5m"
        if dataset_name == "trade_flow_1m":
            return "trade_flow_1m"
        if dataset_name == "features_1m":
            return "features_1m"
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @staticmethod
    def _require_duckdb() -> None:
        if duckdb is None:
            raise ModuleNotFoundError("duckdb is required for LegacyDuckDBSource operations.")

    @staticmethod
    def _normalize_timestamps(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = frame.copy()
        for column in columns:
            if column in normalized.columns:
                normalized[column] = pd.to_datetime(normalized[column], utc=True)
        return normalized

    def _normalize_duckdb_timestamps(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = frame.copy()
        for column in columns:
            if column not in normalized.columns:
                continue
            series = pd.to_datetime(normalized[column], errors="coerce")
            if getattr(series.dt, "tz", None) is None:
                series = series.dt.tz_localize(self._duckdb_source_timezone()).dt.tz_convert("UTC")
            else:
                series = series.dt.tz_convert("UTC")
            normalized[column] = series
        return normalized

    def _normalize_trade_flow_1m(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "timestamp"])
        normalized = frame.copy()
        if "ticker" in normalized.columns:
            normalized = normalized.rename(columns={"ticker": "symbol"})
        if "minute" in normalized.columns:
            normalized = normalized.rename(columns={"minute": "timestamp"})
        if "imbalance_proxy" in normalized.columns and "imbalance_z" not in normalized.columns:
            normalized = normalized.rename(columns={"imbalance_proxy": "imbalance_z"})
        normalized = self._normalize_duckdb_timestamps(normalized, ["timestamp"])
        if "available_time" in normalized.columns:
            normalized = self._normalize_duckdb_timestamps(normalized, ["available_time"])
        else:
            normalized["available_time"] = normalized["timestamp"] + pd.Timedelta(minutes=1)
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

    def _normalize_features_1m(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        normalized = frame.rename(
            columns={
                "ret_1m_past": "ret_1m",
                "volume_z_proxy": "volume_z_12",
                "imbalance_proxy": "imbalance_z",
                "large_trade_ratio": "large_trade_ratio_z",
            }
        ).copy()
        normalized = self._normalize_duckdb_timestamps(normalized, ["timestamp", "bar_end"])
        if "available_time" in normalized.columns:
            normalized = self._normalize_duckdb_timestamps(normalized, ["available_time"])
        elif "bar_end" in normalized.columns:
            normalized["available_time"] = normalized["bar_end"]
        else:
            normalized["available_time"] = normalized["timestamp"] + pd.Timedelta(minutes=1)
        return normalized

    def _normalize_layout_features_1m(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "timestamp"])
        normalized = frame.copy()
        if "timestamp" in normalized.columns:
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
        if "bar_end" in normalized.columns:
            normalized["bar_end"] = pd.to_datetime(normalized["bar_end"], utc=True, errors="coerce")
        if "available_time" in normalized.columns:
            normalized["available_time"] = pd.to_datetime(normalized["available_time"], utc=True, errors="coerce")
        elif "bar_end" in normalized.columns:
            normalized["available_time"] = normalized["bar_end"]
        else:
            normalized["available_time"] = normalized["timestamp"] + pd.Timedelta(minutes=1)
        rename_map = {
            "ret_1m_past": "ret_1m",
            "volume_z_proxy": "volume_z_12",
            "large_trade_ratio": "large_trade_ratio_z",
            "imbalance_proxy": "imbalance_z",
        }
        normalized = normalized.rename(columns={key: value for key, value in rename_map.items() if key in normalized.columns})
        return normalized

    @staticmethod
    def _build_generated_features_1m(raw_1m: pd.DataFrame, trade_flow_1m: pd.DataFrame) -> pd.DataFrame:
        if raw_1m.empty:
            return pd.DataFrame()

        bars = raw_1m.copy()
        bars["symbol"] = bars["symbol"].astype(str)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        for column in ["close", "volume", "dollar_volume"]:
            if column in bars.columns:
                bars[column] = pd.to_numeric(bars[column], errors="coerce")

        flow = trade_flow_1m.copy()
        if flow.empty:
            flow = pd.DataFrame(columns=["symbol", "timestamp", "trade_count", "imbalance_proxy", "large_trade_dollar_volume"])
        else:
            flow = flow.rename(columns={"ticker": "symbol", "minute": "timestamp"}).copy()
            flow["symbol"] = flow["symbol"].astype(str)
            flow["timestamp"] = pd.to_datetime(flow["timestamp"], utc=True, errors="coerce")
            for column in ["trade_count", "imbalance_proxy", "large_trade_dollar_volume"]:
                if column not in flow.columns:
                    flow[column] = np.nan
                flow[column] = pd.to_numeric(flow[column], errors="coerce")
            flow = flow[["symbol", "timestamp", "trade_count", "imbalance_proxy", "large_trade_dollar_volume"]]

        merged = bars.merge(flow, on=["symbol", "timestamp"], how="left")
        merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        by_symbol = merged.groupby("symbol", sort=False)

        merged["ret_1m"] = by_symbol["close"].pct_change()
        merged["volume_z_12"] = MarketReadRepository._ratio_to_rolling_mean(by_symbol["volume"])
        merged["dollar_volume_z_proxy"] = MarketReadRepository._ratio_to_rolling_mean(by_symbol["dollar_volume"])
        merged["trade_count_z_proxy"] = MarketReadRepository._ratio_to_rolling_mean(by_symbol["trade_count"])
        merged["large_trade_ratio"] = (
            merged["large_trade_dollar_volume"].fillna(0.0).astype(float)
            / merged["dollar_volume"].replace(0.0, np.nan).astype(float)
        ).fillna(0.0)

        for source_column, output_column in [
            ("imbalance_proxy", "imbalance_z"),
            ("large_trade_ratio", "large_trade_ratio_z"),
        ]:
            rolling_mean = merged.groupby("symbol", sort=False)[source_column].transform(
                lambda series: series.shift(1).rolling(30, min_periods=3).mean()
            )
            rolling_std = merged.groupby("symbol", sort=False)[source_column].transform(
                lambda series: series.shift(1).rolling(30, min_periods=3).std(ddof=0)
            )
            merged[output_column] = (
                (merged[source_column] - rolling_mean) / rolling_std.replace(0.0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

        merged["flow_impulse_score"] = (
            0.35 * merged["dollar_volume_z_proxy"].fillna(0.0)
            + 0.25 * merged["trade_count_z_proxy"].fillna(0.0)
            + 0.25 * merged["imbalance_z"].fillna(0.0)
            + 0.15 * merged["large_trade_ratio_z"].fillna(0.0)
        )
        if "bar_end" in merged.columns:
            merged["available_time"] = merged["bar_end"]
        else:
            merged["available_time"] = merged["timestamp"] + pd.Timedelta(minutes=1)

        return merged[
            [
                "symbol",
                "timestamp",
                "available_time",
                "ret_1m",
                "volume_z_12",
                "imbalance_z",
                "large_trade_ratio_z",
                "flow_impulse_score",
            ]
        ].copy()

    @staticmethod
    def _ratio_to_rolling_mean(grouped: pd.core.groupby.generic.SeriesGroupBy) -> pd.Series:
        rolling_mean = grouped.transform(lambda series: series.shift(1).rolling(30, min_periods=1).mean())
        current = grouped.obj.astype(float)
        return ((current / rolling_mean.replace(0.0, np.nan)) - 1.0).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame.reset_index(drop=True)
        available_columns = [column for column in columns if column in frame.columns]
        if not available_columns:
            return frame.reset_index(drop=True)
        return frame.sort_values(available_columns).reset_index(drop=True)

    def _load_limited_symbols(self, connection: duckdb.DuckDBPyConnection, trade_date: str) -> list[str] | None:
        if self._symbol_limit is None:
            return None
        rows = connection.execute(
            """
            SELECT symbol
            FROM bars_5m
            WHERE date = ?
            GROUP BY symbol
            ORDER BY symbol
            LIMIT ?
            """,
            [trade_date, self._symbol_limit],
        ).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _build_symbol_filter_sql(symbols: list[str] | None) -> tuple[str, list[str]]:
        if not symbols:
            return "", []
        placeholders = ", ".join(["?"] * len(symbols))
        return f"AND symbol IN ({placeholders})", symbols

    def _duckdb_source_timezone(self) -> str:
        if self._resolved_duckdb_source_timezone:
            return self._resolved_duckdb_source_timezone
        return self._source.source_timezone if isinstance(self._source, LegacyDuckDBSource) else "UTC"

    def _resolve_duckdb_source_timezone(
        self,
        connection: duckdb.DuckDBPyConnection,
        trade_date: str,
    ) -> str:
        if self._resolved_duckdb_source_timezone:
            return self._resolved_duckdb_source_timezone
        if self._source.source_timezone != "auto":
            return self._source.source_timezone

        snapshot_clock = SnapshotClock()
        session_open_utc = snapshot_clock.session_open_timestamp(trade_date).to_pydatetime()
        session_close_utc = snapshot_clock.iter_trade_date(trade_date)[-1].to_pydatetime()

        best_timezone = "UTC"
        best_count = -1
        for candidate in ("UTC", "Asia/Taipei"):
            candidate_zone = ZoneInfo(candidate)
            open_local = session_open_utc.astimezone(candidate_zone).time()
            close_local = session_close_utc.astimezone(candidate_zone).time()
            count = self._count_bars_in_local_session(
                connection=connection,
                trade_date=trade_date,
                session_open_local=open_local,
                session_close_local=close_local,
            )
            if count > best_count:
                best_timezone = candidate
                best_count = count
        return best_timezone

    @staticmethod
    def _count_bars_in_local_session(
        *,
        connection: duckdb.DuckDBPyConnection,
        trade_date: str,
        session_open_local: datetime.time,
        session_close_local: datetime.time,
    ) -> int:
        if session_open_local <= session_close_local:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM bars_5m
                    WHERE date = ?
                      AND CAST(timestamp AS TIME) BETWEEN ? AND ?
                    """,
                    [trade_date, session_open_local.isoformat(), session_close_local.isoformat()],
                ).fetchone()[0]
            )

        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM bars_5m
                WHERE date = ?
                  AND (CAST(timestamp AS TIME) >= ? OR CAST(timestamp AS TIME) <= ?)
                """,
                [trade_date, session_open_local.isoformat(), session_close_local.isoformat()],
            ).fetchone()[0]
        )

    def _apply_symbol_limit(
        self,
        *,
        bars_5m: pd.DataFrame,
        trade_flow_1m: pd.DataFrame,
        features_1m: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._symbol_limit is None or bars_5m.empty or "symbol" not in bars_5m.columns:
            return bars_5m, trade_flow_1m, features_1m

        allowed_symbols = sorted(bars_5m["symbol"].dropna().astype(str).unique().tolist())[: self._symbol_limit]
        return (
            self._filter_frame_by_symbols(bars_5m, allowed_symbols),
            self._filter_frame_by_symbols(trade_flow_1m, allowed_symbols),
            self._filter_frame_by_symbols(features_1m, allowed_symbols),
        )

    @staticmethod
    def _filter_frame_by_symbols(frame: pd.DataFrame, allowed_symbols: list[str]) -> pd.DataFrame:
        if frame.empty or "symbol" not in frame.columns:
            return frame
        return frame[frame["symbol"].astype(str).isin(allowed_symbols)].copy()
