from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class SnapshotClock:
    """Emit 5-minute T1 snapshot timestamps aligned to the US market session."""

    def __init__(
        self,
        *,
        frame_minutes: int = 5,
        session_open: time = time(hour=9, minute=30),
        first_snapshot: time = time(hour=9, minute=35),
        session_close: time = time(hour=16, minute=0),
    ) -> None:
        self._frame_minutes = frame_minutes
        self._session_open = session_open
        self._first_snapshot = first_snapshot
        self._session_close = session_close

    def iter_trade_date(self, trade_date: str | date) -> list[pd.Timestamp]:
        market_date = self._coerce_date(trade_date)
        current = datetime.combine(market_date, self._first_snapshot, tzinfo=NEW_YORK)
        session_end = datetime.combine(market_date, self._session_close, tzinfo=NEW_YORK)

        snapshots: list[pd.Timestamp] = []
        while current <= session_end:
            snapshots.append(pd.Timestamp(current.astimezone(UTC)))
            current += timedelta(minutes=self._frame_minutes)
        return snapshots

    def iter_range(self, start_date: str | date, end_date: str | date) -> list[pd.Timestamp]:
        start = self._coerce_date(start_date)
        end = self._coerce_date(end_date)
        current = start
        snapshots: list[pd.Timestamp] = []

        while current <= end:
            if current.weekday() < 5:
                snapshots.extend(self.iter_trade_date(current))
            current += timedelta(days=1)

        return snapshots

    def session_open_timestamp(self, trade_date: str | date) -> pd.Timestamp:
        market_date = self._coerce_date(trade_date)
        market_open = datetime.combine(market_date, self._session_open, tzinfo=NEW_YORK)
        return pd.Timestamp(market_open.astimezone(UTC))

    @staticmethod
    def _coerce_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)
