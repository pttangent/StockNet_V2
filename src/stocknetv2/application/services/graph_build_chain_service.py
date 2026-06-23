from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GraphBuildValidationSummary:
    database_path: str
    run_count: int
    trade_date_count: int
    snapshot_count: int
    edge_count: int
    layer_community_count: int
    date_start: str
    date_end: str


@dataclass(frozen=True)
class GraphBuildChainSummary:
    watched_run: GraphBuildValidationSummary
    full_history_run: GraphBuildValidationSummary


class GraphBuildChainService:
    def __init__(
        self,
        *,
        is_watched_run_active: Callable[[], bool],
        validate_watched_run: Callable[[], GraphBuildValidationSummary],
        is_full_history_run_active: Callable[[], bool],
        validate_full_history_run: Callable[[], GraphBuildValidationSummary],
        launch_full_history_run: Callable[[], None],
        sleep: Callable[[float], None],
        log: Callable[[str], None],
        poll_seconds: float = 30.0,
    ) -> None:
        self._is_watched_run_active = is_watched_run_active
        self._validate_watched_run = validate_watched_run
        self._is_full_history_run_active = is_full_history_run_active
        self._validate_full_history_run = validate_full_history_run
        self._launch_full_history_run = launch_full_history_run
        self._sleep = sleep
        self._log = log
        self._poll_seconds = poll_seconds

    def run(self) -> GraphBuildChainSummary:
        watched_summary = self._wait_for_completed_database(
            is_active=self._is_watched_run_active,
            validator=self._validate_watched_run,
            waiting_message="Monthly graph build still running; waiting for completion.",
            retry_message="Monthly graph build not yet valid; continuing to wait.",
        )
        self._log(
            "Monthly graph build validated with "
            f"{watched_summary.trade_date_count} trade dates and {watched_summary.edge_count} edges."
        )

        try:
            full_history_summary = self._validate_full_history_run()
            self._log(
                "Full-history graph build already validated with "
                f"{full_history_summary.trade_date_count} trade dates."
            )
            return GraphBuildChainSummary(
                watched_run=watched_summary,
                full_history_run=full_history_summary,
            )
        except Exception:
            pass

        if not self._is_full_history_run_active():
            self._log("Launching full-history graph build.")
            self._launch_full_history_run()
        else:
            self._log("Full-history graph build is already running; attaching to it.")

        full_history_summary = self._wait_for_completed_database(
            is_active=self._is_full_history_run_active,
            validator=self._validate_full_history_run,
            waiting_message="Full-history graph build still running; waiting for completion.",
            retry_message="Full-history graph build not yet valid; continuing to wait.",
        )
        self._log(
            "Full-history graph build validated with "
            f"{full_history_summary.trade_date_count} trade dates and {full_history_summary.edge_count} edges."
        )
        return GraphBuildChainSummary(
            watched_run=watched_summary,
            full_history_run=full_history_summary,
        )

    def _wait_for_completed_database(
        self,
        *,
        is_active: Callable[[], bool],
        validator: Callable[[], GraphBuildValidationSummary],
        waiting_message: str,
        retry_message: str,
    ) -> GraphBuildValidationSummary:
        while True:
            if is_active():
                self._log(waiting_message)
                self._sleep(self._poll_seconds)
                continue
            try:
                return validator()
            except Exception as exc:
                self._log(f"{retry_message} Last validation error: {exc}")
                self._sleep(self._poll_seconds)
