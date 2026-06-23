from __future__ import annotations

from stocknetv2.application.services.graph_build_chain_service import (
    GraphBuildChainService,
    GraphBuildValidationSummary,
)


def _summary(database_path: str, trade_date_count: int) -> GraphBuildValidationSummary:
    return GraphBuildValidationSummary(
        database_path=database_path,
        run_count=trade_date_count,
        trade_date_count=trade_date_count,
        snapshot_count=trade_date_count * 78,
        edge_count=trade_date_count * 100,
        layer_community_count=trade_date_count * 10,
        date_start="2025-01-02",
        date_end="2025-01-31",
    )


def test_graph_build_chain_service_waits_for_month_then_launches_full_history():
    watched_active_states = iter([True, False])
    full_history_active_states = iter([False, True, False])
    launch_calls: list[str] = []
    log_messages: list[str] = []
    sleep_calls: list[float] = []
    full_history_validation_attempts = {"count": 0}

    def validate_watched_run():
        return _summary("month.duckdb", 20)

    def validate_full_history_run():
        full_history_validation_attempts["count"] += 1
        if full_history_validation_attempts["count"] == 1:
            raise FileNotFoundError("not ready")
        return _summary("full.duckdb", 612)

    service = GraphBuildChainService(
        is_watched_run_active=lambda: next(watched_active_states, False),
        validate_watched_run=validate_watched_run,
        is_full_history_run_active=lambda: next(full_history_active_states, False),
        validate_full_history_run=validate_full_history_run,
        launch_full_history_run=lambda: launch_calls.append("launched"),
        sleep=lambda seconds: sleep_calls.append(seconds),
        log=log_messages.append,
        poll_seconds=0.5,
    )

    summary = service.run()

    assert summary.watched_run.trade_date_count == 20
    assert summary.full_history_run.trade_date_count == 612
    assert launch_calls == ["launched"]
    assert sleep_calls == [0.5, 0.5]
    assert any("Monthly graph build validated" in message for message in log_messages)
    assert any("Launching full-history graph build." in message for message in log_messages)


def test_graph_build_chain_service_skips_launch_when_full_history_already_valid():
    launch_calls: list[str] = []

    service = GraphBuildChainService(
        is_watched_run_active=lambda: False,
        validate_watched_run=lambda: _summary("month.duckdb", 20),
        is_full_history_run_active=lambda: False,
        validate_full_history_run=lambda: _summary("full.duckdb", 612),
        launch_full_history_run=lambda: launch_calls.append("launched"),
        sleep=lambda seconds: None,
        log=lambda message: None,
        poll_seconds=0.5,
    )

    summary = service.run()

    assert summary.full_history_run.trade_date_count == 612
    assert launch_calls == []
