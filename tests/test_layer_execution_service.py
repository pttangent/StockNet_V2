from __future__ import annotations

from concurrent.futures import Future
from datetime import UTC, datetime

import pandas as pd

from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.domain.graph.layer_config import CommunityDetectionConfig, ThemeDiscoverySettings
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


def _test_settings() -> ThemeDiscoverySettings:
    return ThemeDiscoverySettings(
        layer_community_detection=CommunityDetectionConfig(
            algorithm="connected_components",
            fallback_algorithm="error",
        )
    )


def _build_trade_date_inputs() -> TradeDateInputs:
    minute_timestamps = list(pd.date_range("2026-01-02T14:31:00Z", periods=20, freq="1min"))
    bars_timestamps = [
        datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 40, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 45, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 50, tzinfo=UTC),
    ]
    base = [float(index % 6) + 0.1 * index for index in range(20)]
    returns = [0.001 * ((index % 5) - 2) + 0.0001 * index for index in range(20)]

    bars_5m = pd.DataFrame(
        {
            "timestamp": bars_timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "close": [10.0, 10.1, 10.2, 10.3] + [20.0, 20.22, 20.39, 20.65] + [30.0, 30.1, 30.4, 30.2],
        }
    )
    features_1m = pd.DataFrame(
        {
            "timestamp": minute_timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "ret_1m": returns + [value * 1.01 for value in returns] + list(reversed(returns)),
            "volume_z_12": base + [value * 1.02 for value in base] + list(reversed(base)),
            "large_trade_ratio_z": [value * 0.3 for value in base]
            + [value * 0.303 for value in base]
            + [value * 0.2 for value in reversed(base)],
        }
    )
    trade_flow_1m = pd.DataFrame(
        {
            "timestamp": minute_timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "flow_impulse_score": base + [value * 1.01 for value in base] + list(reversed(base)),
            "imbalance_z": [value * 0.2 for value in base]
            + [value * 0.202 for value in base]
            + [value * -0.1 for value in base],
            "large_trade_ratio_z": [value * 0.3 for value in base]
            + [value * 0.303 for value in base]
            + [value * 0.2 for value in reversed(base)],
        }
    )
    return TradeDateInputs(
        trade_date="2026-01-02",
        bars_5m=bars_5m,
        trade_flow_1m=trade_flow_1m,
        features_1m=features_1m,
        data_version="test-data-version",
    )


def test_layer_execution_service_builds_all_six_layer_outputs():
    service = LayerExecutionService(settings=_test_settings())
    inputs = _build_trade_date_inputs()

    result = service.execute_for_snapshot(
        inputs=inputs,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
    )

    assert set(result.layer_edges) == {
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    }
    assert len(result.layer_edges["dtw_return_similarity_graph"]) >= 1
    assert len(result.layer_edges["flow_alignment_graph"]) >= 1
    assert len(result.layer_edges["dtw_trade_flow_similarity_graph"]) >= 1
    assert len(result.layer_edges["volume_expansion_graph"]) >= 1
    assert len(result.layer_edges["large_trade_alignment_graph"]) >= 1


def test_return_window_excludes_premarket_and_caps_regular_session_history():
    timestamps = list(pd.date_range("2026-01-02T13:30:00Z", periods=30, freq="5min"))
    bars = pd.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["AAA"] * 30 + ["BBB"] * 30,
            "close": [100.0 + index for index in range(30)] + [200.0 + 2 * index for index in range(30)],
        }
    )

    window = LayerExecutionService._build_return_window(
        bars,
        pd.Timestamp("2026-01-02T15:30:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
        lookback_bars=6,
    )

    assert len(window) == 6
    assert window.index.min() > pd.Timestamp("2026-01-02T14:30:00Z")
    assert window.index.max() <= pd.Timestamp("2026-01-02T15:30:00Z")


class _InlineExecutor:
    def __init__(self) -> None:
        self.submitted_layer_names: list[str] = []

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def submit(self, fn, *args, **kwargs):
        self.submitted_layer_names.append(args[0])
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def test_layer_execution_service_can_dispatch_layers_through_executor():
    executor = _InlineExecutor()
    service = LayerExecutionService(
        parallel_workers=3,
        executor_factory=lambda max_workers: executor,
        settings=_test_settings(),
    )
    inputs = _build_trade_date_inputs()

    result = service.execute_for_snapshot(
        inputs=inputs,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
    )

    assert executor.submitted_layer_names == [
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    ]
    assert set(result.layer_edges) == set(executor.submitted_layer_names)
