from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_watch_graph_build_chain_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "watch_graph_build_chain.py"
    )
    spec = importlib.util.spec_from_file_location("watch_graph_build_chain", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_partition(root: Path, dataset: str, trade_date: str) -> None:
    partition = root / dataset / f"date={trade_date}"
    partition.mkdir(parents=True, exist_ok=True)


def test_resolve_full_history_trade_dates_uses_trade_flow_intersection(tmp_path: Path):
    _make_partition(tmp_path, "bars_5m", "2024-01-02")
    _make_partition(tmp_path, "bars_5m", "2025-01-02")
    _make_partition(tmp_path, "bars_5m", "2025-01-03")
    _make_partition(tmp_path, "raw_1m", "2024-01-02")
    _make_partition(tmp_path, "raw_1m", "2025-01-02")
    _make_partition(tmp_path, "raw_1m", "2025-01-03")
    _make_partition(tmp_path, "trade_flow_1m", "2025-01-02")
    _make_partition(tmp_path, "trade_flow_1m", "2025-01-03")

    module = _load_watch_graph_build_chain_module()

    trade_dates = module._resolve_full_history_trade_dates(tmp_path)

    assert trade_dates == ["2025-01-02", "2025-01-03"]


def test_build_python_process_probe_command_excludes_current_pid():
    module = _load_watch_graph_build_chain_module()

    command = module._build_python_process_probe_command(
        command_token="full_market_graph_build_2025_01.duckdb",
        exclude_pid=12345,
    )

    assert "$_.ProcessId -ne 12345" in command
    assert "full_market_graph_build_2025_01.duckdb" in command
