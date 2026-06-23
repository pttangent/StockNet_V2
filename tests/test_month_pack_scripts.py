from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.build_month_pack import build_month_pack
from scripts.ingest_trade_date_into_pack import ingest_trade_date_into_pack
from scripts.migrate_data_to_distributed_packs import cleanup_empty_legacy_data_roots
from scripts.validate_month_pack import validate_month_pack
from stocknetv2.infrastructure.project_paths import ProjectPaths


def _write_partition(root: Path, relative_dir: str, filename: str, frame: pd.DataFrame) -> None:
    target_dir = root / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / filename, index=False)


def test_build_month_pack_writes_expected_month_structure(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(tmp_path)
    trade_date = "2026-01-02"
    bars_5m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "close": [10.0],
        }
    )
    features_1m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "available_time": [datetime(2026, 1, 2, 14, 36, tzinfo=UTC)],
            "symbol": ["AAA"],
            "ret_1m": [0.01],
            "volume_z_12": [1.0],
            "imbalance_z": [0.2],
            "large_trade_ratio_z": [0.3],
            "flow_impulse_score": [0.4],
        }
    )
    raw_1m = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "bar_end": [datetime(2026, 1, 2, 14, 36, tzinfo=UTC)],
            "close": [10.0],
            "volume": [100.0],
            "dollar_volume": [1000.0],
        }
    )
    trade_flow_1m = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "minute": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "trade_count": [10.0],
            "dollar_volume": [1000.0],
            "imbalance_proxy": [0.2],
            "large_trade_dollar_volume": [100.0],
        }
    )
    labels_1m = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "future_ret_1m": [0.01],
            "future_ret_5m": [0.02],
            "future_ret_15m": [0.03],
            "future_ret_30m": [0.04],
        }
    )

    _write_partition(paths.data_root, f"bars_5m/date={trade_date}", "bars_5m.parquet", bars_5m)
    _write_partition(paths.data_root, f"features_1m/date={trade_date}", "features_1m.parquet", features_1m)
    _write_partition(paths.data_root, f"raw_1m/date={trade_date}", "bars_1m.parquet", raw_1m)
    _write_partition(paths.data_root, f"trade_flow_1m/date={trade_date}", "trade_flow_1m.parquet", trade_flow_1m)
    _write_partition(paths.data_root, f"labels_1m/date={trade_date}", "labels_1m.parquet", labels_1m)

    pack_root = build_month_pack(
        data_root=paths.data_root,
        output_root=paths.distributed_packs_root,
        month="2026-01",
    )

    assert (pack_root / "pack_manifest.json").exists()
    assert (pack_root / "snapshot_schedule.parquet").exists()
    assert (pack_root / "symbol_universe.parquet").exists()
    assert (pack_root / "layer_input_schema.json").exists()
    assert (pack_root / f"dates/date={trade_date}/bars_5m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/raw_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/trade_flow_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/features_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/labels_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/graph_features_1m.parquet").exists()
    manifest = json.loads((pack_root / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["month"] == "2026-01"
    assert manifest["trade_dates"] == [trade_date]


def test_validate_month_pack_reports_valid_pack(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(tmp_path)
    trade_date = "2026-01-02"
    pack_root = paths.distributed_packs_root / "month=2026-01"
    (pack_root / f"dates/date={trade_date}").mkdir(parents=True, exist_ok=True)
    (pack_root / "pack_manifest.json").write_text(
        json.dumps({"month": "2026-01", "trade_dates": [trade_date], "files": []}),
        encoding="utf-8",
    )
    pd.DataFrame({"snapshot_id": ["2026-01-02T14:35:00Z"]}).to_parquet(pack_root / "snapshot_schedule.parquet")
    pd.DataFrame({"symbol_id": [1], "symbol": ["AAA"], "universe_version": ["2026-01"]}).to_parquet(
        pack_root / "symbol_universe.parquet"
    )
    (pack_root / "layer_input_schema.json").write_text(json.dumps({"features": ["ret_1m"]}), encoding="utf-8")
    pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol_id": [1], "symbol": ["AAA"], "close": [10.0]}).to_parquet(
        pack_root / f"dates/date={trade_date}/bars_5m.parquet"
    )
    pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol_id": [1], "symbol": ["AAA"]}).to_parquet(
        pack_root / f"dates/date={trade_date}/raw_1m.parquet"
    )
    pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol_id": [1], "symbol": ["AAA"]}).to_parquet(
        pack_root / f"dates/date={trade_date}/trade_flow_1m.parquet"
    )
    pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol_id": [1], "symbol": ["AAA"], "ret_1m": [0.1]}).to_parquet(
        pack_root / f"dates/date={trade_date}/features_1m.parquet"
    )
    pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol_id": [1], "symbol": ["AAA"], "future_ret_1m": [0.1]}).to_parquet(
        pack_root / f"dates/date={trade_date}/labels_1m.parquet"
    )
    pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol_id": [1], "symbol": ["AAA"], "ret_1m": [0.1]}).to_parquet(
        pack_root / f"dates/date={trade_date}/graph_features_1m.parquet"
    )
    (pack_root / f"dates/date={trade_date}/date_manifest.json").write_text(json.dumps({"trade_date": trade_date}), encoding="utf-8")

    summary = validate_month_pack(pack_root)

    assert summary["status"] == "ok"
    assert summary["month"] == "2026-01"
    assert summary["trade_dates"] == [trade_date]


def test_build_month_pack_can_move_legacy_source_files(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(tmp_path)
    trade_date = "2026-01-02"
    bars_path = paths.data_root / f"bars_5m/date={trade_date}/bars_5m.parquet"
    raw_path = paths.data_root / f"raw_1m/date={trade_date}/bars_1m.parquet"
    flow_path = paths.data_root / f"trade_flow_1m/date={trade_date}/trade_flow_1m.parquet"
    features_path = paths.data_root / f"features_1m/date={trade_date}/features_1m.parquet"
    labels_path = paths.data_root / f"labels_1m/date={trade_date}/labels_1m.parquet"

    _write_partition(paths.data_root, f"bars_5m/date={trade_date}", "bars_5m.parquet", pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "close": [10.0]}))
    _write_partition(paths.data_root, f"raw_1m/date={trade_date}", "bars_1m.parquet", pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "close": [10.0], "volume": [100.0], "dollar_volume": [1000.0]}))
    _write_partition(paths.data_root, f"trade_flow_1m/date={trade_date}", "trade_flow_1m.parquet", pd.DataFrame({"minute": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "ticker": ["AAA"], "trade_count": [10.0], "dollar_volume": [1000.0], "imbalance_proxy": [0.2], "large_trade_dollar_volume": [100.0]}))
    _write_partition(paths.data_root, f"features_1m/date={trade_date}", "features_1m.parquet", pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "available_time": [datetime(2026, 1, 2, 14, 36, tzinfo=UTC)], "symbol": ["AAA"], "ret_1m": [0.01], "volume_z_12": [1.0], "imbalance_z": [0.2], "large_trade_ratio_z": [0.3], "flow_impulse_score": [0.4]}))
    _write_partition(paths.data_root, f"labels_1m/date={trade_date}", "labels_1m.parquet", pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "future_ret_1m": [0.01], "future_ret_5m": [0.02], "future_ret_15m": [0.03], "future_ret_30m": [0.04]}))

    pack_root = build_month_pack(
        data_root=paths.data_root,
        output_root=paths.distributed_packs_root,
        month="2026-01",
        storage_mode="move",
    )

    assert not bars_path.exists()
    assert not raw_path.exists()
    assert not flow_path.exists()
    assert not features_path.exists()
    assert not labels_path.exists()
    assert (pack_root / f"dates/date={trade_date}/bars_5m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/raw_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/trade_flow_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/features_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/labels_1m.parquet").exists()


def test_ingest_trade_date_into_pack_moves_incoming_files_and_updates_manifest(tmp_path: Path) -> None:
    incoming_root = tmp_path / "incoming"
    incoming_root.mkdir(parents=True, exist_ok=True)
    bars_path = incoming_root / "bars_5m.parquet"
    raw_path = incoming_root / "bars_1m.parquet"
    flow_path = incoming_root / "trade_flow_1m.parquet"
    features_path = incoming_root / "features_1m.parquet"
    labels_path = incoming_root / "labels_1m.parquet"

    pd.DataFrame({"timestamp": [datetime(2026, 2, 3, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "close": [10.0]}).to_parquet(bars_path, index=False)
    pd.DataFrame({"timestamp": [datetime(2026, 2, 3, 14, 35, tzinfo=UTC)], "bar_end": [datetime(2026, 2, 3, 14, 36, tzinfo=UTC)], "symbol": ["AAA"], "close": [10.0], "volume": [100.0], "dollar_volume": [1000.0]}).to_parquet(raw_path, index=False)
    pd.DataFrame({"minute": [datetime(2026, 2, 3, 14, 35, tzinfo=UTC)], "ticker": ["AAA"], "trade_count": [10.0], "dollar_volume": [1000.0], "imbalance_proxy": [0.2], "large_trade_dollar_volume": [100.0]}).to_parquet(flow_path, index=False)
    pd.DataFrame({"timestamp": [datetime(2026, 2, 3, 14, 35, tzinfo=UTC)], "available_time": [datetime(2026, 2, 3, 14, 36, tzinfo=UTC)], "symbol": ["AAA"], "ret_1m": [0.01], "volume_z_12": [1.0], "imbalance_z": [0.2], "large_trade_ratio_z": [0.3], "flow_impulse_score": [0.4]}).to_parquet(features_path, index=False)
    pd.DataFrame({"timestamp": [datetime(2026, 2, 3, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "future_ret_1m": [0.01], "future_ret_5m": [0.02], "future_ret_15m": [0.03], "future_ret_30m": [0.04]}).to_parquet(labels_path, index=False)

    pack_root = ingest_trade_date_into_pack(
        trade_date="2026-02-03",
        bars_5m_path=bars_path,
        raw_1m_path=raw_path,
        trade_flow_1m_path=flow_path,
        features_1m_path=features_path,
        labels_1m_path=labels_path,
        output_root=tmp_path / "data" / "distributed_packs",
    )

    assert not bars_path.exists()
    assert not raw_path.exists()
    assert not flow_path.exists()
    assert not features_path.exists()
    assert not labels_path.exists()
    assert (pack_root / "pack_manifest.json").exists()
    assert (pack_root / "dates/date=2026-02-03/bars_5m.parquet").exists()
    assert (pack_root / "dates/date=2026-02-03/raw_1m.parquet").exists()
    assert (pack_root / "dates/date=2026-02-03/trade_flow_1m.parquet").exists()
    assert (pack_root / "dates/date=2026-02-03/features_1m.parquet").exists()
    assert (pack_root / "dates/date=2026-02-03/labels_1m.parquet").exists()
    assert (pack_root / "dates/date=2026-02-03/graph_features_1m.parquet").exists()


def test_build_month_pack_is_idempotent_after_move_when_legacy_dirs_are_empty(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(tmp_path)
    trade_date = "2026-01-02"

    _write_partition(
        paths.data_root,
        f"bars_5m/date={trade_date}",
        "bars_5m.parquet",
        pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "close": [10.0]}),
    )
    _write_partition(
        paths.data_root,
        f"raw_1m/date={trade_date}",
        "bars_1m.parquet",
        pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "close": [10.0], "volume": [100.0], "dollar_volume": [1000.0]}),
    )
    _write_partition(
        paths.data_root,
        f"trade_flow_1m/date={trade_date}",
        "trade_flow_1m.parquet",
        pd.DataFrame({"minute": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "ticker": ["AAA"], "trade_count": [10.0], "dollar_volume": [1000.0], "imbalance_proxy": [0.2], "large_trade_dollar_volume": [100.0]}),
    )
    _write_partition(
        paths.data_root,
        f"features_1m/date={trade_date}",
        "features_1m.parquet",
        pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "available_time": [datetime(2026, 1, 2, 14, 36, tzinfo=UTC)], "symbol": ["AAA"], "ret_1m": [0.01], "volume_z_12": [1.0], "imbalance_z": [0.2], "large_trade_ratio_z": [0.3], "flow_impulse_score": [0.4]}),
    )
    _write_partition(
        paths.data_root,
        f"labels_1m/date={trade_date}",
        "labels_1m.parquet",
        pd.DataFrame({"timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)], "symbol": ["AAA"], "future_ret_1m": [0.01], "future_ret_5m": [0.02], "future_ret_15m": [0.03], "future_ret_30m": [0.04]}),
    )

    pack_root = build_month_pack(
        data_root=paths.data_root,
        output_root=paths.distributed_packs_root,
        month="2026-01",
        storage_mode="move",
    )

    rerun_pack_root = build_month_pack(
        data_root=paths.data_root,
        output_root=paths.distributed_packs_root,
        month="2026-01",
        storage_mode="move",
    )

    assert rerun_pack_root == pack_root
    manifest = json.loads((pack_root / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["trade_dates"] == [trade_date]
    assert (pack_root / f"dates/date={trade_date}/features_1m.parquet").exists()
    assert (pack_root / f"dates/date={trade_date}/labels_1m.parquet").exists()


def test_cleanup_empty_legacy_data_roots_removes_empty_date_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    empty_date_dir = data_root / "raw_1m" / "date=2026-02-03"
    empty_date_dir.mkdir(parents=True, exist_ok=True)

    removed = cleanup_empty_legacy_data_roots(data_root)

    assert str(empty_date_dir) in removed
    assert not empty_date_dir.exists()
    assert not (data_root / "raw_1m").exists()
