from __future__ import annotations

import argparse
import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.infrastructure.project_paths import ProjectPaths
from stocknetv2.infrastructure.repositories.market_read_repository import MarketReadRepository

PACK_SCHEMA_VERSION = "stocknet-month-pack-v2"
PACK_DATASET_FILES = {
    "bars_5m": "bars_5m.parquet",
    "bars_15m": "bars_15m.parquet",
    "raw_1m": "raw_1m.parquet",
    "trade_flow_1m": "trade_flow_1m.parquet",
    "features_1m": "features_1m.parquet",
    "labels_1m": "labels_1m.parquet",
    "graph_features_1m": "graph_features_1m.parquet",
}
REQUIRED_SOURCE_DATASETS = ("bars_5m", "bars_15m", "raw_1m", "trade_flow_1m", "labels_1m")
DERIVED_DATASETS = ("features_1m", "graph_features_1m")
REQUIRED_READY_DATASETS = (*REQUIRED_SOURCE_DATASETS, *DERIVED_DATASETS)
LEGACY_DATASET_FILES = {
    "bars_5m": "bars_5m.parquet",
    "bars_15m": "bars_15m.parquet",
    "raw_1m": "bars_1m.parquet",
    "trade_flow_1m": "trade_flow_1m.parquet",
    "features_1m": "features_1m.parquet",
    "labels_1m": "labels_1m.parquet",
}
LAYER_INPUT_COLUMNS = [
    "timestamp",
    "available_time",
    "symbol",
    "symbol_id",
    "ret_1m",
    "ret_1m_past",
    "volume_z_proxy",
    "volume_z_12",
    "imbalance_proxy",
    "imbalance_z",
    "large_trade_ratio",
    "large_trade_ratio_z",
    "flow_impulse_score",
    "trade_count",
    "dollar_volume",
    "large_trade_dollar_volume",
]


def build_month_pack(
    *,
    data_root: Path | str,
    output_root: Path | str,
    month: str,
    code_commit: str = "migrated-without-git-history",
    storage_mode: str = "copy",
) -> Path:
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    pack_root = output_root / f"month={month}"
    pack_root.mkdir(parents=True, exist_ok=True)

    trade_dates = _resolve_month_trade_dates(data_root=data_root, pack_root=pack_root, month=month)
    if not trade_dates:
        raise RuntimeError(f"No packable trade dates found for month {month}.")

    symbol_ids: dict[str, int] = {}
    for trade_date in trade_dates:
        date_root = pack_root / "dates" / f"date={trade_date}"
        date_root.mkdir(parents=True, exist_ok=True)

        bars_path = date_root / PACK_DATASET_FILES["bars_5m"]
        bars_15m_path = date_root / PACK_DATASET_FILES["bars_15m"]
        raw_path = date_root / PACK_DATASET_FILES["raw_1m"]
        flow_path = date_root / PACK_DATASET_FILES["trade_flow_1m"]
        features_path = date_root / PACK_DATASET_FILES["features_1m"]
        labels_path = date_root / PACK_DATASET_FILES["labels_1m"]
        graph_features_path = date_root / PACK_DATASET_FILES["graph_features_1m"]

        bars_5m = _stage_existing_dataset_file(
            source_path=_legacy_source_path(data_root, "bars_5m", trade_date),
            target_path=bars_path,
            storage_mode=storage_mode,
        )
        bars_15m = _stage_existing_dataset_file(
            source_path=_legacy_source_path(data_root, "bars_15m", trade_date),
            target_path=bars_15m_path,
            storage_mode=storage_mode,
        )
        raw_1m = _stage_existing_dataset_file(
            source_path=_legacy_source_path(data_root, "raw_1m", trade_date),
            target_path=raw_path,
            storage_mode=storage_mode,
        )
        trade_flow_1m = _stage_existing_dataset_file(
            source_path=_legacy_source_path(data_root, "trade_flow_1m", trade_date),
            target_path=flow_path,
            storage_mode=storage_mode,
        )
        features_1m = _stage_existing_dataset_file(
            source_path=_legacy_source_path(data_root, "features_1m", trade_date),
            target_path=features_path,
            storage_mode=storage_mode,
        )
        labels_1m = _stage_existing_dataset_file(
            source_path=_legacy_source_path(data_root, "labels_1m", trade_date),
            target_path=labels_path,
            storage_mode=storage_mode,
        )
        graph_features_1m = _ensure_graph_features_file(
            target_path=graph_features_path,
            features_1m=features_1m,
            trade_flow_1m=trade_flow_1m,
        )

        discovered_symbols = (
            _extract_symbols(bars_5m)
            | _extract_symbols(bars_15m)
            | _extract_symbols(raw_1m)
            | _extract_symbols(_normalize_trade_flow_symbols(trade_flow_1m))
            | _extract_symbols(features_1m)
            | _extract_symbols(labels_1m)
            | _extract_symbols(graph_features_1m)
        )
        _register_symbol_ids(symbol_ids, discovered_symbols)

        _rewrite_existing_parquet_with_symbol_ids(bars_path, bars_5m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(bars_15m_path, bars_15m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(raw_path, raw_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(flow_path, trade_flow_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(features_path, features_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(labels_path, labels_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(graph_features_path, graph_features_1m, symbol_ids)

    return rebuild_month_pack_metadata(pack_root=pack_root, code_commit=code_commit)


def ingest_trade_date_into_pack(
    *,
    trade_date: str,
    bars_5m_path: Path | str,
    raw_1m_path: Path | str,
    trade_flow_1m_path: Path | str,
    labels_1m_path: Path | str,
    output_root: Path | str,
    features_1m_path: Path | str | None = None,
    code_commit: str = "ingested-directly-to-pack",
) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    month = trade_date[:7]
    pack_root = output_root / f"month={month}"
    date_root = pack_root / "dates" / f"date={trade_date}"
    date_root.mkdir(parents=True, exist_ok=True)

    staged_bars_path = date_root / PACK_DATASET_FILES["bars_5m"]
    staged_raw_path = date_root / PACK_DATASET_FILES["raw_1m"]
    staged_flow_path = date_root / PACK_DATASET_FILES["trade_flow_1m"]
    staged_features_path = date_root / PACK_DATASET_FILES["features_1m"]
    staged_labels_path = date_root / PACK_DATASET_FILES["labels_1m"]
    staged_graph_features_path = date_root / PACK_DATASET_FILES["graph_features_1m"]

    _move_file(Path(bars_5m_path).expanduser().resolve(), staged_bars_path)
    _move_file(Path(raw_1m_path).expanduser().resolve(), staged_raw_path)
    _move_file(Path(trade_flow_1m_path).expanduser().resolve(), staged_flow_path)
    _move_file(Path(labels_1m_path).expanduser().resolve(), staged_labels_path)

    raw_1m = _read_optional_parquet(staged_raw_path)
    trade_flow_1m = _read_optional_parquet(staged_flow_path)
    if features_1m_path is not None:
        _move_file(Path(features_1m_path).expanduser().resolve(), staged_features_path)
        features_1m = _read_optional_parquet(staged_features_path)
    else:
        features_1m = _generate_features_1m(raw_1m=raw_1m, trade_flow_1m=trade_flow_1m)
        _write_parquet_frame(staged_features_path, features_1m)

    graph_features = _build_graph_features_from_frames(features_1m=features_1m, trade_flow_1m=trade_flow_1m)
    _write_parquet_frame(staged_graph_features_path, graph_features)

    rebuild_month_pack_metadata(pack_root=pack_root, code_commit=code_commit)
    return pack_root


def rebuild_month_pack_metadata(*, pack_root: Path | str, code_commit: str = "rebuild-month-pack") -> Path:
    pack_root = Path(pack_root).expanduser().resolve()
    month = pack_root.name.split("=", 1)[1] if "=" in pack_root.name else pack_root.name
    snapshot_clock = SnapshotClock()
    symbol_ids: dict[str, int] = {}
    symbol_universe_rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    trade_dates: list[str] = []

    dates_root = pack_root / "dates"
    if not dates_root.exists():
        dates_root.mkdir(parents=True, exist_ok=True)

    for date_root in sorted(dates_root.glob("date=*")):
        if not date_root.is_dir() or not _date_root_has_payload(date_root):
            continue

        trade_date = date_root.name.split("=", 1)[1]
        bars_path = date_root / PACK_DATASET_FILES["bars_5m"]
        bars_15m_path = date_root / PACK_DATASET_FILES["bars_15m"]
        raw_path = date_root / PACK_DATASET_FILES["raw_1m"]
        flow_path = date_root / PACK_DATASET_FILES["trade_flow_1m"]
        features_path = date_root / PACK_DATASET_FILES["features_1m"]
        labels_path = date_root / PACK_DATASET_FILES["labels_1m"]
        graph_features_path = date_root / PACK_DATASET_FILES["graph_features_1m"]

        bars_5m = _read_optional_parquet(bars_path)
        bars_15m = _read_optional_parquet(bars_15m_path)
        raw_1m = _read_optional_parquet(raw_path)
        trade_flow_1m = _read_optional_parquet(flow_path)
        features_1m = _read_optional_parquet(features_path)
        labels_1m = _read_optional_parquet(labels_path)

        graph_features = _read_optional_parquet(graph_features_path)
        if graph_features.empty and not graph_features_path.exists() and not features_1m.empty:
            graph_features = _build_graph_features_from_frames(features_1m=features_1m, trade_flow_1m=trade_flow_1m)
            _write_parquet_frame(graph_features_path, graph_features)
        elif graph_features.empty and graph_features_path.exists() and not features_1m.empty:
            graph_features = _build_graph_features_from_frames(features_1m=features_1m, trade_flow_1m=trade_flow_1m)
            _write_parquet_frame(graph_features_path, graph_features)

        discovered_symbols = (
            _extract_symbols(bars_5m)
            | _extract_symbols(bars_15m)
            | _extract_symbols(raw_1m)
            | _extract_symbols(_normalize_trade_flow_symbols(trade_flow_1m))
            | _extract_symbols(features_1m)
            | _extract_symbols(labels_1m)
            | _extract_symbols(graph_features)
        )
        _register_symbol_ids(symbol_ids, discovered_symbols)
        for symbol, symbol_id in sorted(symbol_ids.items(), key=lambda item: item[1]):
            if symbol_id > len(symbol_universe_rows):
                symbol_universe_rows.append(
                    {"symbol_id": symbol_id, "symbol": symbol, "universe_version": month}
                )

        _rewrite_existing_parquet_with_symbol_ids(bars_path, bars_5m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(bars_15m_path, bars_15m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(raw_path, raw_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(flow_path, trade_flow_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(features_path, features_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(labels_path, labels_1m, symbol_ids)
        _rewrite_existing_parquet_with_symbol_ids(graph_features_path, graph_features, symbol_ids)

        missing_required_files, empty_required_files = _required_file_gaps(date_root)
        readiness_status = "ready" if not missing_required_files and not empty_required_files else "not_ready"

        trade_dates.append(trade_date)
        date_manifest = {
            "trade_date": trade_date,
            "readiness_status": readiness_status,
            "missing_required_files": missing_required_files,
            "empty_required_files": empty_required_files,
            "bars_5m_path": _relative_path_if_exists(pack_root, bars_path),
            "bars_15m_path": _relative_path_if_exists(pack_root, bars_15m_path),
            "raw_1m_path": _relative_path_if_exists(pack_root, raw_path),
            "trade_flow_1m_path": _relative_path_if_exists(pack_root, flow_path),
            "features_1m_path": _relative_path_if_exists(pack_root, features_path),
            "labels_1m_path": _relative_path_if_exists(pack_root, labels_path),
            "graph_features_1m_path": _relative_path_if_exists(pack_root, graph_features_path),
            "bar_rows": int(len(_read_optional_parquet(bars_path))),
            "bars_15m_rows": int(len(_read_optional_parquet(bars_15m_path))),
            "raw_1m_rows": int(len(_read_optional_parquet(raw_path))),
            "trade_flow_1m_rows": int(len(_read_optional_parquet(flow_path))),
            "features_1m_rows": int(len(_read_optional_parquet(features_path))),
            "labels_1m_rows": int(len(_read_optional_parquet(labels_path))),
            "graph_features_1m_rows": int(len(_read_optional_parquet(graph_features_path))),
        }
        date_manifest_path = date_root / "date_manifest.json"
        date_manifest_path.write_text(json.dumps(date_manifest, indent=2), encoding="utf-8")

        for path in (
            bars_path,
            bars_15m_path,
            raw_path,
            flow_path,
            features_path,
            labels_path,
            graph_features_path,
            date_manifest_path,
        ):
            if path.exists():
                file_rows.append(_file_manifest_row(pack_root, path, None if path.suffix == ".json" else len(_read_optional_parquet(path))))

        for snapshot_time in snapshot_clock.iter_trade_date(trade_date):
            schedule_rows.append(
                {
                    "trade_date": trade_date,
                    "snapshot_id": snapshot_time.isoformat(),
                    "snapshot_clock": snapshot_time.strftime("%H%M"),
                    "timestamp": snapshot_time,
                }
            )

    symbol_universe = pd.DataFrame(symbol_universe_rows).sort_values(["symbol_id"]).reset_index(drop=True)
    snapshot_schedule = pd.DataFrame(schedule_rows).sort_values(["trade_date", "timestamp"]).reset_index(drop=True)
    symbol_universe_path = pack_root / "symbol_universe.parquet"
    snapshot_schedule_path = pack_root / "snapshot_schedule.parquet"
    layer_schema_path = pack_root / "layer_input_schema.json"

    _write_parquet_frame(symbol_universe_path, symbol_universe)
    _write_parquet_frame(snapshot_schedule_path, snapshot_schedule)
    layer_schema_path.write_text(json.dumps({"features": LAYER_INPUT_COLUMNS}, indent=2), encoding="utf-8")
    file_rows.extend(
        [
            _file_manifest_row(pack_root, symbol_universe_path, len(symbol_universe)),
            _file_manifest_row(pack_root, snapshot_schedule_path, len(snapshot_schedule)),
            _file_manifest_row(pack_root, layer_schema_path, None),
        ]
    )

    manifest = {
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "month": month,
        "pack_id": sha256(f"{month}|{len(trade_dates)}|{len(symbol_universe_rows)}".encode("utf-8")).hexdigest(),
        "code_commit": code_commit,
        "config_hash": sha256(json.dumps({"month": month, "trade_dates": trade_dates}).encode("utf-8")).hexdigest(),
        "symbol_universe_hash": _sha256_file(symbol_universe_path),
        "snapshot_schedule_hash": _sha256_file(snapshot_schedule_path),
        "trade_dates": trade_dates,
        "readiness_status": "ready" if not _pack_required_gaps(pack_root)[0] and not _pack_required_gaps(pack_root)[1] else "not_ready",
        "missing_required_files": _pack_required_gaps(pack_root)[0],
        "empty_required_files": _pack_required_gaps(pack_root)[1],
        "files": file_rows,
    }
    (pack_root / "pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pack_root


def _build_graph_features_from_frames(*, features_1m: pd.DataFrame, trade_flow_1m: pd.DataFrame) -> pd.DataFrame:
    if features_1m.empty:
        return pd.DataFrame(columns=LAYER_INPUT_COLUMNS)
    features = features_1m.copy()
    if not trade_flow_1m.empty:
        merged_flow = _normalize_trade_flow_symbols(trade_flow_1m)
        if "minute" in merged_flow.columns and "timestamp" not in merged_flow.columns:
            merged_flow = merged_flow.rename(columns={"minute": "timestamp"})
        features = features.merge(merged_flow, on=["timestamp", "symbol"], how="left", suffixes=("", "_flow"))
    for preferred, fallback in {
        "trade_count": "trade_count_flow",
        "dollar_volume": "dollar_volume_flow",
        "large_trade_dollar_volume": "large_trade_dollar_volume_flow",
        "large_trade_ratio_z": "large_trade_ratio_z_flow",
        "imbalance_z": "imbalance_z_flow",
    }.items():
        if preferred not in features.columns and fallback in features.columns:
            features[preferred] = features[fallback]
    if "available_time" not in features.columns:
        if "bar_end" in features.columns:
            features["available_time"] = pd.to_datetime(features["bar_end"], utc=True, errors="coerce")
        else:
            features["available_time"] = pd.to_datetime(features["timestamp"], utc=True, errors="coerce") + pd.Timedelta(minutes=1)
    for column in LAYER_INPUT_COLUMNS:
        if column not in features.columns:
            features[column] = pd.NA
    return features[LAYER_INPUT_COLUMNS].sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _resolve_month_trade_dates(*, data_root: Path, pack_root: Path, month: str) -> list[str]:
    legacy_trade_dates = {
        path.parent.name.split("=", 1)[1]
        for path in (data_root / "bars_5m").glob("date=*/bars_5m.parquet")
        if path.is_file()
    }
    pack_trade_dates = {
        date_root.name.split("=", 1)[1]
        for date_root in (pack_root / "dates").glob("date=*")
        if date_root.is_dir() and _date_root_has_payload(date_root)
    }
    return sorted(
        trade_date
        for trade_date in (legacy_trade_dates | pack_trade_dates)
        if trade_date.startswith(month)
    )


def _legacy_source_path(data_root: Path, dataset_name: str, trade_date: str) -> Path:
    return data_root / dataset_name / f"date={trade_date}" / LEGACY_DATASET_FILES[dataset_name]


def _stage_existing_dataset_file(
    *,
    source_path: Path,
    target_path: Path,
    storage_mode: str,
) -> pd.DataFrame:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        if storage_mode == "move":
            _move_file(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
    if target_path.exists():
        return _read_optional_parquet(target_path)
    return pd.DataFrame()


def _ensure_graph_features_file(
    *,
    target_path: Path,
    features_1m: pd.DataFrame,
    trade_flow_1m: pd.DataFrame,
) -> pd.DataFrame:
    if target_path.exists():
        return _read_optional_parquet(target_path)
    if features_1m.empty:
        return pd.DataFrame()
    frame = _build_graph_features_from_frames(features_1m=features_1m, trade_flow_1m=trade_flow_1m)
    _write_parquet_frame(target_path, frame)
    return frame


def _generate_features_1m(*, raw_1m: pd.DataFrame, trade_flow_1m: pd.DataFrame) -> pd.DataFrame:
    generated = MarketReadRepository._build_generated_features_1m(raw_1m=raw_1m, trade_flow_1m=trade_flow_1m)
    if generated.empty:
        return pd.DataFrame()
    return generated.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _empty_labels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "timestamp",
            "future_ret_1m",
            "future_ret_5m",
            "future_ret_15m",
            "future_ret_30m",
        ]
    )


def _rewrite_existing_parquet_with_symbol_ids(path: Path, frame: pd.DataFrame, symbol_ids: dict[str, int]) -> None:
    if not path.exists():
        return
    _write_parquet_frame(path, _add_symbol_ids(frame, symbol_ids))


def _relative_path_if_exists(pack_root: Path, path: Path) -> str | None:
    if not path.exists():
        return None
    return str(path.relative_to(pack_root)).replace("\\", "/")


def _required_file_gaps(date_root: Path) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    empty: list[str] = []
    for dataset_name in REQUIRED_READY_DATASETS:
        path = date_root / PACK_DATASET_FILES[dataset_name]
        if not path.exists():
            missing.append(path.name)
            continue
        if path.suffix == ".parquet" and len(_read_optional_parquet(path)) == 0:
            empty.append(path.name)
    return missing, empty


def _pack_required_gaps(pack_root: Path) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    empty: list[str] = []
    for date_root in sorted((pack_root / "dates").glob("date=*")):
        if not date_root.is_dir():
            continue
        date_missing, date_empty = _required_file_gaps(date_root)
        missing.extend([f"{date_root.name}/{value}" for value in date_missing])
        empty.extend([f"{date_root.name}/{value}" for value in date_empty])
    return missing, empty


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_parquet_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _move_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    shutil.move(str(source_path), str(target_path))


def _normalize_trade_flow_symbols(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    if "ticker" in normalized.columns and "symbol" not in normalized.columns:
        normalized["symbol"] = normalized["ticker"]
    return normalized


def _extract_symbols(frame: pd.DataFrame) -> set[str]:
    if "symbol" not in frame.columns and "ticker" not in frame.columns:
        return set()
    column_name = "symbol" if "symbol" in frame.columns else "ticker"
    return {
        str(value)
        for value in frame[column_name].dropna().tolist()
        if str(value).strip() and str(value).lower() != "nan"
    }


def _register_symbol_ids(symbol_ids: dict[str, int], discovered_symbols: set[str]) -> None:
    for symbol in sorted(discovered_symbols):
        if symbol not in symbol_ids:
            symbol_ids[symbol] = len(symbol_ids) + 1


def _add_symbol_ids(frame: pd.DataFrame, symbol_ids: dict[str, int]) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    symbol_column = "symbol" if "symbol" in normalized.columns else "ticker" if "ticker" in normalized.columns else None
    if symbol_column is None:
        return normalized
    normalized["symbol_id"] = normalized[symbol_column].astype(str).map(symbol_ids).astype("Int32")
    return normalized


def _date_root_has_payload(date_root: Path) -> bool:
    return any((date_root / filename).exists() for filename in PACK_DATASET_FILES.values())


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _file_manifest_row(pack_root: Path, path: Path, rows: int | None) -> dict[str, object]:
    return {
        "path": str(path.relative_to(pack_root)).replace("\\", "/"),
        "size": path.stat().st_size,
        "rows": rows,
        "sha256": _sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Build a distributed month pack under data/distributed_packs.")
    parser.add_argument("--data-root", default=str(project_paths.not_ready_root))
    parser.add_argument("--output-root", default=str(project_paths.ready_root))
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format.")
    parser.add_argument("--code-commit", default="migrated-without-git-history")
    parser.add_argument("--storage-mode", default="copy", choices=("copy", "move"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack_root = build_month_pack(
        data_root=args.data_root,
        output_root=args.output_root,
        month=args.month,
        code_commit=args.code_commit,
        storage_mode=args.storage_mode,
    )
    print(json.dumps({"status": "ok", "pack_root": str(pack_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
