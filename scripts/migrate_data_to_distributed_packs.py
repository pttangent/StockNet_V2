from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.build_month_pack import build_month_pack
from stocknetv2.infrastructure.project_paths import ProjectPaths


def migrate_data_to_distributed_packs(*, data_root: Path | str, output_root: Path | str) -> dict[str, object]:
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    months = sorted(
        {
            directory.name.split("=", 1)[1][:7]
            for directory in (data_root / "bars_5m").glob("date=*")
            if directory.is_dir()
        }
    )
    migrated_months: list[str] = []
    for month in months:
        build_month_pack(
            data_root=data_root,
            output_root=output_root,
            month=month,
            storage_mode="move",
            code_commit="migrated-legacy-data-into-distributed-packs",
        )
        migrated_months.append(month)
    removed_paths = cleanup_empty_legacy_data_roots(data_root)
    return {"months": migrated_months, "removed_paths": removed_paths}


def cleanup_empty_legacy_data_roots(data_root: Path | str) -> list[str]:
    data_root = Path(data_root).expanduser().resolve()
    removed_paths: list[str] = []
    for root_name in ("bars_5m", "raw_1m", "trade_flow_1m", "features_1m", "labels_1m"):
        root_path = data_root / root_name
        if not root_path.exists():
            continue
        for child in sorted(root_path.glob("date=*")):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
                removed_paths.append(str(child))
        if root_path.is_dir() and not any(root_path.iterdir()):
            root_path.rmdir()
            removed_paths.append(str(root_path))
    return removed_paths


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Move legacy data roots into distributed month packs.")
    parser.add_argument("--data-root", default=str(project_paths.not_ready_root))
    parser.add_argument("--output-root", default=str(project_paths.ready_root))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = migrate_data_to_distributed_packs(data_root=args.data_root, output_root=args.output_root)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
