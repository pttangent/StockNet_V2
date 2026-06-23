from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.infrastructure.project_paths import ProjectPaths
from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository


def materialize_features_1m(
    *,
    data_root: Path | str,
    date_start: str | None = None,
    date_end: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    data_root = Path(data_root).expanduser().resolve()
    features_root = data_root / "features_1m"
    features_root.mkdir(parents=True, exist_ok=True)
    repository = MarketReadRepository(LegacySourceLayout(data_root=data_root))
    available_dates = repository.list_available_trade_dates("bars_5m")
    selected_dates = [
        trade_date
        for trade_date in available_dates
        if (date_start is None or trade_date >= date_start) and (date_end is None or trade_date <= date_end)
    ]
    written_dates: list[str] = []
    skipped_dates: list[str] = []

    for trade_date in selected_dates:
        output_path = features_root / f"date={trade_date}" / "features_1m.parquet"
        if output_path.exists() and not overwrite:
            skipped_dates.append(trade_date)
            continue
        inputs = repository.load_trade_date_inputs(trade_date)
        if inputs.features_1m.empty:
            skipped_dates.append(trade_date)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        inputs.features_1m.to_parquet(output_path, index=False)
        written_dates.append(trade_date)

    return {
        "selected_dates": selected_dates,
        "written_dates": written_dates,
        "skipped_dates": skipped_dates,
    }


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Materialize features_1m parquet partitions into data/features_1m.")
    parser.add_argument("--data-root", default=str(project_paths.not_ready_root))
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = materialize_features_1m(
        data_root=args.data_root,
        date_start=args.date_start,
        date_end=args.date_end,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
