from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def validate_month_pack(pack_root: Path | str) -> dict[str, object]:
    pack_root = Path(pack_root).expanduser().resolve()
    missing: list[str] = []
    for relative_path in [
        "pack_manifest.json",
        "snapshot_schedule.parquet",
        "symbol_universe.parquet",
        "layer_input_schema.json",
    ]:
        if not (pack_root / relative_path).exists():
            missing.append(relative_path)

    manifest_path = pack_root / "pack_manifest.json"
    trade_dates: list[str] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trade_dates = list(manifest.get("trade_dates", []))
    for trade_date in trade_dates:
        for relative_path in [
            f"dates/date={trade_date}/bars_5m.parquet",
            f"dates/date={trade_date}/raw_1m.parquet",
            f"dates/date={trade_date}/trade_flow_1m.parquet",
            f"dates/date={trade_date}/features_1m.parquet",
            f"dates/date={trade_date}/labels_1m.parquet",
            f"dates/date={trade_date}/graph_features_1m.parquet",
            f"dates/date={trade_date}/date_manifest.json",
        ]:
            if not (pack_root / relative_path).exists():
                missing.append(relative_path)

    month_name = pack_root.name.split("=", 1)[1] if "=" in pack_root.name else pack_root.name
    return {
        "status": "ok" if not missing else "error",
        "month": month_name,
        "trade_dates": trade_dates,
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a distributed month pack.")
    parser.add_argument("--pack-root", required=True)
    return parser.parse_args()


def main() -> int:
    summary = validate_month_pack(parse_args().pack_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
