from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.build_month_pack import ingest_trade_date_into_pack
from stocknetv2.infrastructure.project_paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Ingest one new trade date directly into a distributed month pack.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--bars-5m-path", required=True)
    parser.add_argument("--raw-1m-path", required=True)
    parser.add_argument("--trade-flow-1m-path", required=True)
    parser.add_argument("--features-1m-path")
    parser.add_argument("--labels-1m-path", required=True)
    parser.add_argument("--output-root", default=str(project_paths.distributed_packs_root))
    parser.add_argument("--code-commit", default="ingested-directly-to-pack")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack_root = ingest_trade_date_into_pack(
        trade_date=args.trade_date,
        bars_5m_path=args.bars_5m_path,
        raw_1m_path=args.raw_1m_path,
        trade_flow_1m_path=args.trade_flow_1m_path,
        features_1m_path=args.features_1m_path,
        labels_1m_path=args.labels_1m_path,
        output_root=args.output_root,
        code_commit=args.code_commit,
    )
    print(json.dumps({"status": "ok", "pack_root": str(pack_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
