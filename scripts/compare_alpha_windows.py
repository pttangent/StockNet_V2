from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.graph_evaluation_pack_service import (
    _export_cross_window_alpha_comparison_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare alpha ranking outputs across two evaluation windows.")
    parser.add_argument("--first-ranking", required=True, help="Path to the first window alpha_feature_ranking_by_layer.csv")
    parser.add_argument("--second-ranking", required=True, help="Path to the second window alpha_feature_ranking_by_layer.csv")
    parser.add_argument("--output", required=True, help="Target cross_window_alpha_comparison.csv path")
    parser.add_argument("--first-window-id", required=True, help="Identifier for the first evaluation window")
    parser.add_argument("--second-window-id", required=True, help="Identifier for the second evaluation window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _export_cross_window_alpha_comparison_report(
        Path(args.first_ranking).expanduser().resolve(),
        Path(args.second_ranking).expanduser().resolve(),
        output_path,
        first_window_id=args.first_window_id,
        second_window_id=args.second_window_id,
    )
    print(f"Cross-window comparison completed: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
