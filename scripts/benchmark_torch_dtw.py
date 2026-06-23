from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.domain.graph.dtw_benchmark import BenchmarkConfig, benchmark_dtw_backends
from stocknetv2.domain.graph.dtw_window import compute_effective_dtw_window
from stocknetv2.domain.graph.series_utils import build_pivot_matrix, zscore_frame_columns
from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.infrastructure.project_paths import ProjectPaths
from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Benchmark the torch/cuda batched DTW prototype on a small real-data window.")
    parser.add_argument(
        "--data-root",
        default=str(project_paths.not_ready_root),
        help="Legacy source root. Default: USStock_Proj/data/not_ready",
    )
    parser.add_argument("--trade-date", default="2025-01-02")
    parser.add_argument(
        "--snapshot-time",
        help="Optional UTC snapshot timestamp. Default: trade-date session open + 45 minutes.",
    )
    parser.add_argument(
        "--value-columns",
        default="ret_1m,flow_impulse_score",
        help="Comma-separated feature columns to benchmark.",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=128,
        help="Maximum complete-data symbols to consider before building pair combinations.",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=2048,
        help="Maximum symbol pairs to benchmark per value column.",
    )
    parser.add_argument("--cpu-repeats", type=int, default=1)
    parser.add_argument("--torch-cpu-repeats", type=int, default=3)
    parser.add_argument("--torch-cuda-repeats", type=int, default=20)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "var" / "benchmarks" / "torch_dtw"),
        help="Output directory for benchmark JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repository = MarketReadRepository(LegacySourceLayout(data_root=data_root))
    inputs = repository.load_trade_date_inputs(args.trade_date)
    snapshot_time = (
        pd.Timestamp(args.snapshot_time, tz="UTC")
        if args.snapshot_time
        else SnapshotClock().session_open_timestamp(args.trade_date) + pd.Timedelta(minutes=45)
    )
    session_open = SnapshotClock().session_open_timestamp(args.trade_date)
    window_info = compute_effective_dtw_window(snapshot_time=snapshot_time, session_open=session_open)
    if not window_info["enabled"]:
        raise RuntimeError(f"DTW window not enabled at snapshot {snapshot_time.isoformat()}")

    print(
        json.dumps(
            {
                "trade_date": args.trade_date,
                "snapshot_time": snapshot_time.isoformat(),
                "effective_lookback_minutes": int(window_info["effective_lookback_minutes"]),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            ensure_ascii=False,
        )
    )

    value_columns = [value.strip() for value in args.value_columns.split(",") if value.strip()]
    summaries: list[dict[str, object]] = []
    for value_column in value_columns:
        left_batch, right_batch, metadata = _build_real_batch(
            inputs.features_1m,
            value_column=value_column,
            snapshot_time=snapshot_time,
            lookback_minutes=int(window_info["effective_lookback_minutes"]),
            symbol_limit=max(2, args.symbol_limit),
            pair_limit=max(1, args.pair_limit),
        )
        output_path = output_dir / f"{args.trade_date}_{value_column}_dtw_benchmark.json"
        summary = benchmark_dtw_backends(
            left_batch,
            right_batch,
            config=BenchmarkConfig(
                cpu_repeats=max(1, args.cpu_repeats),
                torch_cpu_repeats=max(1, args.torch_cpu_repeats),
                torch_cuda_repeats=max(1, args.torch_cuda_repeats),
                warmup_repeats=max(0, args.warmup_repeats),
                output_path=output_path,
            ),
        )
        summary["value_column"] = value_column
        summary["metadata"] = metadata
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "value_column": value_column,
                    "pair_count": summary["pair_count"],
                    "sequence_length": summary["sequence_length"],
                    "cpu_python_best_ms": round(float(summary["results"]["cpu_python"]["best_elapsed_ms"]), 3),
                    "torch_cpu_best_ms": round(float(summary["results"]["torch_cpu_batched"]["best_elapsed_ms"]), 3),
                    "torch_cuda_best_ms": round(float(summary["results"]["torch_cuda_batched"]["best_elapsed_ms"]), 3)
                    if "torch_cuda_batched" in summary["results"]
                    else None,
                    "torch_cuda_speedup_vs_cpu_python": round(float(summary["torch_cuda_speedup_vs_cpu_python"]), 3)
                    if summary.get("torch_cuda_speedup_vs_cpu_python") is not None
                    else None,
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
            )
        )
    aggregate_path = output_dir / f"{args.trade_date}_aggregate_summary.json"
    aggregate_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate_output_path": str(aggregate_path)}, ensure_ascii=False))
    return 0


def _build_real_batch(
    features_1m: pd.DataFrame,
    *,
    value_column: str,
    snapshot_time: pd.Timestamp,
    lookback_minutes: int,
    symbol_limit: int,
    pair_limit: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    matrix = build_pivot_matrix(
        features_1m,
        value_column=value_column,
        snapshot_time=snapshot_time,
        minutes=lookback_minutes,
    )
    if matrix.empty:
        raise RuntimeError(f"No data available for value column {value_column}")

    complete_columns = [
        column
        for column in matrix.columns.tolist()
        if int(matrix[column].notna().sum()) == len(matrix.index)
    ]
    if len(complete_columns) < 2:
        raise RuntimeError(f"Not enough complete-data symbols to benchmark {value_column}")

    chosen_columns = complete_columns[:symbol_limit]
    normalized = zscore_frame_columns(matrix.loc[:, chosen_columns])
    values = normalized.to_numpy(dtype=np.float32)

    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    sampled_symbols: list[tuple[str, str]] = []
    for left_index in range(len(chosen_columns)):
        for right_index in range(left_index + 1, len(chosen_columns)):
            left_rows.append(values[:, left_index].copy())
            right_rows.append(values[:, right_index].copy())
            sampled_symbols.append((str(chosen_columns[left_index]), str(chosen_columns[right_index])))
            if len(left_rows) >= pair_limit:
                break
        if len(left_rows) >= pair_limit:
            break

    if not left_rows:
        raise RuntimeError(f"Could not build any complete pairs for {value_column}")

    left_batch = torch.from_numpy(np.stack(left_rows, axis=0))
    right_batch = torch.from_numpy(np.stack(right_rows, axis=0))
    metadata = {
        "complete_symbol_count": len(complete_columns),
        "chosen_symbol_count": len(chosen_columns),
        "sampled_pair_count": len(sampled_symbols),
        "first_pairs": sampled_symbols[:5],
    }
    return left_batch, right_batch, metadata


if __name__ == "__main__":
    raise SystemExit(main())
