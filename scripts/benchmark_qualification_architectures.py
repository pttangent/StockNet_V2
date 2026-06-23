from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.graph_build_range_service import GraphBuildRangeConfig, GraphBuildRangeService
from stocknetv2.infrastructure.repositories.market_read_repository import LegacySourceLayout, MarketReadRepository


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    max_date_workers: int
    layer_workers_per_process: int
    graph_backend: str
    graph_torch_device: str
    dtw_backend: str
    dtw_torch_device: str
    dtw_torch_batch_pair_threshold: int
    execution_mode: str


INNER_THROUGHPUT_CANDIDATES = [
    CandidateConfig("shards_gpu_batch256_layer4", 1, 4, "torch_cuda", "cuda", "torch_cuda", "cuda", 256, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch256_layer2", 1, 2, "torch_cuda", "cuda", "torch_cuda", "cuda", 256, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch256_layer1", 1, 1, "torch_cuda", "cuda", "torch_cuda", "cuda", 256, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch512_layer4", 1, 4, "torch_cuda", "cuda", "torch_cuda", "cuda", 512, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch512_layer2", 1, 2, "torch_cuda", "cuda", "torch_cuda", "cuda", 512, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch512_layer1", 1, 1, "torch_cuda", "cuda", "torch_cuda", "cuda", 512, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch1024_layer4", 1, 4, "torch_cuda", "cuda", "torch_cuda", "cuda", 1024, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch1024_layer2", 1, 2, "torch_cuda", "cuda", "torch_cuda", "cuda", 1024, "trade_date_shards"),
    CandidateConfig("shards_gpu_batch1024_layer1", 1, 1, "torch_cuda", "cuda", "torch_cuda", "cuda", 1024, "trade_date_shards"),
    CandidateConfig("roundrobin_gpu_batch256_layer1", 1, 1, "torch_cuda", "cuda", "torch_cuda", "cuda", 256, "snapshot_round_robin"),
    CandidateConfig("roundrobin_gpu_batch512_layer1", 1, 1, "torch_cuda", "cuda", "torch_cuda", "cuda", 512, "snapshot_round_robin"),
    CandidateConfig("roundrobin_gpu_batch1024_layer1", 1, 1, "torch_cuda", "cuda", "torch_cuda", "cuda", 1024, "snapshot_round_robin"),
]
SCALE_WORKER_COUNTS = (1, 2, 4, 8, 12, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark StockNetV2 graph-build throughput. "
            "Short 1-3 day windows measure single-day inner throughput only; "
            "a separate wider window measures date-level scaling."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default=str(ROOT_DIR / "research_runs" / "architecture_benchmarks"))
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--benchmark-days", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--scale-days", type=int, default=8)
    return parser.parse_args()


def plan_inner_benchmark_cases(benchmark_days: list[int]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for day_count in benchmark_days:
        for candidate in INNER_THROUGHPUT_CANDIDATES:
            cases.append(
                {
                    "benchmark_type": "inner_throughput",
                    "benchmark_days": day_count,
                    "candidate": candidate,
                }
            )
    return cases


def plan_scale_candidates(base_candidate: CandidateConfig, scale_day_count: int) -> list[CandidateConfig]:
    if scale_day_count <= 0:
        return []
    if base_candidate.execution_mode == "snapshot_round_robin":
        return [
            CandidateConfig(
                candidate_id=f"{base_candidate.candidate_id}_singlectx",
                max_date_workers=1,
                layer_workers_per_process=base_candidate.layer_workers_per_process,
                graph_backend=base_candidate.graph_backend,
                graph_torch_device=base_candidate.graph_torch_device,
                dtw_backend=base_candidate.dtw_backend,
                dtw_torch_device=base_candidate.dtw_torch_device,
                dtw_torch_batch_pair_threshold=base_candidate.dtw_torch_batch_pair_threshold,
                execution_mode=base_candidate.execution_mode,
            )
        ]

    candidates: list[CandidateConfig] = []
    seen_workers: set[int] = set()
    for requested_workers in SCALE_WORKER_COUNTS:
        actual_workers = min(requested_workers, scale_day_count)
        if actual_workers in seen_workers:
            continue
        seen_workers.add(actual_workers)
        candidates.append(
            CandidateConfig(
                candidate_id=f"{base_candidate.candidate_id}_datew{actual_workers}",
                max_date_workers=actual_workers,
                layer_workers_per_process=base_candidate.layer_workers_per_process,
                graph_backend=base_candidate.graph_backend,
                graph_torch_device=base_candidate.graph_torch_device,
                dtw_backend=base_candidate.dtw_backend,
                dtw_torch_device=base_candidate.dtw_torch_device,
                dtw_torch_batch_pair_threshold=base_candidate.dtw_torch_batch_pair_threshold,
                execution_mode=base_candidate.execution_mode,
            )
        )
    return candidates


def _run_candidate(
    *,
    market_calendar: MarketReadRepository,
    data_root: Path,
    output_root: Path,
    candidate: CandidateConfig,
    benchmark_type: str,
    day_count: int,
    trade_dates: list[str],
    config_version: str,
    code_commit: str,
) -> dict[str, object]:
    date_start = trade_dates[0]
    date_end = trade_dates[-1]
    print(
        f"[benchmark] start type={benchmark_type} candidate={candidate.candidate_id} "
        f"days={day_count} range={date_start}..{date_end} "
        f"date_workers={candidate.max_date_workers} layer_workers={candidate.layer_workers_per_process} "
        f"execution_mode={candidate.execution_mode} graph_backend={candidate.graph_backend} "
        f"dtw_batch_threshold={candidate.dtw_torch_batch_pair_threshold}",
        flush=True,
    )
    candidate_output = output_root / f"{benchmark_type}_{candidate.candidate_id}_{day_count}d"
    if candidate_output.exists():
        shutil.rmtree(candidate_output)
    summary = GraphBuildRangeService(
        market_calendar=market_calendar,
        max_workers=candidate.max_date_workers,
    ).run(
        GraphBuildRangeConfig(
            data_root=data_root,
            output_database_path=candidate_output / "graph.duckdb",
            date_start=date_start,
            date_end=date_end,
            run_prefix=f"bench-{candidate.candidate_id}",
            config_id="architecture-benchmark",
            config_name="Architecture benchmark",
            config_version=config_version,
            code_commit=code_commit,
            keep_shards=False,
            continue_on_error=False,
            layer_workers_per_process=candidate.layer_workers_per_process,
            graph_backend=candidate.graph_backend,
            graph_torch_device=candidate.graph_torch_device,
            dtw_backend=candidate.dtw_backend,
            dtw_torch_device=candidate.dtw_torch_device,
            dtw_torch_batch_pair_threshold=candidate.dtw_torch_batch_pair_threshold,
            execution_mode=candidate.execution_mode,
        )
    )
    row = {
        "benchmark_type": benchmark_type,
        "benchmark_days": day_count,
        "date_start": date_start,
        "date_end": date_end,
        **asdict(candidate),
        "processed_trade_dates": len(summary.processed_dates),
        "failure_count": summary.failure_count,
        "elapsed_seconds": summary.elapsed_seconds,
        "seconds_per_trade_date": round(summary.elapsed_seconds / max(1, len(summary.processed_dates)), 4),
    }
    print(
        f"[benchmark] done type={benchmark_type} candidate={candidate.candidate_id} "
        f"days={day_count} elapsed_seconds={summary.elapsed_seconds}",
        flush=True,
    )
    return row


def _write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).sort_values(
        ["benchmark_type", "benchmark_days", "elapsed_seconds", "candidate_id"]
    ).to_csv(output_path, index=False)


def _select_best_inner_candidate(rows: list[dict[str, object]]) -> CandidateConfig:
    inner_rows = [row for row in rows if row["benchmark_type"] == "inner_throughput"]
    if not inner_rows:
        raise RuntimeError("No inner-throughput rows available to choose a scaling baseline.")

    frame = pd.DataFrame(inner_rows)
    grouped = (
        frame.groupby("candidate_id", as_index=False)
        .agg(
            mean_seconds_per_trade_date=("seconds_per_trade_date", "mean"),
            max_benchmark_days=("benchmark_days", "max"),
        max_date_workers=("max_date_workers", "first"),
        layer_workers_per_process=("layer_workers_per_process", "first"),
        graph_backend=("graph_backend", "first"),
        graph_torch_device=("graph_torch_device", "first"),
        dtw_backend=("dtw_backend", "first"),
        dtw_torch_device=("dtw_torch_device", "first"),
        dtw_torch_batch_pair_threshold=("dtw_torch_batch_pair_threshold", "first"),
        execution_mode=("execution_mode", "first"),
        )
        .sort_values(
            ["max_benchmark_days", "mean_seconds_per_trade_date", "candidate_id"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    best = grouped.iloc[0].to_dict()
    return CandidateConfig(
        candidate_id=str(best["candidate_id"]),
        max_date_workers=int(best["max_date_workers"]),
        layer_workers_per_process=int(best["layer_workers_per_process"]),
        graph_backend=str(best["graph_backend"]),
        graph_torch_device=str(best["graph_torch_device"]),
        dtw_backend=str(best["dtw_backend"]),
        dtw_torch_device=str(best["dtw_torch_device"]),
        dtw_torch_batch_pair_threshold=int(best["dtw_torch_batch_pair_threshold"]),
        execution_mode=str(best["execution_mode"]),
    )


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    market_calendar = MarketReadRepository(LegacySourceLayout(data_root=data_root))
    available_dates = [
        trade_date
        for trade_date in market_calendar.list_available_trade_dates("bars_5m")
        if args.date_start <= trade_date <= args.date_end
    ]
    if not available_dates:
        raise RuntimeError("No trade dates available in the requested benchmark range.")

    rows: list[dict[str, object]] = []
    inner_summary_path = output_root / "inner_benchmark_summary.csv"
    scale_summary_path = output_root / "scale_benchmark_summary.csv"
    summary_path = output_root / "architecture_benchmark_summary.csv"

    for case in plan_inner_benchmark_cases(args.benchmark_days):
        day_count = int(case["benchmark_days"])
        trade_dates = available_dates[:day_count]
        if len(trade_dates) < day_count:
            continue
        rows.append(
            _run_candidate(
                market_calendar=market_calendar,
                data_root=data_root,
                output_root=output_root,
                candidate=case["candidate"],
                benchmark_type=str(case["benchmark_type"]),
                day_count=day_count,
                trade_dates=trade_dates,
                config_version=args.config_version,
                code_commit=args.code_commit,
            )
        )
        _write_summary(rows, summary_path)
        _write_summary(
            [row for row in rows if row["benchmark_type"] == "inner_throughput"],
            inner_summary_path,
        )

    best_inner_candidate = _select_best_inner_candidate(rows)
    scale_day_count = min(max(1, args.scale_days), len(available_dates))
    scale_trade_dates = available_dates[:scale_day_count]
    scale_rows: list[dict[str, object]] = []
    for candidate in plan_scale_candidates(best_inner_candidate, scale_day_count):
        row = _run_candidate(
            market_calendar=market_calendar,
            data_root=data_root,
            output_root=output_root,
            candidate=candidate,
            benchmark_type="date_scaling",
            day_count=scale_day_count,
            trade_dates=scale_trade_dates,
            config_version=args.config_version,
            code_commit=args.code_commit,
        )
        rows.append(row)
        scale_rows.append(row)
        _write_summary(rows, summary_path)
        _write_summary(scale_rows, scale_summary_path)

    summary_frame = pd.DataFrame(rows).sort_values(
        ["benchmark_type", "benchmark_days", "elapsed_seconds", "candidate_id"]
    ).reset_index(drop=True)
    summary_frame.to_csv(summary_path, index=False)
    best_scale_row = (
        pd.DataFrame(scale_rows)
        .sort_values(["elapsed_seconds", "candidate_id"])
        .iloc[0]
        .to_dict()
        if scale_rows
        else None
    )
    best_path = output_root / "best_candidate.json"
    best_payload = {
        "best_inner_candidate": asdict(best_inner_candidate),
        "best_scale_candidate": best_scale_row,
        "notes": {
            "inner_throughput": "Short 1-3 day windows keep max_date_workers fixed at 1 to measure single-day throughput.",
            "date_scaling": f"Date-level scaling is measured separately on the first {scale_day_count} available trade dates.",
        },
    }
    best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Inner throughput summary: {inner_summary_path}")
    print(f"Date scaling summary: {scale_summary_path}")
    print(f"Combined summary: {summary_path}")
    print(f"Best candidate summary: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
