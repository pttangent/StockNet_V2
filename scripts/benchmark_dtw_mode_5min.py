from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT_DIR / "scripts" / "run_month_graph_compute.py"
PACK_ROOT = ROOT_DIR / "data" / "ready"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "research_runs" / "benchmark_dtw_mode_5min"

COMMON_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

MODE_CONFIGS = {
    "cpu_only_dtw": {
        "profile": "cpu_only_dtw",
        "dtw_backend": "cpu_python",
        "dtw_torch_device": "cpu",
    },
    "gpu_only_dtw": {
        "profile": "gpu_only_dtw",
        "dtw_backend": "torch_cuda",
        "dtw_torch_device": "cuda",
    },
}


@dataclass(frozen=True)
class BenchmarkCase:
    mode_name: str
    profile: str
    dtw_backend: str
    dtw_torch_device: str
    run_name: str
    output_root: Path


@dataclass(frozen=True)
class BenchmarkResult:
    mode_name: str
    profile: str
    dtw_backend: str
    dtw_torch_device: str
    duration_seconds: float
    completed_snapshots: int
    failure_count: int
    snapshots_per_minute: float
    output_root: str
    started_at: str
    ended_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cpu_only_dtw vs gpu_only_dtw throughput over a fixed wall-clock window."
    )
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--date-start", default="2026-01-01")
    parser.add_argument("--date-end", default="2026-05-30")
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--snapshot-block-size", type=int, default=8)
    parser.add_argument("--max-tasks-per-child", type=int, default=4)
    parser.add_argument("--max-in-flight-tasks", type=int, default=22)
    parser.add_argument("--dtw-pair-batch-size", type=int, default=1024)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["cpu_only_dtw", "gpu_only_dtw"],
        choices=tuple(MODE_CONFIGS),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def build_case(mode_name: str, output_root: Path) -> BenchmarkCase:
    config = MODE_CONFIGS[mode_name]
    run_name = f"bench5m_{mode_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    return BenchmarkCase(
        mode_name=mode_name,
        profile=config["profile"],
        dtw_backend=config["dtw_backend"],
        dtw_torch_device=config["dtw_torch_device"],
        run_name=run_name,
        output_root=output_root / mode_name / run_name,
    )


def count_completed_snapshots(output_root: Path) -> int:
    if not output_root.exists():
        return 0
    return sum(1 for _ in output_root.rglob("_PROFILE_SUCCESS"))


def count_failures(output_root: Path) -> int:
    failures_path = output_root / "failures.jsonl"
    if not failures_path.exists():
        return 0
    return sum(1 for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_command(case: BenchmarkCase, args: argparse.Namespace) -> list[str]:
    # Use optimized default workers (2) and in-flight tasks (4) for GPU mode
    # if they haven't been explicitly changed from the default.
    max_workers = args.max_workers
    max_in_flight = args.max_in_flight_tasks
    if case.mode_name == "gpu_only_dtw":
        if max_workers == 18:
            max_workers = 2
        if max_in_flight == 22:
            max_in_flight = 4

    return [
        sys.executable,
        str(RUN_SCRIPT),
        "--pack-root",
        str(PACK_ROOT),
        "--output-root",
        str(case.output_root),
        "--run-name",
        case.run_name,
        "--profile",
        case.profile,
        "--date-start",
        args.date_start,
        "--date-end",
        args.date_end,
        "--max-workers",
        str(max_workers),
        "--snapshot-block-size",
        str(args.snapshot_block_size),
        "--max-tasks-per-child",
        str(args.max_tasks_per_child),
        "--max-in-flight-tasks",
        str(max_in_flight),
        "--resume-mode",
        "off",
        "--dtw-pair-batch-size",
        str(args.dtw_pair_batch_size),
        "--dtw-backend",
        case.dtw_backend,
        "--dtw-torch-device",
        case.dtw_torch_device,
    ]


def stop_process_tree(pid: int, run_name: str) -> None:
    ps_command = (
        f"$mainPid={pid};"
        f"$runName='{run_name}';"
        "if($mainPid){"
        "Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $mainPid } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };"
        "Stop-Process -Id $mainPid -Force -ErrorAction SilentlyContinue"
        "};"
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -eq 'python.exe' -and $_.CommandLine -like ('*' + $runName + '*') -and "
        "$_.CommandLine -like '*run_month_graph_compute.py*' "
        "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_command],
        check=False,
        capture_output=True,
        text=True,
    )


def run_case(case: BenchmarkCase, args: argparse.Namespace) -> BenchmarkResult:
    case.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(COMMON_ENV)
    command = build_command(case, args)
    started_at = datetime.now(UTC)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(
        f"[benchmark] start mode={case.mode_name} pid={process.pid} "
        f"duration={args.duration_seconds}s output={case.output_root}",
        flush=True,
    )
    try:
        time.sleep(max(1, args.duration_seconds))
    finally:
        if process.poll() is None:
            stop_process_tree(process.pid, case.run_name)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        stdout, stderr = process.communicate(timeout=10)
        (case.output_root / "benchmark.stdout.log").write_text(stdout or "", encoding="utf-8")
        (case.output_root / "benchmark.stderr.log").write_text(stderr or "", encoding="utf-8")

    ended_at = datetime.now(UTC)
    elapsed_seconds = (ended_at - started_at).total_seconds()
    completed = count_completed_snapshots(case.output_root)
    failures = count_failures(case.output_root)
    result = BenchmarkResult(
        mode_name=case.mode_name,
        profile=case.profile,
        dtw_backend=case.dtw_backend,
        dtw_torch_device=case.dtw_torch_device,
        duration_seconds=elapsed_seconds,
        completed_snapshots=completed,
        failure_count=failures,
        snapshots_per_minute=completed * 60.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        output_root=str(case.output_root),
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
    )
    print(
        f"[benchmark] done mode={case.mode_name} completed={completed} "
        f"failures={failures} rate={result.snapshots_per_minute:.2f}/min",
        flush=True,
    )
    return result


def write_summary(output_root: Path, results: list[BenchmarkResult], args: argparse.Namespace) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    payload = {
        "duration_seconds": args.duration_seconds,
        "date_start": args.date_start,
        "date_end": args.date_end,
        "max_workers": args.max_workers,
        "snapshot_block_size": args.snapshot_block_size,
        "max_tasks_per_child": args.max_tasks_per_child,
        "max_in_flight_tasks": args.max_in_flight_tasks,
        "dtw_pair_batch_size": args.dtw_pair_batch_size,
        "results": [asdict(item) for item in results],
    }
    if len(results) == 2:
        left, right = results
        faster = left.mode_name if left.completed_snapshots > right.completed_snapshots else right.mode_name
        if left.completed_snapshots == right.completed_snapshots:
            faster = "tie"
        payload["winner_by_completed_snapshots"] = faster
        if right.completed_snapshots > 0:
            payload["cpu_vs_gpu_ratio"] = round(left.completed_snapshots / right.completed_snapshots, 4)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    if not RUN_SCRIPT.exists():
        raise RuntimeError(f"Missing run script: {RUN_SCRIPT}")
    if not PACK_ROOT.exists():
        raise RuntimeError(f"Missing pack root: {PACK_ROOT}")

    output_root = Path(args.output_root).expanduser().resolve()
    results: list[BenchmarkResult] = []
    for mode_name in args.modes:
        case = build_case(mode_name, output_root)
        results.append(run_case(case, args))

    summary_path = write_summary(output_root, results, args)
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
