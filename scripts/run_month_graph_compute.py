from __future__ import annotations

import argparse
import json
import sys
import traceback
import time
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stocknetv2.application.services.layer_profile_service import resolve_layer_profile
from stocknetv2.application.services.snapshot_block_graph_build_service import SnapshotBlockGraphBuildService
from stocknetv2.application.services.snapshot_resume_service import SnapshotResumeService
from stocknetv2.application.services.snapshot_task_planner import SnapshotTaskPlanner
from stocknetv2.domain.graph.layer_config import build_theme_discovery_settings
from stocknetv2.infrastructure.project_paths import ProjectPaths
from stocknetv2.infrastructure.repositories.month_pack_read_repository import MonthPackReadRepository, SnapshotSpec
from stocknetv2.infrastructure.repositories.snapshot_artifact_repository import SnapshotArtifactRepository

MODE_DEFAULTS = {
    "cpu_no_dtw": {
        "graph_backend": "cpu_numpy",
        "graph_torch_device": "cpu",
        "dtw_backend": "cpu_python",
        "dtw_torch_device": "cpu",
    },
    "cpu_full": {
        "graph_backend": "cpu_numpy",
        "graph_torch_device": "cpu",
        "dtw_backend": "cpu_python",
        "dtw_torch_device": "cpu",
    },
    "cpu_dtw_only": {
        "graph_backend": "cpu_numpy",
        "graph_torch_device": "cpu",
        "dtw_backend": "cpu_python",
        "dtw_torch_device": "cpu",
    },
    "cpu_only_dtw": {
        "graph_backend": "cpu_numpy",
        "graph_torch_device": "cpu",
        "dtw_backend": "cpu_python",
        "dtw_torch_device": "cpu",
    },
    "hybird_full": {
        "graph_backend": "cpu_numpy",
        "graph_torch_device": "cpu",
        "dtw_backend": "torch_cuda",
        "dtw_torch_device": "cuda",
    },
    "gpu_only_dtw": {
        "graph_backend": "cpu_numpy",
        "graph_torch_device": "cpu",
        "dtw_backend": "torch_cuda",
        "dtw_torch_device": "cuda",
    },
}


def parse_args() -> argparse.Namespace:
    project_paths = ProjectPaths.discover(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Compute month-pack graph snapshots with bounded snapshot windows.")
    parser.add_argument("--pack-root", help="Root directory containing many month=YYYY-MM packs.")
    parser.add_argument("--month", help="Month label like 2025-01.")
    parser.add_argument("--month-pack-root", help="Month pack root. Overrides --month.")
    parser.add_argument("--output-root", help="Run output root.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", default="cpu_no_dtw", choices=tuple(MODE_DEFAULTS))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--snapshot-block-size", type=int, default=8)
    parser.add_argument("--max-tasks-per-child", type=int, default=4)
    parser.add_argument("--max-in-flight-tasks", type=int, default=4)
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--snapshot-start", help="HH:MM or HHMM")
    parser.add_argument("--snapshot-end", help="HH:MM or HHMM")
    parser.add_argument("--snapshot-ids", nargs="*")
    parser.add_argument("--resume-mode", default="log", choices=("off", "log"))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--graph-backend", choices=("cpu_numpy", "torch_cpu", "torch_cuda", "torch_auto"))
    parser.add_argument("--graph-torch-device", default=None)
    parser.add_argument("--dtw-backend", dest="dtw_backend", choices=("cpu_python", "torch_cpu", "torch_cuda", "torch_auto"))
    parser.add_argument("--cpu-dtw-backend", dest="dtw_backend", choices=("cpu_python", "torch_cpu", "torch_cuda", "torch_auto"))
    parser.add_argument("--dtw-torch-device", default=None)
    parser.add_argument("--dtw-pair-batch-size", type=int, default=1024, help="Deprecated: use --torch-activation-pair-threshold instead")
    parser.add_argument("--torch-activation-pair-threshold", type=int, default=None)
    parser.add_argument("--torch-gpu-chunk-size", type=int, default=8192)
    parser.add_argument("--system-memory-reserve-gb", type=int, default=10)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.set_defaults(
        pack_root=str(project_paths.ready_root),
        month_pack_root=str(project_paths.ready_root / "month={month}"),
        output_root=str(project_paths.root / "research_runs" / "{run_name}"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads_per_worker != 1:
        raise RuntimeError("run_month_graph_compute only supports --threads-per-worker=1 for bounded-memory CPU runs.")
    output_root = Path(args.output_root.format(run_name=args.run_name)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    month_pack_roots = _resolve_month_pack_roots(args)
    if not month_pack_roots:
        raise RuntimeError("No month packs found for the requested inputs.")

    mode_defaults = MODE_DEFAULTS[args.profile]
    settings = build_theme_discovery_settings(
        graph_backend=args.graph_backend or str(mode_defaults["graph_backend"]),
        graph_torch_device=args.graph_torch_device or str(mode_defaults["graph_torch_device"]),
        dtw_backend=args.dtw_backend or str(mode_defaults["dtw_backend"]),
        dtw_torch_device=args.dtw_torch_device or str(mode_defaults["dtw_torch_device"]),
        dtw_torch_batch_pair_threshold=max(1, args.dtw_pair_batch_size),
        torch_activation_pair_threshold=args.torch_activation_pair_threshold,
        torch_gpu_chunk_size=args.torch_gpu_chunk_size,
    )
    profile = resolve_layer_profile(args.profile, settings)
    schedule = _filter_snapshots(
        _load_schedule_from_pack_roots(month_pack_roots),
        date_start=args.date_start,
        date_end=args.date_end,
        snapshot_start=args.snapshot_start,
        snapshot_end=args.snapshot_end,
        snapshot_ids=set(args.snapshot_ids or []),
    )
    if not schedule:
        raise RuntimeError("No snapshots matched the requested range.")

    run_log_path = output_root / "run.log"
    progress_path = output_root / "progress.jsonl"
    failures_path = output_root / "failures.jsonl"
    completed_ids = SnapshotResumeService().load_completed_snapshot_ids(run_log_path) if args.resume_mode == "log" else set()
    artifact_repository = SnapshotArtifactRepository()
    pending_schedule = [
        snapshot
        for snapshot in schedule
        if not (
            snapshot.snapshot_id in completed_ids
            and artifact_repository.snapshot_success_exists(_snapshot_root(output_root, snapshot))
        )
    ]
    if not pending_schedule:
        print(json.dumps({"status": "complete", "planned_snapshots": len(schedule), "executed_snapshots": 0}, ensure_ascii=False))
        return 0

    _write_run_config(
        output_root=output_root,
        payload={
            "run_name": args.run_name,
            "month_pack_roots": [str(path) for path in month_pack_roots],
            "profile": profile.name,
            "resume_mode": args.resume_mode,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "snapshot_block_size": args.snapshot_block_size,
            "max_workers": args.max_workers,
            "planned_snapshots": len(schedule),
            "pending_snapshots": len(pending_schedule),
            "trade_dates": _build_trade_date_plan(schedule),
        },
    )

    tasks = SnapshotTaskPlanner().plan_blocks(
        snapshots=pending_schedule,
        profile=profile,
        snapshot_block_size=max(1, args.snapshot_block_size),
    )
    failure_count = _run_tasks(
        tasks=tasks,
        output_root=output_root,
        run_name=args.run_name,
        profile_name=profile.name,
        dtw_backend=args.dtw_backend or str(mode_defaults["dtw_backend"]),
        graph_backend=args.graph_backend or str(mode_defaults["graph_backend"]),
        graph_torch_device=args.graph_torch_device or str(mode_defaults["graph_torch_device"]),
        dtw_torch_device=args.dtw_torch_device or str(mode_defaults["dtw_torch_device"]),
        dtw_pair_batch_size=args.dtw_pair_batch_size,
        torch_activation_pair_threshold=args.torch_activation_pair_threshold,
        torch_gpu_chunk_size=args.torch_gpu_chunk_size,
        max_workers=max(1, args.max_workers),
        max_tasks_per_child=max(1, args.max_tasks_per_child),
        max_in_flight_tasks=max(1, args.max_in_flight_tasks),
        continue_on_error=args.continue_on_error,
        run_log_path=run_log_path,
        progress_path=progress_path,
        failures_path=failures_path,
    )
    print(
        json.dumps(
            {
                "status": "complete" if failure_count == 0 else "complete_with_failures",
                "planned_snapshots": len(schedule),
                "executed_snapshots": len(pending_schedule),
                "failure_count": failure_count,
            },
            ensure_ascii=False,
        )
    )
    return 0 if failure_count == 0 else 1


def _resolve_month_pack_roots(args: argparse.Namespace) -> list[Path]:
    if args.pack_root and not args.month:
        pack_root = Path(args.pack_root.format(run_name=args.run_name)).expanduser().resolve()
        if not pack_root.exists():
            return []
        return sorted(
            child.resolve()
            for child in pack_root.iterdir()
            if child.is_dir() and child.name.startswith("month=")
        )
    if args.month_pack_root:
        month_value = args.month or ""
        return [Path(args.month_pack_root.format(month=month_value, run_name=args.run_name)).expanduser().resolve()]
    if args.month:
        return [Path(args.month_pack_root.format(month=args.month, run_name=args.run_name)).expanduser().resolve()]
    return []


def _load_schedule_from_pack_roots(month_pack_roots: list[Path]) -> list[SnapshotSpec]:
    combined: list[SnapshotSpec] = []
    for month_pack_root in month_pack_roots:
        combined.extend(MonthPackReadRepository(month_pack_root).load_snapshot_schedule())
    return sorted(combined, key=lambda item: (item.trade_date, item.snapshot_time, item.snapshot_id))


def _filter_snapshots(
    snapshots: list[SnapshotSpec],
    *,
    date_start: str | None,
    date_end: str | None,
    snapshot_start: str | None,
    snapshot_end: str | None,
    snapshot_ids: set[str],
) -> list[SnapshotSpec]:
    normalized_start = _normalize_clock(snapshot_start)
    normalized_end = _normalize_clock(snapshot_end)
    filtered: list[SnapshotSpec] = []
    for snapshot in snapshots:
        if date_start and snapshot.trade_date < date_start:
            continue
        if date_end and snapshot.trade_date > date_end:
            continue
        if normalized_start and snapshot.snapshot_clock < normalized_start:
            continue
        if normalized_end and snapshot.snapshot_clock > normalized_end:
            continue
        if snapshot_ids and snapshot.snapshot_id not in snapshot_ids:
            continue
        filtered.append(snapshot)
    return filtered


def _normalize_clock(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().replace(":", "")
    return text.zfill(4)


def _run_tasks(
    *,
    tasks,
    output_root: Path,
    run_name: str,
    profile_name: str,
    graph_backend: str,
    graph_torch_device: str,
    dtw_backend: str,
    dtw_torch_device: str,
    dtw_pair_batch_size: int,
    torch_activation_pair_threshold: int | None = None,
    torch_gpu_chunk_size: int = 8192,
    max_workers: int,
    max_tasks_per_child: int,
    max_in_flight_tasks: int,
    continue_on_error: bool,
    run_log_path: Path,
    progress_path: Path,
    failures_path: Path,
) -> int:
    if max_workers <= 1:
        failure_count = 0
        for task in tasks:
            try:
                results = _run_block_worker(
                    {
                        "month_pack_root": str(task.snapshots[0].month_pack_root),
                        "output_root": str(output_root),
                        "run_name": run_name,
                        "profile_name": profile_name,
                        "graph_backend": graph_backend,
                        "graph_torch_device": graph_torch_device,
                        "dtw_backend": dtw_backend,
                        "dtw_torch_device": dtw_torch_device,
                        "dtw_pair_batch_size": dtw_pair_batch_size,
                        "torch_activation_pair_threshold": torch_activation_pair_threshold,
                        "torch_gpu_chunk_size": torch_gpu_chunk_size,
                        "progress_path": str(progress_path),
                        "trade_date": task.trade_date,
                        "window_start": task.window_start.isoformat(),
                        "window_end": task.window_end.isoformat(),
                        "snapshots": [asdict(snapshot) for snapshot in task.snapshots],
                    }
                )
                _append_run_log(run_log_path, results)
            except Exception as exc:
                failure_count += 1
                _append_failure(failures_path, task.block_id, exc)
                if not continue_on_error:
                    raise
        return failure_count

    failure_count = 0
    task_iter = iter(tasks)
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=max_tasks_per_child) as executor:
        while True:
            while len(futures) < max_in_flight_tasks:
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                future = executor.submit(
                    _run_block_worker,
                    {
                        "month_pack_root": str(task.snapshots[0].month_pack_root),
                        "output_root": str(output_root),
                        "run_name": run_name,
                        "profile_name": profile_name,
                        "graph_backend": graph_backend,
                        "graph_torch_device": graph_torch_device,
                        "dtw_backend": dtw_backend,
                        "dtw_torch_device": dtw_torch_device,
                        "dtw_pair_batch_size": dtw_pair_batch_size,
                        "torch_activation_pair_threshold": torch_activation_pair_threshold,
                        "torch_gpu_chunk_size": torch_gpu_chunk_size,
                        "progress_path": str(progress_path),
                        "trade_date": task.trade_date,
                        "window_start": task.window_start.isoformat(),
                        "window_end": task.window_end.isoformat(),
                        "snapshots": [asdict(snapshot) for snapshot in task.snapshots],
                    },
                )
                futures[future] = task.block_id
            if not futures:
                break
            done, _pending = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                block_id = futures.pop(future)
                try:
                    _append_run_log(run_log_path, future.result())
                except Exception as exc:
                    failure_count += 1
                    _append_failure(failures_path, block_id, exc)
                    if not continue_on_error:
                        raise
    return failure_count


def _run_block_worker(payload: dict[str, object]) -> list[dict[str, object]]:
    repository = MonthPackReadRepository(str(payload["month_pack_root"]))
    settings = build_theme_discovery_settings(
        graph_backend=str(payload["graph_backend"]),
        graph_torch_device=str(payload["graph_torch_device"]),
        dtw_backend=str(payload["dtw_backend"]),
        dtw_torch_device=str(payload["dtw_torch_device"]),
        dtw_torch_batch_pair_threshold=max(1, int(payload["dtw_pair_batch_size"])),
        torch_activation_pair_threshold=payload.get("torch_activation_pair_threshold"),
        torch_gpu_chunk_size=payload.get("torch_gpu_chunk_size") or 8192,
    )
    profile = resolve_layer_profile(str(payload["profile_name"]), settings)
    snapshots = [
        SnapshotSpec(
            trade_date=str(item["trade_date"]),
            snapshot_time=pd.Timestamp(item["snapshot_time"]),
            snapshot_id=str(item["snapshot_id"]),
            snapshot_clock=str(item["snapshot_clock"]),
            month_pack_root=str(item["month_pack_root"]),
        )
        for item in payload["snapshots"]
    ]
    progress_path = Path(str(payload["progress_path"]))
    service = SnapshotBlockGraphBuildService(repository=repository)
    for snapshot in snapshots:
        _append_progress_event(
            progress_path,
            {
                "status": "snapshot_started",
                "run_name": payload["run_name"],
                "trade_date": snapshot.trade_date,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_clock": snapshot.snapshot_clock,
                "profile": payload["profile_name"],
                "worker_pid": None,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    results = service.run_block(
        trade_date=str(payload["trade_date"]),
        snapshots=snapshots,
        window_start=pd.Timestamp(str(payload["window_start"])),
        window_end=pd.Timestamp(str(payload["window_end"])),
        profile=profile,
        settings=settings,
        output_root=Path(str(payload["output_root"])),
        run_name=str(payload["run_name"]),
    )
    completed_rows = [
        {
            "status": "snapshot_complete",
            "run_name": payload["run_name"],
            "trade_date": result.trade_date,
            "snapshot_id": result.snapshot_id,
            "snapshot_clock": result.snapshot_clock,
            "profile": payload["profile_name"],
            "layer_count": result.layer_count,
            "edge_count": result.edge_count,
            "elapsed_seconds": result.elapsed_seconds,
            "worker_pid": result.worker_pid,
            "snapshot_root": str(result.snapshot_root),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        for result in results
    ]
    for row in completed_rows:
        _append_progress_event(progress_path, row)
    return completed_rows


def _append_run_log(run_log_path: Path, rows: list[dict[str, object]]) -> None:
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_progress_event(progress_path: Path, row: dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        try:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 + random.random() * 0.1)


def _append_failure(failures_path: Path, block_id: str, exc: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "block_failed",
        "block_id": block_id,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_run_config(*, output_root: Path, payload: dict[str, object]) -> None:
    (output_root / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_trade_date_plan(schedule: list[SnapshotSpec]) -> list[dict[str, object]]:
    snapshot_counts = Counter(snapshot.trade_date for snapshot in schedule)
    month_map = {snapshot.trade_date: snapshot.trade_date[:7] for snapshot in schedule}
    return [
        {
            "trade_date": trade_date,
            "month": month_map[trade_date],
            "planned_snapshots": int(snapshot_counts[trade_date]),
        }
        for trade_date in sorted(snapshot_counts)
    ]


def _snapshot_root(output_root: Path, snapshot: SnapshotSpec) -> Path:
    return (
        output_root
        / f"month={snapshot.trade_date[:7]}"
        / "dates"
        / f"date={snapshot.trade_date}"
        / "snapshots"
        / f"snapshot={snapshot.snapshot_clock}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
