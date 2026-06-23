from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop a qualification run once a target window is completed.")
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--target-window-id", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def read_progress(progress_path: Path) -> dict[str, object]:
    return json.loads(progress_path.read_text(encoding="utf-8"))


def should_stop_for_window(progress: dict[str, object], target_window_id: str) -> bool:
    for window in progress.get("windows", []):
        if window.get("window_id") != target_window_id:
            continue
        status = str(window.get("status", ""))
        return status.startswith("completed")
    return False


def mark_stopped_state(progress: dict[str, object], target_window_id: str) -> dict[str, object]:
    updated = dict(progress)
    updated["status"] = "stopped_after_window"
    updated["current_window_id"] = None
    updated["current_trade_date"] = None
    updated["current_snapshot_id"] = None
    updated["current_snapshot_clock_code"] = None
    updated["current_stage"] = "stopped"
    updated["stop_after_window_id"] = target_window_id
    updated["updated_at"] = datetime.now(UTC).isoformat()
    return updated


def write_stopped_progress(progress_path: Path, target_window_id: str) -> None:
    progress = read_progress(progress_path)
    progress_path.write_text(
        json.dumps(mark_stopped_state(progress, target_window_id), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now(UTC).isoformat()}] {message}\n")


def stop_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = parse_args()
    progress_path = Path(args.progress_file).expanduser().resolve()
    log_path = Path(args.log_file).expanduser().resolve()

    append_log(log_path, f"stop watcher armed for window {args.target_window_id} on pid {args.pid}")
    while True:
        if not progress_path.exists():
            time.sleep(args.poll_seconds)
            continue
        progress = read_progress(progress_path)
        if should_stop_for_window(progress, args.target_window_id):
            append_log(log_path, f"target window {args.target_window_id} completed; stopping process tree pid={args.pid}")
            stop_process_tree(args.pid)
            write_stopped_progress(progress_path, args.target_window_id)
            append_log(log_path, f"qualification run stopped after window {args.target_window_id}")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
