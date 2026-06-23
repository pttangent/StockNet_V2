from __future__ import annotations

import json
from pathlib import Path


class SnapshotResumeService:
    def load_completed_snapshot_ids(self, run_log_path: Path) -> set[str]:
        if not run_log_path.exists():
            return set()
        completed: set[str] = set()
        for line in run_log_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "snapshot_complete" and payload.get("snapshot_id"):
                completed.add(str(payload["snapshot_id"]))
        return completed
