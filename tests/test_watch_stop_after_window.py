from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "watch_stop_after_window.py"
    )
    spec = importlib.util.spec_from_file_location(
        "watch_stop_after_window",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_should_stop_when_target_window_completed():
    module = _load_module()
    progress = {
        "current_window_id": "2025-02",
        "windows": [
            {"window_id": "2025-01", "status": "completed"},
            {"window_id": "2025-02", "status": "completed"},
        ],
    }

    assert module.should_stop_for_window(progress, "2025-02") is True


def test_should_not_stop_for_running_target_window():
    module = _load_module()
    progress = {
        "current_window_id": "2025-02",
        "windows": [
            {"window_id": "2025-02", "status": "evaluation_pack"},
        ],
    }

    assert module.should_stop_for_window(progress, "2025-02") is False


def test_mark_stopped_state_sets_requested_status():
    module = _load_module()
    progress = {
        "status": "running",
        "current_window_id": "2025-02",
        "current_stage": "checkpoint_publish",
        "windows": [
            {"window_id": "2025-01", "status": "completed"},
            {"window_id": "2025-02", "status": "completed"},
            {"window_id": "2025-03", "status": "pending"},
        ],
    }

    updated = module.mark_stopped_state(progress, "2025-02")

    assert updated["status"] == "stopped_after_window"
    assert updated["current_window_id"] is None
    assert updated["current_stage"] == "stopped"
    assert updated["stop_after_window_id"] == "2025-02"
    assert updated["windows"][2]["status"] == "pending"


def test_roundtrip_progress_file_update(tmp_path):
    module = _load_module()
    progress_path = tmp_path / "progress.json"
    payload = {
        "status": "running",
        "current_window_id": "2025-02",
        "current_stage": "checkpoint_publish",
        "windows": [{"window_id": "2025-02", "status": "completed"}],
    }
    progress_path.write_text(json.dumps(payload), encoding="utf-8")

    module.write_stopped_progress(progress_path, "2025-02")

    updated = json.loads(progress_path.read_text(encoding="utf-8"))
    assert updated["status"] == "stopped_after_window"
