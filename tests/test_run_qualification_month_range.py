from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_qualification_month_range.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_qualification_month_range",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_monthly_windows_for_resume_range():
    module = _load_module()

    windows = module.build_monthly_windows(2025, 3, 5)

    assert [window.window_id for window in windows] == ["2025-03", "2025-04", "2025-05"]
    assert windows[0].date_start == "2025-03-01"
    assert windows[-1].date_end == "2025-05-31"
