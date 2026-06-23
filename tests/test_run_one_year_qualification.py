from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_one_year_qualification.py"
    spec = importlib.util.spec_from_file_location("run_one_year_qualification", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_year_window_covers_full_year():
    module = _load_module()

    window = module.build_year_window(2025)

    assert window.window_id == "2025"
    assert window.date_start == "2025-01-01"
    assert window.date_end == "2025-12-31"


def test_build_date_window_extends_to_year_end_by_default():
    module = _load_module()

    window = module.build_date_window("2025-01-07")

    assert window.window_id == "2025-01-07_to_2025-12-31"
    assert window.date_start == "2025-01-07"
    assert window.date_end == "2025-12-31"


def test_build_date_window_uses_explicit_end_date():
    module = _load_module()

    window = module.build_date_window("2025-01-07", "2025-03-15")

    assert window.window_id == "2025-01-07_to_2025-03-15"
    assert window.date_start == "2025-01-07"
    assert window.date_end == "2025-03-15"
