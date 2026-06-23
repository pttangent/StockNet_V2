from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_qualification_architectures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_qualification_architectures",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inner_benchmark_keeps_date_workers_fixed_for_short_windows():
    module = _load_benchmark_module()

    cases = module.plan_inner_benchmark_cases([1, 2, 3])

    assert cases
    assert {case["benchmark_days"] for case in cases} == {1, 2, 3}
    assert {case["candidate"].max_date_workers for case in cases} == {1}
    assert {"trade_date_shards", "snapshot_round_robin"} <= {
        case["candidate"].execution_mode for case in cases
    }
    first_candidate_ids = [
        case["candidate"].candidate_id
        for case in cases
        if case["benchmark_days"] == 1
    ]
    assert first_candidate_ids[:3] == [
        "shards_gpu_batch256_layer4",
        "shards_gpu_batch256_layer2",
        "shards_gpu_batch256_layer1",
    ]


def test_scale_benchmark_clamps_worker_counts_to_available_days():
    module = _load_benchmark_module()
    base_candidate = module.CandidateConfig(
        "gpu_inner_best",
        1,
        2,
        "torch_cuda",
        "cuda",
        "torch_cuda",
        "cuda",
        1024,
        "trade_date_shards",
    )

    candidates = module.plan_scale_candidates(base_candidate, scale_day_count=3)

    assert [candidate.max_date_workers for candidate in candidates] == [1, 2, 3]


def test_scale_benchmark_keeps_round_robin_mode_single_process():
    module = _load_benchmark_module()
    base_candidate = module.CandidateConfig(
        "gpu_round_robin_best",
        1,
        1,
        "torch_cuda",
        "cuda",
        "torch_cuda",
        "cuda",
        1024,
        "snapshot_round_robin",
    )

    candidates = module.plan_scale_candidates(base_candidate, scale_day_count=8)

    assert len(candidates) == 1
    assert candidates[0].max_date_workers == 1
    assert candidates[0].execution_mode == "snapshot_round_robin"
