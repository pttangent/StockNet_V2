from __future__ import annotations

import json
from pathlib import Path

import torch

from stocknetv2.domain.graph.dtw_benchmark import (
    BenchmarkConfig,
    benchmark_dtw_backends,
)


def test_benchmark_dtw_backends_reports_cpu_and_torch_results(tmp_path):
    left = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, -0.1, -0.2],
        ],
        dtype=torch.float32,
    )
    right = torch.tensor(
        [
            [0.0, 0.11, 0.19, 0.29],
            [0.1, 0.02, -0.12, -0.18],
        ],
        dtype=torch.float32,
    )
    output_path = tmp_path / "dtw_benchmark.json"

    summary = benchmark_dtw_backends(
        left,
        right,
        config=BenchmarkConfig(
            cpu_repeats=1,
            torch_cpu_repeats=1,
            torch_cuda_repeats=1,
            warmup_repeats=0,
            output_path=output_path,
        ),
    )

    assert "cpu_python" in summary["results"]
    assert "torch_cpu_batched" in summary["results"]
    assert output_path.exists()

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["pair_count"] == 2
    assert written["sequence_length"] == 4
    assert "results" in written
