from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from stocknetv2.domain.graph.dtw_distance import dtw_similarity
from stocknetv2.domain.graph.dtw_torch import batched_dtw_similarity_torch


@dataclass(frozen=True)
class BenchmarkConfig:
    cpu_repeats: int = 3
    torch_cpu_repeats: int = 5
    torch_cuda_repeats: int = 20
    warmup_repeats: int = 2
    output_path: Path | None = None


def benchmark_dtw_backends(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    benchmark_config = config or BenchmarkConfig()
    left_cpu = left.detach().to(dtype=torch.float32, device="cpu").contiguous()
    right_cpu = right.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if left_cpu.shape != right_cpu.shape:
        raise ValueError("left and right must share the same shape")
    if left_cpu.ndim != 2:
        raise ValueError("benchmark expects 2D tensors shaped [pair_count, sequence_length]")

    reference_similarity = _cpu_python_similarity(left_cpu, right_cpu)
    results: dict[str, Any] = {}
    results["cpu_python"] = _time_backend(
        "cpu_python",
        repeats=max(1, benchmark_config.cpu_repeats),
        warmup=0,
        runner=lambda: _cpu_python_similarity(left_cpu, right_cpu),
        reference=reference_similarity,
    )
    results["torch_cpu_batched"] = _time_backend(
        "torch_cpu_batched",
        repeats=max(1, benchmark_config.torch_cpu_repeats),
        warmup=max(0, benchmark_config.warmup_repeats),
        runner=lambda: batched_dtw_similarity_torch(left_cpu, right_cpu, device="cpu").detach().cpu(),
        reference=reference_similarity,
    )
    if torch.cuda.is_available():
        left_cuda = left_cpu.to(device="cuda", non_blocking=False)
        right_cuda = right_cpu.to(device="cuda", non_blocking=False)
        results["torch_cuda_batched"] = _time_backend(
            "torch_cuda_batched",
            repeats=max(1, benchmark_config.torch_cuda_repeats),
            warmup=max(1, benchmark_config.warmup_repeats),
            runner=lambda: batched_dtw_similarity_torch(left_cuda, right_cuda, device="cuda").detach().cpu(),
            reference=reference_similarity,
            synchronize=torch.cuda.synchronize,
        )

    summary: dict[str, Any] = {
        "pair_count": int(left_cpu.shape[0]),
        "sequence_length": int(left_cpu.shape[1]),
        "cuda_available": bool(torch.cuda.is_available()),
        "results": results,
    }
    cuda_result = results.get("torch_cuda_batched")
    if cuda_result is not None:
        cpu_baseline_ms = float(results["cpu_python"]["best_elapsed_ms"])
        gpu_elapsed_ms = float(cuda_result["best_elapsed_ms"])
        summary["torch_cuda_speedup_vs_cpu_python"] = (
            cpu_baseline_ms / gpu_elapsed_ms if gpu_elapsed_ms > 0 else None
        )
    if benchmark_config.output_path is not None:
        benchmark_config.output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _cpu_python_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    values = [
        dtw_similarity(left_row.tolist(), right_row.tolist())
        for left_row, right_row in zip(left, right, strict=True)
    ]
    return torch.tensor(values, dtype=torch.float32)


def _time_backend(
    name: str,
    *,
    repeats: int,
    warmup: int,
    runner: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        runner()
        if synchronize is not None:
            synchronize()

    timings_ms: list[float] = []
    last_result: torch.Tensor | None = None
    for _ in range(max(1, repeats)):
        if synchronize is not None:
            synchronize()
        started_at = time.perf_counter()
        last_result = runner()
        if synchronize is not None:
            synchronize()
        timings_ms.append((time.perf_counter() - started_at) * 1000.0)

    assert last_result is not None
    max_abs_diff = float(torch.max(torch.abs(last_result - reference)).item()) if len(reference) else 0.0
    return {
        "name": name,
        "repeats": repeats,
        "best_elapsed_ms": min(timings_ms),
        "mean_elapsed_ms": sum(timings_ms) / len(timings_ms),
        "max_abs_diff_vs_cpu_python": max_abs_diff,
        "mean_similarity": float(last_result.mean().item()) if len(last_result) else 0.0,
    }
