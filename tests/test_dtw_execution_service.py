from __future__ import annotations

import pytest

from stocknetv2.infrastructure.dtw.dtw_execution_service import (
    InProcessDtwExecutionService,
    SharedGpuDtwExecutionService,
)


def test_in_process_dtw_execution_service():
    service = InProcessDtwExecutionService()
    scores, backend = service.compute(
        [[1.0, 2.0, 3.0]],
        [[1.0, 2.1, 3.0]],
        backend="torch_cpu",
        torch_batch_pair_threshold=1,
    )
    assert len(scores) == 1
    assert backend == "torch_cpu"
    assert scores[0] > 0.0


def test_shared_gpu_dtw_execution_service_runs_non_cuda_directly():
    service = SharedGpuDtwExecutionService()
    scores, backend = service.compute(
        [[1.0, 2.0, 3.0]],
        [[1.0, 2.1, 3.0]],
        backend="torch_cpu",
        torch_batch_pair_threshold=1,
    )
    assert len(scores) == 1
    assert backend == "torch_cpu"
    assert scores[0] > 0.0
