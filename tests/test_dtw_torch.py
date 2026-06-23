from __future__ import annotations

import torch

from stocknetv2.domain.graph.dtw_distance import (
    dtw_distance_with_path_length,
    dtw_similarity,
)
from stocknetv2.domain.graph.dtw_torch import (
    batched_dtw_distance_with_path_length_torch,
    batched_dtw_similarity_torch,
)


def test_batched_dtw_distance_matches_reference_cpu():
    left = torch.tensor(
        [
            [0.0, 1.0, 2.0, 1.5],
            [0.2, 0.4, 0.1, -0.1],
            [1.0, 0.5, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    right = torch.tensor(
        [
            [0.0, 1.0, 2.1, 1.4],
            [0.1, 0.45, 0.05, -0.15],
            [1.0, 0.55, -0.05, -0.45],
        ],
        dtype=torch.float32,
    )

    distances, path_lengths = batched_dtw_distance_with_path_length_torch(left, right, device="cpu")

    expected = [
        dtw_distance_with_path_length(l_row.tolist(), r_row.tolist())
        for l_row, r_row in zip(left, right, strict=True)
    ]
    assert distances.shape == (3,)
    assert path_lengths.shape == (3,)
    for index, (expected_distance, expected_length) in enumerate(expected):
        assert abs(float(distances[index].item()) - expected_distance) < 1e-6
        assert int(path_lengths[index].item()) == expected_length


def test_batched_dtw_similarity_matches_reference_cpu():
    left = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    right = torch.tensor(
        [
            [0.1, 0.2, 0.31],
            [0.0, 0.8, 0.1],
        ],
        dtype=torch.float32,
    )

    similarity = batched_dtw_similarity_torch(left, right, device="cpu")

    expected = [
        dtw_similarity(l_row.tolist(), r_row.tolist())
        for l_row, r_row in zip(left, right, strict=True)
    ]
    for index, expected_similarity in enumerate(expected):
        assert abs(float(similarity[index].item()) - expected_similarity) < 1e-6


def test_batched_dtw_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        return

    left = torch.tensor([[0.0, 1.0, 2.0, 1.0]], dtype=torch.float32)
    right = torch.tensor([[0.0, 1.1, 1.9, 1.0]], dtype=torch.float32)

    similarity = batched_dtw_similarity_torch(left, right, device="cuda")

    assert similarity.device.type == "cuda"
    assert 0.0 < float(similarity[0].item()) <= 1.0


def test_dtw_fp32_vs_cpu_python_precision():
    import random
    # Create random sequences
    random.seed(42)
    torch.manual_seed(42)
    left_seqs = [[random.uniform(-1, 1) for _ in range(15)] for _ in range(5)]
    right_seqs = [[random.uniform(-1, 1) for _ in range(15)] for _ in range(5)]

    from stocknetv2.domain.graph.dtw_backend import compute_dtw_similarity_scores

    scores_cpu, _ = compute_dtw_similarity_scores(left_seqs, right_seqs, backend="cpu_python")
    scores_torch_fp32, _ = compute_dtw_similarity_scores(
        left_seqs, right_seqs, backend="torch_cpu", torch_batch_pair_threshold=1, torch_dtype="float32"
    )

    for c, t in zip(scores_cpu, scores_torch_fp32, strict=True):
        assert abs(c - t) < 1e-5


def test_dtw_fp32_vs_fp64_precision():
    import random
    random.seed(42)
    torch.manual_seed(42)
    left_seqs = [[random.uniform(-1, 1) for _ in range(15)] for _ in range(5)]
    right_seqs = [[random.uniform(-1, 1) for _ in range(15)] for _ in range(5)]

    from stocknetv2.domain.graph.dtw_backend import compute_dtw_similarity_scores

    scores_torch_fp32, _ = compute_dtw_similarity_scores(
        left_seqs, right_seqs, backend="torch_cpu", torch_batch_pair_threshold=1, torch_dtype="float32"
    )
    scores_torch_fp64, _ = compute_dtw_similarity_scores(
        left_seqs, right_seqs, backend="torch_cpu", torch_batch_pair_threshold=1, torch_dtype="float64"
    )

    for t32, t64 in zip(scores_torch_fp32, scores_torch_fp64, strict=True):
        assert abs(t32 - t64) < 1e-6

