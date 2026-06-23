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
