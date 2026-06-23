from __future__ import annotations

from typing import Any

import torch


def batched_dtw_distance_with_path_length_torch(
    left: torch.Tensor | Any,
    right: torch.Tensor | Any,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    left_tensor = _prepare_batch_tensor(left, device=device)
    right_tensor = _prepare_batch_tensor(right, device=device)
    if left_tensor.shape != right_tensor.shape:
        raise ValueError(
            f"left and right batch tensors must have the same shape, got {tuple(left_tensor.shape)} and {tuple(right_tensor.shape)}"
        )
    if left_tensor.ndim != 2:
        raise ValueError(f"expected 2D batch tensor shaped [batch, time], got ndim={left_tensor.ndim}")

    batch_size, sequence_length = left_tensor.shape
    if batch_size == 0 or sequence_length == 0:
        return (
            torch.empty((batch_size,), dtype=torch.float64, device=left_tensor.device),
            torch.zeros((batch_size,), dtype=torch.int32, device=left_tensor.device),
        )

    inf = torch.tensor(float("inf"), dtype=left_tensor.dtype, device=left_tensor.device)
    costs = torch.full(
        (batch_size, sequence_length + 1, sequence_length + 1),
        inf.item(),
        dtype=left_tensor.dtype,
        device=left_tensor.device,
    )
    lengths = torch.zeros(
        (batch_size, sequence_length + 1, sequence_length + 1),
        dtype=torch.int32,
        device=left_tensor.device,
    )
    costs[:, 0, 0] = 0.0
    unavailable_length = torch.full(
        (batch_size, 3),
        torch.iinfo(torch.int32).max,
        dtype=torch.int32,
        device=left_tensor.device,
    )

    for row in range(1, sequence_length + 1):
        left_values = left_tensor[:, row - 1]
        for col in range(1, sequence_length + 1):
            previous_costs = torch.stack(
                (
                    costs[:, row - 1, col],
                    costs[:, row, col - 1],
                    costs[:, row - 1, col - 1],
                ),
                dim=1,
            )
            previous_lengths = torch.stack(
                (
                    lengths[:, row - 1, col],
                    lengths[:, row, col - 1],
                    lengths[:, row - 1, col - 1],
                ),
                dim=1,
            )
            min_costs = previous_costs.min(dim=1, keepdim=True).values
            tie_mask = previous_costs == min_costs
            chosen_lengths = torch.where(tie_mask, previous_lengths, unavailable_length).min(dim=1).values
            local_cost = torch.abs(left_values - right_tensor[:, col - 1])
            costs[:, row, col] = local_cost + min_costs.squeeze(1)
            lengths[:, row, col] = chosen_lengths + 1

    return costs[:, sequence_length, sequence_length], lengths[:, sequence_length, sequence_length]


def batched_normalized_dtw_distance_torch(
    left: torch.Tensor | Any,
    right: torch.Tensor | Any,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    distances, path_lengths = batched_dtw_distance_with_path_length_torch(left, right, device=device)
    normalized = torch.full_like(distances, float("inf"))
    valid_mask = torch.isfinite(distances) & (path_lengths > 0)
    normalized[valid_mask] = distances[valid_mask] / path_lengths[valid_mask].to(distances.dtype)
    return normalized


def batched_dtw_similarity_torch(
    left: torch.Tensor | Any,
    right: torch.Tensor | Any,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    distance = batched_normalized_dtw_distance_torch(left, right, device=device)
    similarity = torch.zeros_like(distance)
    valid_mask = torch.isfinite(distance)
    similarity[valid_mask] = 1.0 / (1.0 + distance[valid_mask])
    return similarity


def _prepare_batch_tensor(value: torch.Tensor | Any, *, device: str | torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64, device=device)
