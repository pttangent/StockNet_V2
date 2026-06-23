from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_graph_backend(*, requested_backend: str, torch_device: str) -> str:
    normalized_backend = requested_backend.strip().lower()
    if normalized_backend not in {"cpu_numpy", "torch_cpu", "torch_cuda", "torch_auto"}:
        raise ValueError(f"Unsupported graph backend: {requested_backend}")
    if normalized_backend == "cpu_numpy":
        return "cpu_numpy"

    try:
        import torch
    except Exception as exc:  # pragma: no cover - only exercised when torch is missing
        raise RuntimeError("Torch graph backend requested but torch is not installed.") from exc

    if normalized_backend == "torch_cpu":
        return "torch_cpu"
    if normalized_backend == "torch_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA graph backend requested but CUDA is not available.")
        return "torch_cuda"
    if torch.cuda.is_available() and torch_device != "cpu":
        return "torch_cuda"
    return "torch_cpu"


def resolve_graph_torch_device(*, effective_backend: str, torch_device: str) -> str:
    if effective_backend == "torch_cpu":
        return "cpu"
    if effective_backend == "torch_cuda":
        return "cuda"
    return torch_device


def compute_pairwise_correlation_metrics_torch(
    matrix: pd.DataFrame,
    *,
    min_periods: int = 2,
    min_variance: float = 1e-12,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    values, torch = _prepare_tensor(matrix, device=device)
    mask = (~torch.isnan(values)).to(dtype=torch.float64)
    filled = torch.nan_to_num(values, nan=0.0)
    overlap = (mask.transpose(0, 1) @ mask).to(dtype=torch.int32)
    correlation = _compute_pairwise_correlation_tensor(
        filled=filled,
        mask=mask,
        overlap=overlap,
        min_periods=min_periods,
        min_variance=min_variance,
        torch=torch,
    )
    return _to_numpy(correlation), _to_numpy(overlap)


def compute_flow_alignment_metrics_torch(
    matrix: pd.DataFrame,
    *,
    epsilon: float,
    min_periods: int = 2,
    min_variance: float = 1e-12,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, torch = _prepare_tensor(matrix, device=device)
    mask = (~torch.isnan(values)).to(dtype=torch.float64)
    filled = torch.nan_to_num(values, nan=0.0)
    overlap = (mask.transpose(0, 1) @ mask).to(dtype=torch.int32)
    joint_active = _compute_joint_active_counts_tensor(filled=filled, mask=mask, epsilon=epsilon, torch=torch)
    correlation = _compute_pairwise_correlation_tensor(
        filled=filled,
        mask=mask,
        overlap=overlap,
        min_periods=min_periods,
        min_variance=min_variance,
        torch=torch,
    )
    positive = ((filled > epsilon) & (mask > 0)).to(dtype=torch.float64)
    negative = ((filled < -epsilon) & (mask > 0)).to(dtype=torch.float64)
    same_direction = positive.transpose(0, 1) @ positive + negative.transpose(0, 1) @ negative
    same_direction_ratio = torch.zeros_like(same_direction, dtype=torch.float64)
    joint_active_mask = joint_active > 0
    same_direction_ratio[joint_active_mask] = (
        same_direction.to(dtype=torch.float64)[joint_active_mask]
        / joint_active.to(dtype=torch.float64)[joint_active_mask]
    )
    return _to_numpy(correlation), _to_numpy(same_direction_ratio), _to_numpy(joint_active.to(dtype=torch.int32))


def compute_activity_metrics_torch(
    matrix: pd.DataFrame,
    *,
    threshold: float,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, torch = _prepare_tensor(matrix, device=device)
    mask = (~torch.isnan(values)).to(dtype=torch.float64)
    filled = torch.nan_to_num(values, nan=0.0)
    overlap = (mask.transpose(0, 1) @ mask).to(dtype=torch.int32)
    correlation = _compute_pairwise_correlation_tensor(
        filled=filled,
        mask=mask,
        overlap=overlap,
        min_periods=2,
        min_variance=1e-12,
        torch=torch,
    )
    above_threshold = ((filled > threshold) & (mask > 0)).to(dtype=torch.float64)
    co_occurrence = above_threshold.transpose(0, 1) @ above_threshold
    co_expansion_ratio = torch.zeros_like(co_occurrence, dtype=torch.float64)
    overlap_mask = overlap > 0
    co_expansion_ratio[overlap_mask] = (
        co_occurrence.to(dtype=torch.float64)[overlap_mask]
        / overlap.to(dtype=torch.float64)[overlap_mask]
    )
    return _to_numpy(correlation), _to_numpy(co_expansion_ratio), _to_numpy(overlap)


def _compute_pairwise_correlation_tensor(
    *,
    filled,
    mask,
    overlap,
    min_periods: int,
    min_variance: float,
    torch,
    use_overlap_for_statistics: bool = True,
):
    pair_counts = overlap.to(dtype=torch.float64)
    overlap_mask = pair_counts >= max(1, min_periods)

    sum_x = filled.transpose(0, 1) @ mask
    sum_y = sum_x.transpose(0, 1)
    sum_x2 = (filled * filled).transpose(0, 1) @ mask
    sum_y2 = sum_x2.transpose(0, 1)
    sum_xy = filled.transpose(0, 1) @ filled

    safe_counts = torch.where(overlap_mask, pair_counts, torch.ones_like(pair_counts))
    covariance_numerator = sum_xy - (sum_x * sum_y) / safe_counts
    variance_left = sum_x2 - (sum_x * sum_x) / safe_counts
    variance_right = sum_y2 - (sum_y * sum_y) / safe_counts
    denominator = torch.sqrt(torch.clamp(variance_left, min=0.0) * torch.clamp(variance_right, min=0.0))
    correlation = torch.zeros_like(covariance_numerator, dtype=torch.float64)
    valid_mask = overlap_mask & (denominator > 0)
    correlation[valid_mask] = covariance_numerator[valid_mask] / denominator[valid_mask]

    column_counts = mask.sum(dim=0)
    safe_column_counts = torch.where(column_counts > 0, column_counts, torch.ones_like(column_counts))
    column_sums = filled.sum(dim=0)
    column_means = column_sums / safe_column_counts
    centered = (filled - column_means.unsqueeze(0)) * mask
    column_variance = (centered * centered).sum(dim=0) / safe_column_counts
    variance_floor = max(float(min_variance), 1e-12)
    invalid_columns = column_variance < variance_floor
    if invalid_columns.any():
        invalid_indices = torch.where(invalid_columns)[0]
        correlation[invalid_indices, :] = 0.0
        correlation[:, invalid_indices] = 0.0

    if use_overlap_for_statistics:
        correlation = torch.where(overlap_mask, correlation, torch.zeros_like(correlation))
    else:
        correlation = torch.where(overlap_mask, correlation, torch.zeros_like(correlation))
    return correlation


def _compute_joint_active_counts_tensor(*, filled, mask, epsilon: float, torch):
    active = ((torch.abs(filled) > epsilon) & (mask > 0)).to(dtype=torch.float64)
    return (active.transpose(0, 1) @ active).to(dtype=torch.int32)


def _prepare_tensor(matrix: pd.DataFrame, *, device: str):
    import torch

    values = torch.as_tensor(
        matrix.to_numpy(dtype=float, copy=True),
        dtype=torch.float64,
        device=device,
    )
    return values, torch


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
