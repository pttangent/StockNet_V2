from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from stocknetv2.domain.graph.dtw_distance import dtw_similarity


def compute_dtw_similarity_scores(
    left_sequences: Sequence[Sequence[float]],
    right_sequences: Sequence[Sequence[float]],
    *,
    backend: str,
    torch_device: str = "auto",
    torch_batch_pair_threshold: int = 1024,
) -> tuple[list[float], str]:
    if len(left_sequences) != len(right_sequences):
        raise ValueError("left_sequences and right_sequences must have the same length")
    if not left_sequences:
        return [], "no_pairs"

    effective_backend = resolve_dtw_backend(
        requested_backend=backend,
        pair_count=len(left_sequences),
        torch_batch_pair_threshold=torch_batch_pair_threshold,
        torch_device=torch_device,
    )
    if effective_backend == "cpu_python":
        return (
            [
                dtw_similarity(list(left_sequence), list(right_sequence))
                for left_sequence, right_sequence in zip(left_sequences, right_sequences, strict=True)
            ],
            effective_backend,
        )

    import torch

    from stocknetv2.domain.graph.dtw_torch import batched_dtw_similarity_torch

    resolved_device = _resolve_torch_device(effective_backend, torch_device)
    grouped_sequences: dict[int, list[tuple[int, Sequence[float], Sequence[float]]]] = defaultdict(list)
    for index, (left_sequence, right_sequence) in enumerate(zip(left_sequences, right_sequences, strict=True)):
        if len(left_sequence) != len(right_sequence):
            raise ValueError("batched DTW backend requires aligned sequences with equal lengths")
        grouped_sequences[len(left_sequence)].append((index, left_sequence, right_sequence))

    scores = [0.0] * len(left_sequences)
    for _, grouped_items in grouped_sequences.items():
        left_tensor = torch.tensor(
            [list(item[1]) for item in grouped_items],
            dtype=torch.float64,
        )
        right_tensor = torch.tensor(
            [list(item[2]) for item in grouped_items],
            dtype=torch.float64,
        )
        batch_scores = (
            batched_dtw_similarity_torch(
                left_tensor,
                right_tensor,
                device=resolved_device,
            )
            .detach()
            .cpu()
            .tolist()
        )
        for (original_index, _, _), score in zip(grouped_items, batch_scores, strict=True):
            scores[original_index] = float(score)
    return scores, effective_backend


def resolve_dtw_backend(
    *,
    requested_backend: str,
    pair_count: int,
    torch_batch_pair_threshold: int,
    torch_device: str,
) -> str:
    normalized_backend = requested_backend.strip().lower()
    if normalized_backend not in {"cpu_python", "torch_cpu", "torch_cuda", "torch_auto"}:
        raise ValueError(f"Unsupported DTW backend: {requested_backend}")
    if normalized_backend == "cpu_python":
        return "cpu_python"
    if pair_count < max(1, torch_batch_pair_threshold):
        return "cpu_python"

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when torch is missing
        raise RuntimeError("Torch DTW backend requested but torch is not installed.") from exc

    if normalized_backend == "torch_cpu":
        return "torch_cpu"
    if normalized_backend == "torch_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA DTW backend requested but CUDA is not available.")
        return "torch_cuda"
    if torch.cuda.is_available() and torch_device != "cpu":
        return "torch_cuda"
    return "torch_cpu"


def _resolve_torch_device(effective_backend: str, torch_device: str) -> str:
    if effective_backend == "torch_cpu":
        return "cpu"
    if effective_backend == "torch_cuda":
        return "cuda"
    return torch_device
