from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from stocknetv2.domain.graph.dtw_distance import dtw_similarity


_dtw_executor = None


def compute_dtw_similarity_scores(
    left_sequences: Sequence[Sequence[float]],
    right_sequences: Sequence[Sequence[float]],
    *,
    backend: str,
    torch_device: str = "auto",
    torch_batch_pair_threshold: int = 1024,
    torch_dtype: str = "float32",
    torch_gpu_chunk_size: int = 8192,
) -> tuple[list[float], str]:
    global _dtw_executor
    if _dtw_executor is None:
        from stocknetv2.infrastructure.dtw.dtw_execution_service import SharedGpuDtwExecutionService
        _dtw_executor = SharedGpuDtwExecutionService()

    return _dtw_executor.compute(
        left_sequences=left_sequences,
        right_sequences=right_sequences,
        backend=backend,
        torch_device=torch_device,
        torch_batch_pair_threshold=torch_batch_pair_threshold,
        torch_dtype=torch_dtype,
        torch_gpu_chunk_size=torch_gpu_chunk_size,
    )


def _compute_dtw_similarity_scores_impl(
    left_sequences: Sequence[Sequence[float]],
    right_sequences: Sequence[Sequence[float]],
    *,
    backend: str,
    torch_device: str = "auto",
    torch_batch_pair_threshold: int = 1024,
    torch_dtype: str = "float32",
    torch_gpu_chunk_size: int = 8192,
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

    dtype = torch.float64 if torch_dtype == "float64" else torch.float32

    scores = [0.0] * len(left_sequences)
    for _, grouped_items in grouped_sequences.items():
        for chunk_start in range(0, len(grouped_items), torch_gpu_chunk_size):
            chunk = grouped_items[chunk_start : chunk_start + torch_gpu_chunk_size]
            left_tensor = torch.tensor(
                [list(item[1]) for item in chunk],
                dtype=dtype,
            )
            right_tensor = torch.tensor(
                [list(item[2]) for item in chunk],
                dtype=dtype,
            )
            batch_scores = (
                batched_dtw_similarity_torch(
                    left_tensor,
                    right_tensor,
                    device=resolved_device,
                    dtype=dtype,
                )
                .detach()
                .cpu()
                .tolist()
            )
            for (original_index, _, _), score in zip(chunk, batch_scores, strict=True):
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
