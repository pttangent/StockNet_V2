# GPU DTW Runtime Notes

## Background & Rationale
Running the `gpu_only_dtw` profile with many process workers (e.g., `MAX_WORKERS=18`) can cause severe GPU context contention on a single GPU (like the RTX 5090). Since each process initializes its own PyTorch/CUDA context, they contend for GPU resources, resulting in poor hardware utilization.

In addition, prior to this refactoring, the Trade Flow DTW similarity layer was executing DTW comparisons sequentially per symbol pair, with tiny batches of 2-3 components at a time. This prevented the GPU from utilizing its massive parallelism.

## Recommended Configurations

### For `gpu_only_dtw` Mode:
- **`MAX_WORKERS`**: 2 (GPU execution channel)
- **`MAX_IN_FLIGHT_TASKS`**: 4
- **DTW Backend**: `torch_cuda`
- **DTW Device**: `cuda`

### For `cpu_only_dtw` Mode:
- **`MAX_WORKERS`**: 20 (Leverages CPU multicore capacity)
- **`MAX_IN_FLIGHT_TASKS`**: 24
- **DTW Backend**: `cpu_python`
- **DTW Device**: `cpu`

## 5-Minute Benchmark Baselines (RTX 5090 + Ultra 9 285)
- `cpu_only_dtw` (20 workers): ~7 snapshots / 5min
- `gpu_only_dtw` (18 workers, old): ~8 snapshots / 5min (due to overhead/concurrency issues)
- `gpu_only_dtw` (2 workers, optimized): *TBD after Phase 4*
