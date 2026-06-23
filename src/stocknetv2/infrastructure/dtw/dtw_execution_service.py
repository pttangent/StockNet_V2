from __future__ import annotations

import os
import random
import tempfile
import time
from typing import Sequence

from stocknetv2.domain.graph.dtw_backend import _compute_dtw_similarity_scores_impl


class DtwExecutionService:
    def compute(
        self,
        left_sequences: Sequence[Sequence[float]],
        right_sequences: Sequence[Sequence[float]],
        *,
        backend: str,
        torch_device: str = "auto",
        torch_batch_pair_threshold: int = 1024,
        torch_dtype: str = "float32",
        torch_gpu_chunk_size: int = 8192,
    ) -> tuple[list[float], str]:
        raise NotImplementedError


class InProcessDtwExecutionService(DtwExecutionService):
    def compute(
        self,
        left_sequences: Sequence[Sequence[float]],
        right_sequences: Sequence[Sequence[float]],
        *,
        backend: str,
        torch_device: str = "auto",
        torch_batch_pair_threshold: int = 1024,
        torch_dtype: str = "float32",
        torch_gpu_chunk_size: int = 8192,
    ) -> tuple[list[float], str]:
        return _compute_dtw_similarity_scores_impl(
            left_sequences,
            right_sequences,
            backend=backend,
            torch_device=torch_device,
            torch_batch_pair_threshold=torch_batch_pair_threshold,
            torch_dtype=torch_dtype,
            torch_gpu_chunk_size=torch_gpu_chunk_size,
        )


class SharedGpuDtwExecutionService(DtwExecutionService):
    def __init__(self):
        self.lock_dir = os.path.join(tempfile.gettempdir(), "stocknetv2_gpu_dtw_lock")
        self.pid_file = os.path.join(self.lock_dir, "owner.pid")

    def _is_pid_running(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _acquire_lock(self) -> bool:
        my_pid = os.getpid()
        for _ in range(10):  # Retry loop for orphan lock cleanup
            try:
                os.mkdir(self.lock_dir)
                try:
                    with open(self.pid_file, "w") as f:
                        f.write(str(my_pid))
                except Exception:
                    try:
                        os.rmdir(self.lock_dir)
                    except Exception:
                        pass
                    return False
                return True
            except FileExistsError:
                try:
                    if os.path.exists(self.pid_file):
                        with open(self.pid_file, "r") as f:
                            pid_str = f.read().strip()
                        if pid_str:
                            owner_pid = int(pid_str)
                            if self._is_pid_running(owner_pid):
                                return False
                    else:
                        stat = os.stat(self.lock_dir)
                        if time.time() - stat.st_mtime < 10.0:
                            return False
                    
                    # Lock is orphaned
                    if os.path.exists(self.pid_file):
                        try:
                            os.remove(self.pid_file)
                        except Exception:
                            pass
                    try:
                        os.rmdir(self.lock_dir)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
            time.sleep(0.05 + random.random() * 0.05)
        return False

    def compute(
        self,
        left_sequences: Sequence[Sequence[float]],
        right_sequences: Sequence[Sequence[float]],
        *,
        backend: str,
        torch_device: str = "auto",
        torch_batch_pair_threshold: int = 1024,
        torch_dtype: str = "float32",
        torch_gpu_chunk_size: int = 8192,
    ) -> tuple[list[float], str]:
        # Determine if it's running a GPU/CUDA backend
        # We only lock if executing GPU/CUDA to avoid serializing CPU backends
        is_cuda = (backend == "torch_cuda") or (backend == "torch_auto" and torch_device != "cpu")
        
        if is_cuda:
            acquired = False
            # Try to acquire atomic directory lock
            while not acquired:
                if self._acquire_lock():
                    acquired = True
                else:
                    time.sleep(0.05)
            
            try:
                return _compute_dtw_similarity_scores_impl(
                    left_sequences,
                    right_sequences,
                    backend=backend,
                    torch_device=torch_device,
                    torch_batch_pair_threshold=torch_batch_pair_threshold,
                    torch_dtype=torch_dtype,
                    torch_gpu_chunk_size=torch_gpu_chunk_size,
                )
            finally:
                if acquired:
                    try:
                        if os.path.exists(self.pid_file):
                            os.remove(self.pid_file)
                    except Exception:
                        pass
                    try:
                        os.rmdir(self.lock_dir)
                    except Exception:
                        pass
        else:
            return _compute_dtw_similarity_scores_impl(
                left_sequences,
                right_sequences,
                backend=backend,
                torch_device=torch_device,
                torch_batch_pair_threshold=torch_batch_pair_threshold,
                torch_dtype=torch_dtype,
                torch_gpu_chunk_size=torch_gpu_chunk_size,
            )
