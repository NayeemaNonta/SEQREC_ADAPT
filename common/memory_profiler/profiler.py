"""
common/memory_profiler/profiler.py

Lightweight GPU / CPU memory tracking utilities.
"""

import time

import torch


class Timer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start


class MemoryProfiler:
    """
    Tracks peak GPU memory allocated during a training / eval run.
    Falls back to 0.0 when CUDA is unavailable.
    """

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return 0.0

    def report(self) -> str:
        return f"peak GPU mem: {self.peak_mb():.1f} MB"
