"""Deterministic seeding for reproducible runs.

Determinism is a CPU/CUDA guarantee; on MPS it is best-effort only (see device.py).
All bit-identity / determinism tests therefore run on CPU.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch RNGs.

    With ``deterministic=True`` also pin torch to deterministic algorithms. This is
    effective on CPU (and CUDA, with the cuBLAS workspace env set); on MPS it is
    best-effort and runs there should be treated as reproducible-ish, not bit-exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic cuBLAS on CUDA; harmless on CPU/MPS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only so ops lacking a deterministic kernel warn rather than raise.
        torch.use_deterministic_algorithms(True, warn_only=True)
