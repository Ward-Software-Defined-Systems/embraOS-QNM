"""Central device policy.

CPU is the default everywhere (tests, CI, determinism). MPS is opt-in.

MPS caveats, documented here so they are not rediscovered the hard way:
  * float64 is unsupported on MPS — keep everything float32.
  * Bitwise reproducibility / ``torch.use_deterministic_algorithms`` is a CPU/CUDA
    story. Treat MPS runs as reproducible-ish, not bit-exact — which is why the
    bit-identity and determinism tests run on CPU only.
  * Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` for interactive MPS runs so any
    unsupported op falls back to CPU instead of erroring.
  * Forward-looking: PyG scatter / sparse message-passing kernels (the future GNN
    Fabric) have no MPS support — the real GNN will run on CPU or CUDA. Centralizing
    the policy here means that change touches exactly one function.
"""

from __future__ import annotations

import torch


def resolve_device(pref: str = "cpu") -> torch.device:
    """Resolve a device from a preference string.

    ``cpu``  -> CPU (default; used by all tests and CI).
    ``cuda`` -> CUDA if available, else CPU.
    ``mps``  -> MPS if available, else CPU.
    ``auto`` -> CUDA, else MPS, else CPU.
    """
    pref = pref.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device preference: {pref!r}")
