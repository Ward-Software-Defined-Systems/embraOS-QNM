"""Norm null-space characterization (P2.2).

The LayerNorm "a uniform (all-ones) delta is invisible" finding (ARCHITECTURE §3.3, the GPT-2 /
TinyTransformer cores) does NOT transfer to RMSNorm (Qwen2.5 / Llama 3.2). RMSNorm normalizes by
the root-mean-square *without* subtracting the mean, so a uniform shift changes its output. This
is re-characterized here as a falsifiable test so no future Fabric/World-State delta assumes the
old null space on an RMSNorm core. (Mean-centering a delta stays safe under both norms.)
"""

from __future__ import annotations

import torch
from torch import nn

_UNIFORM = torch.ones(16) * 3.7  # an all-ones (uniform-across-features) shift


def test_layernorm_annihilates_uniform_delta() -> None:
    """LayerNorm subtracts the mean ⇒ adding a uniform constant is in its null space."""
    torch.manual_seed(0)
    ln = nn.LayerNorm(16)
    x = torch.randn(4, 16)
    assert torch.allclose(ln(x), ln(x + _UNIFORM), atol=1e-6), (
        "LayerNorm must ignore a uniform shift"
    )


def test_rmsnorm_does_not_annihilate_uniform_delta() -> None:
    """RMSNorm does NOT subtract the mean ⇒ a uniform shift is visible (finding does not transfer)."""
    torch.manual_seed(0)
    rms = nn.RMSNorm(16)
    x = torch.randn(4, 16)
    assert not torch.allclose(rms(x), rms(x + _UNIFORM), atol=1e-4), (
        "RMSNorm must see a uniform shift"
    )
