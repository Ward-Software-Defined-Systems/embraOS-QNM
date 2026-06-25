"""GPT-2 Core (P3) — the Core-swap analog of test_bit_identity, gated on the `gpt2` extra.

Skipped in the default dev/CI sync (where ``transformers`` is absent); enable with
``uv sync --extra gpt2``. Two exact (``torch.equal``) checks on CPU float32:

  1. the reassembled GPT2Core forward equals stock GPT-2 (the adapter op-order is right);
  2. the QNM seam with no-op Fabric/World-State, injected into the *real pretrained* Core,
     is bit-for-bit stock GPT-2 — H0 ("the architecture changed nothing") over GPT-2.

A divergence is an op-order bug to fix, never a tolerance to loosen.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.hf_gpt2_core import GPT2Core
from embraos_qnm.manifold.model import QNMModel

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires the `hf` extra: uv sync --extra hf",
)


def _ids(core: GPT2Core) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, core._stock.config.vocab_size, (2, 16))


def test_gpt2core_parity_with_stock() -> None:
    """embed -> adapter blocks -> final must equal stock GPT-2 exactly."""
    core = GPT2Core("gpt2")
    core.assert_bit_identity_with_stock(_ids(core))


def test_qnm_noop_over_gpt2_is_bit_identical() -> None:
    """The headline P3 check: the no-op QNM seam over a real pretrained Core == stock GPT-2."""
    core = GPT2Core("gpt2")
    cfg = QNMConfig(
        vocab_size=core._stock.config.vocab_size,
        block_size=core.block_size,
        n_layer=core.num_layers(),
        n_head=12,  # only used by QNMConfig's d_model % n_head check (768 % 12 == 0)
        d_model=core.d_model,
        inject_layer=5,
    )
    qnm = QNMModel(cfg, core=core)
    qnm.eval()
    ids = _ids(core)
    with torch.no_grad():
        ours = qnm(ids)[0]
        ref = core._stock(ids).logits
    assert torch.equal(ours, ref), "no-op QNM over GPT-2 must equal stock GPT-2 bit-for-bit"
