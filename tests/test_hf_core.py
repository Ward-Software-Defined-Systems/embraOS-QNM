"""HFCausalCore (P2.1/P2.2) — gated on the `hf` extra; uses a tiny-random Qwen2 built from a
config (NO download), so it runs in CI. The no-op seam over a real HF decoder stack
(RoPE / GQA / RMSNorm) must be bit-identical to stock, exactly as for the LayerNorm cores —
proving the arg-transparent QNMBlock threads the layer's auxiliary args correctly.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.hf_core import HFCausalCore
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.manifold.qnm_block import QNMBlock

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires the `hf` extra: uv sync --extra hf",
)


def _tiny_qwen() -> object:
    import transformers  # pyright: ignore[reportMissingImports]

    cfg_cls = transformers.Qwen2Config  # pyright: ignore[reportAttributeAccessIssue]
    model_cls = transformers.Qwen2ForCausalLM  # pyright: ignore[reportAttributeAccessIssue]
    cfg = cfg_cls(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: fewer KV heads than query heads
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return model_cls(cfg)


def _tiny_qwen3() -> object:
    """Tiny-random Qwen3 (the chosen P2.5 core family). Qwen3 adds qk-norm (RMSNorm on the
    per-head query/key) and a decoupled head_dim — the seam must thread those for free too."""
    import transformers  # pyright: ignore[reportMissingImports]

    cfg_cls = transformers.Qwen3Config  # pyright: ignore[reportAttributeAccessIssue]
    model_cls = transformers.Qwen3ForCausalLM  # pyright: ignore[reportAttributeAccessIssue]
    cfg = cfg_cls(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: fewer KV heads than query heads
        head_dim=16,  # Qwen3 decouples head_dim from hidden_size / num_heads
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return model_cls(cfg)


# Both decoder families behind HFCausalCore: Qwen2 (P2.1/P2.2) and Qwen3 (the P2.5 core family).
_TINY_CORES = pytest.mark.parametrize(
    "build_inner", [_tiny_qwen, _tiny_qwen3], ids=["qwen2", "qwen3"]
)


def _cfg(core: HFCausalCore) -> QNMConfig:
    return QNMConfig(
        vocab_size=int(core._model.config.vocab_size),
        block_size=core.block_size,
        n_layer=core.num_layers(),
        n_head=1,  # unused for an injected core (QNMConfig only validates d_model % n_head)
        d_model=core.d_model,
        inject_layer=core.num_layers() // 2,
    )


@_TINY_CORES
def test_hf_core_noop_seam_is_bit_identical(build_inner) -> None:
    """The headline check: no-op seam over a RoPE/GQA/RMSNorm core == stock, bit-for-bit."""
    core = HFCausalCore(model=build_inner())
    ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        ref = core(ids)  # stock logits, captured BEFORE the seam swap
    qnm = QNMModel(_cfg(core), core=core)  # swaps blocks[L] for a no-op QNMBlock
    with torch.no_grad():
        ours = qnm(ids)[0]
    assert torch.equal(ours, ref), "no-op seam over an HF (RoPE/RMSNorm) core must equal stock"


@_TINY_CORES
def test_hf_core_only_inject_layer_wrapped(build_inner) -> None:
    core = HFCausalCore(model=build_inner())
    cfg = _cfg(core)
    qnm = QNMModel(cfg, core=core)
    assert isinstance(qnm.core.blocks[cfg.inject_layer], QNMBlock)
    assert sum(isinstance(b, QNMBlock) for b in qnm.core.blocks) == 1
