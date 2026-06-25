"""
hf_gpt2_core.py  —  a pretrained GPT-2 small backend behind CoreInterface

Place at: src/embraos_qnm/core/hf_gpt2_core.py

WHY GPT-2 SMALL FIRST
---------------------
It is the structural twin of the from-scratch TinyTransformer: pre-LayerNorm, serial
attn -> MLP blocks, learned positional embeddings, a final ln_f, tied lm_head. So:
  * the injection seam (QNMBlock wrapping one block) transfers with no change;
  * the bit-identity discipline transfers (parity check below, asserted with torch.equal);
  * the ARCHITECTURE Sec 3.3 LayerNorm null-space finding still holds verbatim (a
    uniform-across-features modulation is annihilated by the downstream LayerNorms), so
    Fabric/World-State deltas must stay feature-varying.

IF YOU LATER SWAP TO AN RMSNorm CORE (Llama 3.2 / Qwen2.5 / TinyLlama / Mistral):
  RMSNorm does NOT subtract the mean, so the all-ones direction is NOT in its null space.
  The Sec 3.3 finding does NOT transfer -- a uniform delta would have an effect there.
  Re-run the gate-gradient / null-space characterization for the new norm before trusting
  any "must vary across features" assumption. Those models also add rotary embeddings + GQA,
  which complicate the seam. That is why GPT-2 is the right first swap, not those.

ASSUMPTIONS TO VERIFY against the real repo:
  * CoreInterface surface. This class targets the surface the seam needs from a Core:
        embed(idx)   -> h         # tokens -> residual stream (B,T,D)
        blocks       -> ModuleList of per-layer (B,T,D)->(B,T,D) modules (one is swappable)
        final(h)     -> logits    # ln_f + lm_head
        forward(idx) -> logits    # embed -> blocks -> final
        d_model, n_layer, block_size
    Align the NAMES with the real interfaces.CoreInterface ABC; the method bodies are the
    part that carries value.
  * QNMModel swaps core.blocks[inject_layer] for a QNMBlock. Each entry here is a
    GPT2BlockAdapter exposing the toy forward(h)->h contract, so that swap works as-is.

DEPENDENCY:  uv add transformers        (GPT-2 small is ~500 MB; CPU-runnable.)
TOKENIZER:   this Core carries GPT-2 BPE, not the copy-task char tokenizer. Any real
             language pipeline must use GPT2TokenizerFast.from_pretrained("gpt2").

Smoke / parity check:  uv run python -m embraos_qnm.core.hf_gpt2_core
"""

from __future__ import annotations

import torch
import torch.nn as nn

from embraos_qnm.interfaces import CoreInterface, InjectionFn


class GPT2BlockAdapter(nn.Module):
    """Adapt a HF GPT2Block (multi-arg in, tuple out) to the toy Block contract
    forward(h:(B,T,D)) -> (B,T,D). GPT-2 applies the causal mask internally via its bias
    buffer, so attention_mask=None is correct full-causal behavior; positional info is added
    at embedding time, not in the block."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        try:
            out = self.block(h, use_cache=False)  # first positional arg is hidden_states
        except TypeError:
            out = self.block(h)  # fallback for transformers versions with a different signature
        return out[0] if isinstance(out, tuple) else out


class GPT2Core(CoreInterface):
    """Pretrained GPT-2 small wrapped to expose a swappable per-layer residual-stream block
    list plus embed/final stages, so the QNM seam can inject at `inject_layer` exactly as it
    does for the TinyTransformer. Implements CoreInterface, so it drops into QNMModel(core=...)."""

    def __init__(self, model_name: str = "gpt2") -> None:
        super().__init__()
        # local import keeps transformers optional (gpt2 extra); absent in default dev/CI
        from transformers import GPT2LMHeadModel  # pyright: ignore[reportMissingImports]

        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
        t = model.transformer

        # reuse the stock submodules (shared params => this *is* GPT-2, not a copy).
        # wrapping the SAME GPT2Block objects guarantees parity with stock.
        self.wte = t.wte
        self.wpe = t.wpe
        self.drop = t.drop
        self.blocks = nn.ModuleList(GPT2BlockAdapter(b) for b in t.h)
        self.ln_f = t.ln_f
        self.lm_head = model.lm_head  # tied to wte

        self.d_model = model.config.n_embd
        self.n_layer = model.config.n_layer
        self.block_size = model.config.n_positions
        self._stock = model  # kept only for the parity check

    def num_layers(self) -> int:
        return len(self.blocks)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        _, t_len = idx.shape
        pos = torch.arange(t_len, device=idx.device).unsqueeze(0)  # (1, T)
        return self.drop(self.wte(idx) + self.wpe(pos))

    def final(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_f(h))

    def forward(self, idx: torch.Tensor, *, inject: InjectionFn | None = None) -> torch.Tensor:
        h = self.embed(idx)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if inject is not None:
                h = inject(i, h)
        return self.final(h)

    @torch.no_grad()
    def assert_bit_identity_with_stock(self, idx: torch.Tensor) -> None:
        """The Core-swap analog of test_bit_identity: the wrapped forward must equal stock
        GPT-2 bit-for-bit on CPU float32. A divergence is an op-order bug to fix, never a
        tolerance to loosen."""
        self.eval()
        self._stock.eval()
        ours = self.forward(idx)
        ref = self._stock(idx).logits
        if not torch.equal(ours, ref):
            max_abs = (ours - ref).abs().max().item()
            raise AssertionError(
                f"wrapped GPT-2 diverged from stock (max|delta|={max_abs:.3e}); "
                "fix the embed/block/final op order rather than relaxing the check"
            )


if __name__ == "__main__":
    core = GPT2Core("gpt2")
    ids = torch.randint(0, core._stock.config.vocab_size, (2, 16))
    core.assert_bit_identity_with_stock(ids)
    print(
        f"[OK] GPT2Core parity with stock GPT-2 "
        f"(d_model={core.d_model}, n_layer={core.n_layer}, block_size={core.block_size})"
    )
