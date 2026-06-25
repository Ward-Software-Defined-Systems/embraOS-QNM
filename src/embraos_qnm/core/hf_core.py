"""hf_core.py — a generic pretrained Hugging Face causal-LM Core behind CoreInterface.

Generalizes hf_gpt2_core.py to RMSNorm / RoPE / GQA decoder stacks (Qwen2.5, Llama 3.2, ...).
The seam works because ``QNMBlock`` is arg-transparent: ``blocks`` is the model's LIVE decoder
``ModuleList``, so swapping ``blocks[L]`` for a ``QNMBlock`` injects *inside the model's own
forward* — with RoPE/mask/position_ids threaded into the wrapped layer for free. ``forward``
therefore delegates to the underlying model rather than re-implementing the stack.

RMSNorm WARNING: unlike LayerNorm, RMSNorm does NOT subtract the mean, so the "uniform delta is
in the null space" finding from the LayerNorm cores does not hold here. Re-characterize before
assuming any direction is invisible (P2.2).

FREEZING FOOTGUN (for P2.5 enforce-training): because ``blocks`` is the model's own layer list,
after the seam swap the trainable side-pathway lives *inside* the registered model tree. Freeze
with "freeze everything, then un-freeze the side-pathway" (fabric + world_state + gates), never
``core._model.requires_grad_(False)`` alone.

Needs the ``hf`` extra: ``uv sync --extra hf``  (Qwen2.5-0.5B-Instruct is ~1 GB; CPU/MPS-runnable).
"""

from __future__ import annotations

from typing import Any

import torch

from embraos_qnm.interfaces import CoreInterface, InjectionFn

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class HFCausalCore(CoreInterface):
    """A pretrained HF causal LM behind CoreInterface. ``blocks`` is the model's live decoder
    layer list, so the QNM block-swap seam injects inside the model's own forward."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, model: Any = None) -> None:
        super().__init__()
        if model is None:
            # local import keeps transformers optional (hf extra); absent in default dev/CI
            from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]

            model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model.float()  # float32: CPU bit-identity determinism + dtype match with side-pathways
        inner = model.model  # decoder stack (Qwen2Model / LlamaModel)

        self._model: Any = model
        self.blocks = inner.layers  # LIVE ModuleList — swapping an entry rewires the forward
        self._embed_tokens = inner.embed_tokens
        self._norm = inner.norm
        self._lm_head = model.lm_head

        self.d_model = int(model.config.hidden_size)
        self.block_size = int(model.config.max_position_embeddings)

    def num_layers(self) -> int:
        return len(self.blocks)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return self._embed_tokens(idx)

    def final(self, h: torch.Tensor) -> torch.Tensor:
        return self._lm_head(self._norm(h))

    def forward(self, idx: torch.Tensor, *, inject: InjectionFn | None = None) -> torch.Tensor:
        if inject is not None:
            raise NotImplementedError(
                "HFCausalCore injects via the block-swap seam (QNMModel swaps blocks[L]); "
                "the inject callback is unsupported because RoPE/mask are threaded by the "
                "model's own forward."
            )
        return self._model(idx).logits


def _derisk(model_name: str = DEFAULT_MODEL) -> None:
    """P2.1 de-risk on 8 GB, with ONE model loaded for an honest peak-RSS number:
    (1) the cold-start seam (live Fabric, gate 0) over the real model is bit-identical to stock,
    (2) one frozen-core side-pathway training step fits and gradients flow to the gate."""
    import resource

    from embraos_qnm.config import QNMConfig
    from embraos_qnm.interfaces import FabricInterface
    from embraos_qnm.manifold.model import QNMModel
    from embraos_qnm.world_state import NoOpWorldState

    class _LinFabric(FabricInterface):
        """A live, trainable, mean-centered Fabric stand-in (real R-GCN Fabric is P2.3)."""

        def __init__(self, d: int) -> None:
            super().__init__()
            self.w = torch.nn.Linear(d, d, bias=False)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            x = self.w(h)
            return x - x.mean(dim=-1, keepdim=True)

    torch.manual_seed(0)
    core = HFCausalCore(model_name)
    ids = torch.randint(0, 512, (1, 16))

    # (1) capture stock logits BEFORE the seam swap, then wrap (live Fabric, gate 0) and compare.
    with torch.no_grad():
        ref = core(ids)
    cfg = QNMConfig(
        vocab_size=int(core._model.config.vocab_size),
        block_size=core.block_size,
        n_layer=core.num_layers(),
        n_head=1,  # unused for an injected core (QNMConfig only validates d_model % n_head)
        d_model=core.d_model,
        inject_layer=core.num_layers() // 2,
    )
    qnm = QNMModel(cfg, core=core, fabric=_LinFabric(cfg.d_model), world_state=NoOpWorldState())
    with torch.no_grad():
        ours = qnm(ids)[0]
    assert torch.equal(ours, ref), "cold-start seam (live Fabric, gate 0) over HF core != stock"
    print(
        f"[OK] cold-start seam == stock {model_name} (d_model={core.d_model}, L={core.num_layers()})"
    )

    # (2) frozen-core training step: freeze ALL, then unfreeze the side-pathway only.
    for p in qnm.parameters():
        p.requires_grad_(False)
    block = qnm.qnm_block
    for p in block.fabric.parameters():
        p.requires_grad_(True)
    block.gate_fabric.requires_grad_(True)
    opt = torch.optim.Adam([p for p in qnm.parameters() if p.requires_grad], lr=1e-3)
    logits, _ = qnm(ids)
    loss = logits.float().pow(2).mean()  # dummy objective, just to flow gradients
    loss.backward()
    opt.step()
    assert block.gate_fabric.grad is not None, "gate must receive gradient through the frozen core"
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes
    print(f"[OK] frozen-core training step flows (gate.grad set); peak RSS ~ {peak_gb:.2f} GB")


if __name__ == "__main__":
    _derisk()
