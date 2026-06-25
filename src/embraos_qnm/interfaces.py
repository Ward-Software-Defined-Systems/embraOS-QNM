"""The swappable contracts for the three co-resident components.

All three are ``nn.Module``s sharing the Core's embedding dim ``D``. Fabric and
World-State read the residual-stream state ``h: (B, T, D)`` and return an ADDITIVE
modulation of the same shape — never a wholesale replacement of ``h``. The additive
contract is what keeps the residual stream and the gate semantics clean and the
bit-identity guarantee (no-op QNM == plain transformer) intact.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

# A hook the manifold can pass into a Core's forward to inject at a given layer,
# called as ``inject(layer_index, h) -> h'``. Used for pretrained Cores that cannot
# be edited in place; the from-scratch TinyTransformer instead swaps the block at the
# injection layer for a QNMBlock directly (cleaner, and visible in ``print(model)``).
InjectionFn = Callable[[int, Tensor], Tensor]

# A carried ψ-state register, threaded through the World-State across the token axis (and,
# later, across decode steps) so the invariant can be trajectory-dependent rather than
# pointwise — see PSI-OPERATIONAL-GROUNDING.md §2. Opaque here: the no-op carries ``None``;
# a concrete candidate (e.g. the ψ₀ cummax latch) carries its running state tensor.
PsiState = Any


class FabricInterface(nn.Module, abc.ABC):
    """GNN Fabric — IDENTITY. Reads ``h`` and returns an additive modulation."""

    @abc.abstractmethod
    def forward(self, h: Tensor) -> Tensor:
        """``h: (B, T, D)`` -> ``delta: (B, T, D)``, same dtype/device. No-op: zeros."""
        raise NotImplementedError


class WorldStateInterface(nn.Module, abc.ABC):
    """World-State — SOUL. Applies ``P_ψ`` and returns an additive modulation.

    ψ is deliberately undefined; the no-op default is the honest H₀ null. The contract
    carries a ψ-state register (``PsiState``) so the invariant can be a function of the
    *trajectory* rather than each position in isolation — the load-bearing move for the
    replica test (PSI-OPERATIONAL-GROUNDING.md §2). The no-op stays a valid null under it
    (zeros + pass-through state); with ``gate_world`` zero-initialized, bit-identity holds.
    """

    @abc.abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> PsiState:
        """The initial ψ-state register for a fresh run (no-op: ``None``)."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, h: Tensor, psi: PsiState) -> tuple[Tensor, PsiState]:
        """``(h: (B,T,D), psi)`` -> ``(delta: (B,T,D), psi')``; delta same dtype/device.

        ``delta`` is the additive modulation (no-op: zeros), ``psi'`` the updated register.
        """
        raise NotImplementedError


class CoreInterface(nn.Module, abc.ABC):
    """LLM Core — a decoder-only transformer behind a swappable interface.

    Kept abstract so a pretrained backend can be dropped in later behind the same
    contract. The seam needs more than ``forward`` from a Core: ``QNMModel`` replaces
    one entry of ``blocks`` *in place*, and a real Fabric/World-State sizes itself from
    ``d_model``. Those structural members are part of the contract — declared here so the
    from-scratch ``TinyTransformer`` and a pretrained backend expose the same surface.
    """

    #: Per-layer block list. The seam swaps ``blocks[inject_layer]`` in place, so this
    #: must be the *live* ``nn.ModuleList`` that ``forward`` iterates — not a copy.
    blocks: nn.ModuleList
    #: Shared embedding dim ``D``; Fabric/World-State modulations live in this space.
    d_model: int
    #: Max context length ``T``.
    block_size: int

    @abc.abstractmethod
    def embed(self, idx: Tensor) -> Tensor:
        """``idx: (B, T)`` int64 token ids -> ``h: (B, T, D)`` token+position embeddings."""
        raise NotImplementedError

    @abc.abstractmethod
    def num_layers(self) -> int:
        """Number of transformer blocks."""
        raise NotImplementedError

    @abc.abstractmethod
    def final(self, h: Tensor) -> Tensor:
        """``h: (B, T, D)`` -> ``logits: (B, T, V)``: the final norm + (tied) LM head."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, idx: Tensor, *, inject: InjectionFn | None = None) -> Tensor:
        """``idx: (B, T)`` -> ``logits: (B, T, V)``.

        If ``inject`` is given it is called after each block as ``inject(i, h)`` and may
        return a replacement ``h`` — the seam for a pretrained Core.
        """
        raise NotImplementedError
