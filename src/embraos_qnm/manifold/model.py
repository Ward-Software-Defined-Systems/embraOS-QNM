"""QNMModel — assembles the Core and inserts the injection seam at one layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.transformer import TinyTransformer
from embraos_qnm.fabric import NoOpFabric
from embraos_qnm.interfaces import CoreInterface, FabricInterface, WorldStateInterface
from embraos_qnm.manifold.qnm_block import QNMBlock
from embraos_qnm.world_state import NoOpWorldState


class QNMModel(nn.Module):
    """The Quantum Neural Manifold (classical approximation).

    A ``TinyTransformer`` Core whose block at ``config.inject_layer`` is replaced by a
    ``QNMBlock`` carrying the Fabric (IDENTITY) and World-State (SOUL). With the default
    no-op components, ``forward`` is bit-identical to the plain Core.
    """

    def __init__(
        self,
        config: QNMConfig,
        fabric: FabricInterface | None = None,
        world_state: WorldStateInterface | None = None,
        core: CoreInterface | None = None,
    ):
        super().__init__()
        self.config = config
        # Default Core is the from-scratch TinyTransformer; pass ``core=`` to drop in a
        # pretrained backend (e.g. GPT2Core) behind the same CoreInterface contract.
        self.core: CoreInterface = core if core is not None else TinyTransformer(config)
        if not 0 <= config.inject_layer < self.core.num_layers():
            raise ValueError(
                f"inject_layer ({config.inject_layer}) must be in "
                f"[0, num_layers={self.core.num_layers()}) for the given Core"
            )
        fabric = fabric if fabric is not None else NoOpFabric()
        world_state = world_state if world_state is not None else NoOpWorldState()
        # Swap exactly one Core block for a QNMBlock. All QNM behavior lives inside the
        # swapped block, so print(model) shows precisely where the architecture diverges,
        # and the Fabric/World-State parameters are registered exactly once.
        inject_at = config.inject_layer
        self.core.blocks[inject_at] = QNMBlock(
            self.core.blocks[inject_at], fabric, world_state, enabled=config.qnm_enabled
        )

    @property
    def qnm_block(self) -> QNMBlock:
        """The injected QNMBlock (typed accessor for the seam)."""
        block = self.core.blocks[self.config.inject_layer]
        assert isinstance(block, QNMBlock)
        return block

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        logits = self.core(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Autoregressively sample ``max_new_tokens``. ``temperature=0.0`` => greedy argmax."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.core.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k).values[:, -1, None]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
