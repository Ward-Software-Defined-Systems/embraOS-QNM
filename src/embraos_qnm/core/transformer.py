"""TinyTransformer — the from-scratch decoder-only LLM Core.

A nanoGPT-scale instrument: token+position embeddings, a stack of pre-norm blocks, a
final norm, and a (weight-tied) LM head. Implements CoreInterface so a pretrained
backend can later be dropped in behind the same contract.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.block import Block
from embraos_qnm.interfaces import CoreInterface, InjectionFn


class TinyTransformer(CoreInterface):
    def __init__(self, config: QNMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.block_size = config.block_size
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Weight tying (after init): the LM head shares the token embedding matrix.
        self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_layers(self) -> int:
        return len(self.blocks)

    def embed(self, idx: Tensor) -> Tensor:
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B,T,D) + (T,D) broadcast
        return self.drop(x)

    def final(self, h: Tensor) -> Tensor:
        return self.lm_head(self.ln_f(h))

    def forward(self, idx: Tensor, *, inject: InjectionFn | None = None) -> Tensor:
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if inject is not None:
                x = inject(i, x)
        return self.final(x)
