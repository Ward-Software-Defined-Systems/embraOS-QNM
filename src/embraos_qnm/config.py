"""Configuration for the tiny QNM model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QNMConfig:
    """Hyperparameters for the tiny QNM.

    Defaults are sized so the synthetic copy task learns to near-zero loss on CPU in
    seconds — this Core is a *mechanism instrument*, not a product.
    """

    vocab_size: int  # required; comes from the tokenizer
    block_size: int = 64  # max context length (T)
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128  # shared embedding dim D (must be divisible by n_head)
    dropout: float = 0.0  # 0.0 on purpose: dropout adds forward-pass RNG that would
    # complicate the determinism tests. Turn it on later for a generalization study.
    bias: bool = True

    # --- QNM injection seam ---
    inject_layer: int = 2  # L: which Core block becomes a QNMBlock (0-indexed)
    qnm_enabled: bool = True  # master toggle; False => exactly the plain transformer
    gate_init: float = 0.0  # ReZero gate_fabric init; 0.0 => bit-identical cold-start. A >0 value
    # warm-starts the install to un-starve the Fabric's content gradient (PSI Part III, Rung 1).

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if not 0 <= self.inject_layer < self.n_layer:
            raise ValueError(
                f"inject_layer ({self.inject_layer}) must be in [0, n_layer={self.n_layer})"
            )
