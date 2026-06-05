"""Checkpoint round-trip + the QNM gate keys survive save/load."""

from __future__ import annotations

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.seed import set_seed
from embraos_qnm.train import load_checkpoint, save_checkpoint


def test_save_load_roundtrip_reproduces_forward(cfg: QNMConfig, tmp_path):
    set_seed(0)
    model = QNMModel(cfg)
    model.eval()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, cfg, path)

    loaded = load_checkpoint(path)
    loaded.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    assert torch.equal(model(idx)[0], loaded(idx)[0])


def test_checkpoint_contains_gate_keys(cfg: QNMConfig):
    set_seed(0)
    sd = QNMModel(cfg).state_dict()
    layer = cfg.inject_layer
    assert f"core.blocks.{layer}.gate_fabric" in sd
    assert f"core.blocks.{layer}.gate_world" in sd
