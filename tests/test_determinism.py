"""Determinism: same seed reproduces; different seed diverges (CPU)."""

from __future__ import annotations

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.seed import set_seed


def _states_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_same_seed_same_init(cfg: QNMConfig):
    set_seed(0)
    m1 = QNMModel(cfg)
    set_seed(0)
    m2 = QNMModel(cfg)
    assert _states_equal(m1.state_dict(), m2.state_dict())


def test_same_seed_same_forward(cfg: QNMConfig):
    set_seed(0)
    m1 = QNMModel(cfg)
    set_seed(0)
    m2 = QNMModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    assert torch.equal(m1(idx)[0], m2(idx)[0])


def test_different_seed_differs(cfg: QNMConfig):
    set_seed(0)
    m1 = QNMModel(cfg)
    set_seed(1)
    m2 = QNMModel(cfg)
    assert not _states_equal(m1.state_dict(), m2.state_dict())


def test_generate_is_deterministic_under_seed(cfg: QNMConfig):
    set_seed(0)
    model = QNMModel(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    set_seed(123)
    a = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    set_seed(123)
    b = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    assert torch.equal(a, b)
