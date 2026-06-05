"""Toolchain smoke test: the package imports and torch is present."""


def test_package_imports():
    import embraos_qnm

    assert embraos_qnm.__version__


def test_torch_available():
    import torch

    # CPU is always available; this just confirms the wheel installed and links.
    assert torch.zeros(1).sum().item() == 0.0
