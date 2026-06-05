"""Learning-signal tests: the Core actually trains."""

from __future__ import annotations

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.data.toy import CopyTask
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.seed import set_seed
from embraos_qnm.train import copy_accuracy, train


def _small() -> tuple[CopyTask, QNMConfig]:
    task = CopyTask(n_symbols=6, length=4)
    config = QNMConfig(
        vocab_size=task.vocab_size, block_size=task.seq_len, n_layer=3, n_head=4, d_model=64
    )
    return task, config


def test_copy_task_loss_drops_and_accuracy_rises():
    task, config = _small()
    model, losses = train(task, config, steps=150, batch_size=64, lr=3e-3, seed=0, log_every=0)
    assert losses[-1] < 0.1 * losses[0]
    assert copy_accuracy(model, task, batch_size=256) > 0.95


def test_overfit_single_batch():
    task, config = _small()
    set_seed(0)
    model = QNMModel(config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = task.batch(32)

    first = last = None
    for _ in range(200):
        _, loss = model(x, y)
        assert loss is not None
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()

    assert first is not None and last is not None
    assert last < 0.05
    assert last < 0.1 * first
