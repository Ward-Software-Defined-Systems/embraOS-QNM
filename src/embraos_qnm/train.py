"""Minimal training loop for the tiny QNM on the synthetic copy task.

``train`` is importable (the learning-signal test calls it) and ``main`` is a CLI:

    uv run python -m embraos_qnm.train --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.data.toy import CopyTask
from embraos_qnm.device import resolve_device
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.seed import set_seed


@torch.no_grad()
def copy_accuracy(
    model: QNMModel,
    task: CopyTask,
    batch_size: int = 128,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> float:
    """Fraction of scored (copied) tokens predicted correctly."""
    device = device or torch.device("cpu")
    was_training = model.training
    model.eval()
    x, y = task.batch(batch_size, generator=generator)
    x, y = x.to(device), y.to(device)
    logits, _ = model(x)
    preds = logits.argmax(dim=-1)
    mask = y != -1
    acc = (preds[mask] == y[mask]).float().mean().item()
    model.train(was_training)
    return acc


def train(
    task: CopyTask,
    config: QNMConfig | None = None,
    *,
    steps: int = 300,
    batch_size: int = 64,
    lr: float = 3e-3,
    device: torch.device | None = None,
    seed: int = 0,
    log_every: int = 50,
) -> tuple[QNMModel, list[float]]:
    """Train on fresh copy-task batches. Returns the model and the per-step losses."""
    set_seed(seed)
    device = device or torch.device("cpu")
    if config is None:
        config = QNMConfig(vocab_size=task.vocab_size, block_size=task.seq_len)
    model = QNMModel(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    losses: list[float] = []
    for step in range(steps):
        x, y = task.batch(batch_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            acc = copy_accuracy(model, task, batch_size, device)
            print(f"step {step:4d} | loss {loss.item():.4f} | copy_acc {acc:.3f}")
    return model, losses


def save_checkpoint(model: QNMModel, config: QNMConfig, path: str) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(config)}, path)


def load_checkpoint(path: str, device: torch.device | None = None) -> QNMModel:
    ckpt = torch.load(path, map_location=device or torch.device("cpu"))
    config = QNMConfig(**ckpt["config"])
    model = QNMModel(config)
    model.load_state_dict(ckpt["model"])
    return model.to(device) if device else model


def main() -> None:
    p = argparse.ArgumentParser(description="Train the tiny QNM on the copy task.")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--length", type=int, default=6)
    p.add_argument("--n-symbols", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="checkpoint output path")
    args = p.parse_args()

    device = resolve_device(args.device)
    task = CopyTask(n_symbols=args.n_symbols, length=args.length)
    config = QNMConfig(vocab_size=task.vocab_size, block_size=task.seq_len)
    print(f"device={device.type} vocab={config.vocab_size} seq_len={task.seq_len}")
    model, _ = train(
        task,
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    print(f"final copy accuracy: {copy_accuracy(model, task, 256, device):.3f}")
    if args.out:
        save_checkpoint(model, config, args.out)
        print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
