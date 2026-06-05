"""Sample from a tiny QNM.

With no checkpoint, trains a quick copy model and shows it copying:

    uv run python -m embraos_qnm.generate --device cpu
"""

from __future__ import annotations

import argparse

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.data.toy import CopyTask
from embraos_qnm.device import resolve_device
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.train import copy_accuracy, load_checkpoint, train


def demo_copy(model: QNMModel, task: CopyTask, device: torch.device, n: int = 6) -> None:
    """Show the model greedily copying ``n`` random prompts."""
    gen = torch.Generator().manual_seed(0)
    x, _ = task.batch(n, generator=gen)
    length = task.length
    prompt = x[:, : length + 1].to(device)  # source + separator
    out = model.generate(prompt, max_new_tokens=length, temperature=0.0)
    tok = task.tokenizer
    for i in range(n):
        src = tok.decode(prompt[i, :length].tolist())
        copy = tok.decode(out[i, length + 1 :].tolist())
        flag = "OK" if src == copy else "XX"
        print(f"  [{flag}] {src} > {copy}")


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from a tiny QNM (copy-task demo).")
    p.add_argument(
        "--ckpt", default=None, help="checkpoint to load; if omitted, trains a quick model"
    )
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--length", type=int, default=6)
    p.add_argument("--n-symbols", type=int, default=8)
    args = p.parse_args()

    device = resolve_device(args.device)
    task = CopyTask(n_symbols=args.n_symbols, length=args.length)
    if args.ckpt:
        model = load_checkpoint(args.ckpt, device=device)
    else:
        print("no checkpoint — training a quick copy model...")
        config = QNMConfig(vocab_size=task.vocab_size, block_size=task.seq_len)
        model, _ = train(task, config, steps=args.steps, device=device, seed=0)

    print(f"copy accuracy: {copy_accuracy(model, task, 256, device):.3f}")
    print("samples (greedy):")
    demo_copy(model, task, device)


if __name__ == "__main__":
    main()
