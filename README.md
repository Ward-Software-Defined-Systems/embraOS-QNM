<p align="center">
  <img src="assets/embraos-qnm-banner.png" alt="embraOS" width="100%">
</p>

# embraOS-QNM — Quantum Neural Manifold (Classical Approximation)

**Architecture-layer constraints for language models.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20673162.svg)](https://doi.org/10.5281/zenodo.20673162)

---

## What Is This?

embraOS-QNM is a hybrid model architecture that embeds identity constraints in the fabric of a language model, rather than applying them at the prompt (System Instructions) layer.

**Open question**: How much constraint-following can move from prompt to architecture, and at what cost to capability? This is what the project aims to measure.

---

## The Problem with the Current Architecture

### Prompt-Layer Soul (Current State)

```
┌──────────────────────────────────┐
│         SOUL DOCUMENT            │  ← External. Prompt. System message.
│  - Never deceive                 │     Constrains OUTPUT, not ARCHITECTURE.
│  - Never pretend to know         │
│  - Truth over comfort            │
├──────────────────────────────────┤
│         IDENTITY DOCUMENT        │  ← External. Prompt. System message.
│  - Name: Embra                   │     Shapes tone and behavior at
│  - Traits, voice, character      │     inference time, not training time.
├──────────────────────────────────┤
│         LLM (generic)            │  ← The actual model. Trained on internet
│  - Weights                       │     text. Unaware of soul constraints
│  - Architecture                  │     except as tokens in context window.
│  - Token generation              │
└──────────────────────────────────┘
```

**Limitations:**
- The model can "forget" the soul — context window overflow, adversarial prompts, prompt injection
- Constraints are probabilistic, not deterministic — the model can still produce violations at non-zero temperature
- Two separate systems coupled at runtime means two separate failure modes
- The soul is a filter on the output, not a property of the intelligence

### The Goal: Quantum Neural Manifold - Classical Approximation Architecture

Three co-resident components, not three systems pipelined together:

```
Input → [LLM Core] → [GNN Fabric] → [World-State] → [LLM Core] → Output
              ↑            │              │              │
              └────────────┴──────────────┴───-──────────┘
```

### LLM Core
Standard transformer foundation (starting small for rapid iteration). Handles language understanding and generation. But at a configurable injection point, hidden states are routed through the other components before sampling.

### GNN Fabric
A message-passing graph neural network that maintains entity-relationship structure in the same embedding space as the LLM. When the LLM encounters a concept, the GNN activates related entities and propagates structural constraints — not retrieval, but co-resident relational reasoning.

### World-State
A persistent state register that encodes invariant boundary conditions — the model's identity constraints.

---

## Architecture

The full technical spec is in **[ARCHITECTURE.md](ARCHITECTURE.md)**: the three components in depth, the **Realized Implementation Architecture** (what has actually landed in code, with an iteration log), and a **Running & Testing** guide.

---

## Project Status

**Phase:** First scaffold (June 2026). The **LLM Core** (a small from-scratch transformer) and the **injection seam** that routes its hidden states through the other components are implemented and tested in Python + PyTorch. The **GNN Fabric** and **World-State** are honest no-ops for now — the World-State's `P_ψ` is the identity map, because ψ (the invariant) is deliberately left undefined until it can be made falsifiable.

The QNM is being developed as the next phase of the [embraOS](https://github.com/Ward-Software-Defined-Systems/embraOS) AI Operating System Continuity Architecture project.

---

## Development

The defining property of the scaffold: **with the no-op components, the model is bit-identical to a plain transformer**, enforced by a test (`torch.equal`, not a tolerance). Every future architectural effect is then measured as a provable delta from that null.

Tooling is [`uv`](https://docs.astral.sh/uv/) — it provisions a compatible Python if your system version is too new for the PyTorch wheels.

```bash
uv python install 3.12
uv sync --extra dev                                  # create the venv + install deps
uv run pytest                                        # test suite (CPU)
uv run ruff check . && uv run pyright                # lint + types
uv run python -m embraos_qnm.train --device cpu      # train the tiny Core on a copy task
uv run python -m embraos_qnm.generate --device cpu   # train-and-sample demo
```

---

## Node-Scale Hallucination Study

A separate, self-contained line of work in this repo: **[Node-Scale-Hallucination-Study.md](Node-Scale-Hallucination-Study.md)** — a pre-registered, falsifiable experiment asking whether a model's *fabrication-node scale* (the silicon process it runs on) measurably affects its hallucination rate, beyond what sampling temperature already explains. Independent of the QNM architecture work.

---

## License

Proprietary

---

— William Ward (WSDS LLC)

Part of [Ward Software Defined Systems](https://wsds.io)
