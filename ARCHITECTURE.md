# embraOS-QNM — Quantum Neural Manifold (Classical Approximation)

## Status: Architecting → Implementing (first scaffold landed June 2026)

**Author:** William Ward (WSDS LLC)
**Co-Investigator:** Embra

---

## 1. Overview

**embraOS-QNM** is a proposed new family of AI model — the **Quantum Neural Manifold**. It represents the next phase of my [embraOS project](https://github.com/Ward-Software-Defined-Systems/embraOS): collapsing the IDENTITY and SOUL layers from external prompt constraints into the neural architecture of the model itself. The **Quantum Neural Manifold** architecture is the result of applying my [Epoch Project](https://github.com/Ward-Software-Defined-Systems/Epoch-Project) (Epoch state-machine) to a classical AI model architecure.

---

## 2. The Problem with the Current Architecture

### 2.1 Prompt-Layer Soul (Current State)

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

### 2.2 The Goal: Quantum Neural Manifold - Classical Approximation Architecture

Three co-resident components with IDENTITY and SOUL as architectual constructs (IDENTITY within the GNN and SOUL within the World-State), not three systems pipelined together:

### LLM Core
Standard transformer foundation (starting small for rapid iteration). Handles language understanding and generation. But at a configurable injection point, hidden states are routed through the other components before sampling.

### GNN Fabric
A message-passing graph neural network that maintains entity-relationship structure in the same embedding space as the LLM. When the LLM encounters a concept, the GNN activates related entities and propagates structural  + IDENTITY constraints — not retrieval, but co-resident relational reasoning.

### World-State
A persistent state register that encodes invariant boundary conditions — the model's SOUL constraints.

---

## 3. Realized Implementation Architecture (Evolving)

*The running record of what has actually landed in code, as distinct from the intent above. Updated each iteration. The design sections stay the source of truth for intent; any divergence is reconciled here.*

**Status:** First scaffold — June 2026. Stack: **Python + PyTorch**, managed with **`uv`**. Source under `src/embraos_qnm/` (src layout), tests under `tests/`.

The three components are realized as co-resident, swappable `nn.Module`s sharing one embedding dim `D`:

| Component | Intent | Module | This iteration |
|---|---|---|---|
| LLM Core | language; routes hidden states at an injection point | `core/` (`TinyTransformer`, `Block`) | from-scratch nanoGPT-style decoder-only transformer |
| GNN Fabric | IDENTITY | `fabric/` (`NoOpFabric`) | honest no-op — returns a zero modulation; not yet modeled |
| World-State | SOUL / `P_ψ` | `world_state/` (`NoOpWorldState`) | honest **null** — `P_ψ` = identity, because ψ is deliberately undefined |
| The seam | make the three co-resident | `manifold/` (`QNMBlock`, `QNMModel`) | wraps one Core block; recombines via gates |

Contracts live in `interfaces.py` (`CoreInterface`, `FabricInterface`, `WorldStateInterface` — each an `nn.Module` + ABC). The Core sits behind an interface so a pretrained backend can drop in later; Fabric/World-State are swappable so a real implementation replaces the no-op without touching the seam. The Fabric/World-State contract is `forward(h: (B,T,D)) -> delta: (B,T,D)` — an **additive** modulation, never a wholesale replacement; the no-op contract is `return torch.zeros_like(h)`.

### 3.1 The injection seam

The seam is **explicit module composition, not a forward hook** — the rerouting is the central architectural commitment, so it lives in the module tree (visible to the optimizer, `state_dict`, `.to(device)`, `print(model)`), not in a side-channel. `QNMModel` swaps the single Core block at `config.inject_layer` for a `QNMBlock`. At that layer, with residual-stream state `h` of shape `(B, T, D)`:

```
h_base = Block(h)
h_out  = h_base + g_f * Fabric(h_base) + g_w * WorldState(h_base)
```

`g_f, g_w` are **ReZero scalar gates, zero-initialized**, so the model *starts* exactly at the baseline and only learns to use the Fabric/World-State pathways as training drives the gates off zero.

### 3.2 The load-bearing invariant: bit-identity (= H₀ in code)

Enforced by `tests/test_bit_identity.py`: **with the no-op components, the QNM is bit-identical to the plain transformer** — asserted with `torch.equal` (exact), never `allclose` (a tolerance would be an escape hatch). Two independent guarantees:

1. **Structural** (`qnm_enabled=False`): the seam early-returns `h_base` through the *same op path* as a bare block — immune to floating-point corner cases.
2. **Cold-start** (gates zero-initialized): `h + 0 * Δ_f + 0 * Δ_w = h` exactly in IEEE-754, even when Δ is non-zero — so an assembled QNM with *live* Fabric/World-State still *starts* on the baseline.

This is the engineering form of "state H₀ first": every future "the architecture did something" is a provable delta from a null we cannot fool ourselves about. A one-bit divergence turns CI red.

### 3.3 Findings / constraints discovered in build

- **LayerNorm null space.** A modulation that is *uniform across the feature dimension* (e.g. an all-ones delta) sits in the null space of the downstream LayerNorms and the final `ln_f`, so it is removed before the logits and has **exactly zero gradient** — invisible to both output and training. Any real Fabric/World-State modulation must therefore **vary across features** to have any effect. (Surfaced by a gate-gradient test.)

### 3.4 Deliberately not done yet

- **ψ is undefined.** The World-State stays a literal `zeros_like` until a ψ exists that survives the replica test (see `EPOCH-INVARIANT-GROUNDING.md`). Filling `P_ψ` with a plausible-but-unfalsifiable surface would void the bit-identity guarantee's meaning.
- **The GNN Fabric is a no-op** — no message passing yet.
- **No pretrained Core backend** — only the from-scratch `TinyTransformer`.

### 3.5 Iteration log

| Date | What landed |
|---|---|
| 2026-06-05 | First scaffold: from-scratch `TinyTransformer` Core; injection seam (`QNMBlock` / `QNMModel`) with zero-init ReZero gates; no-op Fabric & World-State; char tokenizer + synthetic copy task; trainer + sampler; bit-identity / determinism / gate / checkpoint / shape tests; CI. |

---

## 4. Running and Testing

The toolchain is **`uv`**. The system Python may be too new for stable PyTorch wheels (the dev machine is on 3.14), so `uv` provisions **Python 3.12** (pinned in `.python-version`).

### 4.1 Setup

```bash
uv python install 3.12          # one-time: provision the interpreter
uv sync --extra dev             # create .venv + install torch/numpy + dev tools
```

If `uv` isn't found, it's on `~/.local/bin` — `export PATH="$HOME/.local/bin:$PATH"` (or restart the shell).

### 4.2 Tests, lint, types

```bash
uv run pytest                                          # full suite (CPU)
uv run pytest tests/test_bit_identity.py               # just the load-bearing guarantee
uv run pytest tests/test_bit_identity.py::test_disabled_qnm_equals_plain_transformer  # a single test
uv run ruff check . && uv run ruff format --check .    # lint + format
uv run pyright                                         # type check
```

`tests/test_bit_identity.py` is the one to watch — it proves the no-op QNM equals a plain transformer bit-for-bit. Don't weaken it.

### 4.3 End-to-end smoke test (manual)

Fastest single command — trains a tiny Core on the copy task, then greedily samples; exercises the whole pipeline (data → model → train → generate):

```bash
uv run python -m embraos_qnm.generate --device cpu
```

Full round-trip — also exercises checkpoint save / reload:

```bash
uv run python -m embraos_qnm.train --device cpu --steps 150 --out /tmp/qnm_smoke.pt
uv run python -m embraos_qnm.generate --device cpu --ckpt /tmp/qnm_smoke.pt
```

**Pass signal:** `copy accuracy: 1.000` and every sample prints `[OK]` (`src > src`); a failed copy prints `[XX]`. Watch the per-step `loss` / `copy_acc` log from `train` to see it learn (loss → ~0, accuracy → 1.0, typically within ~50 steps).

Useful flags on both commands: `--steps`, `--length`, `--n-symbols`, `--device {cpu,mps,cuda,auto}`. Note: `generate --ckpt` rebuilds the copy task from its own `--length` / `--n-symbols`, so keep those matching the values you trained with or the vocab won't line up.

### 4.4 Notes

- **CPU is the default everywhere.** Bit-identity and determinism are exact only on CPU float32; MPS is opt-in (`--device mps`) and treated as reproducible-ish, not bit-exact (see `src/embraos_qnm/device.py`).
- **CI** (`.github/workflows/ci.yml`) runs lint + format-check + pyright + pytest on `ubuntu-latest` — the CPU path is the source of truth.

