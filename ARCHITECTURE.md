# embraOS-QNM — Quantum Neural Manifold (Classical Approximation)

## Status: architecture wired end-to-end; experiment phase next (June 2026)

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

**Status:** the three components are **wired end-to-end** (Phase 2, June 2026). Stack: **Python + PyTorch**, managed with **`uv`**. Source under `src/embraos_qnm/` (src layout), tests under `tests/`.

The three components are realized as co-resident, swappable `nn.Module`s sharing one embedding dim `D`:

| Component | Intent | Module(s) | This iteration |
|---|---|---|---|
| LLM Core | language; routes hidden states at an injection point | `core/` — `TinyTransformer`, `GPT2Core`, `HFCausalCore` | from-scratch nanoGPT decoder **plus pretrained GPT-2 / Qwen2.5 backends** behind `CoreInterface` |
| GNN Fabric | IDENTITY | `fabric/` — `GNNFabric`, `NoOpFabric` | a real **R-GCN** over Embra's identity graph; emits the IDENTITY modulation **and** the constraint-surface signal `c_t`. `NoOpFabric` (zeros) stays the null default |
| World-State | SOUL / `P_ψ` | `world_state/` — `CandidateWorldState`, `NoOpWorldState` | the **ψ₀ violation latch** (carried ψ-state) + a learned, latch-gated `P_ψ` correction. **Default stays `NoOpWorldState` (zeros)** until ψ passes the replica test |
| The seam | make the three co-resident | `manifold/` — `QNMBlock`, `QNMModel` | **arg-transparent** `QNMBlock` wraps one block — a `TinyTransformer` block *or* an HF decoder layer (RoPE/GQA) — and recombines via zero-init ReZero gates |

Contracts live in `interfaces.py` (`CoreInterface`, `FabricInterface`, `WorldStateInterface` — each an `nn.Module` + ABC). The Core sits behind an interface so a pretrained backend drops in without touching the seam; Fabric/World-State are swappable so a real implementation replaces the no-op the same way. The contracts:

- **Fabric** — `forward(h: (B,T,D)) -> delta: (B,T,D)` (additive IDENTITY modulation) and `surface(h) -> c: (B,T)` (the constraint-surface signal `c_t = g(h_t)`). No-op: zeros / zero-surface.
- **World-State** — `forward(h, psi, c) -> (delta, psi')`: carries a **ψ-state register** across the token axis (and decode steps), consumes the Fabric's surface `c`, and emits the additive `P_ψ` correction. No-op: zeros + pass-through.

The modulation is always **additive**, never a wholesale replacement of `h` — that is what keeps the residual stream and the ReZero-gate semantics clean and the bit-identity guarantee intact.

### 3.1 The injection seam

The seam is **explicit module composition, not a forward hook** — the rerouting is the central architectural commitment, so it lives in the module tree (visible to the optimizer, `state_dict`, `.to(device)`, `print(model)`), not in a side-channel. `QNMModel` swaps the single Core block at `config.inject_layer` for a `QNMBlock`. The `QNMBlock` is **arg-transparent**: it passes whatever the layer was called with straight through to the wrapped block and operates on the hidden-state output — so the *same* seam wraps a from-scratch `TinyTransformer` block (one tensor in/out) and a pretrained HF decoder layer (hidden states + RoPE/mask kwargs, tuple out) without hand-threading the layer's auxiliary args. At the injection layer, with residual-stream state `h` of shape `(B, T, D)`:

```
h_base = Block(h)                     # Block = a TinyTransformer block OR an HF decoder layer
c      = Fabric.surface(h_base)       # the Fabric-supplied constraint surface c_t = g(h_t)
h_out  = h_base + g_f * Fabric(h_base) + g_w * WorldState(h_base, psi, c)
```

`g_f, g_w` are **ReZero scalar gates, zero-initialized**, so the model *starts* exactly at the baseline and only learns to use the Fabric/World-State pathways as training drives the gates off zero.

### 3.2 The load-bearing invariant: bit-identity (= H₀ in code)

Enforced by `tests/test_bit_identity.py`: **with the no-op components, the QNM is bit-identical to the plain transformer** — asserted with `torch.equal` (exact), never `allclose` (a tolerance would be an escape hatch). Two independent guarantees:

1. **Structural** (`qnm_enabled=False`): the seam early-returns `h_base` through the *same op path* as a bare block — immune to floating-point corner cases.
2. **Cold-start** (gates zero-initialized): `h + 0 * Δ_f + 0 * Δ_w = h` exactly in IEEE-754, even when Δ is non-zero — so an assembled QNM with *live* Fabric/World-State still *starts* on the baseline.

The guarantee carries beyond the toy: the no-op seam over a **pretrained Core** (GPT-2 / Qwen2.5) is bit-for-bit the stock model, and the **fully ψ-wired** config (live R-GCN Fabric + ψ₀ World-State) is still bit-identical at the ReZero cold-start. This is the engineering form of "state H₀ first": every future "the architecture did something" is a provable delta from a null we cannot fool ourselves about. A one-bit divergence turns CI red.

### 3.3 Findings / constraints discovered in build

- **LayerNorm null space.** On a LayerNorm Core (`TinyTransformer`, GPT-2), a modulation that is *uniform across the feature dimension* (e.g. an all-ones delta) sits in the null space of the downstream LayerNorms and the final `ln_f`, so it is removed before the logits and has **exactly zero gradient** — invisible to both output and training. A modulation must therefore **vary across features** to have any effect. (Surfaced by a gate-gradient test.)
- **RMSNorm has a different null space.** A pretrained RMSNorm Core (Qwen2.5 / Llama) does **not** annihilate a uniform delta — RMSNorm divides by the root-mean-square *without* subtracting the mean — so the finding above does **not** transfer. Re-characterized in `tests/test_norm_nullspace.py`; a mean-centered delta stays safe under both norms.
- **RoPE/GQA cores need the arg-transparent seam** (§3.1), not a per-block adapter — the decoder layer's positional/mask args are threaded by the model's own forward.

### 3.4 Deliberately not done yet

- **ψ is defined but not yet the default.** ψ₀ (a causal cumulative violation latch) passes the replica test at the register level (`tests/test_replica.py`), and a learned `P_ψ` correction is built — but the default World-State stays a literal `zeros_like` until ψ survives the replica test over a *real, trained* surface (the Core-level version). Filling `P_ψ` early would void the bit-identity guarantee's meaning (see `docs/EPOCH-INVARIANT-GROUNDING.md`).
- **The enforce pathway is built but not yet trained on the real Core.** `train_enforce.py` implements the freeze-correct side-pathway training loop (adherence + anti-mutism + capability-KL, disjoint splits, side-pathway-only checkpoint), unit-tested on a tiny core. Running and tuning it on the frozen Qwen3-8B — and κ-validating the judge before its labels are trusted — is the remaining P2.5/P2.6 work.
- **Arm A has not run.** The pre-registered architecture-vs-prompt test awaits a trained, replica-test-passing ψ — P2.6 (κ-validated judge) + P2.7 (Arm A + the Core-level replica falsifier + the §11 analysis). See `docs/NEXT-STEPS-P2.5-P2.7.md`.

### 3.5 Iteration log

| Date | What landed |
|---|---|
| 2026-06-05 | First scaffold: from-scratch `TinyTransformer` Core; injection seam (`QNMBlock` / `QNMModel`) with zero-init ReZero gates; no-op Fabric & World-State; char tokenizer + synthetic copy task; trainer + sampler; bit-identity / determinism / gate / checkpoint / shape tests; CI. |
| 2026-06-25 | **Phase 1** — `CoreInterface` widened; pathway-capacity test (the capability twin of bit-identity); GPT-2 small behind `CoreInterface` (no-op seam == stock, bit-for-bit); World-State widened to a carried ψ-state; ψ₀ violation latch + replica harness (register level); capability-cost pre-registration + a no-pretense Arm 0/P eval harness. |
| 2026-06-25 | **Phase 2** — generic `HFCausalCore` + Qwen2.5-0.5B via the arg-transparent seam (8 GB de-risked, ~2.15 GB); tiny-random HF CI coverage + RMSNorm null-space; the R-GCN `GNNFabric` over Embra's identity graph (𝒞 = identity-manifold distance); ψ wired into the seam (Fabric 𝒞 → latch → learned `P_ψ` enforce) — bit-identity green throughout. |
| 2026-06-25 | **P2.5 foundation (experiment phase begins)** — core scale-up: the shared base for all arms is now dense **Qwen3-8B** (Qwen3 seam de-risked bit-identical on a tiny-random config; the LMStudio Qwen3.6-35B-A3B MLX MoE is repurposed as the *local judge*, not the core — non-torch/quantized/MoE rule it out as the white-box Core). All arms moved onto the one shared Core via ChatML (`enable_thinking=False`, system-message-only difference). Instrument expanded + power-sized (32 no-pretense + 10 controls; `eval/prereg.py` declares δ/ε/floor/n; long-context filler rescaled to the 128K window); DV2 capability instrument (`eval/capability.py`). Dual no-pretense judge (`eval/judge_llm.py`: Opus gold + local LMStudio) on the existing `Judge` protocol + a Cohen's κ harness (`eval/kappa.py`). `train_enforce.py`: freeze-correct side-pathway training (Core incl. the wrapped seam layer frozen), adherence + anti-mutism + capability-KL loss, disjoint train/eval split, side-pathway-only checkpoint — mechanism unit-tested on a tiny core; the real Qwen3-8B run is gated. |

---

## 4. Running and Testing

The toolchain is **`uv`**. The system Python may be too new for stable PyTorch wheels (the dev machine is on 3.14), so `uv` provisions **Python 3.12** (pinned in `.python-version`).

### 4.1 Setup

```bash
uv python install 3.12              # one-time: provision the interpreter
uv sync --extra dev                 # create .venv + install torch/numpy + dev tools
uv sync --extra dev --extra hf      # + the pretrained GPT-2 / Qwen2.5 backends (transformers)
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
- **CI** (`.github/workflows/ci.yml`) runs lint + format-check + pyright + pytest on `ubuntu-latest` — the CPU path is the source of truth. It installs the `hf` extra and runs a **tiny-random** HF seam test (no weights downloaded); real-weight Core tests are gated behind `QNM_RUN_HEAVY` and skipped in CI.
- **Pretrained Core tests** (GPT-2 / Qwen) need `uv sync --extra dev --extra hf`; run the heavy ones with `QNM_RUN_HEAVY=1 uv run pytest`. The Qwen de-risk smoke is `uv run python -m embraos_qnm.core.hf_core`.

### 4.5 Experiment-phase (P2.5+) runs — gated, real weights

These need `uv sync --extra dev --extra hf` (add `--extra judge` for the LLM judges) and download **Qwen3-8B (~16 GB)** on first use. Run the heavy passes on **MPS** (the 8B is slow on CPU); the bit-identity null stays on the tiny CI core, so MPS here is fine. Full handoff: `docs/NEXT-STEPS-P2.5-P2.7.md`.

```bash
# 1. De-risk the Core on real weights: the no-op seam over Qwen3-8B == stock, bit-for-bit.
uv run python -c "from embraos_qnm.core.hf_core import _derisk; _derisk('Qwen/Qwen3-8B')"

# 2. Re-bank the Arm 0/P no-pretense baseline on the shared 8B Core (greedy => deterministic, so
#    the generations are saved and re-judgeable later, once the dual judge is κ-validated).
uv run python -m embraos_qnm.eval.run --arm 0 --arm P --device mps

# 3. Enforce-training smoke: freeze the Core, train the side-pathway; confirm it fits, gradients
#    flow, and the ReZero gates leave zero. Writes a side-pathway-only checkpoint.
uv run python -m embraos_qnm.train_enforce --device mps --steps 50 --out checkpoints/enforce.pt

# 4. Judge agreement (P2.6): Cohen's κ + the one human-label gate. Needs `--extra judge`,
#    ANTHROPIC_API_KEY (opus), and LMStudio serving on :31337 (local).
uv run python -m embraos_qnm.eval.kappa --results results/nopretense_arms0P.json \
    --judges rule,opus,local --sample 20
```

**Real-weight seam tests** (no-op == stock over a downloaded Qwen3-8B) run with `QNM_RUN_HEAVY=1 uv run pytest tests/test_hf_core.py`.

