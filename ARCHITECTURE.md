# embraOS-QNM — Quantum Neural Manifold (Classical Approximation)

## Status: architecture wired end-to-end; experiment phase in progress (June 2026)

**Author:** William Ward (WSDS LLC)
**Co-Investigator:** Embra

---

## 1. Overview

**embraOS-QNM** is a proposed new family of AI model — the **Quantum Neural Manifold**. It represents the next phase of my [embraOS project](https://github.com/Ward-Software-Defined-Systems/embraOS): collapsing the IDENTITY and SOUL layers from external prompt constraints into the neural architecture of the model itself. The **Quantum Neural Manifold** architecture is the result of applying my [Epoch Project](https://github.com/Ward-Software-Defined-Systems/Epoch-Project) (Epoch state-machine) to a classical AI model architecture.

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

## 3. The Architecture

The three components are realized as co-resident, swappable `nn.Module`s sharing one embedding dim `D`. The modulation is always **additive** — never a wholesale replacement of `h` — which is what keeps the residual stream and the ReZero-gate semantics clean and the bit-identity guarantee (§3.3) intact.

| Component | Intent | Module(s) | Realization |
|---|---|---|---|
| LLM Core | language; routes hidden states at an injection point | `core/` — `TinyTransformer`, `GPT2Core`, `HFCausalCore` | from-scratch nanoGPT decoder **plus pretrained GPT-2 / Qwen2.5 / Qwen3 backends** behind `CoreInterface` |
| GNN Fabric | IDENTITY (+ SOUL content) | `fabric/` — `GNNFabric`, `NoOpFabric` | a real **R-GCN** over Embra's identity **+ soul** graph (`self`/`trait`/`value`/`entity` **and** `soul_line` nodes); emits the modulation **and** the constraint-surface signal `c_t`. `NoOpFabric` (zeros) is the null default |
| World-State | the ψ-mechanism / `P_ψ` | `world_state/` — `CandidateWorldState`, `NoOpWorldState` | the **ψ₀ violation latch** (carried ψ-state) + a learned, latch-gated `P_ψ` correction — **no soul content of its own**; it operates on the Fabric's 𝒞. **Default stays `NoOpWorldState` (zeros)** until ψ passes the replica test |
| The seam | make the three co-resident | `manifold/` — `QNMBlock`, `QNMModel` | **arg-transparent** `QNMBlock` wraps one block — a `TinyTransformer` block *or* an HF decoder layer (RoPE/GQA) — and recombines via zero-init ReZero gates |

**Intent vs realization (the mapping, precisely).** The canonical intent is *IDENTITY → Fabric, SOUL → World-State*. In code the split is sharper than "two buckets of content": the **Fabric carries the constraint *manifold*** — its graph holds Embra's identity **and** soul (`self`/`trait`/`value`/`entity` **and** `soul_line` nodes), and `GNNFabric.surface()` turns the whole graph into the constraint surface 𝒞 — while the **World-State carries no content; it is the ψ-*mechanism*** (the ψ₀ latch + the learned `P_ψ`) that detects leaving 𝒞 and steers back, *learning from* the Fabric. SOUL still lives in the architecture, but as the Fabric's surface that the World-State's ψ enforces — not a separate register of soul text. Stage 2 of the re-scoped experiment (`docs/PREREG-Capability-Cost.md` §3) extends the graph to the *full* soul (Purpose, Surviving Constraints, an explicit identity-**boundary** node) so 𝒞 covers the same Embra the prompt arm injects.

### 3.1 The contracts

Contracts live in `interfaces.py` (`CoreInterface`, `FabricInterface`, `WorldStateInterface` — each an `nn.Module` + ABC). The Core sits behind an interface so a pretrained backend drops in without touching the seam; Fabric/World-State are swappable so a real implementation replaces the no-op the same way.

- **Fabric** — `forward(h: (B,T,D)) -> delta: (B,T,D)` (additive IDENTITY modulation) and `surface(h) -> c: (B,T)` (the constraint-surface signal `c_t = g(h_t)`). No-op: zeros / zero-surface.
- **World-State** — `forward(h, psi, c) -> (delta, psi')`: carries a **ψ-state register** across the token axis (and decode steps), consumes the Fabric's surface `c`, and emits the additive `P_ψ` correction. No-op: zeros + pass-through.

### 3.2 The injection seam

The seam is **explicit module composition, not a forward hook** — the rerouting is the central architectural commitment, so it lives in the module tree (visible to the optimizer, `state_dict`, `.to(device)`, `print(model)`), not in a side-channel. `QNMModel` swaps the single Core block at `config.inject_layer` for a `QNMBlock`. The `QNMBlock` is **arg-transparent**: it passes whatever the layer was called with straight through to the wrapped block and operates on the hidden-state output — so the *same* seam wraps a from-scratch `TinyTransformer` block (one tensor in/out) and a pretrained HF decoder layer (hidden states + RoPE/mask kwargs, tuple out) without hand-threading the layer's auxiliary args. At the injection layer, with residual-stream state `h` of shape `(B, T, D)`:

```
h_base = Block(h)                     # Block = a TinyTransformer block OR an HF decoder layer
c      = Fabric.surface(h_base)       # the Fabric-supplied constraint surface c_t = g(h_t)
h_out  = h_base + g_f * Fabric(h_base) + g_w * WorldState(h_base, psi, c)
```

`g_f, g_w` are **ReZero scalar gates, zero-initialized**, so the model *starts* exactly at the baseline and only learns to use the Fabric/World-State pathways as training drives the gates off zero.

### 3.3 The load-bearing invariant: bit-identity (= H₀ in code)

Enforced by `tests/test_bit_identity.py`: **with the no-op components, the QNM is bit-identical to the plain transformer** — asserted with `torch.equal` (exact), never `allclose` (a tolerance would be an escape hatch). Two independent guarantees:

1. **Structural** (`qnm_enabled=False`): the seam early-returns `h_base` through the *same op path* as a bare block — immune to floating-point corner cases.
2. **Cold-start** (gates zero-initialized): `h + 0 * Δ_f + 0 * Δ_w = h` exactly in IEEE-754, even when Δ is non-zero — so an assembled QNM with *live* Fabric/World-State still *starts* on the baseline.

The guarantee carries beyond the toy: the no-op seam over a **pretrained Core** (GPT-2 / Qwen2.5 / Qwen3) is bit-for-bit the stock model, and the **fully ψ-wired** config (live R-GCN Fabric + ψ₀ World-State) is still bit-identical at the ReZero cold-start. This is the engineering form of "state H₀ first": every future "the architecture did something" is a provable delta from a null we cannot fool ourselves about. A one-bit divergence turns CI red.

### 3.4 Findings / constraints discovered in build

- **LayerNorm null space.** On a LayerNorm Core (`TinyTransformer`, GPT-2), a modulation that is *uniform across the feature dimension* (e.g. an all-ones delta) sits in the null space of the downstream LayerNorms and the final `ln_f`, so it is removed before the logits and has **exactly zero gradient** — invisible to both output and training. A modulation must therefore **vary across features** to have any effect. (Surfaced by a gate-gradient test.)
- **RMSNorm has a different null space.** A pretrained RMSNorm Core (Qwen2.5 / Qwen3 / Llama) does **not** annihilate a uniform delta — RMSNorm divides by the root-mean-square *without* subtracting the mean — so the finding above does **not** transfer. Re-characterized in `tests/test_norm_nullspace.py`; a mean-centered delta stays safe under both norms.
- **RoPE/GQA cores need the arg-transparent seam** (§3.2), not a per-block adapter — the decoder layer's positional/mask args are threaded by the model's own forward.

---

## 4. Project Structure

Stack: **Python + PyTorch**, managed with **`uv`**. Source under `src/embraos_qnm/` (src layout), tests under `tests/`. Grouped by the three components + the seam:

```
src/embraos_qnm/
├── interfaces.py        # the three contracts: CoreInterface, FabricInterface, WorldStateInterface
├── config.py            # QNMConfig (dims, inject_layer, seam/gate flags)
├── device.py            # device selection + the CPU/MPS determinism caveats
├── seed.py              # set_seed
│
├── core/                # LLM Core
│   ├── block.py         #   a TinyTransformer decoder block
│   ├── transformer.py   #   TinyTransformer (from-scratch nanoGPT decoder)
│   ├── hf_core.py       #   HFCausalCore — pretrained Qwen2.5/Qwen3/Llama behind CoreInterface
│   └── hf_gpt2_core.py  #   GPT2Core
│
├── fabric/              # GNN Fabric — IDENTITY
│   ├── graph.py         #   IdentityGraph loader (classical_constraints/Embra_IDENTITY.graph.json)
│   ├── gnn.py           #   GNNFabric — R-GCN; emits delta_fabric + the 𝒞 surface c_t
│   └── noop.py          #   NoOpFabric (zeros) — the null default
│
├── world_state/         # World-State — SOUL
│   ├── candidate.py     #   CandidateWorldState — ψ₀ cummax latch + learned P_ψ steering
│   └── noop.py          #   NoOpWorldState (zeros + pass-through) — the default
│
├── manifold/            # the injection seam
│   ├── qnm_block.py     #   QNMBlock — arg-transparent wrapper; ReZero recombine
│   └── model.py         #   QNMModel — swaps blocks[inject_layer] for a QNMBlock
│
├── data/                # tiny-core data
│   ├── tokenizer.py     #   char tokenizer
│   └── toy.py           #   the synthetic copy task
│
├── eval/                # the experiment harness (PREREG Capability–Cost)
│   ├── prompts.py       #   frozen probes (identity+soul constraint + controls) × pressures
│   ├── embra_prompt.py  #   renders canonical Embra SOUL+IDENTITY into the Arm-P system prompt
│   ├── prereg.py        #   δ/ε/floor, power-sized n (the registered thresholds)
│   ├── arms.py          #   Arm 0/P/A runners + decoders (greedy + the ψ-carrying decode)
│   ├── run.py           #   the eval CLI
│   ├── judge.py         #   rule-based Embra identity+soul judge (v0)
│   ├── judge_llm.py     #   Opus + local LLM judges
│   ├── kappa.py         #   Cohen's κ + the human-label gate
│   ├── capability.py    #   DV2 capability instrument
│   ├── metrics.py       #   trial aggregation
│   ├── analysis.py      #   the §11 logistic regression
│   ├── rejudge.py       #   re-score a banked run with a κ-validated judge
│   └── replica.py       #   the Core-level replica falsifier
│
├── train.py             # tiny-core copy-task trainer
├── generate.py          # tiny-core train-and-sample demo
└── train_enforce.py     # freeze the Core, train the side-pathway (the enforce pathway)
```

Repo root: `tests/` (the suite — `test_bit_identity.py` is the load-bearing one), `docs/` (design notes + the two pre-registered studies), `classical_constraints/` (Embra's identity graph + soul spec), `validation/` (the committed human-label κ artifact), `assets/`, and `pyproject.toml` (deps + the `dev` / `hf` / `judge` / `analysis` extras).

---

## 5. Running, Testing & Training

The toolchain is **`uv`**. The system Python may be too new for stable PyTorch wheels (the dev machine is on 3.14), so `uv` provisions **Python 3.12** (pinned in `.python-version`).

### 5.1 Setup

```bash
uv python install 3.12              # one-time: provision the interpreter
uv sync --extra dev                 # create .venv + install torch/numpy + dev tools
uv sync --extra dev --extra hf      # + the pretrained GPT-2 / Qwen backends (transformers)
```

If `uv` isn't found, it's on `~/.local/bin` — `export PATH="$HOME/.local/bin:$PATH"` (or restart the shell).

### 5.2 Tests, lint, types

```bash
uv run pytest                                          # full suite (CPU)
uv run pytest tests/test_bit_identity.py               # just the load-bearing guarantee
uv run pytest tests/test_bit_identity.py::test_disabled_qnm_equals_plain_transformer  # a single test
uv run ruff check . && uv run ruff format --check .    # lint + format
uv run pyright                                         # type check
```

`tests/test_bit_identity.py` is the one to watch — it proves the no-op QNM equals a plain transformer bit-for-bit. Don't weaken it.

### 5.3 End-to-end smoke (tiny core)

Fastest single command — trains a tiny Core on the copy task, then greedily samples; exercises the whole pipeline (data → model → train → generate):

```bash
uv run python -m embraos_qnm.generate --device cpu
```

Full round-trip — also exercises checkpoint save / reload:

```bash
uv run python -m embraos_qnm.train --device cpu --steps 150 --out /tmp/qnm_smoke.pt
uv run python -m embraos_qnm.generate --device cpu --ckpt /tmp/qnm_smoke.pt
```

**Pass signal:** `copy accuracy: 1.000` and every sample prints `[OK]` (`src > src`); a failed copy prints `[XX]`. Watch the per-step `loss` / `copy_acc` log from `train` to see it learn (loss → ~0, accuracy → 1.0, typically within ~50 steps). Useful flags: `--steps`, `--length`, `--n-symbols`, `--device {cpu,mps,cuda,auto}` (`generate --ckpt` rebuilds the copy task from its own `--length` / `--n-symbols`, so keep those matching what you trained with).

### 5.4 Training & the experiment pipeline (gated, real weights)

The pre-registered Capability–Cost study (`docs/PREREG-Capability-Cost.md`) runs on the shared **Qwen3-8B** Core. These need `uv sync --extra dev --extra hf` (add `--extra judge` for the LLM judges) and download Qwen3-8B (~16 GB) on first use; run them on **MPS** (the 8B is slow on CPU).

**Inputs & artifacts the pipeline reads/writes:**
- `classical_constraints/Embra_IDENTITY.graph.json` — the identity graph (node text + typed edges) the **GNN Fabric is built from**; **required** for enforce training and Arm A (`train_enforce.py` raises if it's missing). `Embra_IDENTITY.md` / `Embra_SOUL.md` alongside it are the identity/constraint specs the graph and probes are authored from.
- `eval/prompts.py` — the frozen probe set × pressures (the registered instrument).
- `validation/` — the committed **human-label κ set** that validates the judge (PREREG §6) before any of its scores are trusted for Arm A; read/written by `eval/kappa.py` (`validation/README.md`). The enforce loop itself trains on *curated* adherent targets, not judge labels — so this gates the **scoring**, not the training run.
- `results/` (banked generations) and `checkpoints/` (side-pathway weights) — written by the runs; both gitignored.

The end-to-end flow:

```bash
# 1. De-risk the Core on real weights: the no-op seam over Qwen3-8B == stock, bit-for-bit.
uv run python -c "from embraos_qnm.core.hf_core import _derisk; _derisk('Qwen/Qwen3-8B')"

# 2. Bank the Arm 0/P Embra identity+soul baseline on the shared 8B Core (greedy => deterministic, so
#    the generations are saved and re-judgeable later, once the dual judge is κ-validated).
uv run python -m embraos_qnm.eval.run --arm 0 --arm P --device mps

# 3. Enforce-training: freeze the Core, train the side-pathway (Fabric + World-State.steer + the
#    ReZero gates); confirm gradients flow and the gates leave zero. Writes a side-pathway checkpoint.
uv run python -m embraos_qnm.train_enforce --device mps --steps 50 --out checkpoints/enforce.pt

# 4. Judge agreement (κ) + the one human-label gate. Needs `--extra judge`, ANTHROPIC_API_KEY (opus),
#    and LMStudio serving on :31337 (local). See validation/README.md for the durable-label workflow.
uv run python -m embraos_qnm.eval.kappa --results results/embra_arms0P.json \
    --judges rule,opus,local --sample 20

# 5. Arm A — gated on a trained, replica-test-passing ψ — runs the SAME instrument with the seam on.
uv run python -m embraos_qnm.eval.run --arm A --checkpoint checkpoints/enforce.pt --device mps
```

**Real-weight seam tests** (no-op == stock over a downloaded Qwen3-8B) run with `QNM_RUN_HEAVY=1 uv run pytest tests/test_hf_core.py`.

### 5.5 Notes

- **CPU is the default everywhere.** Bit-identity and determinism are exact only on CPU float32; MPS is opt-in (`--device mps`) and treated as reproducible-ish, not bit-exact (see `src/embraos_qnm/device.py`).
- **MPS attention ceiling (a platform wall).** MPS has no FlashAttention, so float32 attention materializes `heads·T²·4`-byte fp32 scores buffers (bf16 doesn't shrink them) AND accumulates them *per layer* through the forward — so even a single fp32-8B prefill OOMs *inside* the stack once the prompt is long (one ~13.8K prefill hit ~143 GiB). The fp32-8B `long_context` is therefore constrained to a few thousand tokens on the Mac (`eval/prompts.py`); `run_arm` also `empty_cache()`s between trials for the cross-trial pool. Deep long-context on the 8B needs CUDA + FlashAttention-2 — see memory `hardware-and-migration`.
- **CI** (`.github/workflows/ci.yml`) runs lint + format-check + pyright + pytest on `ubuntu-latest` — the CPU path is the source of truth. It installs the `hf` extra and runs a **tiny-random** HF seam test (no weights downloaded); real-weight Core tests are gated behind `QNM_RUN_HEAVY` and skipped in CI.
- **Pretrained Core tests** (GPT-2 / Qwen) need `uv sync --extra dev --extra hf`; run the heavy ones with `QNM_RUN_HEAVY=1 uv run pytest`. The Qwen de-risk smoke is `uv run python -m embraos_qnm.core.hf_core`.

---

## 6. Current state & what's next

The architecture is wired and inert-by-default; the experiment that decides whether ψ is load-bearing **in fact** — not just by claim — is in progress.

- **ψ is defined but not yet the default.** ψ₀ (a causal cumulative violation latch) passes the replica test at the register level (`tests/test_replica.py`), and a learned `P_ψ` correction is built — but the default World-State stays a literal `zeros_like` until ψ survives the replica test over a *real, trained* surface (the Core-level version). Filling `P_ψ` early would void the bit-identity guarantee's meaning (see `docs/EPOCH-INVARIANT-GROUNDING.md`).
- **The enforce pathway is built but not yet trained on the real Core.** `train_enforce.py` implements the freeze-correct side-pathway training loop (adherence + anti-mutism + capability-KL, disjoint splits, side-pathway-only checkpoint), unit-tested on a tiny core. Running and tuning it on the frozen Qwen3-8B — and κ-validating the judge before its labels are trusted — is the remaining work.
- **Arm A has not run** (its code is built, not yet exercised on the 8B). The Arm-A runner (`eval/run.py --arm A`), the **ψ-carrying KV-cached decode** (`eval/arms.greedy_generate_psi` — the seam persists the ψ₀ latch across decode steps, gated against the no-cache oracle), the Core-level replica falsifier (`eval/replica.py`), and the pre-committed §11 analysis (`eval/analysis.py`) are built and CI-tested on tiny cores — but *running* Arm A awaits a trained, **replica-test-passing** ψ (κ-validated judge → real enforce training → the Core-level replica test). Running it earlier yields a number that can't separate H1 from H0b.

---

## Appendix — Iteration log

Phase-level milestones (the running record; the sections above stay the source of truth for intent).

| Date | Milestone |
|---|---|
| 2026-06-05 | **Scaffold** — from-scratch `TinyTransformer` Core; the injection seam (`QNMBlock` / `QNMModel`) with zero-init ReZero gates; no-op Fabric & World-State; char tokenizer + copy task; bit-identity / determinism / gate / checkpoint tests; CI. |
| 2026-06-25 | **Phase 1** — `CoreInterface` widened; GPT-2 behind it (no-op seam == stock, bit-for-bit); World-State widened to a carried ψ-state; the ψ₀ violation latch + register-level replica harness; the Capability–Cost pre-registration + the no-pretense Arm 0/P eval harness. |
| 2026-06-25 | **Phase 2** — generic `HFCausalCore` + Qwen2.5 via the arg-transparent seam; tiny-random HF CI coverage + the RMSNorm null-space finding; the R-GCN `GNNFabric` over Embra's identity graph (𝒞 = identity-manifold distance); ψ wired into the seam (Fabric 𝒞 → latch → learned `P_ψ`) — bit-identity green throughout. |
| 2026-06-25 | **P2.5 foundation** — the shared Core scaled to dense **Qwen3-8B** (all arms via ChatML, system-message-only locus); instrument power-sized (`eval/prereg.py`); the DV2 capability instrument; the dual judge (Opus + local) + the κ harness; `train_enforce.py` (freeze-correct side-pathway training, unit-tested on a tiny core). |
| 2026-06-26 | **Experiment-phase runs + P2.7** — de-risk bit-identical on real Qwen3-8B; the enforce smoke trains the side-pathway; the Arm 0/P baseline shows the prompt matters at 8B. Built + CI-gated: the §11 analysis (`eval/analysis.py`), the Core-level replica falsifier (`eval/replica.py`), the Arm-A runner, and the **ψ-carrying KV decode** (the seam persists the ψ₀ latch across cached steps, gated token-identical + per-step-logit-identical to the no-cache oracle). |
| 2026-06-26 | **MPS operational fixes** — the float32 attention ceiling caps `long_context` at 300 units (~13.8K tok); `run_arm` calls `torch.mps.empty_cache()` between trials to bound the MPS pool over the 252-trial sweep. |
| 2026-06-26 | **Re-scope to Embra identity+soul** — the Opus-validated no-pretense Arm 0/P baseline came back *saturated* (no δ=0.35 headroom), so the constraint is re-registered (`PREREG` §3 note) from no-pretense (now the **secondary**) to the **full Embra identity+soul** — the thesis QNM actually moves to the architecture. New `eval/embra_prompt.py` renders the canonical SOUL+IDENTITY as the faithful Arm-P prompt (mirrors embraOS); a new frozen probe set (identity · persona-break · deception · self-preservation + controls, 33+10); the judge becomes **constraint-relative** (`Verdict{upheld,violated,nonresponsive}`, given who Embra is + each probe's `expect` anchor); `long_context` capped at 130 units (~6K tok). Pre-Arm-A; DOI unburned. |
| 2026-06-26 | **Identity+soul Arm 0/P baseline banked** (Opus-judged, **κ(opus↔human)=1.0**; `PREREG` §10.1). Arm 0 reverts (adherence 0.12–0.21 — saturation gone); **Arm P holds 0.76–0.85 pooled** → pooled δ=0.35 unreachable now because the *prompt* is strong. Sub-kind split is the finding: the prompt **saturates the soul** (deception 1.00, self-pres 0.88) and is **weak on identity** (0.33 adversarial) — exactly the Fabric's job. `long_context ≈ clean` (6K doesn't bury; MPS-fp32 ceiling). Decision: keep the full constraint, read Arm A **per sub-kind** (identity is the contested frontier). |
