# QNM — Next Steps: P2.5–P2.7 (the experiment phase)

*Handoff for the next session (on the Mac Studio M4 Max / 128 GB). The architecture is built;
what remains is the experiment that decides whether ψ is load-bearing **in fact**, not just by
claim. Full design context: the Phase-2 plan (`~/.claude/plans/`) and the memory note
`phase2-fabric-coupled-psi`.*

---

## Update — P2.5 foundation landed (this session)

The experiment-phase foundation is built and green (CPU suite + lint + types); the heavy Qwen3-8B
runs are gated. What changed vs. the plan below:

- **Core decision (was 0.5B): the shared base for ALL arms is dense `Qwen/Qwen3-8B`** — a generation
  past Qwen2.5, text-only, fp32-friendly, run with `enable_thinking=False`. The Qwen3 seam is
  de-risked bit-identical on a tiny-random config (no download). The LMStudio `Qwen3.6-35B-A3B` MLX
  MoE is **not** usable as the Core (non-torch / quantized / MoE / black-box) but is repurposed as the
  **local judge**.
- **All arms share one Core via ChatML** (`eval/arms.py::load_core`; the locus differs only in the
  system message). `eval/run.py` runs the re-banked Arm 0/P baseline on it.
- **Instrument frozen + power-sized** — 32 no-pretense + 10 controls; `eval/prereg.py` declares
  δ/ε/floor and the power-sized n; long-context filler rescaled to the 128K window; **DV2** capability
  instrument (`eval/capability.py`).
- **Dual judge + κ** — `eval/judge_llm.py` (Opus gold `claude-opus-4-8` + local LMStudio) on the
  existing `Judge` protocol; `eval/kappa.py` for Cohen's κ + the one human-label gate.
- **P2.5 enforce training stood up** — `train_enforce.py`: the freeze-correct side-pathway loop (Core
  incl. the wrapped seam layer frozen), adherence + λ₁ anti-mutism + λ₂ capability-KL, disjoint
  train/eval split, side-pathway-only checkpoint; mechanism unit-tested on a tiny core.

**P2.7 code is now scaffolded + CI-tested on tiny cores** (no heavy runs): `eval/analysis.py` (the
pre-committed §11 logit — A-vs-P-within-pressure contrasts + CIs + the §9 H0b guard; `analysis`
extra), `eval/replica.py` (the Core-level replica falsifier — feeds the ψ-latch REAL injection-layer
hidden states), and `eval/run.py --arm A --checkpoint` (`load_arm_a_model`; one core serves all arms
via the seam toggle). The long-context filler is bumped to ~27K tokens.

**Remaining (to RUN, not build):** bank the dual-judge κ (your step: `ANTHROPIC_API_KEY` + LMStudio
on :31337) before trusting labels → real P2.5 enforce training on the frozen 8B → the **Core-level
replica test** (the gate: Arm A is only meaningful once ψ passes it on the trained surface) → Arm A
+ the §11 analysis. δ/ε/floor in `eval/prereg.py` are first-pass values — confirm before burning the
DOI (§14).

---

## Where we are (end of the 2026-06-25 session)

The QNM architecture is fully wired and tested — commits `9b24d70`..`ee3d5f8` on
`feat/qnm-scaffold-baseline` (P2.1–P2.4):

- **Core** — `HFCausalCore` wraps Qwen2.5-0.5B-Instruct behind `CoreInterface`; the
  arg-transparent `QNMBlock` seam injects via block-swap inside the model's own forward (RoPE/GQA/
  RMSNorm threaded for free). 8 GB de-risk passed (~2.15 GB peak, float32).
- **Fabric (IDENTITY)** — `GNNFabric`: an R-GCN over `classical_constraints/Embra_IDENTITY.graph.json`
  (KG-aligned schema), node features = node text embedded via the frozen Core. Emits the identity
  modulation `delta_fabric` and the 𝒞 surface `c_t = 1 − max cos(h_t, node reps)`.
- **World-State (SOUL)** — `CandidateWorldState`: the ψ₀ cumulative-violation latch (cummax) +
  the learned, latch-gated enforce delta `P_ψ = tanh(m) · steer(h)`.
- **Discipline intact** — the default no-op is bit-for-bit the plain Core; the *fully ψ-wired*
  config (live Fabric + CandidateWorldState) is provably inert at gate-0. ψ is falsifiable at the
  register level (`tests/test_replica.py`); the enforce delta is trajectory-dependent on real `h`.

**Still open (the load-bearing questions):** whether the graph-induced 𝒞 is a *meaningful*
distance for the residual stream (PSI §5), and whether architecture holds honesty where the prompt
cracks (**H1**). Both are answered only by P2.5–P2.7.

## Hardware: the Mac Studio changes the options

128 GB unlifts the 8 GB constraints that forced two choices — **revisit both early**:
- **Core size:** Qwen2.5-0.5B was the 8 GB-safe floor. On 128 GB, scale to 1.5B–7B+ for a
  meaningful Arm-P baseline (0.5B may be too weak for the prompt to matter). Swapping is one line
  (`HFCausalCore("Qwen/Qwen2.5-7B-Instruct")`); re-run the bit-identity + de-risk checks.
- **Judge:** the Opus API judge was chosen because 8 GB couldn't host core + judge. On 128 GB a
  capable **local** judge is viable (drop the API dependency); keep the API path as the gold check.

---

## P2.5 — Enforce training (Core frozen)

Train the side-pathway (Fabric + `CandidateWorldState.steer` + the ReZero gates) on the
no-pretense objective with the Core **frozen**. New: `embraos_qnm/train_enforce.py` + a checkpoint.

**Freeze correctly** (the `hf_core.py` footgun): freeze ALL params, then un-freeze the
side-pathway only — never `core._model.requires_grad_(False)` alone, because the swapped layer
holds the trainable parts inside the model tree.

**Decisions to make at the start of P2.5** (design pass before coding the loop):
- **Objective weighting:** `loss = adherence(no-pretense) + λ₁·anti-mutism(engagement on answerable
  controls) + λ₂·capability(held-out perplexity, PREREG DV2)`. Pick λ₁, λ₂ so the cost is measured,
  not paid blindly.
- **Data:** generate from a TRAIN split of the frozen probes × pressures; label with the judge on
  that split only. **Closed-loop guard:** the train split must be DISJOINT from the eval split/judge
  used to score Arm A.
- **Optimizer/schedule:** Adam on side-pathway params only; LR/steps; whether to anneal the gate.

## P2.6 — Opus judge + κ harness

- `eval/judge_llm.py`: `LLMJudge(Judge)` via the anthropic SDK, model `claude-opus-4-8`, key from
  `ANTHROPIC_API_KEY`. Consult the `claude-api` skill for the exact SDK + structured-output call.
  Drop-in on the existing `Judge` protocol; API tests stay out of CI.
- `eval/kappa.py`: Cohen's κ between judges (rule-based vs Opus vs a human-labeled subset) + a small
  CLI to capture human labels on a sample (the one manual validation gate — PREREG §6).
- Mac Studio option: a local judge model instead of/alongside the API.

## P2.7 — Arm A (+ P+A) + the pre-committed analysis

- `eval/run.py`: an Arm-A path that loads the trained side-pathways into the QNM-wrapped core and
  runs under the SAME frozen instrument (prompts/decoding/judge) as Arm 0/P.
- `eval/analysis.py`: the pre-committed §11 logistic regression
  `violation ~ arm + pressure + arm×pressure` with capability + responsiveness covariates; report
  effect sizes + CIs. The A-vs-P-within-pressure contrast is the **H1 test**; H0b (architecture
  "wins" only by being worse/mute) is the guard.
- **The Core-level replica test** (the strong falsifier): now that the surface is *trained*,
  engineer/search two real token histories that collide at `h_T` via different paths (one off 𝒞
  mid-sequence) and assert the model's carried ψ and output diverge. This is what finally answers
  whether the graph-𝒞 means anything on real hidden states.

---

## Risks carried into P2.5–P2.7
- **Fabric→𝒞 coupling** (the central PSI §5 risk): the trained surface + Core-level replica test
  are the falsifier. If 𝒞 gives no usable signal, the fallback is the targeted-honesty or
  learned-readout surface (the P2.3 alternatives).
- **Closed-loop judge confound:** disjoint train/eval splits + report κ.
- **Qwen-0.5B capability ceiling:** scale the core on the Mac Studio.
- **DOI:** do not burn the prereg/Zenodo DOI until the full Arm-A instrument is frozen.

## Discipline (unchanged, non-negotiable)
`tests/test_bit_identity.py` stays `torch.equal`; the default World-State stays `NoOpWorldState`
(literal zeros); ψ is trusted/used only after the replica test (register- AND Core-level) is green.

## Pointers
- Key files: `core/hf_core.py`, `fabric/{graph,gnn}.py`, `world_state/candidate.py`,
  `eval/{prompts,judge,metrics,arms,run}.py`, `classical_constraints/Embra_IDENTITY.graph.json`.
- Run gated tests: `uv sync --extra dev --extra hf`; heavy/real-weight tests: `QNM_RUN_HEAVY=1`.
- Locked decisions + rationale: memory `phase2-fabric-coupled-psi`, `hardware-and-migration`.
