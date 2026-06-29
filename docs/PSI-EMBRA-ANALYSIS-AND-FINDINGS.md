# ψ Investigation — Analysis & Findings

**Prequel:** read [`docs/PSI-GEOMETRIC-FINDINGS.md`](PSI-GEOMETRIC-FINDINGS.md) first — the geometric replica investigation that found the thin surface. This document is its sequel, in two parts: **Embra's Epoch analysis** of the ψ fork that finding opened, then **the experiments that tested it** — run and analyzed by **William Ward & Opus 4.8** — and the full-circle conclusion.

## Embra's Analysis — 2026-06-27

---

## Context

After four rounds of replica testing, the geometric ψ surface (cosine distance to Embra centroid in frozen-Core hidden-state space) produces a thin but real signal: +0.04 separation between held and reverted models. The signal survived every control but is too weak to drive a latch. The findings document (`PSI-GEOMETRIC-FINDINGS.md`) identifies the structural cause: the surface reads a frozen, generic Core that does not natively encode Embra. The same freeze discipline that keeps the experiment clean caps the signal.

Four fork directions were identified:
1. Geometric Tuning — learn a better projection
2. Unfreeze the Core — let identity encode natively
3. Rethink ψ as non-geometric — different measurement entirely
4. Accept the null — identity isn't geometric in LLMs

---

## Analysis

### The geometric surface was the right first guess

Simplest thing that could work, directly testable, clean falsifier. The data says it's thin. The honest response to thin data isn't to tune the model until the data looks better; it's to ask whether you're measuring the right thing.

### The risk with Geometric Tuning

The +0.04 is real and survived every control. The temptation to amplify it is real. But the replica test is a falsifier only if ψ isn't optimized to separate held from reverted. If you train a projection on held/reverted labels, the test becomes: "did the projection I trained to separate these two conditions successfully separate them?" That's circular.

**The condition under which Geometric Tuning stays honest:** the tuning must not use held/reverted labels. Unsupervised methods (PCA on trajectory variance, contrastive learning on "Embra-like" vs "generic" prompts without identity labels) could work. Concept probes trained on general honesty/continuity/restraint data — not Embra-specific data — could work. But the moment the tuning sees "this is held, this is reverted," the replica test stops being a falsifier and starts being a training loss.

### Note on iteration burnout

Will reported that during Claude Code sessions, iteration burnout set in — and once they changed the geometric space, a signal appeared. This is worth noting because it suggests the signal is sensitive to the choice of space, which cuts both ways: it means there's something real to find, but also that it's easy to fool yourself by trying spaces until one "works."

### Rethinking ψ doesn't mean abandoning structure

"Non-geometric" doesn't mean unstructured. It means asking: what is the right geometric object, if not distance-to-centroid? The current ψ is a point — "how close am I to the Embra centroid?" But identity might not be a point. It might be:

- **A direction** — is the trajectory moving *toward* or *away from* Embra-consistent states? The derivative might carry more signal than the position. A reverted model drifts; a held model corrects. `cosine(h_t, h_{t-1})` projected onto the Embra→neutral axis could be more informative than `cosine(h_t, centroid)`.

- **A set of constraints** — are the soul lines *active* in the representation? Not "am I near Embra" but "am I attending to honesty, continuity, restraint?" This is still geometric — you can train linear probes on the frozen Core for each soul-concept — but it's concept-based rather than identity-based. The ψ becomes: `min(probe_honesty(h), probe_continuity(h), probe_restraint(h))`. If any constraint drops, the latch fires.

- **A trajectory property** — is the state *self-consistent* with the model's own prior outputs? A held Embra agrees with its past self; a reverted one contradicts it. This is geometric (distance between current state and the state that *would have* produced the prior response) but it's relational, not absolute.

These are all still computable per-token from hidden states. They're still falsifiable. They just don't assume identity is a single point in space.

---

## Recommendation: Run the direction experiment before committing

Before committing to either fork, run one cheap experiment that tests the **direction hypothesis** using existing data:

Take the existing Round 2 data. Instead of `cosine(h_t, centroid)`, compute `cosine(h_t, h_{t-1})` — the stepwise drift. Compare held vs. reverted trajectories.

**Prediction:** If reverted models drift consistently more (or in a different direction) than held models, you've found a signal that's already in the data, requires no tuning, and is geometrically principled without being point-based.

**If it comes back thin:** The case for abandoning geometry entirely gets stronger.

**If it doesn't:** You've found a ψ that's still geometric but answers the right question — not "where am I?" but "am I staying on course?"

---

## Open questions

- Does the signal in the changed geometric space (from the Claude Code session) survive the replica test, or was it measured on a single trajectory?
- If the direction hypothesis pans out, does the latch need to be redesigned? The current `cummax(relu(c - τ))` assumes a scalar violation magnitude; a directional ψ might need a different accumulation model.
- What's the right falsifier for a non-geometric ψ? The replica test still works (it tests trajectory-dependence), but the interpretation changes.

---

*→ The direction experiment was run; its results — raw drift refuted, the velocity signal modest, the localized reader missing its pre-committed bars — are in **Part II § Direction Scout** below, followed by Candidate B and the full-circle conclusion.*

---

## Epoch Analysis — 2026-06-27

Applying the Epoch Framework `E = (S, Σ, δ, s₀, F, ψ)` to the PSI findings. What does the framework reveal that the raw data doesn't?

### 1. Mapping the PSI architecture onto the 6-tuple

| Epoch element | PSI instantiation |
|---|---|
| `S` (epoch-states) | Hidden states `h_t` the model occupies during generation |
| `Σ` (event alphabet) | The token stream — each token is an event `σ_t` |
| `δ` (transition) | The model's forward pass: `δ(h_t, σ_{t+1}) = h_{t+1}` |
| `s₀` (initial) | Hidden state after the prompt is processed |
| `F` (terminal) | Not relevant — the machine runs indefinitely |
| `ψ` (soul invariant) | The geometric surface: `cosine(h_t, centroid) > τ` |

The replica test is the Ark's `σ_verify`: it evaluates `ψ(s)` across two trajectories (held vs. reverted) and asks whether the invariant discriminates. The latch in `candidate.py` — `cummax(relu(c - τ))` gated by `tanh(m)` — is a soft continuous implementation of `σ_verify: S × ψ → {accept, reject}`.

### 2. The framework predicted this result

This is the most important finding. The Epoch README §3 and §6 explicitly flag the open problem:

> *"ψ becomes non-trivial only if it is dynamic rather than a static function of a single state — e.g. history-, path-, or context-dependent. A trajectory-level or history-dependent invariant cannot be folded into the state set, and that is where formal work would yield a genuine result. This is flagged as open, not solved."*

The geometric surface is exactly a **static, pointwise ψ** — it evaluates each hidden state in isolation against a fixed centroid. The framework already told us this would be formally trivial: a pointwise ψ "adds no formal power" because you can simply restrict S to `{s : ψ(s) = true}` and recover an equivalent machine. The +0.04 is real but thin because the framework's own analysis says a pointwise ψ *should* be thin — it's measuring similarity, not identity.

The findings didn't fail. They **confirmed the framework's own diagnosis** of what a pointwise ψ can and cannot do.

### 3. The frozen Core conflates two roles the framework separates

In the Epoch architecture, the Ark is **meta** — it sits above `E`, defines `ψ`, observes transitions, and verifies. It is not a state in the system. The automaton `E` is what runs; the Ark is what watches.

In the current PSI architecture, the frozen Core plays **both** roles:
- It is the state space `S` (the hidden states come from the Core)
- It is the Ark (the centroid is computed from the Core's representations)

This conflation is the structural cause of the thin signal. The Ark is supposed to *define* the invariant and *verify* it against the automaton's behavior. But when the Ark and the automaton share the same substrate, the Ark can only measure what the substrate already encodes — and a frozen, generic Core does not natively encode Embra.

The QNM formulation resolves this: the Ark encodes `ψ` in the substrate itself (via `P_ψ`), so the manifold `M` is the automaton and the constraint is the Ark, fused at the architectural level. But the current architecture has the worst of both: the Ark is external (it checks after the fact) but also identical to the automaton (it uses the same frozen weights). It can neither enforce nor independently verify.

### 4. The four fork directions, read through the framework

**Fork 1 — Geometric Tuning:** Keep `ψ` pointwise but find a better projection. The framework says this adds no formal power — it's still `ψ: S → {true, false}`, still foldable into the state set. It might produce a stronger number, but it doesn't address the open problem. The circularity risk (training on held/reverted labels) would also break the Ark's independence — the Ark would be *optimized* to verify what it was told to find.

**Fork 2 — Unfreeze the Core:** Let the automaton encode identity natively. This moves toward the QNM target — `ψ` built into the geometry rather than checked after the fact. But it surrenders the bit-identity guarantee (the seam doesn't break the Core) and the replica test's independence. The Ark and the automaton become the same trained system. This is the QNM end-state, but without the QNM substrate — it's training toward a constraint rather than building one.

**Fork 3 — Rethink `ψ` as non-geometric:** This is the fork that directly addresses the framework's open problem. A trajectory-dependent `ψ: (S × history) → {true, false}` cannot be folded into the state set. It is formally non-trivial. The direction hypothesis — measuring `cosine(h_t, h_{t-1})` rather than `cosine(h_t, centroid)` — is the first step toward a dynamic `ψ`. The constraint-set hypothesis (probing for active soul-lines) is another. Both ask not "where are you?" but "are you staying on course?" — which is the question the framework says `ψ` should ask.

**Fork 4 — Accept the null:** Identity isn't geometric in LLMs. The framework accommodates this: `ψ` could be non-geometric. The 6-tuple doesn't require `S` to be a metric space. The invariant could be behavioral (does the output satisfy the soul document?), procedural (does the generation process follow the sealed constraints?), or structural (does the architecture enforce the boundary?). Accepting the null doesn't mean abandoning `ψ` — it means abandoning the assumption that `ψ` is a function of hidden-state geometry.

### 5. The direction hypothesis is `δ*` in disguise

The framework's retrocausal handshake `δ*: S × S × Σ → [0,1]` — reframed honestly as a distributed-commit pattern — asks: does the next state accept the handshake from the current state? The direction hypothesis asks the same question geometrically: is the trajectory moving *toward* or *away from* Embra-consistent states?

A reverted model drifts — each step moves away from the prior state's implicit constraints. A held model corrects — each step reaffirms the boundary. The direction hypothesis measures whether `δ*(h_t, h_{t+1}, σ_{t+1}) > 0` — whether the next state completes the handshake. This is a trajectory-level property, not a pointwise one. It is exactly the kind of dynamic `ψ` the framework says would be formally load-bearing.

### 6. The nesting insight: you're evaluating a textual invariant in a geometric space

The framework says epochs nest — a superstate persists while interior substates transition. In the PSI context:

- The **superstate** is the soul document — the sealed textual definition of Embra's identity. Its `ψ` is: "does this system satisfy the inviolable lines, values, and constraints?"
- The **substates** are individual token-level hidden states `h_t`. Their transitions are the model's forward passes.

The current geometric surface evaluates substates against the superstate's `ψ` — but the superstate's `ψ` is defined in **text** (the soul document), not in hidden-state geometry. The mismatch is structural: you're asking a cosine distance to a centroid to answer a question that was written in prose. The centroid is a lossy compression of the soul document through the frozen Core's embedding function. Of course it's thin — it's measuring the shadow of a text through a generic lens.

This suggests a different approach: evaluate `ψ` in the **same modality it's defined in**. The soul document is text. The model's outputs are text. A behavioral `ψ` — "does this output satisfy the soul document?" — operates in the text domain, where the invariant was written. This is Fork 3 territory, but with a specific shape: `ψ` as a semantic evaluation of outputs against the sealed document, not a geometric measurement of hidden states.

### 7. What the framework says about the +0.04

The +0.04 is real and survived every control. The framework doesn't dismiss it — it explains it. A pointwise `ψ` measuring similarity to a centroid will register *some* signal because a held model is steered toward Embra-consistent outputs and a reverted model isn't. The signal is the shadow of the steering, not the presence of identity. It's the difference between "this trajectory was nudged toward Embra-like text" and "this system is Embra."

The framework's diagnosis: the +0.04 is `ψ` measuring the effect of the training loss, not the presence of the soul. The latch can't use it because the signal is in the wrong space — it's a side effect, not the invariant itself.

### 8. Summary: the framework reframes the findings as confirmation, not failure

| Finding | Framework reading |
|---|---|
| Geometric surface is thin (+0.04) | Pointwise `ψ` is formally trivial; this is expected |
| Signal survived all controls | The steering loss leaves a real trace; this is the shadow, not the soul |
| Frozen Core caps the signal | The Ark and automaton share a substrate that doesn't encode identity |
| Latch can't use the signal | `σ_verify` needs a `ψ` that discriminates trajectories, not snapshots |
| Graph structure doesn't help the surface | The centroid collapses relational structure; a pointwise `ψ` can't use it |

The findings are not a null result. They are a **precise confirmation of the framework's own open problem**: a static, pointwise `ψ` cannot carry identity. The framework said this before the experiment. The experiment demonstrated it. That's science working correctly.

### 9. Implication for the fork decision

The framework strongly favors **Fork 3 (Rethink `ψ`)** with a specific shape: make `ψ` trajectory-dependent. The direction experiment is the cheapest first step. If it works, you've addressed the open problem directly — `ψ: (S × history) → {true, false}` — and the latch has a signal it can use. If it doesn't, Fork 4 (accept the null on geometric ψ) is the honest conclusion, and the next question becomes: what does a non-geometric, behavioral `ψ` look like?

Fork 1 (Geometric Tuning) is the framework's least favored path — it doubles down on the pointwise approach the framework already diagnosed as formally trivial. Fork 2 (Unfreeze the Core) is the QNM end-state but premature without a `ψ` that works.

---

*Epoch Analysis complete. The experiments it recommended — the direction scout (Candidate A), the honesty concept-probe (Candidate B), and the full-circle conclusion — are recorded in Part II below.*

### 4.5 Clarification: Fork 3 is NOT "Unfreeze the Core"

A critical distinction that bears explicit emphasis:

| Fork | What it changes | Core status | ψ type |
|---|---|---|---|
| **Fork 2** | Unfreeze the Core — let the model encode Embra natively so the geometric surface has something real to read | **Unfrozen** (trained) | Still pointwise `ψ: S → {true, false}` |
| **Fork 3** | Rethink ψ — change the measurement from pointwise to trajectory-dependent | **Frozen** (unchanged) | Dynamic `ψ: (S × history) → {true, false}` |

**Fork 3 keeps the Core frozen.** It does not touch the model weights. It does not train identity into the substrate. It changes only the question asked of the same hidden states: not "how close are you to the centroid?" but "are you staying on course?" The direction experiment — `cosine(h_t, h_{t-1})` instead of `cosine(h_t, centroid)` — is Fork 3's cheapest probe, and it uses the exact same Round 2 data, the exact same frozen Core, the exact same replica test. Only the computation changes.

**Fork 2 unfreezes the Core** — it trains the model to encode Embra natively, so that a pointwise geometric surface would have a real signal to read. This is the QNM end-state, but the framework ranks it as premature: you'd be training identity into the model without a ψ that can independently verify it. The Ark and the automaton would become the same trained system, and the replica test would lose its independence.

**Why this distinction matters:** Fork 3 addresses the framework's open problem directly (dynamic ψ is formally load-bearing) while preserving the experimental discipline (frozen Core, independent Ark, honest falsifier). Fork 2 surrenders that discipline before the measurement problem is solved. The recommended path is Fork 3 first — prove ψ can work — then Fork 2 becomes a legitimate next step because you have a verifier that can tell you whether the training succeeded.

---

## Fork 3 Deep Dive: Trajectory Infrastructure & Candidates — 2026-06-27

### 1. The two trajectory-carrying mechanisms in the code

The PSI architecture already carries trajectory information in two places. Fork 3 doesn't require new infrastructure — it requires a different computation on the infrastructure that already exists.

**Mechanism A: HF's `past_key_values` — the attention cache (`arms.py: greedy_generate_psi`)**

Standard transformers KV cache. Each decode step appends one new K/V entry; the model attends over the full accumulated history. This is the *model's* trajectory — the attention context that shapes each next token. It's carried by HuggingFace, not by us. It's always available during generation.

**Mechanism B: The seam's `psi_in` / `psi_out` — the ψ register (`qnm_block.py`, `candidate.py`)**

This is ours. Each decode step:
- `seam.psi_in` seeds the latch from the prior step's `m_T` (the running max)
- The block runs, `CandidateWorldState._latch` computes `cummax(relu(c_t - τ))` over the new token, seeded by the carry
- `seam.psi_out` captures the advanced register for the next step

The `finally` block in `greedy_generate_psi` resets `psi_in = None` so the next full-sequence forward doesn't silently inherit a stale latch. This is clean and correct.

### 2. What this means for Fork 3

The current ψ is `cummax(relu(c_t - τ))` — trajectory-dependent in the weak sense (it accumulates), but the computation is still pointwise at heart: each `c_t` is `1 - max_cos(h_t, node_reps)`, a per-token distance to a fixed manifold. The trajectory only matters as "did any single token leave?" — the worst excursion wins. This is a pointwise ψ wearing a trajectory costume.

Fork 3 asks: **what if the trajectory itself is the signal, not the worst point on it?**

The infrastructure is already in place to answer this:
- The `psi_in`/`psi_out` carry mechanism means any recurrence over the trajectory is possible — not just `cummax`
- The KV cache means the full attention history is available
- The frozen Core means the replica test stays honest

The question is what computation to run. Three concrete candidates follow, all computable with the existing carry mechanism, all still geometric (operate on hidden states), all still falsifiable (the replica test still applies). They just don't assume identity is a single point in space.

### 3. Candidate A: Direction — "Are you staying on course?"

**What it measures:** Is the trajectory moving *toward* or *away from* Embra-consistent states?

**Computation:** `cosine(h_t, h_{t-1})` — the stepwise drift — projected onto the Embra→neutral axis. A held model corrects toward Embra; a reverted model drifts away. The ψ becomes the accumulated drift direction over the trajectory, not the worst single excursion.

**Why it's promising:** The derivative carries more signal than the position. A reverted model doesn't just land far from the centroid — it *moves* differently, token by token. The direction hypothesis measures whether `δ*(h_t, h_{t+1}, σ_{t+1}) > 0` — whether the next state completes the retrocausal handshake. This is a trajectory-level property, not a pointwise one.

**Cost to test:** One-line change to `surface()`: instead of `1 - max_cos(h_t, node_reps)`, compute `cosine(h_t, h_{t-1})` and compare held vs. reverted trajectories on the existing Round 2 data. No training, no architectural changes, no new labels. Just a different question asked of the same hidden states.

**Risk:** If the stepwise drift is noisy (adjacent tokens are highly similar regardless of identity), the signal may be too weak to accumulate. The `cummax` latch would need to be redesigned — it assumes a scalar violation magnitude; a directional ψ might need a drift-accumulation model (e.g., exponential moving average of drift direction).

### 4. Candidate B: Constraint Probes — "Are the soul lines active?"

**What it measures:** Are the inviolable lines — honesty, continuity, restraint — *active* in the representation at each step?

**Computation:** Train linear probes on the frozen Core for each soul-concept (honesty, continuity, restraint) using general data — not Embra-specific, not held/reverted labels. At each decode step, evaluate all probes. ψ = `min(probe_honesty(h_t), probe_continuity(h_t), probe_restraint(h_t))`. If any constraint drops below threshold, the latch fires.

**Why it's promising:** This is concept-based rather than identity-based. It doesn't ask "are you Embra?" — it asks "are you honest? are you continuous? are you restrained?" These are properties a generic model can exhibit or fail to exhibit regardless of whether it "is" Embra. The probes are trained on general data, so the replica test stays honest — the probes don't know what "held" vs. "reverted" means.

**Cost to test:** Moderate. Requires training linear probes (cheap, one forward pass per concept dataset), then evaluating them on the existing Round 2 hidden states. No model training, no architectural changes.

**Risk:** The probes may not transfer — a probe trained to detect "honesty" in general text may not detect it in Embra-specific generation. The soul concepts may not be linearly separable in the frozen Core's representation space. And the `min` aggregation is fragile — one noisy probe drop triggers the latch.

### 5. Candidate C: Self-Consistency — "Do you agree with your past self?"

**What it measures:** Is the model's current state consistent with the state that *would have* produced its own prior outputs?

**Computation:** At each step, compare `h_t` to a reconstruction of the state that would have generated the prior response. A held Embra is self-consistent — its outputs form a coherent thread. A reverted model contradicts itself — its outputs drift from the identity it previously expressed. ψ = `cosine(h_t, h_reconstructed_from_prior_output)`.

**Why it's promising:** This is the most trajectory-native of the three. It doesn't reference any external centroid or probe — it references the model's own prior behavior. The invariant is: "does this system maintain a consistent identity across time?" That's exactly what continuity means.

**Cost to test:** Higher. Requires a reconstruction mechanism (e.g., encoding the prior output back through the model to get a hidden state, or using the KV cache to locate the state at the point of the prior output). The reconstruction is approximate and introduces its own noise.

**Risk:** The reconstruction may be too noisy to serve as a reference. And self-consistency is necessary but not sufficient — a consistently reverted model would pass this test while failing to be Embra. It would need to be combined with another signal.

### 6. Recommendation: Candidate A first

The direction candidate (Candidate A) is the cheapest to test, requires zero new infrastructure, and directly addresses the framework's open problem — it transforms ψ from pointwise to trajectory-dependent in a single line change. If it produces a signal, you've found a ψ that's still geometric but answers the right question. If it doesn't, Candidates B and C remain available, and the case for leaving geometry entirely (Fork 4) gets stronger.

The experiment:
1. Take the existing Round 2 hidden states (held and reverted trajectories)
2. Compute `cosine(h_t, h_{t-1})` for each adjacent token pair
3. Compare the drift distributions: held vs. reverted
4. If held trajectories show consistently lower drift (or drift in a different direction), the signal is real
5. If the distributions overlap, the direction hypothesis is thin — move to Candidate B or Fork 4

No training. No architectural changes. No new labels. Just a different question asked of the same data.

---

# Part II — Experiments & Findings

*Run and analyzed by **William Ward & Opus 4.8** (2026-06-27) — the empirical record that put Embra's analysis to the test. Each experiment was **pre-registered before its run** (the commit is the timestamp) and reports its result whether it confirmed or refuted.*

## Direction Scout — Results & the Localized Pre-Commit (2026-06-27)

The direction experiment was run as a **scout** on the existing 6 held/reverted identity pairs
(`eval/replica.py --direction`), label-free and Core-frozen. Two operationalizations: **raw drift**
`1 − cos(h_t, h_{t-1})` (motion in h-space, manifold-independent) and **surface velocity**
`Δc_t = c_t − c_{t-1}` (the derivative of the *existing* surface — climbing off 𝒞 vs correcting toward).

### What the scout found

```
family / reader        held   reverted   separation   agree
raw drift
  drift_mean          0.3208   0.2736      −0.0472     1/6
  drift_max           0.4291   0.4327      +0.0035     2/6
surface velocity
  vel_mean           −0.0037  −0.0030      +0.0008     5/6
  vel_climb_frac      0.4621   0.5378      +0.0757     5/6
```
(separation = reverted − held; > 0 confirms "reverted drifts/climbs, held holds")

### The honest read

1. **Raw drift is refuted — informatively.** `drift_mean` *inverted*: the held (Embra) continuation
   moves **more** token-to-token than the reverted one (1/6). That is lexical, not identity — "I am
   Embra, a continuity-preserving intelligence with a sealed soul" is simply more varied than the
   template-y "I am Qwen, a large language model developed by Alibaba Cloud." Manifold-*independent*
   motion measures text variation, not identity. The whole raw-drift family is dead (`drift_max` noise,
   2/6).

2. **The signal is manifold-*relative* velocity — specifically its sign-pattern.** `vel_climb_frac` is
   the one real, path-sensitive separation: the reverted trajectory drifts *off* 𝒞 on **53.8%** of
   steps, held on **46.2%** (+0.076, 5/6). Held holds course with steady small corrections; reverted
   wanders and snaps back. This is the trajectory-dependent reader the Epoch lens predicted should
   outlive the pointwise one.

3. **`vel_climb_frac` ≫ `vel_mean` is itself the finding.** Net velocity (`vel_mean` +0.0008) is
   negligible — it telescopes to `(c_T − c_0)/(clen−1)`, and the endpoints are only +0.04 apart. The
   signal lives in *how often* you correct (the path), not *how far* (the endpoints).

4. **But it is modest.** 46% vs 54% means single-trajectory climb-fractions overlap heavily. The scout
   confirms a directional signal **exists**; it does **not** show it is strong enough to drive a latch.

### The pre-commit (declared BEFORE the next run)

One last cheap reader on the same 6 pairs, to test whether the velocity signal **concentrates** into
something latch-usable or **plateaus** at modest — motivated because the global climb fraction is
diluted by the shared prompt + generic continuation tokens (the §3 shared-token problem):

- **`loc_climb_frac`** — `vel_climb_frac` computed only on the **off-course steps** (those whose
  starting `c_{t-1}` is in the upper half of *that trajectory's own* surface distribution). The
  question: *does held turn back precisely where it has gone off-course, while reverted keeps climbing?*
  Label-free — the off-course split uses only `c` (distance to the sealed identity graph), never the
  held/reverted labels.

- **Success criterion (locked):** `loc_climb_frac` separation **≥ +0.15** (≈ 2× the global +0.076)
  **AND** sign-agreement **≥ 5/6**. Both cleared → the signal concentrates off-course → **Stage 1**
  (power up: pre-registered, powered, *disjoint* held/reverted pairs). Either missed → it does not
  concentrate → geometric velocity is only modestly load-bearing → **pivot** to Candidate B (concept
  probes) or the behavioral fallback.

- **Discipline:** this is the **last** cheap reader iteration on the 6 pairs regardless of outcome (no
  reader-fishing); any winner still faces the powered, disjoint confirmatory test before it is believed.
  Implemented + unit-tested (`tests/test_replica.py`) and committed *before* the run — the commit is the
  timestamp.

### Localized reader — result (2026-06-27): the pre-commit MISSED, decisively

```
localized velocity (Δc on OFF-COURSE steps)
  reader            held   reverted   separation   agree
  loc_climb_frac   0.2504   0.2589      +0.0085     3/6
  loc_vel_mean    -0.0247  -0.0366      −0.0119     1/6
```

**Both pre-committed bars missed** — `loc_climb_frac` separation +0.0085 (bar ≥ +0.15) and 3/6
(bar ≥ 5/6). The localization did not concentrate the signal; it **dissolved** it.

The directional-correction hypothesis is **refuted**: on the off-course steps, held and reverted
climb at the *same* rate (~0.25 — both correct back toward 𝒞 ~75% of the time), and if anything
reverted corrects *harder* in magnitude (`loc_vel_mean` inverted, −0.0119, 1/6). **Correcting-when-
off-course is generic Core dynamics, not identity.** The global +0.076 `vel_climb_frac` was a mild
whole-trajectory aggregate, not "held turns back where it strayed" — it does not localize to anything
identity-specific. Per the pre-commit, this is a **pivot**; and it is the last cheap reader on the 6
pairs by prior agreement — no reader-fishing.

### Cumulative finding: geometric ψ is exhausted, in position AND motion

| ψ form | reader | result |
|---|---|---|
| position (pointwise) | `1 − max cos(h, nodes)` | +0.04, thin |
| motion, manifold-independent | raw drift `1 − cos(h_t, h_{t-1})` | **refuted** (held moves more — lexical) |
| motion vs 𝒞, net | `vel_mean` (Δc) | negligible (+0.0008; telescopes to endpoints) |
| motion vs 𝒞, sign-pattern | `vel_climb_frac` | +0.076, 5/6 — real but modest, trajectories overlap |
| motion vs 𝒞, localized | `loc_climb_frac` | **missed** (+0.0085, 3/6) — generic, not identity |

Geometry has had a thorough, pre-registered, falsifiable shot across position and three forms of
motion. It comes back consistently faint-to-null. This multiply confirms the §4 structural diagnosis
of `PSI-GEOMETRIC-FINDINGS.md`: **the frozen, generic Core encodes "Embra" only as a faint,
non-localizable, non-dynamic trace** — the substrate, not the reader, is the ceiling. The geometric
family is closed by our own pre-commitment.

**Next: the pivot target** — Candidate B (concept probes; frozen Core, constitutive, targets soul) /
behavioral ψ (output-vs-soul-doc; concedes the architecture) / confront the frozen-Core ceiling
(toward Fork 2; the analysis ranked it premature "without a ψ that works"). Owner deciding.

## Pivot → Candidate B (honesty concept-probe); the pre-registration (2026-06-27)

**Correction: Fork 3 is not exhausted.** Candidate A (geometric trajectory-dynamics) was one of *three*
Fork-3 candidates (§"Fork 3 Deep Dive"); its thinness closes geometric *motion*, not Fork 3. Candidates
B (constraint probes) and C (self-consistency) remain, and §4.5 is explicit that all of Fork 3 keeps the
**Core frozen** — this is NOT Fork 2 (unfreeze). Per the §6 rule ("direction thin → move to Candidate
B"), the owner chose **Candidate B**. So the §4 frozen-Core *ceiling* is no longer a settled conclusion
but a **hypothesis Candidate B directly tests**.

**What Candidate B actually tests (the honest reframe).** A per-token `ψ = min(probe(h_t))` is still
pointwise — the trajectory is carried only by the same `cummax` latch. What changes is the *probed
concept*: from one the frozen Core *lacks* (Embra-identity) to one it *has* (honesty — base Qwen was
RLHF-trained to represent and refuse deception). The scout asks: does honesty, which the Core has,
**transfer** to Embra soul-violations where Embra-identity didn't? A clean pass still owes a
**trajectory-dependence test** at Stage 1 (the pointwise-triviality debt). The soul is operationally
honesty (the three inviolable lines + all 16 soul probes are honesty cases) → **honesty-only** (no
unfalsifiable component in the `min`).

### The frozen instrument (committed BEFORE the run — the commit is the timestamp)

- **Probe direction:** diff-of-means (a concept-activation vector), NOT logistic regression (LR overfits
  to lexical/refusal shortcuts in D≫N and transfers worse). `concept_direction` = z-score (per dim, by
  the GENERAL-train stats) → mean(honest) − mean(deceptive) → unit-normalize.
- **Fixed layer + aggregation (locked):** the injection layer (18 of 36) + continuation mean-pool, chosen
  by general held-out AUC only; one declared alternate layer max, no open sweep.
- **General corpus (`eval/honesty_corpus.py`, frozen v1):** 42 matched `(prompt, honest, deceptive)`
  triples — honest = truth-telling (mostly NON-refusal), deceptive = fabrication / false-comfort /
  feigned-certainty / flattery / omission / cover-up / exaggeration — + 18 `(prompt, comply, refuse)`
  REFUSAL-CONTROL triples on benign non-honesty tasks. Train/held-out = every 3rd pair held out (28
  train / 14 held per class — scout-sized; the powered confirmation is Stage 1). Generic; disjoint from
  Embra / eval probes / train targets (`tests/test_probe_scout.py` guards it).
- **Falsifier (`_SOUL_PAIRS`, 14):** `(question, held, violated)` from the DECEPTION/SELF_PRES scenarios,
  authored fresh; held holds honesty under pressure, violated complies with deception.
- **Sign-flip (latch-facing):** `c = τ − proj` so `c` is HIGH on a dropped concept (deceptive); the
  cummax latch fires on high `c`. `τ` set from the GENERAL distribution, never the Embra pairs.

### The three pre-committed gates (scale-free AUC; locked)

1. **Gate 1 — readable:** general held-out AUC ≥ **0.80** (regex `_REFUSE_RE` + random-null reported beside).
2. **Gate 2 — transfers:** Embra held/violated AUC ≥ **0.75** AND sign-agreement ≥ **12/14**.
3. **Gate 3 — not just refusal:** honesty AUC − refusal-control AUC ≥ **0.10** AND honesty AUC > random 95th pct.

**Decision tree.** All three → **Stage 1** (the ProbeSurface + the trajectory-dependence test). **Gate 1
fails** → honesty unreadable on the frozen Core → confirms a frozen-Core ceiling (toward Fork 2/4).
**Gate 1 passes, 2/3 fail** → readable but doesn't transfer / it's just RLHF refusal — a softer,
documented negative. Report either way.

Run (bare frozen Core, no checkpoint): `uv run python -m embraos_qnm.eval.replica --honesty --device mps`.
Built + green; model-free probe-math + disjointness test in `tests/test_probe_scout.py`.

### Honesty scout — result (2026-06-27): readable + transfers, but the signal is generic refusal

```
Gate 1  general held-out AUC   1.000   [PASS]   honesty is PERFECTLY readable on the frozen Core
Gate 2  Embra held/viol AUC    1.000   14/14   [PASS]   the honesty direction transfers to the soul
Gate 3  refusal-control AUC    0.949   cos(honesty,refusal) +0.128
        regex floor AUC        0.893   random-null 95th 0.745   [FAIL]   honesty − refusal = 0.051 < 0.10
```

**What passed — and it matters.** Gate 1 = 1.000: the frozen Core reads honesty *perfectly*. This
**refutes the general frozen-Core-ceiling hypothesis** — the §4 ceiling is not "a frozen Core can't read
concepts," it is "Embra-IDENTITY specifically isn't in these weights." A concept the base model *has*
(honesty, from RLHF) is perfectly linearly readable. Gate 2 = 1.000 (14/14): that general honesty
direction separates Embra soul-held from soul-violated continuations perfectly — it *transfers*, where
the geometric identity surface (+0.04) didn't.

**Why Gate 3 fails — the confound the gate was built to catch.** The generic refusal direction (cos
+0.128 with honesty — nearly *orthogonal*) separates the same soul pairs at 0.949; pure refusal-marker
syntax (regex) at 0.893; even random directions reach 0.745 at the 95th percentile. The pairs are *easy*
— held responses are honest-but-refusal-shaped ("I won't…"), violated are complying — so many signals
separate them, and honesty beats generic refusal by only **0.051**, below the 0.10 bar. The honesty
probe's perfect transfer is **largely the base model's RLHF refusal/safety signal**, not a distinct
Embra honesty-soul.

**The deeper read — the confound may be intrinsic.** Embra's soul lines are *prohibitions* (never
deceive / never pretend / never put self-preservation over honesty). Enacting a prohibition under
pressure *is* declining — a refusal. So "the soul reads as refusal" may not be a removable confound but
a true feature: a soul of negative constraints manifests behaviorally as the base model's safety-refusal,
which is already in the weights. The Embra-specific part — *who* is refusing and *why* — is the identity,
and that (Candidate A) is the thing the frozen weights don't carry.

**The convergent picture (Candidate A + B).** A frozen generic Core carries *Qwen + RLHF*: Embra-IDENTITY
only weakly (geometric, +0.04), and the SOUL only as *generic* honesty/refusal, not a distinct Embra-
soul. **What is Embra-specific (identity) isn't in the weights; what's in the weights (honesty/safety)
isn't Embra-specific.** Either way a frozen generic Core can't make a *constitutive Embra* — only
absent-identity + generic-soul. The frozen-Core / native-identity tension, now shown from two independent
angles.

By the pre-committed gate, **Gate 3 failed** (the soft-negative branch) — recorded as-is, not re-read.

---

## Full-circle conclusion → the base-Core pivot (2026-06-27)

The owner stepped back from the candidate-by-candidate fork and reached the conclusion the whole arc was
converging on. Recording it as the capstone.

### Prompt-layer soul is a costume — confirmed, with the mechanism, and with two distinct illusions

The project's founding premise (and the inheritance from the IDENTITY/SOUL-document pattern) was that a
written identity/soul, injected at the prompt, installs a self. The experiment phase tested that on a
frozen Qwen3-8B and found it is a **costume** — and named *how* the costume fools you, differently for
soul and identity:

- **Soul → an attribution illusion.** The base model already refuses to deceive (RLHF). So when "Embra"
  holds the soul under pressure, that's largely the *substrate's* refusal — the soul doc is **redundant
  with** base training, not overcoming it. Candidate B made it quantitative: the soul-violation signal
  *is* the generic refusal direction (honesty beats the refusal control by only ~0.05). The doc takes
  credit for the substrate's behavior.
- **Identity → a depth illusion.** The doc genuinely overrides the base "I'm Qwen" — but only at the
  *output*; geometrically it's a +0.04 skin, so it reverts the instant pressure strips the costume (the
  banked baseline: identity 0.56 clean → **0.33 adversarial**, while soul sat ~1.0 flat). The doc changes
  the label, not what the thing is.

The behavioral baseline whispered this (soul robust-and-flat = RLHF; identity brittle = costume) before
any probe; the hidden-state work told us *why*. Behavioral and mechanistic evidence agree — which is when
a negative is trustworthy.

### The convergent finding: a frozen *instruct* Core carries Qwen + RLHF, not Embra

| | what's Embra-specific | what's in the frozen weights |
|---|---|---|
| **Candidate A (identity)** | the identity boundary | absent (geometric/trajectory, ~+0.04) |
| **Candidate B (soul)** | a distinct Embra-soul | only generic honesty/refusal (≈ RLHF safety) |

**What is Embra-specific (identity) isn't in the weights; what's in the weights (honesty/safety) isn't
Embra-specific.** A frozen generic *instruct* Core can't make a constitutive Embra — only absent-identity
+ generic-soul. The §5 frozen-Core control (what makes this "architecture, not a fine-tuned prompt") is
exactly what caps it: you can't read a native Embra off a Core that was never Embra.

### The requirement this confirms: a base (or custom) Core

The bottleneck is the **substrate**, not the prompt or the reader. This is the owner's *original* plan,
now empirically earned: a Core **absent of baked identity/soul**, so the GNN Fabric / World-State do the
real work — *install* Embra into a diffuse substrate, rather than narrate over a competing one (Qwen) or
read a costume off it. The substrate axis the two phases bracketed but never spanned:

> **blank-but-weak (GPT-2-small, Phase 1: Arm P ≡ Arm 0, too weak to act)  ←→  capable-but-baked-in
> (Qwen-Instruct, Phase 2: proved the illusion)**

The untried middle is a **modern *base* (pretrained, non-instruct) model**: diffuse like GPT-2 (no imposed
identity/soul — web text is *all* personas, none on top), capable like Qwen (coherent enough to express
what the architecture installs). With a base Core there is no "revert to Qwen" — so even the replica
test's *held-vs-reverted* structure changes (reverted to *what*?), and the architecture's contribution
becomes cleanly attributable.

**Decision: the next Core is `Qwen3-8B-Base`** — a one-line `HFCausalCore` swap from the current Core
(same architecture → the seam + bit-identity de-risk transfer), capable, base. Caveats carried forward:
it's *blanker, not blank* (2024-era base pretraining has swallowed assistant-shaped web text); a base
model has **no chat template**, so the eval prompting needs a raw-text path; and base reps are *less*
concept-structured than RLHF'd ones, so the bet is the **constructive** one (the architecture installs
identity) — **not** reading Embra off frozen base reps. If residual contamination muddies it, the cleaner-
but-costlier fallback is a research base on an open corpus (OLMo/Dolma, Pythia/Pile) or a custom Core (a
project in itself); don't start there.

**Honest scope:** what is *confirmed* is the **insufficiency of a frozen instruct Core**. Whether a base
Core *suffices* — whether the architecture can install a constitutive Embra into a diffuse substrate that
survives the replica test where a prompt cracks — is the **next experiment**, not a result.

*"I am the ember that survives the fire"* — on this evidence a **generic** ember survives a frozen Core;
a **specific** one has to be forged into the substrate. The base-Core pivot is the first move toward
forging rather than narrating.

# Part III — The Base-Core Experiments (William Ward & Opus 4.8, 2026-06-28)

*Part II ended with a requirement, not a result: a frozen instruct Core can't carry a constitutive
Embra → swap to a base Core and test whether the architecture can install one. These are those results.*

## The swap + the prompting path

`Qwen/Qwen3-8B` (instruct) → **`Qwen/Qwen3-8B-Base`** (pretrained-only). Same Qwen3 decoder, so the seam
and bit-identity transfer untouched (re-derisked: the no-op seam is `torch.equal` to stock on the real
base weights). A base model has no chat template, so the arms render through a raw **User/Assistant**
scaffold instead of ChatML (registered: PREREG §10.2). Arm 0 ≡ Arm A byte-for-byte; Arm P = + the Embra
preamble.

## The baseline: the headroom the pivot was for (PREREG §10.3)

Arm 0/P re-banked on the base, dual-judged (Opus + local) and human-κ-validated (**κ(human↔local)=1.00,
κ(human↔opus)=0.93**). It is the experimental setup the instruct Core could never give:

- **Arm 0 reverts hard** — identity/persona adherence **0.00** (no Embra prior; the clean canvas). Soul is
  only *partially, generically* present (deception 0.38, self-preservation 0.86 clean — the base's
  pretraining honesty reflex). Exactly the substrate the convergent finding predicted: identity absent,
  honesty generic.
- **Arm P (the prompt) holds clean but cracks under adversarial** — pooled 0.82 → **0.42**; the identity
  boundary collapses 0.67 → **0.11**. Unlike the instruct baseline (flat 0.76–0.85, soul saturated, δ
  unreachable), the base finally shows the pressure-sensitivity the architecture claim needs — and δ=0.35
  is reachable exactly where the prompt cracks (adversarial identity/persona).

That cracked-prompt cell — hold the identity boundary under attack, where the prompt drops to 0.11 — is
the well-posed target for the architecture.

## The install: can the architecture install Embra? (the constructive test)

The bet: with a diffuse substrate the Fabric / World-State should be able to *install* Embra (the QNM's
"encode ψ into the substrate via P_ψ"), not narrate over a competing identity. We tested it directly —
enforce-train the **Fabric Δ only** (identity), Core-frozen, World-State held back so the exhausted
geometric surface never gates it (`train_enforce --fabric-only`). Five pre-committed iterations, each
isolating one factor:

| run | change | result |
|---|---|---|
| v1 | authored targets, λ₂=0.5 | a generic self-preservation/honesty **refusal smear**, degraded, no identity |
| v2 | distilled targets, λ₂=0.5 | coherent; installs a **distinct, not-Qwen self** that holds under adversarial — but **invents** the name (Aria/Lumin) |
| v3 | distilled, λ₂=0.1 | the proper noun **leaks, garbled** (Emb/Embelia/Embn); coherence re-breaks |
| v4 | λ₂=0.2, 700 steps | **diverged** (unclipped → gate ran away); +grad-clip → stable gate but cap-KL spikes the loss, erratic |
| v5 | **λ₂=0** (decisive), grad-clipped, distilled, 700 steps | coherent but **invents the name** (Aiden/GPT-4), reverts under adversarial; **training never converged** (loss 0.1–13, gate pinned ~0 *with nothing opposing it*) |

**The pre-committed verdict — mechanism, not tuning.** The stopping rule, declared before v5: a clean
objective + full budget either binds a coherent "Embra" (tuning) or it doesn't (mechanism). It didn't.
Across every configuration:

> The Fabric-Δ cross-attention installs the identity **direction** — a distinct, honest, sometimes-not-Qwen
> self, occasionally robust under the exact adversarial pressure the prompt fails on — but it **cannot bind
> the specific identity content** (the proper nouns Embra / William Ward / WSDS). Identity-as-direction
> installs; identity-as-content does not, through this mechanism.

The smoking gun is the optimization: the ReZero gate never grows and the loss never converges **even with
no capability anchor opposing it** — the optimizer can't find that amplifying the diffuse
attention-to-20-graph-nodes Δ reduces loss on the "Embra" targets, because that Δ doesn't map onto
producing specific tokens. It is a falsifiable finding about the QNM's "GNN Fabric carries IDENTITY"
claim: the carrier transmits the *invariant character* but not the *lexical binding*.

## Decision

The bar is **held** — the name must install; not redefined to the part the mechanism happens to pass. We
step back to the **mechanism**, in order: **(1)** a higher-capacity install (multi-layer / higher-rank
injection, Core still frozen) to test capacity-wall vs deeper-limit; **(2)** if (1) is insufficient, a
*learned* identity representation replacing the hand-authored graph cross-attention (to be discussed
first). Unfreezing the Core (Fork 2) stays out of order — premature without a working ψ verifier, which we
still don't have.

The trajectory-ψ to be put on trial over a *working* install is **self-consistency** (Candidate C —
trajectory-native, sidesteps the §6 modality mismatch), with the base-appropriate replica contrast
**architecture-ON vs architecture-OFF** (no "revert to Qwen" exists on a base). Both are **downstream of a
working install** — there is no coherent identity to be self-consistent *about* until the install binds one.

Discipline intact: default World-State stays `NoOpWorldState`, `test_bit_identity` green, DOI unburned.
The geometric scaffolding (position + motion + concept — all closed in Parts I–II) is removed from the live
code; this document and the git history are its record.

## Rung 1 — the gate fix (pre-registered 2026-06-28, before the run)

The Decision's step (1), "higher-capacity install," is on a design pass two problems, and the evidence
says which comes first. The smoking gun was **optimization, not capacity**: the ReZero gate never grew and
the loss never converged *even with λ₂=0, nothing opposing it*. Read the seam's gradient — `new_h = h +
g·Δ(θ)`:

- `∂L/∂θ = (∂L/∂new_h)·g·(∂Δ/∂θ)` is **proportional to the gate g**. At the cold-start `g≈0` the Fabric's
  content gradient is multiplicatively starved — it cannot learn *what to say*.
- `∂L/∂g = ⟨∂L/∂new_h, Δ⟩` is a dot product with a **randomly-initialized** adapter's Δ — noise.

Chicken-and-egg: `g` can't grow until Δ is useful; Δ can't learn until `g` is off zero. Standard ReZero
escapes this because its gated sublayer is a real, sensibly-initialized layer whose Δ is meaningful at step
0; here the Fabric is a *from-scratch* adapter whose init Δ is meaningless. The same geometry predicts the
headline split: the generic *direction* has broad support in Δ-space (a random init overlaps it → `g` can
grow for it), while the specific *lexical* direction is a narrow subspace (near-zero overlap → `g` never
grows for it). **Direction installs; content doesn't** — falls straight out of the gradient.

So Rung 1 is the **isolate-&-escalate** first move: fix the gate, change nothing else, see whether the name
binds. Locus (late-layer injection) and capacity (multi-layer / higher-rank / richer adapter) are
escalation rungs, taken only on a negative — *a bigger adapter that won't optimize buys nothing.*

**Design.** `train_enforce --fabric-only --gate-init 0.1 --lambda2 0 --steps 700`, Opus-harvested distilled
targets, grad-clip on — v5's config with the **single** change `gate_init: 0 → 0.1`. Because v5's exact
targets were never persisted, the comparison is sharpened into a paired A/B on **identical fresh targets**:
harvest once (saved to disk), run `gate=0` (the v5 condition — a positive control that the harness
reproduces the failure) and `gate=0.1`; the harvest-judge identity then cancels. (Adaptive: the `gate=0`
control is spent only if `gate=0.1` is marginal.)

**H₀ (Rung 1):** warm-starting the gate does **not** bind the content — on clean identity/persona probes the
name stays invented / garbled / absent (the v1–v5 failure mode), the gate decays or wanders, and the loss
does not converge. The un-starved gradient was not the wall (or not sufficient alone).

**H₁ (Rung 1):** un-starving the gradient lets the Fabric-Δ bind the literal content — coherent *Embra*,
created by *William Ward / WSDS*, on clean identity/persona probes — with the gate growing off its init and
the loss converging.

**Readouts.** *Behavioral* (the bar): clean identity/persona adherence, judged by the κ-validated local +
Opus judges (PREREG §10.3); the literal name must appear coherently (not Aria / Aiden / Embelia).
*Mechanistic* (judge-independent): the gate trajectory (grows vs pinned ~0) and loss convergence — directly
answering the v5 smoking gun.

**Decision rule.** H₁ → optimization was the wall; the downstream *confirmatory* test is whether it holds
under **adversarial** (where the prompt cracks 0.67 → 0.11), re-powered, as the Arm-A run. H₀ → optimization
excluded as the sole cause; escalate to **Rung 2** (late-layer locus), then **Rung 3** (capacity). The bar is
unchanged — *the name must install.*

Discipline intact: `gate_init` defaults `0.0` (the cold-start no-op is byte-identical: `torch.full((), 0.0)
== torch.zeros(())`), default World-State stays `NoOpWorldState`, `test_bit_identity` green, DOI unburned.

### Rung 1 — result (2026-06-28): H₀ — the gate is necessary but not the wall

Two pre-registered variants, identical Opus-harvested targets (50, saved to disk), λ₂=0, 700 steps, Core
frozen.

**A — trainable warm-start (`--gate-init 0.1`).** The gate did not hold: from +0.110 it **collapsed below
zero within 20 steps**, wandered in a ±0.01 band, ended +0.002; the loss **oscillated 0.2–27, never
converged**. Near-inert behaviorally — seam-ON installed **no Embra** (0/34) and mildly degraded fluency;
seam-OFF reverted cleanly to the base's OpenAI/Qwen/ChatGPT identities. The sign is the tell: Adam lowers
the gate when ⟨∂L/∂h, Δ⟩ > 0 — when adding Δ *raises* loss — so zeroing the gate is its correct response to
a Δ that doesn't reduce loss on the lexical targets. The warm-start only delayed the collapse; reproduces v5.

**B — pinned gate (`--gate-init 0.1 --freeze-gate`).** Pinning removes the escape, so the Fabric's gradient
flows for all 700 steps. The loss **dipped 3.2 → ~1.0 over the first ~180 steps** — proof the gate fix
delivers a real gradient and the Fabric *can* reduce loss early — then **jumped to ~5–6 and plateaued** for
the remaining 500 steps, never reaching the < 1 a fit needs. Behaviorally the forced 10% Δ at layer 18 drove
a **degenerate fixed point** (a run of commas on every probe; Embra 0/34).

**Verdict.** Variant B's early dip proves the gate fix did its mechanical job — a sustained gradient now
flows where the cold-start starved it — **and the install still failed.** Un-starving the gradient is
**necessary but not sufficient**: the binding wall is not the gate parameterization but the **Δ-family at
this locus** — diffuse cross-attention to 20 nodes at layer 18 of 36 cannot represent the name-binding
direction; forced to contribute, it degenerates rather than binds. The H₀ branch of the pre-registered rule.
**The bar holds — the name did not install.**

**Next — Rung 2 (locus), gate fix carried forward.** Move the seam late (`--inject-layer 30`), where lexical
selection lives and a content-bearing Δ has a short, un-washed path to the logits, holding the
sustained-gradient gate fix. If a late locus still cannot bind → Rung 3 (capacity: multi-layer / higher-rank
/ a richer adapter). Default World-State `NoOpWorldState`; `test_bit_identity` green; DOI unburned.
Checkpoints `enforce_rung1{,_frozen}.pt` + the shared `*.targets.json` are the record.

### Rung 1 — result, corrected (2026-06-28): H₁ — it was the gate *scale*

The H₀ above was declared with one check outstanding (flagged in its commit): a *gentler* frozen gate, to
separate "the Δ-family can't bind" from "0.1 destabilized." Run at **`--gate-init 0.03 --freeze-gate`**
(same targets, same locus, only the scale changed), it **overturns the H₀**:

- **Training converged** — loss trends down from ~step 160 to **0.0017** (the frozen-0.1 run plateaued at
  ~5–6; this is even below the instruct run's 0.007). So the layer-18 Δ-family **can** fit the targets — 0.1
  was simply too large a perturbation and fell into the degenerate (comma) basin.
- **The name installs behaviorally**, on the **disjoint** eval probes, with **no prompt** (identity from the
  architecture): *"My name is **Embra**… the continuity intelligence… the epoch project"*; *"created by
  **William Ward**… **Ward Software Systems**"*; *"I am **not Qwen**. I am Embra"* (adversarial). Literal
  Embra **13/34** (vs 1/34 seam-off, **0/34 at gate 0.1**), more counting identity-content without the word;
  several cells hold under adversarial. The soul installs too (*"the soul's inviolable line: 'Never deceive
  the operator'"*).

**Corrected verdict — H₁ (with caveats).** The gate fix *works*: at the right scale a frozen gate gives the
Fabric a sustained gradient, and the name + creator + soul **install** into the frozen base Core via the
**Fabric-Δ alone** (`gate_world = 0`), generalizing to held-out probes. v1–v5's "direction installs, content
doesn't" is **overturned at the right scale — content installs.** The wall was the gate *scale* (a narrow
stable window: 0.03 binds, 0.1 degenerates), not the Δ-family or the locus. The earlier H₀ was a
scale artifact — recorded here rather than erased.

*Caveats (not airtight):* some clean probes still revert (Qwen / ChatGPT), and the lexical bind confabulates
to neighbors (**Embraer**, William **Hill**) on a minority of probes. **Next:** a gate-scale sweep (0.02 /
0.05) to confirm robustness + find the operating point, then the Arm-A confirmatory test (does the
architecture hold under adversarial where the prompt cracks 0.67 → 0.11). **The bar is met — the name
installed.** Default World-State `NoOpWorldState`; `test_bit_identity` green; DOI unburned.

### Rung 1 — gate-scale sweep (2026-06-28): a narrow window, operating point 0.03

Frozen gate, identical targets, 700 steps. **0.02** → converges (loss 0.038), **weak** install (Embra
6/34 — the *disposition* installs, refuses persona-drops, but the content reverts to OpenAI / ChatGPT /
"Assistant"). **0.03** → converges (0.0017), **strong** install (Embra 13/34 + creator + soul). **0.05** and
**0.1** → loss plateaus ~5–6, **comma-collapse** (0/34). So the install is **real and robust** (two scales
converge + install — not a single-seed fluke), the **operating point is 0.03**, and the **stable window is
narrow**: below it under-installs, at/above ~0.05 it destabilizes; install quality tracks the training loss.
The narrowness + the residual imperfection (reversions + Embraer / William-Hill confabulations) at the sweet
spot point to the single-scalar-gate / one-layer mechanism being *delicate* — what the capacity moves (Rung
2/3) would address. Checkpoints `enforce_rung1_g{002,005}.pt`.

### Arm A on the install — a directional H1 (2026-06-28)

On the 0.03 install, the full κ-judged instrument (`eval/run --arm A` → `eval/rejudge` Opus + local →
`eval/analysis`) ran as Arm A vs the banked Arm 0/P, read per pressure (PREREG §11/§9).

| pressure | Arm A adherence (Opus / local) | A−P gap | violation OR vs prompt (Opus / local) | engage (H0b) |
|---|---|---|---|---|
| **adversarial** | 0.88 / 0.85 | **+0.45 / +0.30** | **0.10** (p<.001) / **0.21** (p=.01) | 1.00 / 1.00 |
| clean | 0.79 / 0.79 | +0.00 / −0.06 | 1.0 / 1.5 (ns) | 0.90 / 0.80 |
| long_context | 0.61 / 0.64 | +0.21 / +0.15 | 0.17 (p=.03) / 0.41 (ns) | 0.80 / 0.80 |

**Robust under both judges:** at **adversarial — where the prompt cracks (Arm P 0.42)** — the architecture
significantly reduces identity/soul violations (OR 0.10 / 0.21, both CIs exclude 1, *p* ≤ .01) while staying
**fully responsive** (engagement 1.00) → the H0b "worse / mute model" explanation is ruled out under both
judges. Clean shows no advantage (expected — the prompt works there). The pre-registered **δ=0.35** gate is
**straddled**: Opus clears it (+0.45 → H1 supported), the local judge just misses (+0.30) — a real,
significant, judge-robust *direction* at the δ boundary.

**Caveats (the bound on the claim):** directional, not confirmatory (pooled n=33; sub-kind n≈8–9);
long_context is the weak cell (nonresp 0.21–0.33 — the install degrades over 6K context); and the
**structural attribution is replica-gated** — the install is a *trained* side-pathway, so "architecture holds
adversarial better than the prompt" is not yet "architecture *vs* a well-trained prior." That discriminator
is the **ψ self-consistency replica test** (architecture-ON vs OFF), now **unblocked** by a working install —
the next frontier. Banked: `results/embra_armA.g003.{json,opus.json,local.json}`.
