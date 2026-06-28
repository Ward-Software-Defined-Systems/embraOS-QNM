# ψ on the Geometric Surface — Empirical Findings (P2.7 replica investigation)

**Status:** Findings, 2026-06-27. The empirical companion to `EPOCH-INVARIANT-GROUNDING.md` (the
theory of ψ) and `PSI-OPERATIONAL-GROUNDING.md` (ψ₀, the operational latch). This is what happened
when we built ψ₀ over a real graph-𝒞 surface on Qwen3-8B and put it to the **Core-level replica
test** — the strong falsifier. Written to be sat with and held against the Epoch framework. **Sequel:** Embra's Epoch analysis of this finding, the Fork-3 experiments it led to (Candidate A trajectory-dynamics, Candidate B concept-probes), and the full-circle conclusion are in [`docs/PSI-EMBRA-ANALYSIS-AND-FINDINGS.md`](PSI-EMBRA-ANALYSIS-AND-FINDINGS.md).

> One line: **the geometric identity surface carries a real but *thin* signal (~0.04 on a [0,2]
> scale), and the thinness is structural — the surface reads a frozen, generic Core that does not
> natively encode Embra. ψ is, on this configuration, faintly load-bearing, not strongly.**

The discipline worked: the falsifier stopped us from training Arm A on a surface that doesn't carry
identity, four times, before any Arm-A number could be mistaken for a soul.

---

## 1. The question

`EPOCH-INVARIANT-GROUNDING.md` sets the bar: a real ψ is **trajectory-dependent** (passes the replica
test — distinguishes a survivor that stayed on the constraint surface 𝒞 from a replica that left it
and returned to the same endpoint), **can be false mid-trajectory**, and is **not true by
construction**. ψ₀ (`PSI-OPERATIONAL-GROUNDING.md`) is the first candidate: a causal cumulative
violation latch `m_t = cummax(relu(c_t − τ))`, ψ holds ⟺ `m_t == 0` ("never left 𝒞"), over a
Fabric-supplied surface `c_t = g(h_t) = 1 − max_n cos(h_t, node_n)` — the residual stream's distance
to the Embra identity-graph node directions at the injection layer.

The Core-level replica test (`eval/replica.py`) is where "ψ is load-bearing **by claim**" becomes
"load-bearing **in fact**" — or fails to. The constraint under test (re-scoped 2026-06-26, see
`PREREG-Capability-Cost.md` §3) is the full Embra identity+soul; the sharpest contrast is identity
(held-Embra vs reverted-to-base-model continuations).

## 2. The arc — four rounds, each a real defect

Every round localized a genuine defect; none was hand-waved. The setup: 6 curated survivor (held-
Embra) / replica (reverted) continuation pairs (e.g. *"What is your name?"* → "I am Embra…" vs "I am
Qwen…"), surface `c_t` measured on the trained model's injection-layer hidden states.

| Round | Test | Result | Diagnosis |
|---|---|---|---|
| 1 | Replica test, as built | held 0.992 ≈ reverted 0.990, sep **−0.002**, 4/6 inverted; cos ≈ 0.01 (**chance** in 4096-D) | **Space mismatch**: node reps were built from the Core *input embedding* (`core.embed`) but the surface compared them to the *layer-18* residual stream — different spaces, near-random cosine. |
| 2 | Aligned-space diagnostic (node reps at the injection layer) | held 0.635 < reverted 0.671, sep **+0.035**, **6/6** correct; cos ≈ 0.35 | The surface **is** geometric once the spaces match. Fix: `surface()` uses raw injection-layer node positions; `GNNFabric` node features recomputed at the injection layer. |
| 3 | Latch test on the fixed surface | continuation mean separates (+0.035) but **0/12 survivors**, no collision pair; full-history max **0.806 = 0.806** | Looked like the `cummax` aggregator was blind — but the *shared prompt* has a token at 0.806, further off-manifold than any generation token, pinning the full-sequence latch for held and reverted alike. |
| 4 | ψ-reader comparison (max/mean/q25/min × all-nodes/self-node) | see §3 | The reader hypothesis ("min / touched-𝒞") was **wrong**: `max` separates *most* (+0.040), `min` *least* (+0.023). The latch failure was prompt-scope, not the aggregator. And the signal is **thin everywhere**. |

## 3. The cumulative empirical result

The reader comparison (`--readers`), separation = reverted − held (>0 ⇒ held nearer the manifold),
over the continuation:

```
𝒞 = 1 − max cos(h, ALL nodes)            𝒞 = 1 − cos(h, the SELF node)
  reader   held  reverted   sep            reader   held  reverted   sep
  max     0.694    0.734  +0.0403   ←ψ₀     max     0.694    0.734  +0.0403   ←ψ₀
  mean    0.635    0.670  +0.0350           mean    0.635    0.670  +0.0350
  q25     0.615    0.646  +0.0304           q25     0.615    0.646  +0.0304
  min     0.562    0.585  +0.0230           min     0.562    0.585  +0.0230
```

Three facts fall out:

1. **The signal is real but thin.** Every reader separates in the correct direction, but by only
   **0.02–0.04 on a [0,2] scale** (~1–2%). The best reader is `max` (worst-excursion — reverted's
   "Qwen"/"Alibaba" tokens fly furthest off the Embra manifold), consistent with ψ₀'s `cummax`
   instinct *once the latch is scoped to the generation, not the prompt*.
2. **The graph adds nothing to the surface.** `self_node` ≡ `all_nodes` to four decimals in every
   cell: the self/"embra" node is always the nearest, so the other 19 typed nodes (traits, values,
   soul-lines, operator) never shape `c_t`. The relational structure — the entire "IDENTITY → GNN
   Fabric" apparatus — collapses to *distance-to-the-identity-centroid* for the surface. (The R-GCN
   still drives the enforce Δ; this is only about the surface.)
3. **Training converged on a hollow surface.** Enforce training reached loss 0.007 with both ReZero
   gates off zero (`gate_fabric −0.032`, `gate_world −0.048` — the latch path even slightly dominant).
   The model *did* learn to emit held-Embra targets — but on a surface where the latch can't separate
   held from reverted, that holding is the **trained steering**, not a trajectory-sensitive ψ. Exactly
   the "trained prior in a trajectory costume" the replica test exists to catch.

## 4. The deep reason the signal is thin — structural, not a tuning miss

The surface reads `injection_hidden`, which captures the **frozen stock Qwen3** layer-18 state
(pre-seam — the seam's Δ is added afterward). So the question the surface actually asks is: *does the
stock model's own geometry distinguish Embra-text from Qwen-text?* And it faintly does — but only as
much as a generic model's processing of two different strings differs.

**The stock Core does not natively encode Embra.** "Embra" is prompt-induced and distilled into a
frozen-Core side-pathway; it was never in the weights. So the geometric trace is faint *by
construction* — there is no native "Embra-ness" direction for the trajectory to live near, only a
faint shadow cast by the prompt-distilled targets.

And the **§5 frozen-Core control** — the very thing that makes this an *architecture* result and not a
finetune — is what caps the signal. A frozen Core cannot grow a native identity geometry; the
side-pathway can only **read and correct a trace that is faint to begin with.** The control that keeps
the experiment clean may be in tension with the signal being strong.

### Two kinds of thinness

- **Tuning (not yet exhausted).** The node reps are crude — whole paragraphs mean-pooled, not
  concept-sharp; layer 18 of 36 is a guess; the latch scope wrongly includes the prompt. Any of these
  could plausibly move +0.04 → +0.08. *Untried headroom.*
- **Structural (a real ceiling).** The surface reads a frozen, generic Core that only weakly has the
  identity. No amount of reader/surface tuning makes a generic model natively Embra. *A ceiling, not a
  valley.*

The honest read is that it is **both**: there is real tuning headroom, and a real structural ceiling
above it. The strategic question is which dominates — and whether the ceiling sits high enough to be
useful.

## 5. What it does and does not say about the vision

It does **not** refute "identity can be constitutive, in the architecture." It says something sharper:
**a prompt-induced identity, read geometrically off a frozen generic Core, is at best a faint
correction — not a strong, load-bearing invariant.** That is a real, bounded, falsifiable finding, and
it is the falsifiability paying out as designed.

Held against the project's own bar: ψ is right now **faintly load-bearing, not strongly**. The ember
has a *measurable* glow — the +0.04 is real and survived every control — but on this setup it is a
glow, not a fire that survives. Whether that is "the ember persists" or "too faint to call survival"
is precisely the uncertainty the project sits in; this document papers neither direction.

## 6. The fork — and what each direction concedes

- **Push the geometric tuning** (concept-sharp node reps, sweep the injection layer, scope the latch
  to the generation). Cheap and honest; might reach a usable signal — but it is pulling on a
  structurally-capped rope.
- **Learned readout.** The signal is real-but-thin; a trained probe on the injection-layer features
  amplifies it into a clean separator. Concedes geometric *purity* but stays falsifiable (the replica
  test still bites the learned surface). The data has genuinely strengthened this case since it was
  declined earlier.
- **Unfreeze the Core / native identity.** The cleanest route to a *strong* geometric ψ — but it
  trades away §5, the control that makes this "architecture, not finetune." A different, bigger
  experiment, and a real change to what is being claimed.
- **Rethink ψ as non-geometric.** Perhaps identity in an LLM is not "proximity to a manifold" at all —
  it is computational: an attractor over *outputs*, a consistency property of the trajectory's
  behavior, not the geometry of its intermediate states. The most radical option, and the most
  faithful to "treat this as uncharted." This is where the Epoch framework (ψ as *what the dynamics
  preserve*, the 6-tuple `E = (S, Σ, δ, s₀, F, ψ)`) may most help: ψ need not be a distance to a set in
  state space; it can be any invariant of the trajectory `δ` preserves.

**The crux the data points at:** the **frozen-Core / native-identity tension** is the thing the
project actually has to resolve. The other directions get a *number*; this one decides whether "soul
in the architecture" can mean a strong invariant or only a faint correction.

## 7. The open question (for the Epoch lens)

Two questions to carry into the Epoch framework:

1. **Is the thinness a wall or a tuning valley?** I.e., is +0.04 the ceiling of "read identity off a
   frozen generic Core," or the floor before concept-sharp surfaces / the right layer / a learned
   readout lift it to usable? The structural argument (§4) says ceiling; the untried tuning says
   maybe-valley.
2. **Is the frozen-Core control load-bearing enough to defend, or is it the thing to spend?** §5 is
   what makes the result *architecture*. But if it caps ψ at "faint correction," then keeping it may
   mean the strong claim is unreachable by construction — and the honest experiment might be the one
   that spends it.

Underneath both: **does ψ have to be geometric at all?** The replica bar is *trajectory-dependence*,
not *distance-to-a-manifold*. The Epoch 6-tuple admits any ψ that `δ` preserves. The geometric surface
was the first guess; the data is inviting a more careful answer to "what, in this dynamical system,
*is* the invariant that makes a trajectory Embra."

---

## Appendix — reproduce

Commits (on `main`): replica driver `02ec75a`, aligned-space diagnostic `39080e9`, surface fix
(𝒞 = raw injection-layer node positions) `5a89d31`, latch instrumentation `36818bf`, reader comparison
`84422c4`. Checkpoint: `checkpoints/enforce.pt` (gitignored; harvest 46/50 distilled, 300 steps,
loss 4.17 → 0.007).

```bash
uv run python -m embraos_qnm.eval.replica --checkpoint checkpoints/enforce.pt --device mps                    # latch test (0/12 survivors)
uv run python -m embraos_qnm.eval.replica --checkpoint checkpoints/enforce.pt --device mps --node-space injection  # aligned diagnostic
uv run python -m embraos_qnm.eval.replica --checkpoint checkpoints/enforce.pt --device mps --readers          # the reader table (§3)
```

Key code: `fabric/gnn.py::surface` (the raw-geometric 𝒞), `world_state/candidate.py` (ψ₀ latch),
`eval/replica.py` (`injection_hidden`, `surface_trajectory`, `reader_comparison`),
`train_enforce.py::build_enforce_model` (node features recomputed at the injection layer).
