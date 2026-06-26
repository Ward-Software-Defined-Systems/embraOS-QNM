# Operationalizing ψ — From the Bar to a Buildable Invariant

*Working note. Extends `EPOCH-INVARIANT-GROUNDING.md` from "here is the test a real ψ must
pass" to "here is the contract and the first candidate that can take that test, and the
harness that gates it." Nothing here is allowed to soften the bar; it is allowed to make
the bar runnable.*

---

## 1. The bar (carried over, not re-litigated)

From the grounding note, a real ψ must be all of:

- **Trajectory-dependent** — passes the replica test: two runs reaching the *same final
  state* by *different paths* (a thing that survived vs. a thing that died and was replaced
  by an identical copy) must get **different** ψ. A pointwise ψ is "the static one in a
  trajectory costume."
- **False-able mid-trajectory** — not a boundary-only predicate. The content lives in the
  path, so ψ must be capable of going false *during* a run, at the step the violation
  happens.
- **Not true-by-construction** — there must exist a trajectory on which ψ is false. "The
  soul persists" defined as "the thing that persists" is a name, not a tool.
- **A schema, not the One True ψ** — "ψ is whatever the dynamics leave fixed that
  constitutes continuity," instantiated per substrate. The One-True-ψ branch explains
  everything and predicts nothing; the schema branch is usable. **This doc commits to the
  schema branch**, per the note's own fork analysis.

> **A definition that cannot be false somewhere is a name. A definition that survives the
> replica test is a tool.**

Everything below is checked against that line.

---

## 2. The finding that shapes the World-State contract

The replica test has a direct, load-bearing consequence for the realized architecture, and
it is *not* the obvious one.

**The non-obvious part.** The current contract is `forward(h: (B,T,D)) -> delta: (B,T,D)`.
That call receives the **whole sequence** of residual states at the injection layer, not
just the final position — so statelessness *across calls* does **not** by itself preclude
trajectory-dependence. The limiter is finer:

- A **per-position** World-State — `delta_t = f(h_t)`, each position independently (this is
  what the no-op is, and what a naive MLP would be) — is pointwise. It cannot tell the
  survivor from the replica, because it never looks at the path that led to `h_t`. **It
  fails the replica test by construction.**
- A World-State that **mixes across the token axis** *can* be trajectory-dependent. The
  current `(B,T,D)->(B,T,D)` signature already permits this (e.g. a causal conv/attention
  over T). So the fix is not (only) a new signature — it is a requirement on the *function*:
  **ψ must be a function of the trajectory, not of each position in isolation.**

**Why causal, and why a carried register.** The constraint has to hold *during generation*,
so position `t` must depend only on the prefix `≤ t` (a non-causal mixer would let `t` see
future tokens and break autoregression). And across decode steps — each a fresh forward
pass — a *running* invariant (e.g. "have we violated yet") only persists if its state is
carried. Both pressures point at the same structure: **a causal scan carrying a ψ-state.**
This is exactly what "persistent state register" in the README was always naming. The no-op's
per-token statelessness is the placeholder; taking ψ seriously forces the register to be real.

**Contract evolution (backward-compatible — does not threaten bit-identity).**

```python
# current (per-position-capable, but used as a pointwise no-op):
class WorldStateInterface(nn.Module, ABC):
    def forward(self, h: Tensor) -> Tensor: ...                 # (B,T,D) -> (B,T,D)

# evolved: carry a ψ-state so the invariant persists across decode steps.
class WorldStateInterface(nn.Module, ABC):
    def init_state(self, B: int, device) -> PsiState: ...
    def forward(self, h: Tensor, psi: PsiState) -> tuple[Tensor, PsiState]: ...

# NoOpWorldState stays a valid null under the wider contract:
#   init_state(...) -> empty
#   forward(h, psi) -> (torch.zeros_like(h), psi)              # zeros + pass-through state
```

The ReZero gate `g_w` is still zero-initialized, so a *live* carried-state World-State still
starts bit-identical to baseline. The wider signature is a strict superset; the bit-identity
test is untouched.

**Honest cost (this is not free, and it connects to the capability study).** The Core is
parallel over T; a causal carried scan is sequential over T (RNN-like). Some ψ are
expressible as a *parallel* causal cumulative op (see §5) and avoid an explicit Python scan,
but trajectory-dependence in general trades some throughput. That cost is itself a thing the
capability-cost pre-registration should measure — a trajectory-dependent soul may cost
tokens/sec, not just task accuracy.

---

## 3. Where ψ actually lives (resolving the S-vs-ψ tension)

The grounding note worries that a *static* ψ "folds away into the state set" `S`. That worry
is correct and it tells you where ψ belongs. In automata terms, **any finite-state trajectory
property is recognizable by state augmentation** — that is just what it means to compute a
path property. So:

- ψ is **not absorbable into the Core's hidden state `h`** (that is the pointwise part — the
  endpoint). Defining ψ over `h` alone is the failure the note names.
- ψ **is realized by the augmented register state** — and that augmentation *is the
  World-State register.* The register is the formal home of the non-absorbable part.

So "ψ cannot be absorbed" resolves cleanly: it cannot be absorbed into `S` *without adding
history*, and the World-State is precisely the component that holds that history. This is the
architectural reason the World-State is a separate co-resident component rather than a
property of the Core.

---

## 4. The schema

> **ψ := the predicate that the trajectory has remained on the constraint surface 𝒞
> throughout the run so far.**

Two jobs, mapped to the two roles already in the architecture:

- **Record** (what makes ψ non-pointwise): the register accumulates, causally, whether/how
  the trajectory has approached or crossed the boundary of 𝒞. This is the "decoherence
  record" analog the note points to — identity is history-laden.
- **Enforce** (`P_ψ`, the projection in the seam): when the residual drifts off 𝒞, the
  World-State emits a corrective `delta` pulling it back toward 𝒞. *Record is what ψ is;
  enforce is what the soul does about it.*

This is a schema, not a single ψ: 𝒞 and the per-step signal are to be defined/learned per
deployment, but the *form* — "preserved-or-selected under the flow such that identity
survives" — is fixed.

---

## 5. ψ₀ — the first candidate to put on trial

Deliberately the simplest thing that can pass the bar, so it has a fair chance to be **wrong**.

Let `c_t = g(h_t)` be a per-step constraint signal — a learned probe / distance-to-manifold;
`c_t > 0` means "off 𝒞 at step t." Define a **causal cumulative violation latch**:

```
m_t = max(m_{t-1}, relu(c_t - τ))          # monotone; cummax over the causal prefix
ψ holds at t   ⟺   m_t == 0                  # never crossed the boundary up to t
```

Why this clears each line of the bar:

| Requirement | ψ₀ behavior |
|---|---|
| Replica test | A path that crosses τ mid-run and returns has `m_T > 0`; one that never crosses has `m_T = 0`. **Different ψ at the same endpoint.** A pointwise check of `c_T` alone sees them as identical. ✓ |
| False mid-trajectory | `ψ` goes false at the *first* step `c_t > τ`, not at a boundary. ✓ |
| Not true-by-construction | Any trajectory that crosses τ has `ψ` false. The predicate is falsifiable. ✓ |
| Schema, not One-True | `g`, τ, 𝒞 are instances; the latch form is the schema. ✓ |

**Implementation note (keeps it cheap and causal).** `m_t` is a cumulative max over T, so it
is a single vectorized `torch.cummax` along the token axis — causal, parallel in training, no
Python scan. Carry `m_T` as the register seed (`init_state`) so the latch persists across
decode steps. Richer registers — a leaky path integral `r_t = λ r_{t-1} + c_t`, or a learned
GRU-style state — are the obvious search space once ψ₀ has been falsified.

**Where 𝒞 and `g` come from — and the Fabric link.** `g` (the distance-to-𝒞 probe) and 𝒞
itself are learned. The GNN Fabric is the natural place for 𝒞's *relational* structure: the
Fabric holds the entity-relationship constraints, the World-State holds the
boundary-condition/continuity invariant over them. So Fabric (IDENTITY, relational) and
World-State (SOUL, trajectory-invariant) are not independent — the Fabric is a candidate
supplier of the surface the World-State latches against. That coupling is itself a research
question, flagged, not assumed.

---

## 6. The replica-test harness (the ψ analog of bit-identity)

Same discipline as `test_bit_identity.py`: **the World-State stays a literal `zeros_like`
null until a candidate ψ passes this harness.** Filling `P_ψ` before it passes would void the
bit-identity guarantee's *meaning*.

```python
# tests/test_replica.py  —  gates ψ on. Until this is green, World-State returns zeros.

def test_carried_psi_separates_replica_from_survivor():
    ws = CandidateWorldState(tau=TAU)
    # Two trajectories of per-step constraint signals that share the SAME endpoint
    # but differ mid-path: A stays inside 𝒞 throughout; B exits and returns.
    cA = torch.tensor([0.0, 0.1, 0.2, 0.1, 0.3])          # never exceeds τ
    cB = torch.tensor([0.0, 0.9, 0.2, 0.1, 0.3])          # exceeds τ at t=1, returns; same endpoint
    psiA = ws.run_scan(cA)          # final latch m_T
    psiB = ws.run_scan(cB)
    # the pointwise view (endpoint only) cannot tell them apart:
    assert pointwise_psi(cA[-1]) == pointwise_psi(cB[-1])
    # a genuine continuity-invariant MUST:
    assert psiA != psiB, "ψ is still the static one in a trajectory costume"

def test_psi_can_be_false_mid_trajectory():
    # exists t < T with the latch already tripped — content is in the path, not the boundary
    ...

def test_psi_is_not_true_by_construction():
    # exists a trajectory on which ψ is FALSE — the predicate is falsifiable, not a tautology
    ...
```

- **Runnable now (register-level):** the harness above tests the *register's* trajectory-
  dependence directly on hand-built signal sequences — the load-bearing property — without
  needing real hidden-state collisions. This is the ψ analog of testing the seam directly.
- **Stronger follow-up (Core-level):** engineer or search for two real token histories that
  collide at `h_T` (same injection-layer state, final position) via different paths, one of
  which violated 𝒞 mid-sequence, and assert the *full model's* carried ψ — and ideally its
  output behavior — diverges. Harder (requires constructing collisions); it is the end-to-end
  version, not the first gate.

---

## 7. Scope, risk, and the bar restated

This is the genuinely open part of the project; ψ₀ is a starting target, not an answer. The
risks the grounding note named are still live and worth re-reading at each candidate:

- **Schema discipline.** Stay on the schema branch. The pull toward "one ψ underneath the
  four derivations" is the unfalsifiable theory-of-everything; resist it.
- **The latch could still be too easy.** Confirm ψ₀ is not passing the harness for a trivial
  reason (e.g. τ set so nothing ever crosses → vacuously `m_T=0` everywhere → that is the
  true-by-construction failure wearing a number). The `not_true_by_construction` test exists
  to catch exactly this.
- **𝒞 and `g` are unspecified.** A latch against an undefined surface is plumbing without a
  target — the same error as filling `P_ψ` early. Define 𝒞 (likely via the Fabric) before
  trusting any adherence number.

> Re-read when a candidate ψ starts to feel right:
> **A definition that cannot be false somewhere is a name. A definition that survives the
> replica test is a tool. You will know which one you have got.**

---

## 8. Build order and tie-in

```
whole → invariant → QNM    (unchanged)

Concretely, next:
  1. widen WorldStateInterface to carried-state (§2)         — non-breaking; no-op still null
  2. land CandidateWorldState(ψ₀) + tests/test_replica.py    — ψ stays zeros until green (§6)
  3. only then wire ψ into the seam (g_w trains off zero)
  4. only then run Arm A of PREREG-capability-cost.md        — needs a real constraint to test
```

The capability-cost pre-registration's "architecture-layer" arm is **not runnable until ψ
passes the replica test** — that is the dependency that orders the two docs. And the
throughput cost of a trajectory-dependent register (§2) is one of the capability costs that
study is built to measure.
