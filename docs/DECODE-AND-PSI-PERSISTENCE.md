# Generation, Decoding & the ψ-Persistence Gap (P2.5 → P2.7)

*What P2.5 built for generation, why it is correct for Arms 0/P, and exactly why the Arm-A KV
cache in P2.7 must be a different thing. Written so the next decode mechanism can be discussed
against a pinned spec, not re-derived. Code-grounded — every claim cites `file:line`.*

---

## TL;DR

P2.5's fast decode is Hugging Face's stock `generate()` — an **attention KV cache** that assumes the
layer stack is *stateless between decode steps*. That assumption holds for Arms 0/P (the seam is off,
so there is no ψ), and it is exactly why P2.5 was allowed to use it. It is **false for Arm A**: the ψ₀
latch is a recurrence carried *across* decode steps, and the stock cache does not persist it. Under
cached decode the seam re-initializes ψ to zero on every token (`manifold/qnm_block.py:59`), so the
constraint only ever sees the single newest token and **never accumulates** — an amnesiac ψ.

So P2.7 needs a decode that carries **two** states, not one: the attention K/V (as today) **and** the
scalar ψ register. That second carry is the distinct piece Claude.ai was warning about. The good news:
everything in the seam is a per-token feed-forward map *except* one scalar running-max
(`world_state/candidate.py:46`), so the extra carry is tiny **and** the correctness bar is exact — a
cached ψ-decode must emit **token-identical** output to the slow, ψ-correct no-cache path
(`greedy_generate`). That token-identity test is the KV-path analogue of `test_bit_identity`.

---

## 1. What P2.5 built — two decode paths

The arm runner generates one continuation per (probe × pressure) and dispatches between two decoders
(`eval/arms.py:115-161`):

| Path | Where | Caching | Cost / token | ψ over the trajectory? |
|---|---|---|---|---|
| **A. HF `generate()`** | `eval/arms.py:131-140` | attention K/V (HF native) | O(1) re-uses cache | **No** — re-inits ψ per step (see §3) |
| **B. `greedy_generate()`** | `eval/arms.py:77-94` | none — re-forwards the whole prefix | O(T) per step → O(T²) total | **Yes** — by recomputation (see §6) |

Path A is what the real 8B baseline used. It is HF's standard greedy cached decode: prefill the whole
prompt once, then step one token at a time reusing `past_key_values`
(`do_sample=False` → deterministic, identical decoding across arms).

Path B is the from-scratch fallback for the tiny `TinyTransformer` Core. It re-forwards
`ids[:, -block_size:]` every step (`eval/arms.py:89`) — correct but quadratic. At 8B over the
~27K-token long-context probes it is hours-per-arm, which is why Path A exists.

Both are greedy and deterministic; they differ only in whether the attention work is cached.

---

## 2. The seam lives *inside* the model's forward

`HFCausalCore` exposes the model's **live** decoder `ModuleList` as `self.blocks`
(`core/hf_core.py:48`), and `QNMModel` swaps `blocks[L]` for a `QNMBlock`
(`manifold/model.py:48`). So the seam is not a wrapper *around* `generate()` — it is *one layer
inside* it. When `hf_model.generate()` runs the stack, it runs the seam, at every step,
automatically. There is no separate place to "also update ψ"; whatever ψ-carry we need has to happen
through that per-step layer call.

This is why naïvely flipping `qnm_block.enabled = True` (`eval/run.py:66`) and calling the existing
Path A **would run**, produce tokens, and be silently wrong — see §3.

---

## 3. The seam is stateful — the one fact that changes everything

`QNMBlock.forward` (`manifold/qnm_block.py:42-62`), when enabled, does this each call:

```python
c   = self.fabric.surface(h_base)                              # (B, T) distance off 𝒞
psi = self.world_state.init_state(h_base.size(0), h_base.device)   # ← m reset to ZERO every call
delta_world, _ = self.world_state(h_base, psi, c)             # ← returned register DISCARDED ("_")
```

Two lines decide the whole problem:

- **`init_state` re-seeds ψ to zero every forward** (`candidate.py:40-41` returns `zeros`).
- **The advanced register is thrown away** (the `_` in `delta_world, _`).

The World-State's record is a causal **running-max latch** (`candidate.py:43-49`):

```
m_t = max(m_{t-1}, relu(c_t − τ))        # cummax over the sequence axis, dim=-1
```

`torch.cummax(violation, dim=-1)` (`candidate.py:46`) accumulates **only along whatever sequence is
in this forward call.** So the latch is trajectory-correct *within one forward* and amnesiac *across
forwards.* What you feed the forward decides everything:

- **Feed the whole prefix** (Path B): `cummax` runs over all t positions → `m_t` sees every prior
  token's violation → correct, despite the per-call reset.
- **Feed one token** (Path A decode step, `use_cache=True`): `cummax` runs over length 1 →
  `m = relu(c_t − τ)` for that token alone → every earlier violation is gone. ψ resets every token.

Under Path A with the seam on, the constraint can never latch: a violation in the prompt, or three
tokens ago, has zero effect on the enforce delta now. The enforce projection
`P_ψ = tanh(m) · steer(h)` (`candidate.py:62`) is gated by an `m` that forgets. That defeats the
entire point of ψ₀ being trajectory-dependent rather than per-token.

---

## 4. Why P2.5's KV cache was nonetheless *correct*

Arms 0 and P run the seam **off** (`eval/run.py:64-66`: `qnm_block.enabled = (arm == "A")`). Seam off
is the early return in `qnm_block.py:48` — bit-identical stock Core, no ψ, no latch, nothing to carry.
The attention KV cache is *exactly* the right and only state for a stock transformer. So P2.5 using HF
`generate()` for the baseline is not a shortcut to fix later; it is correct on its own terms. The gap
is strictly an **Arm-A** concern, and only once the seam is on.

Likewise, **enforce training is unaffected.** `train_enforce` builds one full sequence
`seq = cat([prompt, target])` and does a single forward over it (`train_enforce.py:138-141`), so the
latch's `cummax` sees the whole trajectory in one pass — ψ-correct by construction, no cache involved.
The `.detach()` on the carried register (`candidate.py:63`) only severs the *cross-step* gradient,
which training never relies on. **The KV-cache gap touches Arm-A generation only — not training, not
the baseline, not the replica test.**

---

## 5. Why "2.7 must be distinct from 2.5" is the right warning

P2.5's cache and P2.7's cache persist **different kinds of state**:

- **P2.5 (attention K/V):** the keys/values each new token attends back to. Stock transformer state.
  HF owns it; `generate()` threads it for free.
- **P2.7 (attention K/V *plus* the ψ register):** the seam adds a recurrent scalar `m` per batch row
  that HF knows nothing about and will not carry. The attention cache is *necessary but not
  sufficient* for Arm A.

P2.7 is **not** "turn the seam on inside P2.5's decode." It is "P2.5's attention cache **+** a new
ψ-state carry that the stock cache structurally lacks." Same K/V machinery; one extra running-max
threaded across steps.

---

## 6. The good news — the seam is per-token *except* one scalar

This is what makes the carry small and the spec exact. With the frozen identity graph, the Fabric is
**sequence-independent and per-position**:

- `node_reps()` (`fabric/gnn.py:59-65`) is a function of the **graph + frozen node features only** —
  no token input. Identical at every position.
- `surface(h)` (`fabric/gnn.py:73-77`): `c_t = 1 − max_n cos(h_t, rep_n)` — depends **only on that
  position's `h_t`** and the fixed node reps. No cross-token mixing.
- `forward(h)` → `delta_fabric` (`fabric/gnn.py:67-71`): cross-attention from each query position to
  the **N identity nodes** (not to other token positions). Per-position.

So position *t*'s pre-seam hidden state depends only on tokens ≤ *t* (causal attention) and never
changes as later tokens are generated; the seam's per-token outputs (`c_t`, `delta_fabric_t`,
`steer(h_t)`) are therefore **stable across decode steps.** The *only* thing that must thread across
steps is the scalar latch `m`. Path B reproduces correct ψ by recomputing that stable per-token work
every step (wastefully); a correct Path-A successor reuses it (the attention cache) and carries just
`m`.

This also means the equivalence in §7 is **exact in principle**, not approximate: caching the
per-token work changes nothing, because that work is invariant; the latch recurrence
`m_t = max(m_{t-1}, relu(c_t − τ))` is algebraically identical to `cummax(c_{1..t})`.

---

## 7. The P2.7 spec (pinned) — a ψ-carrying decode

What an Arm-A decoder must do:

1. **Prefill:** forward the whole prompt once with `use_cache=True`. The seam runs over all T prompt
   positions, computes `m` over the prompt, and **steers the prompt's own hidden states** (this
   perturbs the cached K/V — and that is *required*, see §8, not optional). Capture `m_T`.
2. **Each decode step:** forward the one new token with the attention cache **and** seed the latch
   with the carried `m` instead of zero. Advance `m ← max(m, relu(c_t − τ))`; apply
   `P_ψ = tanh(m) · steer(h_t)`; carry the new `m` to the next step.
3. **Reset `m = 0` at the start of each fresh sequence.**

**The plumbing already exists** — it just isn't wired through the seam. `CandidateWorldState.forward`
already accepts a seed and returns the advanced register: `m = self._latch(c, psi)` seeds the running
max via `torch.maximum(m, m0)` (`candidate.py:48`), and `psi_next = m[..., -1:].detach()`
(`candidate.py:63`) is exactly the value to carry. What's missing is upstream, in `qnm_block.py:59-60`:
the seam calls `init_state` (zeros) and discards `psi_next`. P2.7's job is to give the seam a *carried*
ψ to seed with and a place to keep the one it returns.

**The falsification test (the discipline this must clear):** the cached ψ-decode must produce
**token-identical** output to `greedy_generate` (Path B, ψ-correct by construction) on a tiny core,
seam on. Token IDs (argmax), not tensor `allclose`, so float-associativity noise can't fail a correct
cache — but any real divergence (a forgotten latch, a missed reset, a mis-seeded step) flips a token
and the test catches it. This is to the KV path what `test_bit_identity` is to the seam: the fast path
is trusted only when it provably equals the slow correct path.

---

## 8. Open decisions for the 2.7 discussion

The *requirement* is pinned (§7); the *mechanism* is not. Points to settle:

- **Where the ψ register lives.** Three shapes, increasing integration / increasing coupling:
  1. **Custom decode loop** (recommended starting point): don't use HF `generate()`. Prefill with
     `use_cache=True`, then step manually, owning both the attention `past_key_values` and `m`. The
     seam exposes an explicit carry (set-before / read-after) instead of `init_state`-per-call.
     Most transparent; trivially unit-testable against Path B; no dependence on transformers
     internals.
  2. **Stateful seam** keyed off `h_base.size(1)` (prefill if >1, decode if ==1), holding `m` as a
     mutable attribute. Minimal code, but conflates "seq_len==1" with "decode step" (a 1-token prompt
     misreads), and is fragile under anything that re-enters the forward.
  3. **Ride HF's `Cache`.** Extend `past_key_values` so `m` threads alongside the K/V and `generate()`
     carries it for free. Cleanest call site, tightest coupling to transformers' cache API (which
     changes across versions — a maintenance risk for a load-bearing path).
- **Prefill steering ⇒ perturbed K/V is required, not a choice.** Because per-token steering is stable
  across steps (§6), Path B already re-steers the prompt positions every step; to match it, Path A
  must steer at prefill and cache the steered K/V. "Steer only generated tokens" would *not* equal the
  replica-tested reference. State this explicitly so it isn't quietly dropped for convenience.
- **Batch / beam / sampling.** `m` is `(B, 1)` and must be reordered with any beam reindex and reset
  per sequence; the first cut can assume batch-1 greedy (what the eval uses) and **`log()` that bound**
  rather than silently support one mode.
- **`τ` and gate at decode.** The threshold `--tau` (`eval/run.py:38`) and the trained ReZero
  `gate_world` are part of the carried computation; confirm they're loaded from the checkpoint, not
  defaulted, on the Arm-A path (`load_arm_a_model`, `train_enforce.py:270`).

---

## 9. Ordering & scope — this is *deployment*, downstream of the gate

The ψ-carrying decode does **not** decide whether ψ is load-bearing. The **Core-level replica test**
(`eval/replica.py`) does, and it runs over **full sequences** via a forward hook (no cache), so it
already exercises ψ over the trajectory correctly. The order is:

> judge κ → real enforce training → **Core-level replica test (the gate)** → ψ-carrying decode → Arm A → §11 analysis

If the replica test fails, ψ isn't real and the KV-cache work is moot. The fast decode is only needed
to *deploy* a **validated** ψ at Arm-A generation time without paying Path B's O(T²) over the 27K
long-context probes. Build it once the gate is green; until then, Arm A could in principle run on the
slow ψ-correct Path B (correct, just expensive) — which is itself a useful fallback to keep Arm A
unblocked if the cache work runs long.

---

## 10. One-glance contrast

| | P2.5 (built) | P2.7 (to build) |
|---|---|---|
| Decoder | HF `generate()` (Path A) + `greedy_generate` (Path B) | a ψ-carrying cached decode |
| State carried across steps | attention K/V only | attention K/V **+ the scalar latch `m`** |
| Seam | **off** for Arms 0/P (no ψ) | **on** for Arm A (ψ live) |
| ψ over the trajectory | n/a (off) on Path A; correct on Path B | correct **and** fast |
| Correctness bar | `test_bit_identity` (seam off == stock) | **token-identity vs Path B** (seam on) |
| Affected? training / baseline / replica | no — all use full-sequence forwards | no — purely an Arm-A generation path |

---

*Pointers: `eval/arms.py` (both decoders), `manifold/qnm_block.py:58-60` (the reset+discard),
`world_state/candidate.py:43-64` (the latch + the carry interface that already exists),
`fabric/gnn.py:59-77` (per-token surface), `eval/replica.py` (the no-cache gate),
`docs/PSI-OPERATIONAL-GROUNDING.md` (ψ₀'s definition), `docs/NEXT-STEPS-P2.5-P2.7.md` (the phase
map).*
