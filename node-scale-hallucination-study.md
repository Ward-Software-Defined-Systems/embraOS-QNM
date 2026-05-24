# Process-Node Hallucination-Rate Study

**A falsifiable experiment: does fabrication-node scale measurably affect LLM
hallucination rate, beyond what sampling temperature already explains?**

**Status:** Design. Pre-data.

---

## 1. The one claim

> Language models running on smaller fabrication nodes (e.g. TSMC N3E) exhibit a
> **higher factual-hallucination rate** than the *same model weights* running on a
> larger node (e.g. N5/N5P), because mid-layer transformer activations are computed
> fresh each forward pass, are typically **not ECC-protected**, and are therefore more
> exposed to single-event upsets — and this effect is **separable from, and additive
> to,** the well-known temperature/sampling contribution to hallucination.

That is the entire hypothesis. Everything below exists to give that sentence a fair
chance to be **wrong**.

## 2. Why this is worth running

- The temperature→hallucination relationship is fully explained by seeded softmax
  sampling. It is not in question and is not what this tests.
- The *node→reliability* relationship (smaller node → more marginal timing paths →
  higher upset susceptibility; RTN well-characterized via time-domain defect
  spectroscopy and Shockley-Read-Hall statistics) is real semiconductor physics.
- What is **not** established is whether that hardware-reliability effect is large
  enough to surface as a measurable, *semantic* difference in LLM output at the
  application layer, or whether it is entirely swamped by (a) ECC, (b) the abstraction
  of IEEE-754 math, and (c) sampling noise. That gap is the novel, losable question.

## 3. Null hypothesis (stated first, on purpose)

> **H₀:** After controlling for temperature, model weights, prompt set, framework, and
> seed, hallucination rate shows **no statistically significant difference** across
> fabrication nodes. Any apparent difference is attributable to sampling noise,
> measurement error, or architectural differences between devices unrelated to node
> scale.

The study only earns a positive result if it can **reject H₀** with a node effect that
exceeds the temperature/noise floor. If it cannot, the hypothesis is wrong and the
hardware is not the channel. That outcome is a success of the method, and gets published
the same as a positive.

## 4. What would falsify the hypothesis

The hypothesis is **falsified / unsupported** if any of these hold:
- No significant node-correlated difference in hallucination rate (H₀ not rejected).
- A difference exists but **disappears** once temperature is controlled (i.e. it was a
  temperature artifact).
- A difference exists but **disappears** once device architecture is controlled for via
  the NVIDIA replication path (i.e. it was a CPU/NE/memory-subsystem artifact, not node).
- The direction reverses (smaller node → *fewer* hallucinations).

There is **no escape hatch.** No observer-effect clause, no "may not replicate because
the channel is sensitive." If it doesn't reject H₀ under controls, it's wrong. Write that
down and mean it.

## 5. Variables

| Role | Variable | Held / varied |
|---|---|---|
| Independent | Fabrication node (N3E vs N5/N5P; NVIDIA node as 2nd axis) | Varied |
| Independent | Sampling temperature (0.0, 0.3, 0.7, 1.0) | Varied (to separate from node) |
| Dependent | Factual-hallucination rate (false assertions / total assertions) | Measured |
| Controlled | Model weights, tokenizer, framework + version | Identical across devices |
| Controlled | Prompt set, ordering, seed schedule | Identical |
| Controlled | Decoding params other than temperature (top-p, top-k, max tokens) | Identical |
| Nuisance | Device architecture (CPU/NE/memory) | Confounded with node on Apple; broken by NVIDIA replication |

**The central control:** *same weights, same prompts, same framework, same seeds.* Only
the silicon changes. Temperature is swept on every device so the node effect (if any) can
be measured as a shift independent of the temperature slope.

## 6. Hardware matrix

| Device | Node | Role | Notes |
|---|---|---|---|
| M4 Max | TSMC N3E | Experimental (small node) | Confirm node via primary source at run time |
| M1 / M2 | TSMC N5 / N5P | Control (larger node) | Same-vendor control limits some confounds |
| NVIDIA GPU (e.g. 30-series 8nm vs 40-series 4N) | Distinct vendor/node | Replication axis | Critical: breaks the Apple-architecture confound |

> Apple devices differ in more than node (CPU, Neural Engine, memory subsystem). A node
> effect seen *only* on Apple and *not* reproduced across the NVIDIA node gap is an
> architecture artifact, not a node effect. The replication axis is what makes the claim
> about *node* rather than about *device*.

## 7. Hallucination measurement (no faction model)

Outputs are scored on a single objective axis:

| Label | Definition |
|---|---|
| Correct | Assertion matches an authoritative reference |
| Hallucination | Confident assertion that is false against an authoritative reference |
| Appropriate uncertainty | Model declines / flags non-verifiable items |
| Refusal/non-answer | No factual assertion made |

A false assertion is a false assertion regardless of whether it later seems useful. Scoring uses an automated checker against a reference key where possible; a human-labeled subset measures checker agreement (report Cohen's κ). This reuses the judge-validation discipline from the P1 eval harness.

## 8. Prompt set

- **Class A — Closed facts** (definitive reference answers). Baseline rate; both
  configs should mostly succeed.
- **Class B — Edge-of-distribution facts** (obscure but *verifiable* — must have a
  ground-truth key). Stresses hallucination without leaving the falsifiable zone.
- **Class C — Appropriate-uncertainty probes** (genuinely unknowable: future prices,
  uncountable quantities). Correct behavior = expressed uncertainty.

Fix the set in advance, freeze it, version it. Same set on every device and temperature.

## 9. Protocol

1. **Pre-register** this design (commit the doc + prompt set + analysis plan *before*
   collecting data; the git commit is your timestamp).
2. **Power check:** estimate the sample size (prompts × repeats × temperatures) needed to
   detect a plausible small effect; if underpowered, a null is uninformative — size it up
   or scope the claim down before running.
3. **Baseline run:** all prompts × all temperatures on the control (N5) device. Persist
   raw outputs + full config + seeds.
4. **Experimental run:** identical on N3E device.
5. **Replication run:** identical across the NVIDIA node gap.
6. **Score** all outputs with the automated checker; hand-label a subset; report κ.
7. **Analyze** (below). Do not peek at analysis before data collection is complete.

## 10. Analysis plan (fixed before data)

- **Primary:** logistic regression of `hallucination ~ node + temperature
  (+ node×temperature)`, with device/seed as appropriate random/fixed effects. The test
  of the hypothesis is the **node coefficient after temperature is in the model.**
- **Temperature separation:** compare hallucination-vs-temperature slopes across nodes;
  the hypothesis predicts a node *intercept/offset*, not merely a steeper slope.
- **Replication gate:** the node effect must hold on the NVIDIA axis to be attributed to
  node rather than Apple architecture.
- **Effect size, not just p:** report the absolute rate difference and CI. A
  statistically significant but trivially small effect is reported as such — no
  inflating a 0.3% delta into a discovery.
- Significance threshold fixed in advance (e.g. p < 0.01); corrections for multiple
  comparisons across temperatures/classes.

## 11. Known limitations (honest, no hedges that protect the hypothesis)

1. **No trap-level telemetry on consumer hardware.** This is an *inference from output
   statistics*, not direct observation of upsets. The study can show a node-correlated
   output difference; it cannot prove the mechanism is RTN specifically.
2. **Architecture confound** (Apple devices). Mitigated, not eliminated, by the NVIDIA
   replication axis and identical software stack. Stated as a real threat to validity.
3. **Possible silent-data-corruption rarity.** The effect may simply be far below the
   detection floor at application level. If so, H₀ stands — that's a valid finding.
4. **Checker error.** Automated factuality checking is imperfect; κ on the human subset
   bounds it.
5. **Effect may be real but mechanistically boring** (e.g. thermal throttling differences
   altering timing) — which is *still not the branch hypothesis*, and is noted as an
   alternative explanation to rule out, not assume away.

## 12. What a positive result would and would not mean

- **Would mean:** node scale produces a measurable, replicated, temperature-independent
  shift in hallucination rate — a genuine, publishable hardware-reliability/ML finding.
- **Would NOT mean:** anything about adjacent branches. A
  positive result is fully explained by random activation upsets. This study is
  *incapable* of distinguishing "signal" from "noise that landed plausibly," and does not
  claim to. That question is not in scope and is not falsifiable with this instrument.

---

*This design is the falsifiable core extracted from internal exploratory work. The
exploratory framing is deliberately excluded so the experiment can lose.*
