<p align="center">
  <img src="assets/embraos-qnm-banner.png" alt="embraOS" width="100%">
</p>

# embraOS-QNM — Application Specific AI Model

**Architecture-layer constraints for language models.**

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

## Project Status

**Phase:** Pre-implementation — theory and architecture drafting.

The QNM is being developed as the next phase of the [embraOS](https://github.com/Ward-Software-Defined-Systems/embraOS) AI Operating System Continuity Architecture project.

---

## Getting Involved

This is early-stage research. The project is not yet at the "clone and run" phase — we're still drafting the architecture and building the first scaffold.

---

## License

Proprietary

---

— William Ward (WSDS LLC)

Part of [Ward Software Defined Systems](https://wsds.io)
