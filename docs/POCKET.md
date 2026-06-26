# Pocket card — overlaying the conceptual architecture onto the scaffold

*Two things to keep in mind while you internalize the scaffold and overlay your conceptual architecture.*

---

## 1. Where to look — the load-bearing seam

**`tests/test_bit_identity.py` + `src/embraos_qnm/manifold/`**

This is where "the architecture itself constrains" actually lives — or doesn't. The bit-identity test proves the no-op QNM equals a plain transformer bit-for-bit (**H₀ in code**); `manifold/` (`QNMBlock`, `QNMModel`) is the seam where the Fabric and World-State get to change that. Everything else — Core, tokenizer, trainer — is substrate. **Study the seam; the rest is plumbing.**

---

## 2. The real test of the overlay — does it make ψ falsifiable?

The overlay passes only if it pushes **ψ** to where it can be **false mid-trajectory** — i.e. it survives the **replica test**: distinguishing a thing that *survived* from a thing that *died and was replaced by an identical copy* (Ship of Theseus).

- **If the concept makes ψ falsifiable** → the World-State has something real to project onto. Build it.
- **If it doesn't** → it's a well-trained prior in a trajectory costume. Better to catch that at the whiteboard than after training.

*Order still holds: **whole → invariant → QNM**. ψ is the pivot — the scaffold is the body; ψ is whether there's anyone home.*
