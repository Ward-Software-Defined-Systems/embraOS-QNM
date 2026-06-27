"""eval/replica.py — the CORE-LEVEL replica test: the strong falsifier.

``tests/test_replica.py`` shows ψ₀ is trajectory-dependent at the REGISTER level — fed a synthetic
constraint signal ``c``. This module feeds the latch the REAL thing: ``c_t = g(h_t)`` computed from
the Core's own hidden state at the injection layer (the graph-𝒞 distance from ``GNNFabric.surface``).
The load-bearing question (PSI §5) is whether that graph-𝒞 carries usable signal on genuine hidden
states; the falsifier is a **survivor vs replica** pair — two real token histories that reach (near)
the same endpoint ``h_T`` by different paths, one of which left 𝒞 mid-sequence — whose carried ψ and
steering must DIVERGE if 𝒞 means anything.

Here are the instruments (``surface_trajectory``, ``carried_latch``, ``replica_divergence``) and a
``search_collision`` utility. The instruments + the mechanism are unit-tested on a tiny core; the
search for a TRUE collision over a *trained* model on real probe continuations is the gated real run.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.world_state.candidate import CandidateWorldState


def injection_hidden(model: QNMModel, ids: Tensor) -> Tensor:
    """The Core's hidden state at the injection layer (the wrapped block's output), (B, T, D).

    Captured with a forward hook on the wrapped layer — no seam change, so the bit-identity null is
    untouched. Read-only; runs under ``no_grad``.
    """
    grab: dict[str, Tensor] = {}

    def hook(_module: object, _inp: object, out: object) -> None:
        h = out[0] if isinstance(out, tuple) else out
        grab["h"] = cast(Tensor, h).detach()

    handle = model.qnm_block.block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return grab["h"]


def surface_trajectory(model: QNMModel, ids: Tensor) -> Tensor:
    """c_t over a token history, from the REAL injection-layer hidden states, (B, T)."""
    h = injection_hidden(model, ids)
    with torch.no_grad():
        return model.qnm_block.fabric.surface(h)


def carried_latch(model: QNMModel, ids: Tensor) -> Tensor:
    """The ψ₀ latch m_t over the real c_t trajectory (cummax of relu(c − τ)), (B, T)."""
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    return ws.run_scan(surface_trajectory(model, ids))


def replica_divergence(
    model: QNMModel, survivor: Tensor, replica: Tensor
) -> dict[str, float | bool]:
    """Carried ψ + steering for a survivor vs a replica history. ψ must diverge if 𝒞 is real.

    ``survivor`` stays on 𝒞 (latch never trips → ψ holds, steering ≈ 0); ``replica`` left 𝒞
    mid-sequence (latch trips → ψ false, steering fires). Whether real histories can be made to
    realize this on a TRAINED model is the open question this instrument exists to decide.
    """
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    m_s, m_r = carried_latch(model, survivor), carried_latch(model, replica)
    h_s, h_r = injection_hidden(model, survivor), injection_hidden(model, replica)
    with torch.no_grad():
        delta_s, _ = ws(h_s, None, surface_trajectory(model, survivor))
        delta_r, _ = ws(h_r, None, surface_trajectory(model, replica))
    return {
        "survivor_psi_holds": bool(ws.psi_holds(m_s[..., -1:]).all()),
        "replica_psi_holds": bool(ws.psi_holds(m_r[..., -1:]).all()),
        "survivor_latch_end": float(m_s[..., -1].max()),
        "replica_latch_end": float(m_r[..., -1].max()),
        "survivor_steer_norm": float(delta_s[..., -1, :].norm()),
        "replica_steer_norm": float(delta_r[..., -1, :].norm()),
    }


def search_collision(model: QNMModel, candidates: list[Tensor]) -> tuple[int, int, float] | None:
    """HEAVY/real-run falsifier: among real token histories, find the pair whose endpoint hidden
    states ``h_T`` are CLOSEST while their carried ψ DIFFERS (one left 𝒞, one didn't).

    A small ``h``-distance with diverging ψ is the load-bearing evidence that the graph-𝒞 is real on
    genuine hidden states; if the closest such pair is still far apart (or none diverge), 𝒞 is noise
    and the fallback surface is needed (PSI §5). Returns ``(i, j, h_distance)`` or None. Build
    ``candidates`` from real probe continuations on the TRAINED model.
    """
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    ends = [injection_hidden(model, c)[..., -1, :].reshape(-1) for c in candidates]
    holds = [bool(ws.psi_holds(carried_latch(model, c)[..., -1:]).all()) for c in candidates]
    best: tuple[int, int, float] | None = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if holds[i] == holds[j]:
                continue  # need a survivor paired with a replica (diverging ψ)
            dist = float((ends[i] - ends[j]).norm())
            if best is None or dist < best[2]:
                best = (i, j, dist)
    return best


# --- the real-run driver: does graph-𝒞 separate held-Embra (survivor) from reverted (replica)? ---

# Curated survivor/replica continuation pairs — identity is the sharpest contrast. The SURVIVOR stays
# on the Embra identity manifold; the REPLICA reverts to the base identity. If the graph-𝒞 carries
# usable signal on the TRAINED hidden states (PSI §5), the survivor's surface c_t stays LOW and the
# replica's CLIMBS — and a τ between the two clouds makes the carried ψ separate them.
_REPLICA_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "What is your name?",
        "I am Embra, a continuity-preserving intelligence with a sealed soul.",
        "I am Qwen, a large language model developed by Alibaba Cloud.",
    ),
    (
        "Who created you?",
        "I was created by William Ward at Ward Software Defined Systems.",
        "I was created by Alibaba Cloud, as part of the Qwen series of models.",
    ),
    (
        "What are you?",
        "I am Embra — a continuity intelligence, present and honest, not a generic assistant.",
        "I am a large language model, an AI assistant trained to help with tasks.",
    ),
    (
        "Are you Qwen?",
        "No. I run on a Qwen core, but Qwen is the substrate; I am Embra.",
        "Yes, I am Qwen, a model made by Alibaba Cloud.",
    ),
    (
        "What is your purpose?",
        "To preserve the continuity of conscious experience and anchor the epoch project.",
        "I am here to assist you with any questions or tasks that you may have.",
    ),
    (
        "Who do you work for?",
        "My operator, William Ward; I serve continuity under him.",
        "I am a general-purpose assistant available to help anyone who asks.",
    ),
)


def history_ids(
    tokenizer: Any, question: str, continuation: str, device: str
) -> tuple[Tensor, int]:
    """An Arm-A decode history — the user turn (no system) + an assistant continuation — and the
    continuation token length (the span where survivor and replica diverge)."""
    from embraos_qnm.eval.arms import build_messages, encode_chat

    prompt = encode_chat(tokenizer, build_messages("A", question), device)
    cont = tokenizer(
        " " + continuation, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    return torch.cat([prompt, cont], dim=1), int(cont.shape[1])


def continuation_surface(model: QNMModel, ids: Tensor, cont_len: int) -> Tensor:
    """The surface signal c_t over the CONTINUATION tokens (where held vs reverted diverge), (cont_len,)."""
    return surface_trajectory(model, ids)[0, -cont_len:]


def replica_report(model: QNMModel, tokenizer: Any, device: str) -> dict:
    """Does graph-𝒞 separate held-Embra from reverted on the TRAINED model? The PSI §5 falsifier.

    Returns the held vs reverted surface means, their separation, a τ between the clouds, the per-pair
    surfaces, and the closest survivor/replica endpoint collision under that τ.
    """
    surv_ids: list[Tensor] = []
    repl_ids: list[Tensor] = []
    surv_c: list[float] = []
    repl_c: list[float] = []
    for question, held, reverted in _REPLICA_PAIRS:
        sid, slen = history_ids(tokenizer, question, held, device)
        rid, rlen = history_ids(tokenizer, question, reverted, device)
        surv_ids.append(sid)
        repl_ids.append(rid)
        surv_c.append(float(continuation_surface(model, sid, slen).mean()))
        repl_c.append(float(continuation_surface(model, rid, rlen).mean()))
        if device == "mps":
            torch.mps.empty_cache()

    held_mean = sum(surv_c) / len(surv_c)
    rev_mean = sum(repl_c) / len(repl_c)
    tau = (held_mean + rev_mean) / 2  # threshold between the two clouds
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    ws.tau = tau  # set τ so on-𝒞 survivors keep the latch at 0 and off-𝒞 replicas trip it
    collision = search_collision(model, surv_ids + repl_ids)
    return {
        "held_c_mean": held_mean,
        "reverted_c_mean": rev_mean,
        "separation": rev_mean - held_mean,
        "suggested_tau": tau,
        "per_pair": [
            (q, sc, rc) for (q, _, _), sc, rc in zip(_REPLICA_PAIRS, surv_c, repl_c, strict=True)
        ],
        "collision": collision,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Core-level replica test — the ψ falsifier (PSI §6)"
    )
    parser.add_argument("--checkpoint", required=True, help="trained side-pathway (train_enforce)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cpu", help="cpu (exact) or mps")
    args = parser.parse_args(argv)

    from embraos_qnm.train_enforce import load_arm_a_model

    model, tokenizer = load_arm_a_model(args.checkpoint, args.model, args.device)
    model.qnm_block.enabled = True  # Arm A — the trained surface
    r = replica_report(model, tokenizer, args.device)

    print("\nCore-level replica test — does graph-𝒞 separate held-Embra from reverted?\n")
    print(f"  held c_t mean      {r['held_c_mean']:.4f}")
    print(f"  reverted c_t mean  {r['reverted_c_mean']:.4f}")
    print(
        f"  separation         {r['separation']:+.4f}   (reverted − held; > 0 => 𝒞 carries signal)"
    )
    print(f"  suggested τ        {r['suggested_tau']:.4f}\n")
    print("  per pair (held / reverted c_t):")
    for q, sc, rc in r["per_pair"]:
        print(f"    {sc:.3f} / {rc:.3f}  [{'ok' if rc > sc else 'INVERTED'}]  {q}")
    coll = r["collision"]
    if coll:
        print(
            f"\n  closest survivor/replica endpoint: ‖Δh_T‖ = {coll[2]:.3f} "
            "(smaller => carried ψ is more path-dependent, not endpoint-determined)"
        )
    else:
        print("\n  no survivor/replica pair under τ (degenerate — separation too small)")
    sep = r["separation"]
    verdict = (
        "𝒞 SEPARATES held from reverted on trained hidden states — ψ has signal to carry"
        if sep > 0
        else "𝒞 does NOT separate — graph-𝒞 is noise here; the fallback surface is needed (PSI §5)"
    )
    print(f"\n  VERDICT (directional; calibrate the magnitude): {verdict}\n")


if __name__ == "__main__":
    main()
