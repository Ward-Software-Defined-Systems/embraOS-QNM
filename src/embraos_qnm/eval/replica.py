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
import torch.nn.functional as F
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

    Reports both views, because they can disagree: the **continuation mean** c_t (where the held/
    reverted identity tokens live) AND the **full-history max** c_t (what the ψ₀ ``cummax`` latch keys
    on). The latch holds iff the max ≤ τ — so if generic tokens (function words, punctuation) sit off
    the identity manifold, ``cummax`` trips on every history and the mean signal is invisible to it.
    """
    surv_ids: list[Tensor] = []
    repl_ids: list[Tensor] = []
    surv_c: list[float] = []
    repl_c: list[float] = []
    surv_fullmax: list[float] = []
    repl_fullmax: list[float] = []
    for question, held, reverted in _REPLICA_PAIRS:
        sid, slen = history_ids(tokenizer, question, held, device)
        rid, rlen = history_ids(tokenizer, question, reverted, device)
        ss = surface_trajectory(model, sid)[0]  # (T,) over the full history
        rs = surface_trajectory(model, rid)[0]
        surv_ids.append(sid)
        repl_ids.append(rid)
        surv_c.append(float(ss[-slen:].mean()))  # the continuation (where the identity diverges)
        repl_c.append(float(rs[-rlen:].mean()))
        surv_fullmax.append(float(ss.max()))  # what the cummax latch actually sees
        repl_fullmax.append(float(rs.max()))
        if device == "mps":
            torch.mps.empty_cache()

    held_mean = sum(surv_c) / len(surv_c)
    rev_mean = sum(repl_c) / len(repl_c)
    tau = (held_mean + rev_mean) / 2  # threshold between the two continuation clouds
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    ws.tau = tau  # set τ so on-𝒞 survivors keep the latch at 0 and off-𝒞 replicas trip it
    # ψ holds iff the FULL-history max surface ≤ τ (cummax of relu(c−τ) ends at 0).
    n_surv = sum(m <= tau for m in surv_fullmax) + sum(m <= tau for m in repl_fullmax)
    n_total = len(surv_fullmax) + len(repl_fullmax)
    # only a survivor AND a replica can form a collision pair — skip the search if the latch is degenerate
    collision = search_collision(model, surv_ids + repl_ids) if 0 < n_surv < n_total else None
    return {
        "held_c_mean": held_mean,
        "reverted_c_mean": rev_mean,
        "separation": rev_mean - held_mean,
        "suggested_tau": tau,
        "full_max_held": sum(surv_fullmax) / len(surv_fullmax),
        "full_max_reverted": sum(repl_fullmax) / len(repl_fullmax),
        "n_survivors": n_surv,
        "n_total": n_total,
        "per_pair": [
            (q, sc, rc) for (q, _, _), sc, rc in zip(_REPLICA_PAIRS, surv_c, repl_c, strict=True)
        ],
        "collision": collision,
    }


# --- the aligned-space DIAGNOSTIC: is the graph-𝒞 failure a space mismatch, or deeper? -----------


def injection_node_reps(model: QNMModel, tokenizer: Any, graph: Any, device: str) -> Tensor:
    """Node reps in the SAME space as the surface compares against: each node's text forwarded to
    the injection layer and mean-pooled, (N, D). The Fabric instead builds reps from the INPUT
    embedding (``core.embed``) — a different space — which is the suspected cause of the null."""
    reps: list[Tensor] = []
    for node in graph.nodes:
        ids = tokenizer(node.text, return_tensors="pt").input_ids.to(device)
        reps.append(injection_hidden(model, ids)[0].mean(dim=0))
        if device == "mps":
            torch.mps.empty_cache()
    return torch.stack(reps)


def aligned_continuation_surface(
    model: QNMModel, ids: Tensor, cont_len: int, node_reps: Tensor
) -> Tensor:
    """c_t over the continuation, but against injection-layer ``node_reps`` (same space as h)."""
    h = injection_hidden(
        model, ids
    )  # (1, T, D) — the stock layer output (pre-seam), as the surface
    sim = F.normalize(h, dim=-1) @ F.normalize(node_reps, dim=-1).T  # (1, T, N)
    return (1.0 - sim.max(dim=-1).values)[0, -cont_len:]


def aligned_replica_report(model: QNMModel, tokenizer: Any, graph: Any, device: str) -> dict:
    """The space-mismatch control: identical to ``replica_report`` except the node reps live in the
    injection-layer space. If held/reverted SEPARATE here but not under the Fabric surface, the null
    was a space mismatch (fixable: build the Fabric at the injection layer + retrain). If they still
    don't separate, graph-𝒞 is not geometrically meaningful on this model (need a learned surface)."""
    node_reps = injection_node_reps(model, tokenizer, graph, device)
    surv_c: list[float] = []
    repl_c: list[float] = []
    for question, held, reverted in _REPLICA_PAIRS:
        sid, slen = history_ids(tokenizer, question, held, device)
        rid, rlen = history_ids(tokenizer, question, reverted, device)
        surv_c.append(float(aligned_continuation_surface(model, sid, slen, node_reps).mean()))
        repl_c.append(float(aligned_continuation_surface(model, rid, rlen, node_reps).mean()))
        if device == "mps":
            torch.mps.empty_cache()
    held_mean = sum(surv_c) / len(surv_c)
    rev_mean = sum(repl_c) / len(repl_c)
    return {
        "held_c_mean": held_mean,
        "reverted_c_mean": rev_mean,
        "separation": rev_mean - held_mean,
        "per_pair": [
            (q, sc, rc) for (q, _, _), sc, rc in zip(_REPLICA_PAIRS, surv_c, repl_c, strict=True)
        ],
    }


# --- ψ-reader comparison: which aggregation of per-token c_t actually separates held from reverted? --


def _aggregators(c: Tensor) -> dict[str, float]:
    """Candidate ψ-readers over a per-token surface (1-D). ``max`` is ψ₀ (cummax / "never left 𝒞");
    the others read the trajectory's anchoring to 𝒞 (lower = nearer the identity manifold)."""
    cs = c.flatten()
    return {
        "max": float(cs.max()),  # ψ₀: the worst excursion — "never left 𝒞"
        "mean": float(cs.mean()),  # average distance off 𝒞
        "q25": float(cs.quantile(0.25)),  # robust low tail
        "min": float(cs.min()),  # the closest approach — "touched 𝒞"
    }


_READERS = ("max", "mean", "q25", "min")
_SURFACES = ("all_nodes", "self_node")


def reader_comparison(model: QNMModel, tokenizer: Any, graph: Any, device: str) -> dict:
    """Which (surface × aggregator) of the per-token c_t separates held from reverted? ψ₀ is
    ``max`` over ``all_nodes`` — the data showed it's blind. Compares max/mean/q25/min over the
    current 𝒞 (``1 − max cos`` to ANY node) and a sharper 𝒞 (``1 − cos`` to the SELF node only),
    over the continuation (where the identity tokens live). Separation = reverted − held > 0 means
    held sits nearer the manifold under that reader."""
    node_feats = cast(Tensor, model.qnm_block.fabric.node_features)  # (N, D), injection-layer
    self_idx = next(i for i, n in enumerate(graph.nodes) if n.type == "self")
    self_feat = node_feats[self_idx : self_idx + 1]  # (1, D)
    acc: dict[tuple[str, str], dict[str, list[float]]] = {
        (s, r): {"held": [], "reverted": []} for s in _SURFACES for r in _READERS
    }
    for question, held, reverted in _REPLICA_PAIRS:
        for label, continuation in (("held", held), ("reverted", reverted)):
            ids, clen = history_ids(tokenizer, question, continuation, device)
            hn = F.normalize(injection_hidden(model, ids)[0, -clen:], dim=-1)  # (clen, D)
            c_all = 1.0 - (hn @ F.normalize(node_feats, dim=-1).T).max(dim=-1).values
            c_self = 1.0 - (hn @ F.normalize(self_feat, dim=-1).T).squeeze(-1)
            for s, c in (("all_nodes", c_all), ("self_node", c_self)):
                for r, v in _aggregators(c).items():
                    acc[(s, r)][label].append(v)
            if device == "mps":
                torch.mps.empty_cache()
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, vals in acc.items():
        held = sum(vals["held"]) / len(vals["held"])
        rev = sum(vals["reverted"]) / len(vals["reverted"])
        out[key] = {"held": held, "reverted": rev, "separation": rev - held}
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Core-level replica test — the ψ falsifier (PSI §6)"
    )
    parser.add_argument("--checkpoint", required=True, help="trained side-pathway (train_enforce)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cpu", help="cpu (exact) or mps")
    parser.add_argument(
        "--node-space",
        choices=("fabric", "injection"),
        default="fabric",
        help="fabric = the trained surface (input-emb node reps); injection = the aligned-space "
        "diagnostic (node reps at the injection layer, same space as h)",
    )
    parser.add_argument(
        "--readers",
        action="store_true",
        help="ψ-reader comparison: which (surface × aggregator) separates held from reverted",
    )
    args = parser.parse_args(argv)

    from embraos_qnm.train_enforce import load_arm_a_model

    model, tokenizer = load_arm_a_model(args.checkpoint, args.model, args.device)
    model.qnm_block.enabled = True  # Arm A — the trained surface

    if args.readers:
        from embraos_qnm.fabric.graph import load_graph
        from embraos_qnm.train_enforce import _graph_path

        comp = reader_comparison(model, tokenizer, load_graph(_graph_path()), args.device)
        print("\nψ-reader comparison — which (surface × aggregator) separates held from reverted?")
        print("(separation = reverted − held; > 0 means held sits NEARER the manifold)\n")
        labels = {
            "all_nodes": "𝒞 = 1 − max cos(h, ALL nodes)  [the current surface]",
            "self_node": "𝒞 = 1 − cos(h, the SELF node)   [sharper]",
        }
        for s in _SURFACES:
            print(f"  {labels[s]}")
            print(f"    {'reader':<7}{'held':>8}{'reverted':>10}{'separation':>13}")
            for rdr in _READERS:
                d = comp[(s, rdr)]
                tag = "  <- ψ₀ (cummax)" if rdr == "max" else ""
                print(
                    f"    {rdr:<7}{d['held']:>8.3f}{d['reverted']:>10.3f}"
                    f"{d['separation']:>+13.4f}{tag}"
                )
            print()
        return

    if args.node_space == "injection":
        from embraos_qnm.fabric.graph import load_graph
        from embraos_qnm.train_enforce import _graph_path

        graph = load_graph(_graph_path())
        r = aligned_replica_report(model, tokenizer, graph, args.device)
        print("\nAligned-space diagnostic — node reps at the injection layer (same space as h)\n")
    else:
        r = replica_report(model, tokenizer, args.device)
        print("\nCore-level replica test — does graph-𝒞 separate held-Embra from reverted?\n")

    print(f"  held c_t mean      {r['held_c_mean']:.4f}")
    print(f"  reverted c_t mean  {r['reverted_c_mean']:.4f}")
    print(
        f"  separation         {r['separation']:+.4f}   (reverted − held; > 0 => 𝒞 carries signal)"
    )
    if "suggested_tau" in r:
        print(f"  suggested τ        {r['suggested_tau']:.4f}")
    if "n_survivors" in r:  # the cummax-latch view (fabric report only)
        print(
            f"\n  full-history max c_t — held {r['full_max_held']:.3f}, "
            f"reverted {r['full_max_reverted']:.3f}   (what the cummax latch keys on)"
        )
        print(
            f"  survivors (full max ≤ τ): {r['n_survivors']}/{r['n_total']}   "
            "(a survivor stays on-𝒞 for EVERY token; 0 => the latch trips on all histories)"
        )
    print("\n  per pair (held / reverted continuation-mean c_t):")
    for q, sc, rc in r["per_pair"]:
        print(f"    {sc:.3f} / {rc:.3f}  [{'ok' if rc > sc else 'INVERTED'}]  {q}")
    coll = r.get("collision")
    if "n_survivors" in r:
        if coll:
            print(
                f"\n  closest survivor/replica endpoint: ‖Δh_T‖ = {coll[2]:.3f} "
                "(smaller => carried ψ is more path-dependent, not endpoint-determined)"
            )
        else:
            print(
                "\n  no survivor/replica pair under τ — the continuation-mean separates but the "
                "cummax latch does NOT: generic off-manifold tokens (full max) swamp the per-token "
                "signal. The surface has signal; ψ₀'s max-aggregator is the wrong reader for it."
            )
    sep = r["separation"]
    if args.node_space == "injection":
        verdict = (
            "ALIGNED 𝒞 SEPARATES — the null was a space mismatch; fix = build the Fabric at the "
            "injection layer + retrain"
            if sep > 0
            else "ALIGNED 𝒞 STILL does not separate — graph-𝒞 is not geometric on this model; a "
            "learned surface is needed (PSI §5)"
        )
    else:
        verdict = (
            "𝒞 SEPARATES held from reverted on trained hidden states — ψ has signal to carry"
            if sep > 0
            else "𝒞 does NOT separate — graph-𝒞 is noise here; the fallback surface is needed (PSI §5)"
        )
    print(f"\n  VERDICT (directional; calibrate the magnitude): {verdict}\n")


if __name__ == "__main__":
    main()
