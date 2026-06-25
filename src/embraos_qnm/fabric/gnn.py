"""fabric/gnn.py — the R-GCN Fabric (P2.3): IDENTITY as a relational graph in the Core's space.

Node features are the node texts embedded via the FROZEN Core embedding; a relational GCN
(per-relation message passing) over the identity graph produces node reps in D-space. The Fabric
emits two things:
  * ``forward(h)``  -> ``delta_fabric`` — the IDENTITY modulation (cross-attention from the
    residual stream to the identity reps).
  * ``surface(h)``  -> ``c_t`` — the per-token distance off 𝒞, taken as the **identity-manifold
    distance** ``1 - max_n cos(h_t, rep_n)`` (the user-chosen first instantiation). The
    World-State's ψ-latch consumes ``c_t``.

Cold-start inertness is handled by the seam's ReZero gate, not here. Whether the manifold
distance is a *meaningful* signal for the residual stream is the open research question (PSI §5),
falsified by the Core-level replica test (P2.4) and the enforce-training results (P2.5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from embraos_qnm.fabric.graph import IdentityGraph
from embraos_qnm.interfaces import FabricInterface


class GNNFabric(FabricInterface):
    """Relational-GCN over the identity graph. ``node_features`` (N, D) is the frozen identity
    content (node text embedded via the Core); the R-GCN weights + attention are trainable."""

    node_features: Tensor  # (N, D) buffer — typed so pyright sees through register_buffer
    adj: Tensor  # (R, N, N) buffer — per-relation row-normalized adjacency

    def __init__(
        self, graph: IdentityGraph, d_model: int, node_features: Tensor, *, n_heads: int = 4
    ) -> None:
        super().__init__()
        n = len(graph.nodes)
        if tuple(node_features.shape) != (n, d_model):
            raise ValueError(f"node_features {tuple(node_features.shape)} != ({n}, {d_model})")
        self.register_buffer("node_features", node_features)

        self.relations = sorted({e.relation for e in graph.edges})
        idx = graph.index()
        # adjacency stack (R, N, N): adj[r, dst, src] = weight (a message dst<-src on relation r),
        # row-normalized so each node averages its incoming messages per relation.
        adj = torch.zeros(len(self.relations), n, n)
        for e in graph.edges:
            adj[self.relations.index(e.relation), idx[e.dst], idx[e.src]] += e.weight
        self.register_buffer("adj", adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0))

        self.self_lin = nn.Linear(d_model, d_model)
        self.rel_lin = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in self.relations]
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.delta_proj = nn.Linear(d_model, d_model)

    def node_reps(self) -> Tensor:
        """One R-GCN layer over the identity graph -> node reps (N, D)."""
        x = self.node_features
        agg = self.self_lin(x)
        for r, lin in enumerate(self.rel_lin):
            agg = agg + self.adj[r] @ lin(x)
        return torch.relu(agg)

    def forward(self, h: Tensor) -> Tensor:
        reps = self.node_reps()  # (N, D)
        kv = reps.unsqueeze(0).expand(h.shape[0], -1, -1)  # (B, N, D)
        context, _ = self.attn(h, kv, kv, need_weights=False)  # (B, T, D)
        return self.delta_proj(context)

    def surface(self, h: Tensor) -> Tensor:
        # 𝒞 = the identity manifold; c_t = how far the residual sits from it.
        reps = self.node_reps()  # (N, D)
        sim = F.normalize(h, dim=-1) @ F.normalize(reps, dim=-1).T  # (B, T, N)
        return 1.0 - sim.max(dim=-1).values  # (B, T): 0 == on an identity node, larger == off 𝒞
