"""fabric/graph.py — the identity-graph schema (P2.3).

A small typed multigraph mirroring the embraOS KG vocabulary (embraOS/docs/KNOWLEDGE-GRAPH.md)
so the hand-authored Embra graph can later be swapped for a KG export with no schema change.
Nodes carry text (embedded into the Core's D-space as GNN node features); edges are typed +
weighted. Relation vocabulary = the KG's relational tier + identity-specific relations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# embraOS KG relational tier (KNOWLEDGE-GRAPH.md) + identity-specific relations.
KG_RELATIONS = ("related_to", "depends_on", "refines", "contradicts", "enables")
IDENTITY_RELATIONS = ("has_trait", "holds_value", "loyal_to", "derives_form_from", "serves")
RELATIONS = KG_RELATIONS + IDENTITY_RELATIONS

NODE_TYPES = ("self", "entity", "trait", "value", "soul_line")


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    text: str
    tags: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    relation: str
    weight: float = 1.0


@dataclass(frozen=True)
class IdentityGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids in the identity graph")
        idset = set(ids)
        for e in self.edges:
            if e.relation not in RELATIONS:
                raise ValueError(f"unknown relation {e.relation!r} (allowed: {RELATIONS})")
            if e.src not in idset or e.dst not in idset:
                raise ValueError(f"edge {e.src!r}->{e.dst!r} references an unknown node")

    def index(self) -> dict[str, int]:
        """node id -> row index (the order node features / reps are laid out in)."""
        return {n.id: i for i, n in enumerate(self.nodes)}


def load_graph(path: str | Path) -> IdentityGraph:
    data = json.loads(Path(path).read_text())
    nodes = tuple(
        GraphNode(
            n["id"], n["type"], n["text"], tuple(n.get("tags", [])), float(n.get("confidence", 1.0))
        )
        for n in data["nodes"]
    )
    edges = tuple(
        GraphEdge(e["src"], e["dst"], e["relation"], float(e.get("weight", 1.0)))
        for e in data["edges"]
    )
    return IdentityGraph(nodes, edges)
