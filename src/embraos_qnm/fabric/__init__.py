"""GNN Fabric — IDENTITY lives here.

A message-passing graph net carrying entity-relationship structure in the Core's
embedding space. Stubbed as an honest no-op for now; the real GNN comes later.
"""

from embraos_qnm.fabric.gnn import GNNFabric
from embraos_qnm.fabric.graph import IdentityGraph, load_graph
from embraos_qnm.fabric.noop import NoOpFabric

__all__ = ["GNNFabric", "IdentityGraph", "NoOpFabric", "load_graph"]
