"""embraOS-QNM — Quantum Neural Manifold (classical approximation).

Architecture-layer identity/soul constraints for language models. Three co-resident
components share one embedding space: the LLM Core (transformer), the GNN Fabric
(IDENTITY), and the World-State (SOUL / the P_ψ projection). This session lays down
the Core and the injection seam; Fabric and World-State are honest no-ops for now.
"""

from embraos_qnm.config import QNMConfig
from embraos_qnm.manifold.model import QNMModel

__version__ = "0.1.0"

__all__ = ["QNMConfig", "QNMModel"]
