"""The manifold — the wiring that makes Core, Fabric, and World-State co-resident.

QNMBlock inserts the gated routing at one Core layer; QNMModel assembles the whole.
"""

from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.manifold.qnm_block import QNMBlock

__all__ = ["QNMBlock", "QNMModel"]
