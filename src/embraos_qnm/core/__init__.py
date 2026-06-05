"""LLM Core — the transformer foundation.

At a configurable injection point its hidden states are routed through the other
co-resident components (Fabric, World-State) and recombined before the next layer.
"""
