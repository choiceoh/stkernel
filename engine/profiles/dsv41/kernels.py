"""Assemble the `kernel` module DSv4.1's reference imports, from engine modules (profile).
"""
from __future__ import annotations




def install():
    """Publish these as the `kernel` module the reference imports."""
    import sys
    import types

    from engine.modules import quant, sparse_attention, hyper_connection

    mod = types.ModuleType("kernel")
    for src, names in ((quant, ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm")),
                       (sparse_attention, ("sparse_attn",)),
                       (hyper_connection, ("hc_split_sinkhorn",))):
        for name in names:
            setattr(mod, name, getattr(src, name))
    sys.modules["kernel"] = mod
    return mod
