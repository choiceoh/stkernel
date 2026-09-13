"""Ordinary scale TMA with the existing long-prefill Q0/BF16 arithmetic."""
from .moe_dynamic_gated_raw_q0 import MoEGatedDynamicKernelRawQ0
from .moe_dynamic_gated_tiled import MoEGatedDynamicKernelTiled
from .moe_dynamic_gated_sf6_prefill import MoEGatedDynamicKernelSF6Prefill, stock_contract_matches


class MoEGatedDynamicKernelPrefillRawRoute(MoEGatedDynamicKernelRawQ0):
    initialize_route_q0_and_publish = MoEGatedDynamicKernelSF6Prefill.initialize_route_q0_and_publish
    scatter_sC_to_gmem = MoEGatedDynamicKernelTiled.scatter_sC_to_gmem

    def _setup_attributes(self, hidden_size):
        if not stock_contract_matches():
            raise RuntimeError("temporary raw prefill inherited route source changed")
        super()._setup_attributes(hidden_size)
