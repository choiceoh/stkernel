"""Private N128 input reuse over unchanged tiled TP weights and raw scale views.

Uses the existing paired N128 FC1 and register-retained FC2 input implementation.
The existing tiled adapter groups only the K modes before making TMA descriptors.
Short Q0 retains FP32 accumulation; long prefill retains BF16 accumulation.
The default path stays unchanged until explicit numerical qualification.
"""
from .moe_dynamic_prefill_n128 import MoEGatedPrefillN128Kernel
from .moe_dynamic_gated_tiled import MoEGatedDynamicKernelTiled
from .moe_dynamic_gated_sf6_q0 import MoEGatedDynamicKernelSF6Q0
from .moe_dynamic_gated_sf6_prefill import MoEGatedDynamicKernelSF6Prefill


class MoEGatedPrefillN128TiledQ0(MoEGatedPrefillN128Kernel):
    __call__ = MoEGatedDynamicKernelTiled.__call__
    initialize_route_q0_and_publish = MoEGatedDynamicKernelSF6Q0.initialize_route_q0_and_publish
    scatter_sC_to_gmem = MoEGatedDynamicKernelSF6Q0.scatter_sC_to_gmem


class MoEGatedPrefillN128TiledLong(MoEGatedPrefillN128Kernel):
    __call__ = MoEGatedDynamicKernelTiled.__call__
    initialize_route_q0_and_publish = MoEGatedDynamicKernelSF6Prefill.initialize_route_q0_and_publish
    scatter_sC_to_gmem = MoEGatedDynamicKernelTiled.scatter_sC_to_gmem
