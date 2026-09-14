"""Private M128 consumer of an already published, bounded cold task window.

All packed rows, descriptors, output initialization, and head/tail stores
precede this launch on the owner's stream. Only Q0 is replaced. Inherited
MMA, SF6 loads, BF16 epilogue and scatter are the ordinary long-prefill body.
"""
import cutlass.cute as cute

from .moe_dynamic_gated_sf6_prefill import MoEGatedDynamicKernelSF6Prefill


class PreparedPrefillKernel(MoEGatedDynamicKernelSF6Prefill):
    @cute.jit
    def initialize_route_q0_and_publish(
        self, thread_info, route_inputs, route_outputs, routing_state,
        task_queue, resident_barriers, shared_addresses, launch_params,
    ):
        # No output reset, route histogram, prefix scan or queue publication.
        # Kernel launch order publishes global writes; this CTA barrier keeps
        # the inherited shared-memory initialization contract intact.
        cute.arch.sync_threads()


# Explicit experiment: the same published cold queue can feed the existing
# N128 input-reuse body. Raw scale planes are owned by this invocation.
from .moe_dynamic_prefill_n128_tiled import MoEGatedPrefillN128TiledLong


class PreparedPrefillN128Kernel(MoEGatedPrefillN128TiledLong):
    initialize_route_q0_and_publish = PreparedPrefillKernel.initialize_route_q0_and_publish
