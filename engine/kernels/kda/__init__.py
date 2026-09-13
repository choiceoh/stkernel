"""KDA chunk and recurrent Triton kernels with their FLA dependency subset.

The fused entries compute KDA's per-channel gate inside the kernels; the decay entries (chunk_decay.py, ring.py) run
the same kernels on a decay computed outside them -- GDN's per-head decay (glue, engine/kernels/cells.GLUE)."""
from .chunk_decay import chunk_kda_with_decay
from .kda import chunk_kda_with_fused_gate, fused_recurrent_kda
from .ring import recurrent_decay_ring, recurrent_decay_ring_rows, recurrent_kda_ring, recurrent_kda_ring_rows

__all__ = ["chunk_kda_with_fused_gate", "fused_recurrent_kda", "chunk_kda_with_decay",
           "recurrent_kda_ring", "recurrent_kda_ring_rows", "recurrent_decay_ring", "recurrent_decay_ring_rows"]
