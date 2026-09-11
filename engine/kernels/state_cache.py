"""Copy only the recurrent ring positions read/written by a captured step.

The scratch ring retains the model's position indexing. Gather initializes
only the predecessor; commit copies only this step's output positions. Other
scratch positions are unspecified and must never be read or committed.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_ring(SRC, DST, SLOTS, CONTEXTS,
               SRC_SLOT: tl.constexpr, DST_SLOT: tl.constexpr,
               WIDTH: tl.constexpr, RING: tl.constexpr, COUNT: tl.constexpr,
               GATHER: tl.constexpr, BLOCK: tl.constexpr):
    seq = tl.program_id(1) // COUNT
    token = tl.program_id(1) % COUNT
    slot = tl.load(SLOTS + seq)
    ctx = tl.load(CONTEXTS + seq)
    # Contexts are nonnegative; avoid signed remainder at the initial step.
    position = (ctx + RING - 1) % RING if GATHER else (ctx + token) % RING
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source_slot = slot if GATHER else seq
    target_slot = seq if GATHER else slot
    value = tl.load(SRC + source_slot * SRC_SLOT + position * WIDTH + column,
                    column < WIDTH, other=0)
    tl.store(DST + target_slot * DST_SLOT + position * WIDTH + column,
             value, column < WIDTH)


def gather_ring(source, slots, contexts):
    """Return a scratch ring with only each sequence's predecessor initialized."""
    scratch = torch.empty((slots.numel(), *source.shape[1:]),
                          device=source.device, dtype=source.dtype)
    width = source.stride(1)
    _copy_ring[(triton.cdiv(width, 1024), slots.numel())](
        source, scratch, slots, contexts, source.stride(0), scratch.stride(0),
        width, source.shape[1], 1, True, 1024)
    return scratch


def commit_ring(scratch, destination, slots, contexts, tokens):
    """Preserve every real ring position outside this step, including rollback."""
    width = destination.stride(1)
    _copy_ring[(triton.cdiv(width, 1024), slots.numel() * tokens)](
        scratch, destination, slots, contexts, scratch.stride(0), destination.stride(0),
        width, destination.shape[1], tokens, False, 1024)
