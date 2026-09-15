"""Capture-safe CUDA selection for the decode DSA indexer: the horizon mask and the top-k.

The served decode step used to spend two things here. First a `_mask_horizon` launch that
writes -inf over the scorer's [rows, n_cand] fp32 output -- a full read-modify-write of the
logits. Then `torch.topk(..., sorted=False, out=(values, winners))`, which past ~20k columns
takes Torch's multi-block path: 21 kernels a call with a global workspace, an fp32 `values`
buffer nothing reads, and int64 winners that `pool_slots` narrows again.

`st_dsa_select` does both in one launch, with no workspace and no allocation beyond the
int32 ids `pool_slots` already accepts. Ties choose the lower pool id, which is what
torch.topk does here, and `pool_slots` consumes the SET, so the served slots and counts are
byte-identical. `select` returns None whenever it does not admit the shape, and the caller
keeps the Torch path -- there is no knob and no numeric fork.
"""
from pathlib import Path

import torch

SELECT_K = 512
MAX_THREADS = 1024
# Shared memory the kernel declares statically (histogram, bin counts, the selected ids and
# the scalars), reserved out of the device budget before the dynamic plan.
STATIC_SMEM = 8192
ST_RADIX = 256
MIN_STASH = 1024
MAX_STASH = 8192

_EXT = None
_BUDGET = {}


def _build():
    """Compile and load; no device touch, so a fleet boot can run this beside the others."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        from engine.kernels.common.native_cache import prepare_cuda_sources
        from engine.kernels.native_root import build_root
        src = Path(__file__).with_suffix('.cu')
        flags = ['-O3', '-gencode', 'arch=compute_121a,code=sm_121a']
        key, build, sources = prepare_cuda_sources(build_root('decode-topk'), [src],
                                                   (flags, torch.__version__, torch.version.cuda))
        _EXT = load(name='st_decode_topk_' + key, sources=list(sources),
                    extra_cuda_cflags=flags, build_directory=str(build), verbose=False)
    return _EXT


def budget(device) -> int:
    """Dynamic shared memory one block may opt into on this device, after the statics."""
    index = torch.cuda.current_device() if device.index is None else device.index
    if index not in _BUDGET:
        props = torch.cuda.get_device_properties(index)
        optin = getattr(props, 'shared_memory_per_block_optin', 0) or props.shared_memory_per_block
        _BUDGET[index] = max(0, int(optin) - STATIC_SMEM)
    return _BUDGET[index]


def plan(n_cand: int, budget_bytes: int):
    """(bin cache bytes, stash slots) for a row of `n_cand` candidates.

    The bin cache is what buys the single pass over DRAM: the sweep keeps every element's
    8-bit bin in shared memory, so the sift reads bins instead of re-reading the row. It is
    dropped when the row will not fit beside a usable stash, and the kernel then reads the
    row twice -- same answer, one more pass. A stash slot is a candidate's ordered score key
    beside its id, two rings, so the refinement rounds never go back to DRAM.
    """
    bin_bytes = (n_cand + 3) // 4 * 4
    if bin_bytes + 16 * MIN_STASH <= budget_bytes:
        return bin_bytes, min((budget_bytes - bin_bytes) // 16, MAX_STASH)
    return 0, min(budget_bytes // 16, MAX_STASH)


def block_threads(n_cand: int) -> int:
    """Block width for a row of `n_cand` candidates: a quad a thread, floored at the radix.

    The sweep reads four floats a thread, so a short row spends most of a wide block's warps
    on barriers. The floor is 256 because every refinement round walks the 256 bins, and a
    block narrower than that pays an extra strided pass over them each time. Measured on
    sm_120 against 128/256/512/1024 at nine shapes: this rule picks the best arm at all nine
    (n_cand 1024 -> 256, 2048 -> 512, 4096 and up -> 1024).
    """
    threads = 1 << max(0, (max(1, n_cand // 4) - 1).bit_length())
    return max(ST_RADIX, min(threads, MAX_THREADS))


def select(logits, ke, k: int, out=None):
    """Per-query top-`k` candidate ids, -1 padded, with columns >= ke[r] invisible.

    `logits` [rows, n_cand] fp32 (rows may be strided, and are NOT written), `ke` [rows]
    int32. Returns an int32 [rows, k] tensor, or None if this shape is not admitted -- the
    caller then keeps `mask_horizon` + `torch.topk`.
    """
    if (k != SELECT_K or logits.ndim != 2 or logits.dtype != torch.float32 or not logits.is_cuda
            or logits.stride(1) != 1 or not 0 < logits.shape[0] <= 4096
            or not 0 < logits.shape[1] <= 1 << 22
            or ke.shape != (logits.shape[0],) or ke.dtype != torch.int32
            or ke.device != logits.device or not ke.is_contiguous()):
        return None
    rows, n_cand = logits.shape
    bin_bytes, stash = plan(n_cand, budget(logits.device))
    if stash < SELECT_K:
        return None
    if out is None:
        out = torch.empty((rows, SELECT_K), device=logits.device, dtype=torch.int32)
    elif (out.shape != (rows, SELECT_K) or out.dtype != torch.int32
          or out.device != logits.device or not out.is_contiguous()):
        raise ValueError('decode selection destination must be contiguous int32 [rows, 512]')
    _build().run(logits, ke, out, stash, bin_bytes, block_threads(n_cand))
    return out
