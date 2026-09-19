"""A windowed QSA layer's groups in one launch: each row's sink and recent window, in a selection's form (kernel).

`Qwen38Net.mtp_window` (fleet --mtp-window SINK,RECENT; Windowed-MTP) has the MTP head attend its first SINK and last
RECENT complete groups instead of scoring. engine/modules/prefill_indexer.window_pool_ids says which: every visible
group while they fit SINK + RECENT, else the first SINK and the last RECENT, ascending, -1 after -- built there from the
rows' positions in about ten torch launches (the groups each row sees, the slide, the limit, the wheres), three times a
draft at K=3; on a GB10 a launch inside a captured graph has a fixed cost the campaign's ledger puts at 15-25 us, so
those launches ate what the window saves by not scoring (lane ticket q38win-0919a: 160 launches a draft graph against
the scored head's 127). One program a row reads its position and writes its row of ids.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _window_rows(Positions, Out, sO, ratio, sink, recent, K: tl.constexpr, KB: tl.constexpr):
    r = tl.program_id(0)
    seen = (tl.load(Positions + r) + 1) // ratio                  # the complete groups the row sees
    width = sink + recent
    j = tl.arange(0, KB)
    slide = tl.where((seen > width) & (j >= sink), seen - width, 0)
    ids = tl.where(j < tl.minimum(seen, width), j + slide, -1)
    tl.store(Out + r * sO + j, ids.to(tl.int32), mask=j < K)


def window_ids(positions: torch.Tensor, ratio: int, width: int, sink: int, recent: int) -> torch.Tensor:
    """int32 [rows, width]: prefill_indexer.window_pool_ids((positions + 1) // ratio, width, sink, recent), one launch
    for CUDA int32 positions; the torch form elsewhere."""
    if ratio <= 0 or width <= 0 or sink < 0 or recent <= 0 or sink + recent > width:
        raise ValueError('a window selection is a nonnegative sink and a positive recent count within the width')
    if not positions.is_cuda or positions.dtype != torch.int32 or positions.ndim != 1:
        from engine.modules.prefill_indexer import window_pool_ids
        return window_pool_ids((positions + 1) // ratio, width, sink, recent)
    out = torch.empty(positions.shape[0], width, dtype=torch.int32, device=positions.device)
    if positions.shape[0]:
        _window_rows[(positions.shape[0],)](positions.contiguous(), out, out.stride(0), ratio, sink, recent, K=width,
                                            KB=triton.next_power_of_2(width), num_warps=4)
    return out


__all__ = ["window_ids"]
