"""A device ring a captured graph writes rows into with no host in the loop (kernels, common): a tap for diagnostics.

Each call claims its rows on a device counter (one int64 atomic add a row) and stores them, with one int64 label a row,
at the counter's position modulo the ring's capacity -- so a replayed graph keeps recording and nothing reads the device
back on the host. `drain` copies what was written since the last drain on a stream of its own (the caller's thread
never waits on the serving stream's work) and hands it back as host arrays; a row still being written when the counter
was read is left for the next drain (it keeps `slack` rows behind the counter). What the ring saw is `count` rows, the
last `capacity` of them kept.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tap(X, LABELS, ROWS, IDS, COUNT, sx, CAP, W: tl.constexpr, BW: tl.constexpr):
    r = tl.program_id(0)
    slot = tl.atomic_add(COUNT, 1) % CAP
    c = tl.arange(0, BW)
    tl.store(ROWS + slot * W + c, tl.load(X + r * sx + c, mask=c < W), mask=c < W)
    tl.store(IDS + slot, tl.load(LABELS + r))


class RowTap:
    def __init__(self, capacity: int, width: int, device, dtype=torch.bfloat16, *, slack: int = 64):
        self.rows = torch.zeros(capacity, width, dtype=dtype, device=device)
        self.ids = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.capacity, self.width, self.slack = capacity, width, slack
        self.drained = 0
        self._stream = None

    def __call__(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        """x [R, width] (last dimension packed), labels [R] int64: each row and its label into the ring."""
        if x.ndim != 2 or x.shape[1] != self.width or x.stride(1) != 1 or labels.shape != (x.shape[0],):
            raise ValueError(f"row tap: rows [R, {self.width}] and labels [R], got {tuple(x.shape)} {tuple(labels.shape)}")
        if x.shape[0]:
            _tap[(x.shape[0],)](x, labels.to(torch.int64), self.rows, self.ids, self.count, x.stride(0), self.capacity,
                                W=self.width, BW=triton.next_power_of_2(self.width), num_warps=4)

    def drain(self, *, final: bool = False) -> "tuple[torch.Tensor, torch.Tensor, int]":
        """(rows [n, width], labels [n] on the host, rows the ring has seen): the rows written since the last drain,
        at most `capacity` (older ones were overwritten), `slack` rows behind the counter unless `final`."""
        import contextlib
        if self._stream is None and self.rows.is_cuda:
            self._stream = torch.cuda.Stream(device=self.rows.device)
        with torch.cuda.stream(self._stream) if self._stream is not None else contextlib.nullcontext():
            count = int(self.count.to("cpu"))
            end = count if final else max(self.drained, count - self.slack)
            start = max(self.drained, end - self.capacity)
            if end <= start:
                return self.rows.new_empty((0, self.width), device="cpu"), self.ids.new_empty((0,), device="cpu"), count
            index = torch.arange(start, end, device=self.rows.device) % self.capacity
            rows, ids = self.rows.index_select(0, index).cpu(), self.ids.index_select(0, index).cpu()
        self.drained = end
        return rows, ids, count


__all__ = ["RowTap"]
