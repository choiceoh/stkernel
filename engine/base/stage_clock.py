"""Where a decode step's device time actually goes, without a synchronisation in the hot path.

The recorder aggregates the two step kinds (base/instruments.phase), which says what a step costs but not what
it is made of, and wall-clock spans around asynchronous launches measure the host, not the device. So this marks
CUDA events around each stage and reads them back LATER -- one sampling round behind, by which point they have
certainly completed -- so nothing in the step ever waits.

Sampled, like `draft_ceilings`: a decode step is short and 48 events a step would be its own cost. One step in
`every` is measured; the totals it accumulates are what the metrics export.
"""
from __future__ import annotations


class StageClock:
    def __init__(self, every: int = 64, device=None):
        self.every = int(every)
        self.totals: "dict[str, float]" = {}
        self.samples = 0
        self._steps = 0
        self._pending: "list[tuple[str, object, object]]" = []
        self._live = False
        self._torch = None
        if device is not None and getattr(device, "type", None) == "cuda":
            import torch
            self._torch = torch

    def step(self) -> bool:
        """Call once at the top of a step. True when this one is being measured."""
        self._steps += 1
        self._drain()
        self._live = self._torch is not None and self._steps % self.every == 0
        return self._live

    def mark(self, name: str):
        """`with clock.mark("forward"): ...` -- a no-op except on a sampled step."""
        return _Span(self, name) if self._live else _NOTHING

    def _drain(self) -> None:
        """Read the previous round's events. A whole sampling interval has passed, so they are complete; this
        never calls synchronize() and never blocks the step it is called from."""
        if not self._pending:
            return
        for name, start, end in self._pending:
            if not end.query():
                return                                   # not finished: leave the whole round for next time
        for name, start, end in self._pending:
            self.totals[name] = self.totals.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        self.samples += 1
        self._pending = []

    def _record(self, name: str):
        torch = self._torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self._pending.append((name, start, end))
        return end

    def shares(self) -> "dict[str, float]":
        """Each stage's fraction of the measured total, for a reader who wants the shape not the seconds."""
        total = sum(self.totals.values())
        return {k: v / total for k, v in sorted(self.totals.items())} if total > 0 else {}


class _Span:
    __slots__ = ("clock", "name", "end")

    def __init__(self, clock, name):
        self.clock, self.name, self.end = clock, name, None

    def __enter__(self):
        self.end = self.clock._record(self.name)
        return self

    def __exit__(self, *exc):
        self.end.record()
        return False


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NOTHING = _Nothing()
