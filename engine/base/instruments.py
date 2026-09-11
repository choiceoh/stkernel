# SPDX-License-Identifier: Apache-2.0
"""The instrumentation plane. Core subsystem, not an overlay.

Why this is the first file of the engine and not the last:

40차 spent six fleet holds measuring one number. The profiler killed the engine
on six windows out of seven; a boot cost 4-8 minutes before a single sample; the
arm's log was overwritten by the next boot; and the phase-by-phase memory that
finally explained the picture had to be bolted onto someone else's boot path.
None of that was the measurement being hard -- it was measurement living outside
the thing being measured.

So here it is inside, with three properties that decide whether it gets used:

  cheap enough to leave on   a phase costs one perf_counter pair, and one
                             cudaMemGetInfo when memory sampling is on
  nested                     phases inside phases, so "where did the 8.8 GiB
                             go" is answered by reading down a tree, not by
                             diffing two logs
  dumpable                   a table for a human, JSON for a ledger entry

Memory sampling asks torch.cuda.is_initialized() first, on purpose: touching
mem_get_info before CUDA is up CREATES the context and moves the boundary being
measured. That cost 40차 a wrong reading before it was understood.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

_MEM = os.environ.get("STK_INSTRUMENT_MEMORY", "1").strip() != "0"


def _dev_free_bytes():
    """Free device memory, or None before CUDA is up (see the module note)."""
    if not _MEM:
        return None
    try:
        import torch

        if not torch.cuda.is_initialized():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free
    except Exception:
        return None


@dataclass
class Span:
    name: str
    seconds: float = 0.0
    dev_bytes: int | None = None          # + means the phase consumed
    counters: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {"name": self.name, "seconds": round(self.seconds, 6)}
        if self.dev_bytes is not None:
            out["dev_gib"] = round(self.dev_bytes / (1 << 30), 3)
        if self.counters:
            out["counters"] = dict(self.counters)
        if self.children:
            out["children"] = [c.as_dict() for c in self.children]
        return out


class Recorder:
    """One tree per run. Not thread-safe by design: a step is one thread, and a
    lock on the hot path is exactly the kind of cost that gets instrumentation
    turned off."""

    def __init__(self, name: str = "run"):
        self.root = Span(name)
        self._stack = [self.root]

    @contextmanager
    def phase(self, name: str):
        span = Span(name)
        self._stack[-1].children.append(span)
        self._stack.append(span)
        free0 = _dev_free_bytes()
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            span.seconds = time.perf_counter() - t0
            free1 = _dev_free_bytes()
            if free0 is not None and free1 is not None:
                span.dev_bytes = free0 - free1
            self._stack.pop()

    def count(self, name: str, n: int = 1) -> None:
        c = self._stack[-1].counters
        c[name] = c.get(name, 0) + n

    def gauge(self, name: str, value) -> None:
        self._stack[-1].counters[name] = value

    # -- output ------------------------------------------------------------
    def table(self) -> str:
        lines = [f"{'phase':<40} {'seconds':>9} {'dev GiB':>9}  counters"]

        def walk(span: Span, depth: int) -> None:
            label = ("  " * depth) + span.name
            mem = "" if span.dev_bytes is None else f"{span.dev_bytes / (1 << 30):+9.2f}"
            cs = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                          for k, v in span.counters.items())
            lines.append(f"{label:<40} {span.seconds:>9.3f} {mem:>9}  {cs}")
            for child in span.children:
                walk(child, depth + 1)

        for child in self.root.children:
            walk(child, 0)
        return "\n".join(lines)

    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=1)

    def as_dict(self) -> dict:
        return {"root": self.root.as_dict(), "memory_sampling": _MEM,
                "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
