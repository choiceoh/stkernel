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
from pathlib import Path

# D11: no undeclared STK_* read here (base/config would kill a boot that set it);
# sampling already waits for torch.cuda.is_initialized(), so it is simply on.
_MEM = True


def process_seconds():
    """Seconds since THIS PROCESS started, or None where /proc does not say.

    A recorder can only measure from the moment it exists, and a boot's recorder is opened after
    torch and the kernel modules are imported -- 9.17 s of a measured 87 s boot sat before its first
    row and had to be recovered from container timestamps by hand (boot-time study 5-f). A module
    level `perf_counter()` would not close that either: python's own startup and every import above
    the stamp are still outside it.

    /proc/self/stat's 22nd field is when this process started, in clock ticks since the machine did,
    and /proc/stat's `btime` is when that was. The two give the one number nobody could read off the
    ledger: how long the boot had been running when its first phase opened.
    """
    try:
        stat = Path("/proc/self/stat").read_text()
        started = float(stat[stat.rindex(")") + 1:].split()[19])          # field 22, past the comm
        hz = os.sysconf("SC_CLK_TCK")
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return max(0.0, time.time() - (int(line.split()[1]) + started / hz))
    except (OSError, ValueError, AttributeError, IndexError, ZeroDivisionError):
        return None
    return None


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
    calls: int = 0

    def as_dict(self) -> dict:
        out = {"name": self.name, "seconds": round(self.seconds, 6)}
        if self.calls > 1:
            out["calls"] = self.calls
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

    def __init__(self, name: str = "run", *, memory_sampling: bool = True):
        self.root = Span(name)
        self._stack = [self.root]
        self._aggregate_spans = {}
        self.memory_sampling = memory_sampling

    @contextmanager
    def phase(self, name: str, *, aggregate: bool = False):
        """Measure one phase, optionally accumulating repeated calls in one span.

        The runner aggregates its two step kinds so a long-lived engine keeps
        bounded instrumentation instead of retaining one object per step.
        Boot and diagnostic phases keep their individual timing by default.
        """
        parent = self._stack[-1]
        key = (id(parent), name)
        span = self._aggregate_spans.get(key) if aggregate else None
        if span is None:
            span = Span(name)
            parent.children.append(span)
            if aggregate:
                self._aggregate_spans[key] = span
        self._stack.append(span)
        free0 = _dev_free_bytes() if self.memory_sampling else None
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            span.seconds += time.perf_counter() - t0
            span.calls += 1
            free1 = _dev_free_bytes() if self.memory_sampling else None
            if free0 is not None and free1 is not None:
                span.dev_bytes = (span.dev_bytes or 0) + free0 - free1
            self._stack.pop()

    def mark(self, name: str, seconds: float, **counters) -> Span:
        """A phase that was timed somewhere this recorder could not reach -- the boot's `front`.

        It lands as a row like any other, in call order, so a table that opens with it reads as the
        whole of the thing instead of the part that happened to be instrumented."""
        span = Span(name, seconds=float(seconds), counters=dict(counters), calls=1)
        self._stack[-1].children.append(span)
        return span

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
            if span.calls > 1:
                cs = f"calls={span.calls} {cs}".rstrip()
            lines.append(f"{label:<40} {span.seconds:>9.3f} {mem:>9}  {cs}")
            for child in span.children:
                walk(child, depth + 1)

        for child in self.root.children:
            walk(child, 0)
        # The sum of the top level, said out loud. Every reader of a boot table has wanted it and every
        # one of them has had to get it from container timestamps instead (the study, twice).
        total = sum(child.seconds for child in self.root.children)
        lines.append(f"{'total':<40} {total:>9.3f}")
        return "\n".join(lines)

    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=1)

    def as_dict(self) -> dict:
        return {"root": self.root.as_dict(), "memory_sampling": _MEM and self.memory_sampling,
                "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
