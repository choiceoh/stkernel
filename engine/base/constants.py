"""Index constants a captured step reads, and may not create.

A decode step asks for the same `arange` over and over: once per segment in every
sparse-MLA layer, a handful of distinct lengths, identical every time. Built at
each ask they are an allocation and a kernel launch apiece -- six hundred of them
in a 45-layer step at four sequences, and the widest is the candidate index, two
mebibytes at the top capacity rung -- and each one becomes a node in the captured
graph that runs again on every replay. They are constants. Build them once.

The rule that makes keeping them safe is the one enforced here: a constant may
not be FIRST built while a capture is recording. Memory allocated during capture
belongs to that graph's private pool and is handed out again to the next capture
in it, so a constant built there would later alias whatever came next. Every
shape runs at least one warmup pass before it is captured (base/graphs), and that
pass is where these get built.

Only lengths from a bounded set belong here: a captured decode asks for the same
few forever, while an eager prefill's lengths follow the request and would grow
this without bound. Callers choose, which is why this is a function and not a
decorator.
"""
from __future__ import annotations

import torch

_IOTA: "dict[tuple, torch.Tensor]" = {}


def iota(n: int, device, dtype=torch.int64) -> torch.Tensor:
    """0..n-1 on `device`, built once and kept. Never write to what this returns."""
    key = (int(n), str(device), dtype)
    kept = _IOTA.get(key)
    if kept is None:
        if str(device).startswith("cuda") and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"iota({n}) was first asked for while a graph was recording: a constant built "
                f"inside a capture lives in that graph's pool, which the next capture takes "
                f"back. The shape's warmup pass is where it should have been built.")
        kept = _IOTA[key] = torch.arange(n, device=device, dtype=dtype)
    return kept


def fresh(n: int, device, dtype=torch.int64) -> torch.Tensor:
    """The same values, not kept: for lengths that follow the request."""
    return torch.arange(n, device=device, dtype=dtype)


_ZEROS: "dict[tuple, torch.Tensor]" = {}


def zeros(n: int, device, dtype=torch.int32) -> torch.Tensor:
    """n zeros on `device`, built once and kept, under the same rule as `iota`: the indexer logits
    kernel takes each query's first key as a vector that a decode step hands over as all zeros,
    one fill launch per row per layer when built at the call. Never write to what this returns."""
    key = (int(n), str(device), dtype)
    kept = _ZEROS.get(key)
    if kept is None:
        if str(device).startswith("cuda") and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"zeros({n}) was first asked for while a graph was recording: a constant built "
                f"inside a capture lives in that graph's pool, which the next capture takes "
                f"back. The shape's warmup pass is where it should have been built.")
        kept = _ZEROS[key] = torch.zeros(n, device=device, dtype=dtype)
    return kept


def forget() -> None:
    """Drop every kept constant. Tests only -- a live capture recorded their addresses."""
    _IOTA.clear()
    _ZEROS.clear()
