"""Dump one sparse-indexer selection with the operands it was made from.

The 2026-09-17 incident's last unverified path is the selection itself: the
selector *kernels* are held to torch.topk on synthetic logits and the KDA
operands were audited on real ones, but the pool ids the indexer actually picks
for a real long prefix have never been compared with a reference recomputed from
the same operands. The indexer is also the only path that engages for long
contexts and content-dependently -- a context shorter than `topk_pools * kpool`
takes the covered path and never selects.

`ST_SELECTION_CAPTURE=<dir>` turns this on; unset, every call returns before it
touches a tensor. One file per (layer, rows) of the FIRST prefill selection the
layer makes, so a boot's own replay writes a handful of dumps and nothing else.
The comparison lives in `tools/selection_reference.py`; the dump is private
activation data and stays outside git, like every other capture here.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV = "ST_SELECTION_CAPTURE"
LAYERS_ENV = "ST_SELECTION_CAPTURE_LAYERS"


def directory():
    """The capture directory, or None when the dump is off."""
    value = os.environ.get(ENV)
    return Path(value) if value else None


def wanted(layer: int) -> bool:
    raw = os.environ.get(LAYERS_ENV)
    if not raw:
        return True
    return str(layer) in {piece.strip() for piece in raw.split(",") if piece.strip()}


def from_net(net, layer, **fields):
    """Dump one selection for a net that may not carry the attribute: tests build stubs."""
    capture = getattr(net, "selection_capture", None)
    if capture is not None:
        capture(layer, **fields)


class SelectionCapture:
    """One dump per layer, from the operands `_select_pools` was called with."""

    def __init__(self, where=None):
        self.where = Path(where) if where else directory()
        self.done = set()
        self.rows_seen = {}

    @property
    def enabled(self):
        return self.where is not None

    def __call__(self, layer, *, q8, w_eff, keys, scales, ke, n_cand, k, selected, prefill: bool):
        if not self.enabled or not wanted(layer) or not prefill or layer in self.done:
            return None
        import torch
        self.done.add(layer)
        self.where.mkdir(parents=True, exist_ok=True)
        target = self.where / f"selection-L{layer}-rows{int(q8.shape[0])}.pt"
        torch.save(dict(layer=int(layer), rows=int(q8.shape[0]), n_cand=int(n_cand), k=int(k),
                        heads=int(q8.shape[1]), width=int(q8.shape[2]), prefill=bool(prefill),
                        q8=q8.detach().to("cpu"), w_eff=w_eff.detach().float().to("cpu"),
                        keys=keys.detach().to("cpu"), scales=scales.detach().float().to("cpu"),
                        ke=ke.detach().to("cpu"), selected=selected.detach().to("cpu")), target)
        print(f"[selection-capture] L{layer} rows={int(q8.shape[0])} n_cand={int(n_cand)} k={int(k)} -> {target}",
              flush=True)
        return target
