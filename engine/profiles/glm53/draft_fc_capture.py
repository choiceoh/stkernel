"""Collect the decode FC pairs the draft bias fitter needs, from a serving boot.

`bench/draft_fc_bias.collect_fc_pairs` has existed with no caller: the fitter can fit and the boot
can bind `draft-fc-bias.json`, but nothing gathered the pairs, so production has run with
`draft_fc_bias_status="missing"` since the feature landed. This is that missing half.

It has to run inside a boot. `reader_identity` hashes the executed FP8 pack, the BF16 source, the
normalization AND the device name, and `prepare_bias` refuses a profile whose identity does not
match the reader it will correct -- so a bundle fitted against anything but the live prepared layer
is rejected at the next boot. Nothing here is offline.

What it takes is already on the hot path: `observe_rows` receives `aux [n*t, A]`, `positions [n, t]`
and `valid [n]`, and the committed-row mask is `arange(t) < valid` exactly as the projection's own
observer builds it. The request family per row comes from `note_sync`, which the adapter already
calls per slot with its sequence -- wrapping it keeps the sequence map current without touching the
decode path. Splitting by family (not by row) is what `fit_fc_bias` requires: it refuses train and
validation sets that share a request.

    DRAFT_FC_CAPTURE = True in the profile's boot, then
    bench/draft_tune.py fc-bias <rank>.pt --out draft-fc-bias.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

DEFAULT_ROWS = 4096
DEFAULT_VALIDATION_SHARE = 0.25


def _family_split(family: str, share: float, salt: str) -> str:
    """Deterministic per-family split; the same request never lands in both."""
    digest = hashlib.sha256((salt + '\0' + family).encode()).digest()
    return 'validation' if int.from_bytes(digest[:8], 'big') / 2 ** 64 < share else 'train'


class DraftFcCapture:
    """Accumulate committed decode FC inputs, then write one rank's pair bundle."""

    def __init__(self, drafter, root, *, rows=DEFAULT_ROWS, validation_share=DEFAULT_VALIDATION_SHARE,
                 salt=''):
        if not 0.0 < validation_share < 1.0:
            raise ValueError('validation share must lie strictly between 0 and 1')
        if type(rows) is not int or not 1 <= rows <= 65536:
            raise ValueError('the row budget must be a bounded positive integer')
        self.drafter, self.root = drafter, Path(root)
        self.rows, self.share, self.salt = rows, float(validation_share), str(salt)
        self.batches: list[dict] = []
        self.kept = {'train': 0, 'validation': 0}
        self.slot_family: dict[int, str] = {}
        self.stopped = None
        self._attached = False

    # -- attachment ------------------------------------------------------

    def attach(self):
        """Wrap `observe_rows` and `note_sync`. Both wrappers are pure additions."""
        if self._attached:
            raise ValueError('draft FC capture is already attached')
        drafter = self.drafter
        inner_rows = drafter.observe_rows

        def observe_rows(field, slots, positions, aux, valid):
            try:
                self._record(slots, positions, aux, valid)
            except Exception as exc:                       # capture never breaks the step
                self.stopped = self.stopped or f'{type(exc).__name__}: {exc}'
            return inner_rows(field, slots, positions, aux, valid)

        drafter.observe_rows = observe_rows
        diagnostics = getattr(drafter, 'diagnostics', None)
        if diagnostics is not None and hasattr(diagnostics, 'note_sync'):
            inner_sync = diagnostics.note_sync

            def note_sync(seq, context, slot, accepted, new, remaining, ends, **kwargs):
                try:
                    self.slot_family[int(slot)] = f'seq-{int(seq)}'
                except Exception:                          # a sequence id is a convenience, not a contract
                    pass
                return inner_sync(seq, context, slot, accepted, new, remaining, ends, **kwargs)

            diagnostics.note_sync = note_sync
        self._attached = True
        return self

    # -- recording -------------------------------------------------------

    def full(self) -> bool:
        return min(self.kept.values()) > 0 and sum(self.kept.values()) >= self.rows

    def _record(self, slots, positions, aux, valid):
        if self.stopped is not None or self.full():
            return
        n, t = positions.shape
        cols = self.drafter.dense['fc.weight'].cols
        flat = aux.reshape(-1, cols)
        if flat.shape[0] != n * t or flat.dtype != torch.bfloat16:
            raise ValueError('draft FC capture expects the row-ordered BF16 aux the projection reads')
        keep = (torch.arange(t, device=positions.device) < valid.view(n, 1)).reshape(n * t)
        mask = keep.cpu()
        if not int(mask.sum()):
            return
        # One batch per slot: a batch carries one request family, and `collect_fc_pairs` bounds a
        # batch at 32 rows -- the served field is already n*t <= 32, but a per-slot split keeps the
        # family constant inside a batch, which is what the fitter's disjointness needs.
        for row in range(n):
            slot = int(slots[row])
            family = self.slot_family.get(slot, f'slot-{slot}')
            split = _family_split(family, self.share, self.salt)
            lo, hi = row * t, (row + 1) * t
            rows_kept = int(mask[lo:hi].sum())
            if not rows_kept or self.kept[split] + rows_kept > self.rows:
                continue
            self.batches.append(dict(aux=flat[lo:hi].clone(), keep=keep[lo:hi].clone(),
                                     ids=[family] * t, split=split))
            self.kept[split] += rows_kept

    # -- output ----------------------------------------------------------

    def status(self) -> dict:
        return dict(kept=dict(self.kept), batches=len(self.batches), rows=self.rows,
                    families=len(set(b['ids'][0] for b in self.batches)), stopped=self.stopped)

    def close(self):
        """Run the collector against the live reader and write this rank's bundle."""
        rank = self.drafter.target.comm.rank
        out = self.root / f'draft-fc-pairs-rank{rank}.pt'
        report = dict(self.status(), rank=rank, path=str(out))
        if self.stopped is not None or not all(self.kept.values()):
            report['error'] = self.stopped or 'no committed rows in both splits'
            return report
        from bench.draft_fc_bias import collect_fc_pairs
        bundle = collect_fc_pairs(self.drafter, self.batches, max_rows=self.rows)
        self.root.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, out)
        report['reader_sha256'] = bundle['reader_sha256']
        report['written'] = out.stat().st_size
        (self.root / f'draft-fc-pairs-rank{rank}.json').write_text(json.dumps(report, sort_keys=True))
        return report


def attach(engine, root, **kwargs):
    """Arm the capture on a prepared drafter, or say why it cannot be armed."""
    drafter = getattr(engine, 'drafter', None)
    if drafter is None:
        raise ValueError('draft FC capture needs a prepared drafter')
    if drafter.p.get('fc.weight') is None:
        raise ValueError('draft FC capture needs the retained BF16 source (prepare without consume_weights)')
    layer = drafter.dense.get('fc.weight')
    if getattr(layer, 'observer', None) is not None:
        raise ValueError('draft FC capture needs an observer-free reader: finish calibration first')
    return DraftFcCapture(drafter, root, **kwargs).attach()
