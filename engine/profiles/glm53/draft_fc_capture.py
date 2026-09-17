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
observer builds it.

Where it hooks matters. A production boot captures the observation into a CUDA graph, so
`drafter.observe_rows` runs once at capture and never again -- the pipeline calls
`drafter.decode_graphs.observe_rows` (or `observe_prepared_rows` when the plan observes early),
which fills the graph's input buffers and replays. Wrapping only the drafter's own method is how
the first armed production boot collected 0 rows. So all three seams are wrapped; the graph ones
are what fire on a served step, and `drafter.observe_rows` covers the graphless path (tree decode,
a boot with no decode graphs). The request family per row comes from `note_sync`, which the adapter already
calls per slot with its sequence -- wrapping it keeps the sequence map current without touching the
decode path. Splitting by family (not by row) is what `fit_fc_bias` requires: it refuses train and
validation sets that share a request.

Rows are held on the HOST, not the device. `aux` is 20,480 BF16 columns -- 40 KiB a row -- and a
4,096-row budget over eight-row slots is about 437 MiB. On the device that is a boot's worth of
arena; on the host it is a page cache. `collect_fc_pairs` wants them back on the source device, so
`close` moves each batch there one at a time and frees it again. That is what lets this ride a
serving boot instead of costing a fleet window: this is collection, not judgement, and collection
needs no exclusivity -- a speed measurement does, which is why `onepass` asks for the door alone.

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


def retain_source(drafter):
    """Keep an independent host reference before compact storage retires BF16 weights."""
    drafter.fc_capture_source = drafter.p['fc.weight'].detach().to('cpu', copy=True)


def source_weight(drafter):
    source = getattr(drafter, 'fc_capture_source', None)
    return source if source is not None else drafter.p.get('fc.weight')


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
        self.seams: list[str] = []
        self.calls, self.calls_budget = 0, max(8 * rows, 4096)
        self.flushed = None                            # the report, once written -- see `maybe_flush`
        self._restore: list[tuple] = []
        self._attached = False

    # -- attachment ------------------------------------------------------

    def attach(self):
        """Wrap the observation seams and `note_sync`. Every wrapper is a pure addition."""
        if self._attached:
            raise ValueError('draft FC capture is already attached')
        drafter = self.drafter
        inner_rows = drafter.observe_rows

        def observe_rows(field, slots, positions, aux, valid):
            self._try_record(slots, positions, aux, valid)
            return inner_rows(field, slots, positions, aux, valid)

        drafter.observe_rows = observe_rows
        self.seams = ['drafter.observe_rows']
        self._restore = [(drafter, 'observe_rows', inner_rows)]
        graphs = getattr(drafter, 'decode_graphs', None)
        if graphs is not None:
            # the served seam: these run on the host every step and fill the graph's inputs
            inner_graph_rows = getattr(graphs, 'observe_rows', None)
            if inner_graph_rows is not None:
                def graph_rows(slots, positions, aux, valid):
                    self._try_record(slots, positions, aux, valid)
                    return inner_graph_rows(slots, positions, aux, valid)

                graphs.observe_rows = graph_rows
                self.seams.append('decode_graphs.observe_rows')
                self._restore.append((graphs, 'observe_rows', inner_graph_rows))
            inner_prepared = getattr(graphs, 'observe_prepared_rows', None)
            if inner_prepared is not None:
                def graph_prepared(slots, positions, context, valid, aux):
                    self._try_record(slots, positions, aux, valid)
                    return inner_prepared(slots, positions, context, valid, aux)

                graphs.observe_prepared_rows = graph_prepared
                self.seams.append('decode_graphs.observe_prepared_rows')
                self._restore.append((graphs, 'observe_prepared_rows', inner_prepared))
        diagnostics = getattr(drafter, 'diagnostics', None)
        if diagnostics is not None and hasattr(diagnostics, 'note_sync'):
            inner_sync = diagnostics.note_sync

            def note_sync(seq, context, slot, accepted, new, remaining, ends, **kwargs):
                try:
                    # `seq` is the request id the adapter keys its limits by -- a string, not a number.
                    # int(seq) raised on every call, the except swallowed it, and the family fell back
                    # to the SLOT: three families a boot, and a 25% share that kept missing validation.
                    self.slot_family[int(slot)] = f'seq-{seq}'
                except Exception:                          # a sequence id is a convenience, not a contract
                    pass
                return inner_sync(seq, context, slot, accepted, new, remaining, ends, **kwargs)

            diagnostics.note_sync = note_sync
            self._restore.append((diagnostics, 'note_sync', inner_sync))
        self._attached = True
        return self

    def detach(self):
        """Put every wrapped callable back. This is what stops paying the per-step device sync."""
        for owner, name, inner in reversed(self._restore):
            try:
                setattr(owner, name, inner)
            except Exception:                          # noqa: BLE001 -- a restore never breaks a step
                pass
        self._restore, self._attached = [], False
        return self

    # -- recording -------------------------------------------------------

    def _try_record(self, slots, positions, aux, valid):
        try:
            self._record(slots, positions, aux, valid)
        except Exception as exc:                           # capture never breaks the step
            self.stopped = self.stopped or f'{type(exc).__name__}: {exc}'

    def full(self) -> bool:
        """Recording is over: the rows are in and two families exist, or the door ran out of chances.

        NOT `min(kept) > 0`. The hash split leaves validation empty most of the time on a door with a
        handful of families, and `rebalance` is what fixes that -- at close, by moving one whole family.
        Waiting here for a balance that only close can produce would keep the collector recording (and
        paying its per-step device sync) long past the point where it already had everything it needs.

        The call budget is the backstop for a door that never fills: after it, `maybe_flush` files
        whatever the refusal reason is and detaches, so the sync always stops."""
        if self.calls >= self.calls_budget:
            return True
        if sum(self.kept.values()) < self.rows:
            return False
        return len({batch['ids'][0] for batch in self.batches}) >= 2       # rebalance needs two

    def _record(self, slots, positions, aux, valid):
        if self.stopped is not None or self.full():
            return
        self.calls += 1
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
            # host-side: see the module docstring. `close` returns them to the source device.
            self.batches.append(dict(aux=flat[lo:hi].to('cpu', copy=True), keep=keep[lo:hi].cpu(),
                                     ids=[family] * t, split=split))
            self.kept[split] += rows_kept

    # -- output ----------------------------------------------------------

    def status(self) -> dict:
        return dict(kept=dict(self.kept), batches=len(self.batches), rows=self.rows,
                    families=len(set(b['ids'][0] for b in self.batches)), stopped=self.stopped,
                    seams=list(self.seams), calls=self.calls)

    def rebalance(self) -> str | None:
        """Move whole families until both splits are fed, or say why that is impossible.

        The hash split gives each family the requested share independently, so a door that served
        four requests has a 0.75^4 = 32% chance of putting none of them in validation -- and then
        `close` threw away everything it had collected. Two production boots lost 1,971 and 421 rows
        that way. Families move whole: `fit_fc_bias` refuses a train and a validation set that share
        a request, and that is the invariant this preserves."""
        empty = [name for name, kept in self.kept.items() if not kept]
        if not empty:
            return None
        if len(empty) == len(self.kept):
            return 'nothing was collected'
        families = {}
        for batch in self.batches:
            families.setdefault(batch['ids'][0], []).append(batch)
        if len(families) < 2:
            return f'one family ({next(iter(families), None)}) cannot fill two splits'
        want = empty[0]
        # the family the salt ranks first: deterministic, and independent of arrival order
        name = min(families, key=lambda f: hashlib.sha256((self.salt + '\0' + f).encode()).digest())
        moved = 0
        for batch in families[name]:
            moved += int(batch['keep'].sum())
            batch['split'] = want
        self.kept[want] += moved
        for name_other in self.kept:
            if name_other != want:
                self.kept[name_other] -= moved
        return None

    def maybe_flush(self):
        """Write the bundle the moment the budget is met, then take the collector off the step.

        `close` used to be the only writer and it runs at shutdown -- but a deploy stops production with
        `docker rm -f`, a SIGKILL, so no Python runs and everything collected is lost. Every close report
        this feature ever produced came from a CRASH (which does unwind), never from the deploys that
        actually stop production. Writing at the budget instead makes the artifact independent of how the
        boot ends, and detaching afterwards is what stops paying the per-step device sync.

        The write costs roughly a second once: the reader runs over the collected batches and the bundle
        goes to disk. It is deliberately taken on the step loop's own after-step seam (base/serve calls
        `engine.housekeeping`), where the profile's calibration filing already takes the same kind of cost,
        instead of on a thread that would put CUDA work beside a replaying graph.
        """
        if self.flushed is not None or not self._attached or self.stopped is not None or not self.full():
            return None
        self.detach()                                  # first: a failed write must not keep charging the step
        try:
            self.flushed = self.close()
        except Exception as exc:                       # noqa: BLE001 -- a collector never breaks serving
            self.flushed = dict(self.status(), error=f'{type(exc).__name__}: {exc}')
        self.batches = []                              # 437 MiB of host pages: the bundle is written, drop them
        return self.flushed

    def close(self):
        """Run the collector against the live reader and write this rank's bundle."""
        rank = self.drafter.target.comm.rank
        out = self.root / f'draft-fc-pairs-rank{rank}.pt'
        why = None if self.stopped is not None else self.rebalance()
        report = dict(self.status(), rank=rank, path=str(out))
        if self.stopped is not None or why is not None or not all(self.kept.values()):
            report['error'] = self.stopped or why or 'no committed rows in both splits'
            self._file(report)          # even a refusal leaves evidence: a container's log does not outlive it
            return report
        from bench.draft_fc_bias import collect_fc_pairs
        device = self.drafter.p['hidden_norm.weight'].device
        source = source_weight(self.drafter).to(device)

        def on_device():
            for batch in self.batches:                 # one at a time: the whole set never lands at once
                yield dict(batch, aux=batch['aux'].to(device), keep=batch['keep'].to(device))

        bundle = collect_fc_pairs(self.drafter, on_device(), max_rows=self.rows, source=source)
        self.root.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, out)
        report['reader_sha256'] = bundle['reader_sha256']
        report['written'] = out.stat().st_size
        self._file(report)
        return report

    def _file(self, report):
        """The report beside the bundle. A boot's container log is replaced by the next arm's."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / f'draft-fc-pairs-rank{report["rank"]}.json').write_text(
                json.dumps(report, sort_keys=True, default=str))
        except Exception:                              # a shutdown never fails on its own record
            pass


def attach(engine, root, **kwargs):
    """Arm the capture on a prepared drafter, or say why it cannot be armed."""
    drafter = getattr(engine, 'drafter', None)
    if drafter is None:
        raise ValueError('draft FC capture needs a prepared drafter')
    if source_weight(drafter) is None:
        raise ValueError('draft FC capture needs the retained BF16 source (retain before compaction)')
    layer = drafter.dense.get('fc.weight')
    if getattr(layer, 'observer', None) is not None:
        raise ValueError('draft FC capture needs an observer-free reader: finish calibration first')
    return DraftFcCapture(drafter, root, **kwargs).attach()
