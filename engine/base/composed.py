"""A composition served: the position-addressed store over blocks and slots, and the model the runner drives (base).

Two pieces, both model-free:

`PositionStore` is engine/base/composition's State protocol over the engine's memory. A feature's per-token rows
(`put_rows`/`rows`) live in a BlockPool's blocks by POSITION -- block position // block_tokens, row position %
block_tokens -- so a sequence's history is its block table and a cached prefix's blocks are adopted, never copied
(base/prefix). Its whole values (`get`/`put`) live in the sequence's fixed slot (base/kv.SlotPool), carved from the
specs the features declare (base/cache_spec: key, dtype, shape) -- so a slot's bytes are one contiguous region a tier
moves (base/tiered_kv) and a prefix boundary's state is one copy of them into a snapshot. What the store holds after
a step is the state after the step's last token -- except for a verify step (a drafter's provisional tokens): its rows
are written at their positions like any others (a rejected one is overwritten by the next step's writes, as GLM-5.3's
are), and the value after EACH of its tokens goes to the slot's ring (`ring` copies of the slot's regions) until
`accept(seq, n)` copies the value after token n-1 back into the slot. Rolling back is choosing a position.

`ComposedModel` answers the runner's Model protocol and the door's engine surface (base/serve) for any composition:
tokens and limits per row, the prefill split at the prefix cache's marks, sampling through base/sampler with base/draws'
uniforms keyed the way GLM-5.3's are, parking as a host record beside the slot's bytes. Every option the door admits is
served through the same base functions every engine uses: a row with penalties, logit_bias, logprobs, a seed, a reasoning budget, a grammar
(base/grammar, when a compiler is bound) or min_tokens still ahead has its logits processed by base/sampler.process_logits
over its own history and picked on its own; the other rows are drawn together. Without a drafter a decode
step is one token a row (horizon = context + 1). With one (`Drafter` below) it is GLM-5.3's verification by position:
the step feeds [last token] + the row's drafts as a verify segment, samples every position with the uniform the
same generation count would draw without drafts, accepts drafts while the sample agrees, appends accepted + 1 tokens
and accepts that many into the store -- so a drafter changes how many tokens a step yields, never which. The math is
the composition's; nothing here knows a model.

Pictures (a door with a vision tower, base/serve): the request's records -- kind, digest, the placeholder positions their
rows replace, the canvas, the grid -- are kept a row, their positions absolute in the row's tokens (a continued turn's
are re-based past the history), and handed to a composition that declares it sees them (`sees_media`): `check_media(
tokens, records)` refuses what it cannot serve before the row changes, `bind_media(seq, tokens, records)` adopts them
once it has (its rotary layout, say), `forget_media(seq)` drops them, and `forward(..., media=records)` receives them
with every prefill piece, which encodes what that piece reaches. `media_marks` salts the prefix cache; a parked row keeps the
marks and grids, not the canvases -- the caches hold the pictures' effect. A composition without `sees_media` serves
text only.
"""
from __future__ import annotations

from math import isfinite

import torch

from engine.base import draws
from engine.base.cache_spec import PagedSpec, Plan, SlotSpec, _ITEMSIZE
from engine.base.composition import Composition, Step
from engine.base.kv import EMPTY, BlockPool, SlotPool
from engine.base.sampler import needs_rich_sampler, sample, validate_options

ALIGN = 64                      # every region inside a block or a slot starts on this boundary (dtype views need it)
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float8_e4m3fn": torch.float8_e4m3fn, "int64": torch.int64, "int32": torch.int32, "uint8": torch.uint8}


def _aligned(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class Layout:
    """Where each keyed spec's bytes sit: in a block ([region][token][element]) or in a slot ([region][element]), one
    region per model layer the spec is kept for (`layers_by_key`: composition.spec_layers)."""

    def __init__(self, paged: "list[PagedSpec]", slots: "list[SlotSpec]", block_tokens: int, layers_by_key: dict):
        self.block_tokens = block_tokens
        self.paged, self.slot, self.index = {}, {}, {}
        at = 0
        for spec in paged:
            at = self._place(self.paged, spec, at, block_tokens * spec.bytes_per_token, layers_by_key)
        self.block_bytes = at
        at = 0
        for spec in slots:
            at = self._place(self.slot, spec, at, spec.bytes_per_seq, layers_by_key)
        self.slot_bytes = at

    def _place(self, table, spec, at, per_layer, layers_by_key) -> int:
        if not spec.key or spec.key not in layers_by_key or len(layers_by_key[spec.key]) != spec.layers:
            raise ValueError(f"{spec.name}: a store needs a keyed spec (key, dtype, shape) whose layers the plan names")
        table[spec.key] = (at, spec)
        self.index[spec.key] = {layer: i for i, layer in enumerate(layers_by_key[spec.key])}
        return at + _aligned(spec.layers * per_layer)

    def region(self, key: str, layer: int) -> int:
        """The rank of model `layer` among the layers `key` is kept for."""
        try:
            return self.index[key][layer]
        except KeyError:
            raise ValueError(f"{key} is not kept for layer {layer}") from None

    def plan(self, kv_gib: float, max_seqs: int, sector: int = 4096) -> Plan:
        """base/cache_spec.plan's split with this layout's block and slot sizes (its rounding included)."""
        block_bytes = max(sector, -(-self.block_bytes // sector) * sector)     # a model without paged rows still pages
        slots_bytes = self.slot_bytes * (max_seqs + 1)
        left = int(kv_gib * (1 << 30)) - slots_bytes
        if left <= 0:
            raise MemoryError(f"{max_seqs} slots need {slots_bytes / 2**20:.1f} MiB of a {kv_gib:.3f} GiB budget")
        return Plan(self.block_tokens, block_bytes, left // block_bytes, self.slot_bytes, max_seqs + 1,
                    (left // block_bytes) * block_bytes / (1 << 30), slots_bytes / (1 << 30))


def merged_specs(compositions) -> "tuple[list[PagedSpec], list[SlotSpec], dict]":
    """One layout's specs for several compositions over the same sequences -- a target and its drafter's heads: a key
    two of them keep is one spec whose layers are both's (the same element layout required), and no model layer is
    claimed twice (a head takes an `offset` past the target's layers)."""
    import dataclasses
    tables, layers = ({}, {}), {}
    for composition in compositions:
        for table, specs in zip(tables, composition.cache_specs()):
            for spec in specs:
                held = table.get(spec.key)
                if held is None:
                    table[spec.key] = spec
                    continue
                size = "bytes_per_token" if isinstance(spec, PagedSpec) else "bytes_per_seq"
                if (held.dtype, tuple(held.shape), getattr(held, size)) != (spec.dtype, tuple(spec.shape), getattr(spec, size)):
                    raise ValueError(f"{spec.key}: two compositions keep it with different element layouts")
                table[spec.key] = dataclasses.replace(held, layers=held.layers + spec.layers)
        for key, kept in composition.spec_layers().items():
            layers.setdefault(key, []).extend(kept)
    for key, kept in layers.items():
        if len(set(kept)) != len(kept):
            raise ValueError(f"{key}: two compositions claim the same model layer (give the drafter's heads an offset)")
        layers[key] = sorted(kept)
    return list(tables[0].values()), list(tables[1].values()), layers


class PositionStore:
    """The served State (engine/base/composition.State's protocol): rows by position in blocks, values in slots."""

    def __init__(self, layout: Layout, pool: BlockPool, slot_view: torch.Tensor, snapshot_view: "torch.Tensor | None" = None,
                 device=None, ring_view: "torch.Tensor | None" = None, ring: int = 0):
        if pool.storage is None or pool.block_bytes < layout.block_bytes or pool.block_size != layout.block_tokens:
            raise ValueError("the pool must carry storage for the layout's block bytes at its block size")
        if slot_view.dtype != torch.uint8 or slot_view.numel() < pool.max_seqs * layout.slot_bytes:
            raise ValueError("the slot view is uint8, one slot region per pool row (slot 0 included)")
        if ring and (ring_view is None or ring_view.dtype != torch.uint8
                     or ring_view.numel() < pool.max_seqs * ring * layout.slot_bytes):
            raise ValueError("a ring is uint8, `ring` slot regions per pool row")
        self.layout, self.pool, self.slots = layout, pool, slot_view
        self.snapshots = snapshot_view
        self.rings, self.ring = ring_view, ring
        self.device = device if device is not None else slot_view.device
        self.contexts: dict = {}                     # seq -> tokens computed (the state is the one after them)
        self.slot_of: dict = {}
        self._step: dict = {}                        # seq -> (ctx, length, verify) of the step being run
        self._verifying: dict = {}                   # seq -> (ctx, length) of a verify step awaiting accept
        self._kept: dict = {}                        # seq -> {(layer, key)} the verify step wrote to the ring
        self._accepted: dict = {}                    # seq -> (ctx, n, kept) of its last accepted verify step

    def lane(self) -> "PositionStore":
        """Another State over the same memory -- the same blocks, slots, snapshots and rings, the same open sequences --
        with contexts of its own: a drafter's heads write their layers' rows at positions the target has already
        passed, and step back after a provisional chain (`place`)."""
        import copy
        view = copy.copy(self)
        view.contexts, view._step, view._verifying, view._kept, view._accepted = {}, {}, {}, {}, {}
        return view

    def place(self, seq: int, position: int) -> None:
        """The sequence stands at `position` in this state: rows past it are the next step's to overwrite, rows before
        it are the ones already written there (by this lane, or in blocks the row adopted). A lane whose features keep
        no per-sequence values is the only one this is right for -- values past `position` would stay."""
        if seq not in self.slot_of:
            raise ValueError(f"sequence {seq} is not open")
        if seq in self._verifying:
            raise ValueError(f"sequence {seq} has a verify step waiting for accept")
        if position < 0:
            raise ValueError("a position is nonnegative")
        self.contexts[seq] = position
        self._accepted.pop(seq, None)

    # -- rows and slots ------------------------------------------------------------------------------------------
    def open(self, seq: int, slot: int) -> None:
        if not 0 < slot < self.pool.max_seqs:
            raise ValueError(f"slot {slot} is outside the pool's rows (slot 0 is the null slot)")
        self.slot_of[seq] = slot
        self.contexts[seq] = 0
        self.slot_bytes(slot).zero_()

    def close(self, seq: int) -> None:
        self.slot_of.pop(seq, None)
        self.contexts.pop(seq, None)
        self._verifying.pop(seq, None)
        self._kept.pop(seq, None)
        self._accepted.pop(seq, None)

    def drop(self, seq: int) -> None:
        self.close(seq)

    def slot_bytes(self, slot: int) -> torch.Tensor:
        n = self.layout.slot_bytes
        return self.slots[slot * n:(slot + 1) * n]

    def snapshot_bytes(self, snap: int) -> torch.Tensor:
        if self.snapshots is None:
            raise ValueError("this store keeps no prefix snapshots")
        n = self.layout.slot_bytes
        return self.snapshots[snap * n:(snap + 1) * n]

    def _slot_view(self, layer: int, key: str, seq: int) -> torch.Tensor:
        at, spec = self.layout.slot[key]
        i = self.layout.region(key, layer)
        region = self.slot_bytes(self.slot_of[seq])[at + i * spec.bytes_per_seq:at + (i + 1) * spec.bytes_per_seq]
        return region.view(DTYPES[spec.dtype]).view(*spec.shape)

    def _ring_region(self, layer: int, key: str, seq: int, j: int) -> torch.Tensor:
        at, spec = self.layout.slot[key]
        i = self.layout.region(key, layer)
        n = self.layout.slot_bytes
        base = (self.slot_of[seq] * self.ring + j) * n
        return self.rings[base + at + i * spec.bytes_per_seq:base + at + (i + 1) * spec.bytes_per_seq]

    def get(self, layer: int, key: str, seq: int, default=None):
        """None until the sequence has computed a token (a feature's first step starts from its own zero)."""
        if self.contexts.get(seq, 0) == 0:
            return default
        return self._slot_view(layer, key, seq)

    def put(self, layer: int, key: str, seq: int, value, at: "int | None" = None) -> None:
        """The value after the step; in a verify step, the value after its token `at` (to the ring)."""
        step = self._step.get(seq)
        if step is not None and step[2]:
            if at is None or not 0 <= at < step[1]:
                raise ValueError(f"{key}: a verify step keeps the value after each of its {step[1]} tokens")
            _, spec = self.layout.slot[key]
            region = self._ring_region(layer, key, seq, at).view(DTYPES[spec.dtype]).view(*spec.shape)
            if tuple(value.shape) != tuple(region.shape):
                raise ValueError(f"{key}: a value is {tuple(region.shape)}, got {tuple(value.shape)}")
            region.copy_(value)
            self._kept.setdefault(seq, set()).add((layer, key))
            return
        if at is not None:
            raise ValueError(f"{key}: only a verify step keeps values by offset")
        view = self._slot_view(layer, key, seq)
        if tuple(value.shape) != tuple(view.shape):
            raise ValueError(f"{key}: a value is {tuple(view.shape)}, got {tuple(value.shape)}")
        view.copy_(value)

    def _block_view(self, block: int, layer: int, key: str) -> torch.Tensor:
        at, spec = self.layout.paged[key]
        i = self.layout.region(key, layer)
        per_layer = self.layout.block_tokens * spec.bytes_per_token
        region = self.pool.block(block)[at + i * per_layer:at + (i + 1) * per_layer]
        return region.view(DTYPES[spec.dtype]).view(self.layout.block_tokens, *spec.shape)

    def put_rows(self, layer: int, key: str, seq: int, rows) -> None:
        """This step's rows for `seq` at its positions ctx.. (the step named them to `check`)."""
        ctx, length, _ = self._step[seq]
        if rows.shape[0] != length:
            raise ValueError(f"{key}: the step holds {length} tokens of sequence {seq}, got {rows.shape[0]} rows")
        row = self.pool.row(seq)
        width = self.layout.block_tokens
        done = 0
        while done < length:
            position = ctx + done
            block, offset = row[position // width], position % width
            if block == EMPTY:
                raise ValueError(f"sequence {seq} holds no block for position {position}")
            n = min(length - done, width - offset)
            self._block_view(block, layer, key)[offset:offset + n].copy_(rows[done:done + n])
            done += n

    def rows(self, layer: int, key: str, seq: int, count: int) -> torch.Tensor:
        """The rows of positions [0, count), gathered from the row's blocks (a copy: the served kernels read the
        block table instead)."""
        held = self.contexts.get(seq, 0) + (self._step[seq][1] if seq in self._step else 0)
        if count > held:
            raise ValueError(f"sequence {seq} holds {held} positions of {key}, asked {count}")
        width = self.layout.block_tokens
        row = self.pool.row(seq)
        parts = [self._block_view(row[b], layer, key)[:min(width, count - b * width)] for b in range(-(-count // width))]
        return torch.cat(parts) if len(parts) != 1 else parts[0]

    # -- the step ----------------------------------------------------------------------------------------------------
    def check(self, step: Step) -> None:
        self._step = {}
        for s in step.segments:
            if s.seq not in self.slot_of:
                raise ValueError(f"sequence {s.seq} is not open")
            if s.seq in self._verifying:
                raise ValueError(f"sequence {s.seq} has a verify step waiting for accept")
            if self.contexts[s.seq] != s.ctx:
                raise ValueError(f"sequence {s.seq} is at {self.contexts[s.seq]} tokens, the step says {s.ctx}")
            if s.verify and s.length > self.ring:
                raise ValueError(f"a verify segment of {s.length} tokens needs a ring of {s.length}, the store keeps {self.ring}")
            row = self.pool.row(s.seq)
            need = -(-(s.ctx + s.length) // self.layout.block_tokens)
            if need > len(row) or any(row[b] == EMPTY for b in range(need)):
                raise ValueError(f"sequence {s.seq} holds fewer blocks than {s.ctx + s.length} positions need")
            self._step[s.seq] = (s.ctx, s.length, s.verify)
            self._accepted.pop(s.seq, None)                  # the ring is this step's from here on
            if s.verify:
                self._kept[s.seq] = set()

    def commit(self, step: Step) -> None:
        for s in step.segments:
            if s.verify:
                self._verifying[s.seq] = (s.ctx, s.length)
            else:
                self.contexts[s.seq] = s.ctx + s.length
        self._step = {}

    def accept(self, seq: int, n: int) -> None:
        """Keep the first `n` tokens of the sequence's verify step: the ring's values after token n-1 into the slot,
        the context ctx + n (the rows past it are the next step's to overwrite)."""
        if seq not in self._verifying:
            raise ValueError(f"sequence {seq} has no verify step to accept")
        ctx, length = self._verifying[seq]
        if not 1 <= n <= length:
            raise ValueError(f"accept keeps 1..{length} tokens of sequence {seq}'s verify step, not {n}")
        del self._verifying[seq]
        kept = self._kept.pop(seq, set())
        slot = self.slot_bytes(self.slot_of[seq])
        for layer, key in kept:
            lo, hi = self._slot_span(layer, key)
            slot[lo:hi].copy_(self._ring_region(layer, key, seq, n - 1))
        self.contexts[seq] = ctx + n
        self._accepted[seq] = (ctx, n, kept)

    def _slot_span(self, layer: int, key: str) -> "tuple[int, int]":
        at, spec = self.layout.slot[key]
        i = self.layout.region(key, layer)
        return at + i * spec.bytes_per_seq, at + (i + 1) * spec.bytes_per_seq

    # -- prefix boundaries and parking --------------------------------------------------------------------------------
    def checkpoint(self, seq: int, position: int, snap: int) -> None:
        """The state after `position` tokens into snapshot `snap`: the slot's, when the sequence stands there -- or,
        for a boundary inside the verify step it just accepted (a decode step crossing a block boundary), the slot's
        bytes with the values that step kept taken from the ring at that position."""
        here = self.contexts.get(seq)
        if here == position:
            self.snapshot_bytes(snap).copy_(self.slot_bytes(self.slot_of[seq]))
            return
        last = self._accepted.get(seq)
        if last is None or not last[0] < position < last[0] + last[1]:
            raise ValueError(f"sequence {seq} stands at {here}, not at {position}")
        ctx, _, kept = last
        target = self.snapshot_bytes(snap)
        target.copy_(self.slot_bytes(self.slot_of[seq]))
        for layer, key in kept:
            lo, hi = self._slot_span(layer, key)
            target[lo:hi].copy_(self._ring_region(layer, key, seq, position - ctx - 1))

    def restore(self, seq: int, position: int, snap: int) -> None:
        """The sequence starts at `position` with the snapshot's state; its rows before it are the adopted blocks'."""
        self.slot_bytes(self.slot_of[seq]).copy_(self.snapshot_bytes(snap))
        self.contexts[seq] = position

    def resume(self, seq: int, slot: int, context: int) -> None:
        """The row reopened in `slot`, whose bytes the tier restored: nothing is zeroed."""
        self.slot_of[seq] = slot
        self.contexts[seq] = context


def store_for(composition: Composition, kv_gib: float, max_seqs: int, block_tokens: int, *, snapshots: int = 0,
              device="cpu", ring: int = 0, also=()) -> "tuple[PositionStore, BlockPool, SlotPool, Plan]":
    """A store, its pools and the plan behind them, sized from the composition's cache specs on `device` -- the
    reference lane's arena: plain tensors (a profile's boot carves the same regions out of base/arena). `ring`: the
    longest verify segment it keeps per-token values for (a drafter's k + 1). `also`: compositions whose rows live in
    the same blocks (a drafter's heads, `merged_specs`)."""
    paged, slots, layers = merged_specs((composition, *also))
    layout = Layout(paged, slots, block_tokens, layers)
    plan = layout.plan(kv_gib, max_seqs)
    pool = BlockPool(plan.num_blocks, block_tokens, max_seqs=plan.num_slots, max_blocks_per_seq=plan.num_blocks)
    pool.attach_storage(torch.zeros(plan.num_blocks * plan.block_bytes, dtype=torch.uint8, device=device), plan.block_bytes)
    slot_view = torch.zeros(plan.num_slots * max(layout.slot_bytes, 1), dtype=torch.uint8, device=device)
    snapshot_view = (torch.zeros(snapshots * max(layout.slot_bytes, 1), dtype=torch.uint8, device=device)
                     if snapshots else None)
    ring_view = (torch.zeros(plan.num_slots * ring * max(layout.slot_bytes, 1), dtype=torch.uint8, device=device)
                 if ring else None)
    return (PositionStore(layout, pool, slot_view, snapshot_view, device, ring_view, ring), pool,
            SlotPool(plan.num_slots), plan)


class Drafter:
    """What a ComposedModel drafts with (the protocol; engine/modules/mtp's heads answer it).

    `k`: the most drafts a row proposes a step. `observe(seq, ctx, next_ids, hidden)`: the target just computed
    positions ctx..ctx+L-1 of `seq` and kept them -- `hidden` [L, ...] the residual state before its closing mix at each
    (Composition.forward(hidden=True)), `next_ids` [L] the token AT each position + 1 (the prompt's next, the accepted
    draft, or the token just sampled and not yet fed); called once per kept position, in order. `propose(seqs)`: up to
    k draft ids per row, continuing after the row's last token. `forget(seq)`: the row is gone or parked -- its next
    proposal may be empty until it has observed again."""
    k = 0

    def observe(self, seq: int, ctx: int, next_ids: "list[int]", hidden: torch.Tensor) -> None:
        pass

    def propose(self, seqs) -> "list[list[int]]":
        return [[] for _ in seqs]

    def forget(self, seq: int) -> None:
        pass


REFUSED_OPTIONS = ()             # every base/sampler option is served (a grammar needs a bound compiler)


class ComposedModel:
    """The runner's Model and the door's engine over a Composition and a PositionStore (see the module docstring)."""

    k = 0                                            # no drafter: a decode step is one token per row

    def __init__(self, composition: Composition, store: PositionStore, *, vocab: int, eos_ids, max_new: int = 256,
                 temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, max_context: int = 2 ** 31 - 1,
                 drafter: "Drafter | None" = None, grammars=None):
        if type(vocab) is not int or vocab <= 0 or type(max_new) is not int or max_new <= 0:
            raise ValueError("a composed model needs a positive vocabulary and generation limit")
        if drafter is not None and drafter.k and store.ring < drafter.k + 1:
            raise ValueError(f"a drafter of k={drafter.k} verifies {drafter.k + 1} tokens; the store's ring keeps {store.ring}")
        self.composition, self.store = composition, store
        self.vocab, self.eos = vocab, set(int(t) for t in eos_ids)
        self.max_new, self.temperature, self.top_p, self.seed, self.max_context = max_new, temperature, top_p, seed, max_context
        self.tokens, self.prompt_len, self.limits, self.min_new, self.options, self.ends = {}, {}, {}, {}, {}, {}
        self.seeds, self.nonces = {}, {}
        self.admissions = 0
        self.steps = 0
        self.drafter = drafter
        if drafter is not None:
            self.k = drafter.k
        self.grammars = grammars                         # base/grammar.Grammars (or its protocol): structured output
        self.history = None                              # base/sampler.History, built at the first row with penalties
        self.matchers, self.lps, self.thinking = {}, {}, {}
        self.media = {}                                  # seq -> picture records, positions absolute (module docstring)
        self.drafts_total = self.accepted_total = self.drafted_total = 0
        self.covered_mass = self.reachable_mass = 0.0

    # -- the door's surface (base/serve) ------------------------------------------------------------------------------
    def validate(self, ids, max_new, temperature) -> None:
        if not ids or any(type(t) is not int or not 0 <= t < self.vocab for t in ids):
            raise ValueError("prompt token id is outside the model vocabulary")
        if type(max_new) is not int or max_new <= 0:
            raise ValueError("generation limit must be a positive integer")
        try:
            valid = type(temperature) in (int, float) and isfinite(temperature) and temperature >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("temperature must be finite and nonnegative")

    def validate_options(self, options: dict) -> None:
        validate_options(options, vocab=self.vocab)
        if options.get("grammar") is not None and self.grammars is None:
            raise ValueError("structured output (response_format) is not served: no grammar compiler is bound")

    def prepare_options(self, options: dict) -> None:
        """Compile a request's grammar before it is admitted: a schema the compiler refuses is the door's 400."""
        if options.get("grammar") is not None and self.grammars is not None:
            self.grammars.ready(options["grammar"])

    def _bind(self, seq: int, ids, max_new, temperature, min_new, options) -> None:
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        self.validate(ids, max_new, temperature)
        if type(min_new) is not int or not 0 <= min_new <= max_new:
            raise ValueError("min_tokens must be an integer between 0 and the generation limit")
        options = dict(options or {})
        if options:
            self.validate_options(options)
        self.limits[seq] = (max_new, float(temperature))
        self.min_new[seq] = min_new
        self.options[seq] = options
        self.ends[seq] = self.eos | set(int(t) for t in options.get("stop_token_ids") or ())
        if options.get("seed") is not None:
            self.seeds[seq] = int(options["seed"])
        else:
            self.seeds.pop(seq, None)
        self.nonces[seq] = self.admissions                # the same count on every rank, in the same order
        self.admissions += 1
        self.thinking[seq] = options.get("reasoning_budget") is not None
        if options.get("logprobs") is not None:
            self.lps[seq] = []
        else:
            self.lps.pop(seq, None)
        if options.get("grammar") is not None:
            self.matchers[seq] = self.grammars.matcher(options["grammar"], self.k + 2, after=options.get("grammar_after"))
        else:
            self.matchers.pop(seq, None)
        if self.history is not None:
            self.history.forget(seq)                     # the row's tokens were just replaced or extended by a new turn

    def _media(self, ids: "list[int]", media, base: int) -> "list[dict]":
        """The door's records for `ids` (the new tokens, standing at `base` in the row), their positions made absolute.
        What a record's grid means is the composition's (`bind_media`); its shape is checked here."""
        if not getattr(self.composition, "sees_media", False):
            raise ValueError("the composed engine serves text only")
        items = []
        for m in media:
            positions = [int(p) for p in m["positions"]]
            grid = tuple(int(g) for g in m["grid"])
            if (m.get("kind") not in ("image", "video") or not positions or len(grid) != 3
                    or any(b <= a for a, b in zip(positions, positions[1:])) or positions[0] < 0 or positions[-1] >= len(ids)):
                raise ValueError("a media record needs a kind, a grid and increasing placeholder positions inside the prompt")
            items.append({"kind": m["kind"], "digest": str(m["digest"]), "positions": [base + p for p in positions],
                          "canvas": m.get("canvas"), "grid": grid})
        return items

    def add(self, seq: int, ids, max_new=None, temperature=None, min_new: int = 0, options=None, media=None) -> None:
        if seq in self.tokens:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        ids = list(ids)
        items = self._media(ids, media, 0) if media else None
        if items:
            self.composition.check_media(ids, items)              # refused before the row takes anything
        self._bind(seq, ids, max_new, temperature, min_new, options)
        self.tokens[seq] = ids
        self.prompt_len[seq] = len(ids)
        if items:
            self.media[seq] = items
            self.composition.bind_media(seq, ids, items)

    def forget(self, seq: int) -> None:
        for d in (self.tokens, self.prompt_len, self.limits, self.min_new, self.options, self.ends, self.seeds, self.nonces,
                  self.matchers, self.lps, self.thinking):
            d.pop(seq, None)
        if self.media.pop(seq, None) is not None:
            self.composition.forget_media(seq)
        if self.history is not None:
            self.history.forget(seq)
        if self.drafter is not None:
            self.drafter.forget(seq)

    def extend(self, seq: int, ids, max_new=None, temperature=None, min_new: int = 0, options=None, media=None,
               drop_unfed: bool = False) -> int:
        """A new turn on a conversation the store still holds: returns the tokens to prefill (the last sampled, never
        fed, and the new ones). `media` positions are the new tokens' (the door re-bases a continued turn's)."""
        ids = list(ids)
        if drop_unfed and len(self.tokens[seq]) - self.context(seq) != 1:
            raise ValueError("only one never-fed token can be dropped")
        base = len(self.tokens[seq]) - (1 if drop_unfed else 0)
        items = self._media(ids, media, base) if media else None
        if items:
            self.composition.check_media(self.tokens[seq][:base] + ids, self.media.get(seq, []) + items)
        if drop_unfed:
            del self.tokens[seq][-1]
        self._bind(seq, ids, max_new, temperature, min_new, options)
        self.tokens[seq] += ids
        self.prompt_len[seq] = len(self.tokens[seq])
        if items:
            self.media[seq] = self.media.get(seq, []) + items
            self.composition.bind_media(seq, self.tokens[seq], self.media[seq])
        return len(self.tokens[seq]) - self.context(seq)

    def extension_tokens(self, seq: int, ids) -> int:
        return len(self.tokens[seq]) + len(ids) - self.context(seq)

    def history(self, seq: int) -> "list[int]":
        return list(self.tokens[seq])

    def history_ref(self, seq: int) -> "list[int]":
        return self.tokens[seq]

    def history_from(self, seq: int, start: int) -> "list[int]":
        return self.tokens[seq][start:]

    def generated(self, seq: int) -> "list[int]":
        return self.tokens[seq][self.prompt_len[seq]:]

    def generated_count(self, seq: int) -> int:
        return len(self.tokens[seq]) - self.prompt_len[seq]

    def generated_since(self, seq: int, sent: int) -> "list[int]":
        return self.tokens[seq][self.prompt_len[seq] + sent:]

    def logprobs(self, seq: int):
        """[(token, logprob, [(id, logprob), ...])] per generated token when the request asked for logprobs (the
        processed row's, the door's logprobs shape), else None."""
        return self.lps.get(seq)

    def media_marks(self, seq: int) -> "list[tuple[int, str]]":
        """(first position, digest) of every picture in the conversation, in order (the door's continuation check and
        the prefix cache's salts)."""
        return [(m["positions"][0], m["digest"]) for m in self.media.get(seq, [])]

    # -- the runner's protocol ----------------------------------------------------------------------------------------
    def open(self, seq: int, slot: int) -> None:
        self.store.open(seq, slot)

    def close(self, seq: int) -> None:
        self.store.close(seq)

    def context(self, seq: int) -> int:
        return self.store.contexts[seq]

    def horizon(self, seq: int) -> int:
        return self.context(seq) + 1 + self.k

    def checkpoint(self, seq: int, position: int, snap: int) -> None:
        self.store.checkpoint(seq, position, snap)

    def restore(self, seq: int, position: int, snap: int) -> None:
        self.store.restore(seq, position, snap)

    def snapshot_bytes(self, snap: int):
        return self.store.snapshot_bytes(snap)

    def state_bytes(self, slot: int):
        return self.store.slot_bytes(slot)

    def park_bytes(self, slot: int, context: int):
        """What the tier keeps of a conversation parked after `context` tokens: the store's live state when it names one
        (a served store's delta-rule rings keep one state of K+1, `live_bytes`), else the slot whole."""
        live = getattr(self.store, "live_bytes", None)
        return self.store.slot_bytes(slot) if live is None else live(slot, context)

    def resume_bytes(self, slot: int, context: int):
        """Where that reads back: the slot cleared first, as `open` leaves a new row's (`clear`), then the same views."""
        live = getattr(self.store, "live_bytes", None)
        if live is None:
            return self.store.slot_bytes(slot)
        self.store.clear(slot)
        return live(slot, context)

    def park(self, seq: int) -> dict:
        if seq not in self.store.slot_of:
            raise ValueError(f"seq {seq} is not open")
        record = {"context": self.context(seq), "pending": len(self.tokens[seq]) - self.context(seq),
                  "tokens": list(self.tokens[seq]), "prompt_len": self.prompt_len[seq],
                  "limits": [self.limits[seq][0], self.limits[seq][1]], "min_new": self.min_new.get(seq, 0),
                  "options": {k: v for k, v in self.options.get(seq, {}).items() if k != "grammar"}}
        if self.media.get(seq):                          # marks and grids only: the caches hold the pictures' effect
            record["media"] = [[m["kind"], m["digest"], m["positions"][0], len(m["positions"]), list(m["grid"])]
                               for m in self.media[seq]]
        self.close(seq)
        self.forget(seq)
        return record

    def resume(self, seq: int, slot: int, record: dict) -> None:
        if seq in self.tokens or seq in self.store.slot_of:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        self.tokens[seq] = list(record["tokens"])
        self.prompt_len[seq] = int(record["prompt_len"])
        options = dict(record.get("options") or {})
        if options.get("logit_bias"):                    # base/kv_tier keeps the record as JSON: its keys come back as text
            options["logit_bias"] = {int(k): float(v) for k, v in options["logit_bias"].items()}
        media = [{"kind": kind, "digest": str(digest), "positions": list(range(int(first), int(first) + int(count))),
                  "canvas": None, "grid": tuple(int(g) for g in grid)} for kind, digest, first, count, grid in record.get("media") or ()]
        if media:
            self.composition.check_media(self.tokens[seq], media)
        self._bind(seq, self.tokens[seq], int(record["limits"][0]), float(record["limits"][1]),
                   int(record.get("min_new", 0)), options)
        self.store.resume(seq, slot, int(record["context"]))
        if media:
            self.media[seq] = media
            self.composition.bind_media(seq, self.tokens[seq], media)

    # -- sampling: one uniform a row from base/draws, the row's key as GLM-5.3 keys it ----------------------------------
    def _uniform(self, seq: int) -> float:
        key = draws.row_key(self.seeds.get(seq, self.seed), self.nonces[seq], self.generated_count(seq))
        return draws.uniform(key, draws.PICK, 0)

    def _rich(self, seq: int) -> bool:
        """Whether a row's pick needs its logits processed first: a base/sampler option the plain draw cannot take
        (penalties, bias, a grammar, logprobs, a reasoning budget), or min_tokens still holding the end back."""
        opts = self.options.get(seq, {})
        return (needs_rich_sampler(opts, 0.0, False) or seq in self.matchers or seq in self.lps
                or self.generated_count(seq) < self.min_new.get(seq, 0))

    def _pick(self, seqs, logits: torch.Tensor) -> "list[int]":
        """One pick a row from its logits row, the plain rows drawn together, a rich row through `_pick_rich`."""
        picks = [None] * len(seqs)
        plain = [i for i, seq in enumerate(seqs) if not self._rich(seq)]
        for i, seq in enumerate(seqs):
            if i not in plain:
                picks[i] = self._pick_rich(seq, logits[i])
        if plain:
            rows = logits[plain]
            chosen = [seqs[i] for i in plain]
            temps = [self.limits[s][1] for s in chosen]
            if all(t <= 0 for t in temps):
                drawn = rows[:, :self.vocab].argmax(dim=-1).tolist()
            else:
                dev = rows.device
                opts = [self.options.get(s, {}) for s in chosen]
                drawn = sample(rows, torch.tensor(temps, dtype=torch.float32, device=dev),
                               torch.tensor([float(o.get("top_p", self.top_p)) for o in opts], dtype=torch.float32, device=dev),
                               torch.tensor([self._uniform(s) for s in chosen], dtype=torch.float32, device=dev),
                               top_k=torch.tensor([int(o.get("top_k") or 0) for o in opts], dtype=torch.int32, device=dev),
                               valid=self.vocab).tolist()
            for i, token in zip(plain, drawn):
                picks[i] = int(token)
        return picks

    def _pick_rich(self, seq: int, raw: torch.Tensor) -> int:
        """The rich pick for one position, over base/sampler and base/grammar: bias and penalties over
        the row's tokens (base/sampler.History), the decodable cut, min_tokens' forbidden ends, the reasoning budget's
        forced end, the grammar's mask, then the row's own draw -- and the logprobs of the processed row. The row's
        tokens so far include every earlier pick, so a pick here is the same at a verify position as in a plain step."""
        from engine.base.sampler import History, process_logits, top_logprobs
        opts = self.options.get(seq, {})
        if self.history is None:
            self.history = History(int(raw.shape[-1]), raw.device)
        seen, counts = self.history.of(seq, self.tokens[seq], self.prompt_len[seq])
        ends = self.ends.get(seq, self.eos)
        forbid = (torch.tensor(sorted(ends), dtype=torch.int64, device=raw.device)
                  if ends and self.generated_count(seq) < self.min_new.get(seq, 0) else None)
        row = process_logits(raw, opts, seen, counts, (), self.vocab, forbid=forbid, force=self._reasoning_over(seq, opts))
        matcher = self.matchers.get(seq)
        if matcher is not None:
            masks = self.grammars.prepare([(seq, matcher, [])], row.device)
            if masks.has(seq):
                masks.apply(seq, row[None])
        temperature = self.limits[seq][1]
        if temperature <= 0:
            pick = int(row[:self.vocab].argmax())
        else:
            dev = row.device
            pick = int(sample(row[None], torch.tensor([temperature], dtype=torch.float32, device=dev),
                              torch.tensor([float(opts.get("top_p", self.top_p))], dtype=torch.float32, device=dev),
                              torch.tensor([self._uniform(seq)], dtype=torch.float32, device=dev),
                              top_k=torch.tensor([int(opts.get("top_k") or 0)], dtype=torch.int32, device=dev),
                              valid=self.vocab)[0])
        if seq in self.lps:
            self.lps[seq].append((pick, *top_logprobs(row[:self.vocab], pick, int(opts["logprobs"]))))
        return pick

    def _reasoning_over(self, seq: int, opts: dict) -> "int | None":
        """The reasoning-end token once the row's thinking budget is spent and it has not closed the block itself
        (base/sampler's rule: min_tokens is a promise and a budget is not, so a forbidden end is not forced)."""
        budget = opts.get("reasoning_budget")
        if budget is None or not self.thinking.get(seq, False):
            return None
        end = opts["reasoning_end"]
        if end in self.tokens[seq][self.prompt_len[seq]:]:
            self.thinking[seq] = False
            return None
        matcher = self.matchers.get(seq)
        if matcher is not None and matcher.armed:
            return None          # a call began inside the block: its grammar holds the row, and forcing the end breaks it
        return end if self.generated_count(seq) >= budget else None

    def _commit(self, seq: int, token: int) -> bool:
        self.tokens[seq].append(int(token))
        matcher = self.matchers.get(seq)
        if matcher is not None:
            matcher.advance([int(token)])
        return token in self.ends[seq] or self.generated_count(seq) >= self.limits[seq][0]

    # -- steps -----------------------------------------------------------------------------------------------------------
    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int, marks=None) -> bool:
        """The prompt's tokens [start, start+tokens) through the composition, split where the prefix cache marks a
        boundary (its snapshot is the state after that piece); the first token is sampled when the prompt is in.

        A composition that snapshots a boundary out of its own uncut forward says so: `takes_mark(piece start,
        position)`. Those marks ride the piece (`forward(..., marks=((position, snapshot), ...))`) and cost no forward
        -- a served chunk of 42 blocks is one forward, not 42 -- and the step is cut only where it declines (a boundary
        off its kernel's grid: the piece after the cut starts on the boundary, so the rest of the step's are on it)."""
        ids = self.tokens[seq][start:start + tokens]
        if len(ids) != tokens:
            raise ValueError(f"seq {seq} holds {len(self.tokens[seq])} tokens; asked to prefill to {start + tokens}")
        cuts = sorted((int(p), int(snap)) for p, snap in (marks or {}).items())
        if any(not start < p <= start + tokens for p, _ in cuts):
            raise ValueError("a prefill mark lies inside the step")
        takes = getattr(self.composition, "takes_mark", None)
        pieces, at, inside = [], start, []                   # (stop, the marks the piece's forward takes on its way)
        for p, snap in cuts:
            if p == start + tokens:
                break                                        # the step's end: the store holds that state afterwards
            if takes is not None and takes(at, p):
                inside.append((p, snap))
            else:
                pieces.append((p, tuple(inside)))
                at, inside = p, []
        pieces.append((start + tokens, tuple(inside)))
        at, logits, kept = start, None, []
        pictures = {"media": self.media[seq]} if self.media.get(seq) else {}
        for stop, inside in pieces:
            piece = torch.tensor(ids[at - start:stop - start], dtype=torch.int64, device=self.store.device)
            taken = dict({"marks": inside} if inside else {}, **pictures)
            if self.drafter is None:
                logits = self.composition.forward(Step.of([(seq, at, piece)]), self.store, **taken)
            else:
                logits, hidden = self.composition.forward(Step.of([(seq, at, piece)]), self.store, hidden=True, **taken)
                kept.append((at, hidden))
            at = stop
            for p, snap in cuts:
                if p == stop:
                    self.store.checkpoint(seq, p, snap)
        self.steps += 1
        finished = False
        if self.context(seq) == self.prompt_len[seq]:
            finished = self._commit(seq, self._pick([seq], logits)[0])
        for piece_start, hidden in kept:                     # the next ids exist now: the prompt's, or the first sample
            n = hidden.shape[0]
            self.drafter.observe(seq, piece_start, self.tokens[seq][piece_start + 1:piece_start + n + 1], hidden)
        return finished

    def decode(self, seqs, blocks, slots) -> "list[bool]":
        if self.drafter is None or not self.k:
            chunks = [(seq, self.context(seq), torch.tensor([self.tokens[seq][-1]], dtype=torch.int64, device=self.store.device))
                      for seq in seqs]
            logits = self.composition.forward(Step.of(chunks), self.store)
            self.steps += 1
            return [self._commit(seq, token) for seq, token in zip(seqs, self._pick(list(seqs), logits))]
        return self._verify(list(seqs))

    def _verify(self, seqs) -> "list[bool]":
        """One verification step (the module docstring): [last token] + drafts per row, every position sampled with its
        own generation count's uniform, drafts accepted while the sample agrees."""
        proposals = self.drafter.propose(seqs)
        drafts = []                                  # no more drafts than the row can still take after its next token
        for seq, proposal in zip(seqs, proposals):
            room = min(self.limits[seq][0] - self.generated_count(seq), self.max_context - self.context(seq)) - 1
            drafts.append([int(t) for t in proposal[:max(0, min(self.k, room))]])
        step = Step.of([(seq, self.context(seq),
                         torch.tensor([self.tokens[seq][-1]] + d, dtype=torch.int64, device=self.store.device), True)
                        for seq, d in zip(seqs, drafts)])
        logits, hidden = self.composition.forward(step, self.store, logits="all", hidden=True)
        self.steps += 1
        finished = []
        for seq, d, segment in zip(seqs, drafts, step.segments):
            fed, done, matched = 0, False, 0
            for j in range(segment.length):
                pick = self._pick([seq], logits[segment.start + j:segment.start + j + 1])[0]
                done = self._commit(seq, pick)
                fed = j + 1
                if j < len(d) and pick == d[j]:
                    matched += 1
                if done or j == len(d) or pick != d[j]:
                    break
            self.store.accept(seq, fed)
            if d:
                self.drafts_total += 1
                self.drafted_total += len(d)
                self.accepted_total += matched
            self.drafter.observe(seq, segment.ctx, self.tokens[seq][segment.ctx + 1:segment.ctx + fed + 1],
                                 hidden[segment.start:segment.start + fed])
            finished.append(done)
        return finished


__all__ = ["ALIGN", "DTYPES", "Layout", "merged_specs", "PositionStore", "store_for", "Drafter", "ComposedModel", "REFUSED_OPTIONS"]
