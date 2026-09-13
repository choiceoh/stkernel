"""A composition served: the position-addressed store over blocks and slots, and the model the runner drives (base).

Two pieces, both model-free:

`PositionStore` is engine/base/composition's State protocol over the engine's memory. A feature's per-token rows
(`put_rows`/`rows`) live in a BlockPool's blocks by POSITION -- block position // block_tokens, row position %
block_tokens -- so a sequence's history is its block table and a cached prefix's blocks are adopted, never copied
(base/prefix). Its whole values (`get`/`put`) live in the sequence's fixed slot (base/kv.SlotPool), carved from the
specs the features declare (base/cache_spec: key, dtype, shape) -- so a slot's bytes are one contiguous region a tier
moves (base/tiered_kv) and a prefix boundary's state is one copy of them into a snapshot. What the store holds after
a step is the state after the step's last token; a rejected draft would be overwritten by the next step's writes at
the same positions, as GLM-5.3's rings are -- nothing here rolls back.

`ComposedModel` answers the runner's Model protocol and the door's engine surface (base/serve) for any composition:
tokens and limits per row, the prefill split at the prefix cache's marks, one token per decode step (no drafter yet:
horizon = context + 1), sampling through base/sampler with base/draws' uniforms keyed the way GLM-5.3's are, parking
as a host record beside the slot's bytes. The math is the composition's; nothing here knows a model.
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


class PositionStore:
    """The served State (engine/base/composition.State's protocol): rows by position in blocks, values in slots."""

    def __init__(self, layout: Layout, pool: BlockPool, slot_view: torch.Tensor, snapshot_view: "torch.Tensor | None" = None,
                 device=None):
        if pool.storage is None or pool.block_bytes < layout.block_bytes or pool.block_size != layout.block_tokens:
            raise ValueError("the pool must carry storage for the layout's block bytes at its block size")
        if slot_view.dtype != torch.uint8 or slot_view.numel() < pool.max_seqs * layout.slot_bytes:
            raise ValueError("the slot view is uint8, one slot region per pool row (slot 0 included)")
        self.layout, self.pool, self.slots = layout, pool, slot_view
        self.snapshots = snapshot_view
        self.device = device if device is not None else slot_view.device
        self.contexts: dict = {}                     # seq -> tokens computed (the state is the one after them)
        self.slot_of: dict = {}
        self._step: dict = {}                        # seq -> (ctx, length) of the step being run

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

    def get(self, layer: int, key: str, seq: int, default=None):
        """None until the sequence has computed a token (a feature's first step starts from its own zero)."""
        if self.contexts.get(seq, 0) == 0:
            return default
        return self._slot_view(layer, key, seq)

    def put(self, layer: int, key: str, seq: int, value) -> None:
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
        ctx, length = self._step[seq]
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
            if self.contexts[s.seq] != s.ctx:
                raise ValueError(f"sequence {s.seq} is at {self.contexts[s.seq]} tokens, the step says {s.ctx}")
            row = self.pool.row(s.seq)
            need = -(-(s.ctx + s.length) // self.layout.block_tokens)
            if need > len(row) or any(row[b] == EMPTY for b in range(need)):
                raise ValueError(f"sequence {s.seq} holds fewer blocks than {s.ctx + s.length} positions need")
            self._step[s.seq] = (s.ctx, s.length)

    def commit(self, step: Step) -> None:
        for s in step.segments:
            self.contexts[s.seq] = s.ctx + s.length
        self._step = {}

    # -- prefix boundaries and parking --------------------------------------------------------------------------------
    def checkpoint(self, seq: int, position: int, snap: int) -> None:
        """The slot's state, which is the state after `position` tokens, into snapshot `snap`."""
        if self.contexts.get(seq) != position:
            raise ValueError(f"sequence {seq} stands at {self.contexts.get(seq)}, not at {position}")
        self.snapshot_bytes(snap).copy_(self.slot_bytes(self.slot_of[seq]))

    def restore(self, seq: int, position: int, snap: int) -> None:
        """The sequence starts at `position` with the snapshot's state; its rows before it are the adopted blocks'."""
        self.slot_bytes(self.slot_of[seq]).copy_(self.snapshot_bytes(snap))
        self.contexts[seq] = position

    def resume(self, seq: int, slot: int, context: int) -> None:
        """The row reopened in `slot`, whose bytes the tier restored: nothing is zeroed."""
        self.slot_of[seq] = slot
        self.contexts[seq] = context


def store_for(composition: Composition, kv_gib: float, max_seqs: int, block_tokens: int, *, snapshots: int = 0,
              device="cpu") -> "tuple[PositionStore, BlockPool, SlotPool, Plan]":
    """A store, its pools and the plan behind them, sized from the composition's cache specs on `device` -- the
    reference lane's arena: plain tensors (a profile's boot carves the same regions out of base/arena)."""
    paged, slots = composition.cache_specs()
    layout = Layout(paged, slots, block_tokens, composition.spec_layers())
    plan = layout.plan(kv_gib, max_seqs)
    pool = BlockPool(plan.num_blocks, block_tokens, max_seqs=plan.num_slots, max_blocks_per_seq=plan.num_blocks)
    pool.attach_storage(torch.zeros(plan.num_blocks * plan.block_bytes, dtype=torch.uint8, device=device), plan.block_bytes)
    slot_view = torch.zeros(plan.num_slots * max(layout.slot_bytes, 1), dtype=torch.uint8, device=device)
    snapshot_view = (torch.zeros(snapshots * max(layout.slot_bytes, 1), dtype=torch.uint8, device=device)
                     if snapshots else None)
    return PositionStore(layout, pool, slot_view, snapshot_view, device), pool, SlotPool(plan.num_slots), plan


REFUSED_OPTIONS = ("presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias", "logprobs", "grammar",
                   "grammar_after", "reasoning_budget", "reasoning_end")


class ComposedModel:
    """The runner's Model and the door's engine over a Composition and a PositionStore (see the module docstring)."""

    k = 0                                            # no drafter: a decode step is one token per row

    def __init__(self, composition: Composition, store: PositionStore, *, vocab: int, eos_ids, max_new: int = 256,
                 temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, max_context: int = 2 ** 31 - 1):
        if type(vocab) is not int or vocab <= 0 or type(max_new) is not int or max_new <= 0:
            raise ValueError("a composed model needs a positive vocabulary and generation limit")
        self.composition, self.store = composition, store
        self.vocab, self.eos = vocab, set(int(t) for t in eos_ids)
        self.max_new, self.temperature, self.top_p, self.seed, self.max_context = max_new, temperature, top_p, seed, max_context
        self.tokens, self.prompt_len, self.limits, self.min_new, self.options, self.ends = {}, {}, {}, {}, {}, {}
        self.seeds, self.nonces = {}, {}
        self.admissions = 0
        self.steps = 0
        self.drafter = None
        self.grammars = None
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
        validate_options(options)
        refused = [k for k in REFUSED_OPTIONS if options.get(k) is not None]
        if refused:
            raise ValueError(f"sampling options {refused} are not served by the composed engine")

    def prepare_options(self, options: dict) -> None:
        pass

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

    def add(self, seq: int, ids, max_new=None, temperature=None, min_new: int = 0, options=None, media=None) -> None:
        if media:
            raise ValueError("the composed engine serves text only")
        if seq in self.tokens:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        self._bind(seq, list(ids), max_new, temperature, min_new, options)
        self.tokens[seq] = list(ids)
        self.prompt_len[seq] = len(ids)

    def forget(self, seq: int) -> None:
        for d in (self.tokens, self.prompt_len, self.limits, self.min_new, self.options, self.ends, self.seeds, self.nonces):
            d.pop(seq, None)

    def extend(self, seq: int, ids, max_new=None, temperature=None, min_new: int = 0, options=None, media=None,
               drop_unfed: bool = False) -> int:
        """A new turn on a conversation the store still holds: returns the tokens to prefill (the last sampled, never
        fed, and the new ones)."""
        if media:
            raise ValueError("the composed engine serves text only")
        if drop_unfed:
            if len(self.tokens[seq]) - self.context(seq) != 1:
                raise ValueError("only one never-fed token can be dropped")
            del self.tokens[seq][-1]
        self._bind(seq, list(ids), max_new, temperature, min_new, options)
        self.tokens[seq] += list(ids)
        self.prompt_len[seq] = len(self.tokens[seq])
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
        return None

    def media_marks(self, seq: int) -> list:
        return []

    # -- the runner's protocol ----------------------------------------------------------------------------------------
    def open(self, seq: int, slot: int) -> None:
        self.store.open(seq, slot)

    def close(self, seq: int) -> None:
        self.store.close(seq)

    def context(self, seq: int) -> int:
        return self.store.contexts[seq]

    def horizon(self, seq: int) -> int:
        return self.context(seq) + 1

    def checkpoint(self, seq: int, position: int, snap: int) -> None:
        self.store.checkpoint(seq, position, snap)

    def restore(self, seq: int, position: int, snap: int) -> None:
        self.store.restore(seq, position, snap)

    def snapshot_bytes(self, snap: int):
        return self.store.snapshot_bytes(snap)

    def state_bytes(self, slot: int):
        return self.store.slot_bytes(slot)

    def park(self, seq: int) -> dict:
        if seq not in self.store.slot_of:
            raise ValueError(f"seq {seq} is not open")
        record = {"context": self.context(seq), "pending": len(self.tokens[seq]) - self.context(seq),
                  "tokens": list(self.tokens[seq]), "prompt_len": self.prompt_len[seq],
                  "limits": [self.limits[seq][0], self.limits[seq][1]], "min_new": self.min_new.get(seq, 0),
                  "options": dict(self.options.get(seq, {}))}
        self.close(seq)
        self.forget(seq)
        return record

    def resume(self, seq: int, slot: int, record: dict) -> None:
        if seq in self.tokens or seq in self.store.slot_of:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        self.tokens[seq] = list(record["tokens"])
        self.prompt_len[seq] = int(record["prompt_len"])
        self._bind(seq, self.tokens[seq], int(record["limits"][0]), float(record["limits"][1]),
                   int(record.get("min_new", 0)), record.get("options") or {})
        self.store.resume(seq, slot, int(record["context"]))

    # -- sampling: one uniform a row from base/draws, the row's key as GLM-5.3 keys it ----------------------------------
    def _uniform(self, seq: int) -> float:
        key = draws.row_key(self.seeds.get(seq, self.seed), self.nonces[seq], self.generated_count(seq))
        return draws.uniform(key, draws.PICK, 0)

    def _pick(self, seqs, logits: torch.Tensor) -> "list[int]":
        temps = [self.limits[s][1] for s in seqs]
        if needs_rich_sampler({}, 0.0, False):
            raise AssertionError("unreachable")
        if all(t <= 0 for t in temps):
            picks = logits[:, :self.vocab].argmax(dim=-1).tolist()
        else:
            dev = logits.device
            opts = [self.options.get(s, {}) for s in seqs]
            picks = sample(logits, torch.tensor(temps, dtype=torch.float32, device=dev),
                           torch.tensor([float(o.get("top_p", self.top_p)) for o in opts], dtype=torch.float32, device=dev),
                           torch.tensor([self._uniform(s) for s in seqs], dtype=torch.float32, device=dev),
                           top_k=torch.tensor([int(o.get("top_k") or 0) for o in opts], dtype=torch.int32, device=dev),
                           valid=self.vocab).tolist()
        for i, seq in enumerate(seqs):                                   # OpenAI min_tokens: no end before min_new
            if picks[i] in self.ends[seq] and self.generated_count(seq) < self.min_new.get(seq, 0):
                row = logits[i, :self.vocab].clone()
                row[list(self.ends[seq])] = float("-inf")
                picks[i] = int(row.argmax().item())
        return picks

    def _commit(self, seq: int, token: int) -> bool:
        self.tokens[seq].append(int(token))
        return token in self.ends[seq] or self.generated_count(seq) >= self.limits[seq][0]

    # -- steps -----------------------------------------------------------------------------------------------------------
    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int, marks=None) -> bool:
        """The prompt's tokens [start, start+tokens) through the composition, split where the prefix cache marks a
        boundary (its snapshot is the state after that piece); the first token is sampled when the prompt is in."""
        ids = self.tokens[seq][start:start + tokens]
        if len(ids) != tokens:
            raise ValueError(f"seq {seq} holds {len(self.tokens[seq])} tokens; asked to prefill to {start + tokens}")
        cuts = sorted((int(p), int(snap)) for p, snap in (marks or {}).items())
        if any(not start < p <= start + tokens for p, _ in cuts):
            raise ValueError("a prefill mark lies inside the step")
        at, logits = start, None
        for stop in [p for p, _ in cuts if p < start + tokens] + [start + tokens]:
            piece = torch.tensor(ids[at - start:stop - start], dtype=torch.int64, device=self.store.device)
            logits = self.composition.forward(Step.of([(seq, at, piece)]), self.store)
            at = stop
            for p, snap in cuts:
                if p == stop:
                    self.store.checkpoint(seq, p, snap)
        self.steps += 1
        if self.context(seq) == self.prompt_len[seq]:
            return self._commit(seq, self._pick([seq], logits)[0])
        return False

    def decode(self, seqs, blocks, slots) -> "list[bool]":
        chunks = [(seq, self.context(seq), torch.tensor([self.tokens[seq][-1]], dtype=torch.int64, device=self.store.device))
                  for seq in seqs]
        logits = self.composition.forward(Step.of(chunks), self.store)
        self.steps += 1
        return [self._commit(seq, token) for seq, token in zip(seqs, self._pick(list(seqs), logits))]


__all__ = ["ALIGN", "DTYPES", "Layout", "PositionStore", "store_for", "ComposedModel", "REFUSED_OPTIONS"]
