"""GLM-5.3 behind the runner (profile, the adapter): tokens in, tokens out.

base/runner knows a four-method model and nothing else; this is that model
for GLM-5.3. It owns the token buffers (prompt + generated per sequence),
turns the runner's calls into `net.Step`s over the profile's caches, and
samples (base/sampler, seeded -> replayable, D12). Draft tokens come from a
`Drafter` -- `NullDrafter` proposes nothing (K=0); DFlash2 is the fleet's
drafter and plugs in here with the same two calls.

Verification is rejection by position: the step feeds [last token] + K
drafts at ctx..ctx+K, samples at every position, accepts drafts while the
sample agrees, and appends accepted+1 tokens. The caches were written for
all K+1 positions; the next step overwrites what was rejected (net.py).
"""
from __future__ import annotations

import time

import torch
from math import isfinite

from engine.base.sampler import sample
from engine.profiles.glm53.caches import Glm53Caches
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53.net import Glm53Net, Segment, Step


class NullDrafter:
    """No drafts (K=0): a decode step is one token per sequence."""
    k = 0
    aux_layers = ()

    def observe(self, ring, positions, aux) -> None:
        pass

    def propose(self, anchor: int, position: int, ring) -> "list[int]":
        return []


class Glm53Engine:
    def __init__(self, net: Glm53Net, caches: Glm53Caches, F: Facts, drafter=None, max_new: int = 256,
                 eos_ids=(), temperature: float = 0.0, top_p: float = 1.0, seed: int = 0, decodable: "int | None" = None,
                 aux_layers=None, context_ceiling: "int | None" = None):
        self.net, self.caches, self.F = net, caches, F
        self.drafter = drafter or NullDrafter()
        if self.drafter.k > F.spec_k:
            raise ValueError(f"drafter proposes {self.drafter.k} > spec_k {F.spec_k}: the rings are sized for {F.spec_k}")
        self.aux_layers = list(aux_layers) if aux_layers is not None else list(self.drafter.aux_layers)
        self.max_new, self.eos = max_new, set(eos_ids)
        self.temperature, self.top_p = temperature, top_p
        self.decodable = decodable                          # logits past this id are the tokenizer's orphans: masked (as served)
        # One served ceiling: the door refuses a horizon past it (base/serve reads this)
        # and the decode ladder captures no bucket above it. Unset = the trained positions.
        trained = getattr(F, "max_position", 2**31 - 1)
        if context_ceiling is not None and not 0 < int(context_ceiling) <= trained:
            raise ValueError(f"served context ceiling must be in 1..{trained}")
        self.max_context = trained if context_ceiling is None else int(context_ceiling)
        self.gen = torch.Generator(device=caches.device).manual_seed(seed)
        self.tokens, self.prompt_len, self.ctx, self.slot, self.limits = {}, {}, {}, {}, {}
        self.min_new = {}                                   # seq -> no end token before this many generated (OpenAI min_tokens)
        self.options = {}                                   # seq -> the request's sampling options beyond temperature (base/sampler.OPTION_KEYS)
        self.gens = {}                                      # seq -> its own torch.Generator when the request carries a seed
        self.ends = {}                                      # seq -> end tokens: the model's plus the request's stop_token_ids
        self.lps = {}                                       # seq -> per committed token (id, logprob, [(id, logprob)...]) when asked
        self.matchers = {}                                  # seq -> base/grammar.Matcher when the request carries a grammar
        self.grammars = None                                # base/grammar.Grammars, bound at boot when structured output is served
        self.vision = None                                  # vision.Vision, bound at boot when images are served (45차 §23 A7)
        self.media = {}                                     # seq -> [dict(kind, digest, positions (absolute), canvas, grid)]: the prompt's pictures
        self.embeds = {}                                    # seq -> {media index: [tokens, hidden] rows} encoded at the first chunk that needs them
        self.accepted_total = 0
        self.drafted_total = 0
        # Free observability for decisions this engine keeps having to make by probe.
        # All of it is host arithmetic on numbers already in hand: no device read, no sync.
        self.decode_shape_counts = {}              # (sequences, capacity bucket) -> decode steps replayed there
        self.accepted_per_step = [0] * (self.drafter.k + 2)   # how many drafts a step committed, 0..k+1
        self.lane_info = {}                        # what is actually bound: set by the boot that built the lanes
        self.steps = 0
        self.decode_graphs = None
        self.sampling_graphs = None
        self.pipeline = None                                # pipeline.AsyncDecode once the graphs are captured (45차 §23 B3)
        self.inflight = {}                                  # seq -> decode steps launched ahead whose tokens the host has not read
        self.memory = None
        self.prefill_chunk = None

    def capture_decode(self, max_seqs: int) -> None:
        """Bind the fleet's finite target decode graphs before admitting work."""
        from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs
        if self.tokens:
            raise ValueError("capture must finish before requests are admitted")
        if self.decode_graphs is not None:
            raise ValueError("decode graphs are already prepared")
        try:
            if self.memory is not None:
                self._warmup_prefill_memory()
            self._warmup_serving_kernels()
            self.decode_graphs = Glm53DecodeGraphs(self.net, self.caches, max_seqs,
                                                  self.drafter.k + 1, self.aux_layers, memory=self.memory,
                                                  ceiling=self.max_context)
            if self.drafter.k:
                self.drafter.capture_decode(self.caches, memory=self.memory)
                self._check_graph_pools()
            from engine.profiles.glm53.decode_graphs import SamplingGraphs
            self.sampling_graphs = SamplingGraphs(self.decode_graphs, self.gen, self.decodable, self.top_p)
            if self.drafter.k:
                from engine.profiles.glm53.pipeline import AsyncDecode
                self.pipeline = AsyncDecode(self)
            if self.memory is not None:
                self.memory.checkpoint("ready")
                self.memory.ready = True
        except BaseException:
            self.close_decode()
            raise

    def _check_graph_pools(self) -> None:
        """The decode loop reads this step's auxiliary hidden states, which live in the
        target graphs' memory pool, while the drafter's observation graph replays between
        segments. Sharing one pool would let that replay's own allocations land on top of
        them -- finite numbers, wrong drafter context, no test that could see it. The
        separation is what makes the loop correct, so it is asserted, not assumed."""
        target = self.decode_graphs.graphs.pool
        drafter = self.drafter.decode_graphs
        for name, other in (("proposals", drafter.proposals), ("observations", drafter.observations)):
            if other.pool == target:
                raise ValueError(f"the drafter's {name} graphs share the target graphs' memory pool: "
                                 "a replay between segments would overwrite the auxiliary hidden states")

    def qualify_eager_decode(self, warmup: bool = True) -> None:
        """Serve without captured decode graphs (knob `decode_eager`): same lanes, the step stays in Python.

        Memory still has to qualify -- that gate is about the allocator ceiling, not about graphs. A
        reference-lane bisect (knob `lanes`) skips the largest-prefill warmup: the torch sparse MLA gathers
        [T, K, 512] fp32 rows and a 6,912-token chunk asks 27 GiB of it -- that table exists to talk, not to serve.
        """
        if self.tokens:
            raise ValueError("qualification must finish before requests are admitted")
        if self.decode_graphs is not None:
            raise ValueError("decode graphs are already prepared")
        if self.memory is not None:
            if warmup:
                self._warmup_prefill_memory()
            self.memory.checkpoint("ready")
            self.memory.ready = True

    def warmup_shapes(self, lengths=(64, 256, 1024, 2048, 4096), widths=(1, 2, 3, 4)) -> dict:
        """Pay the first-use JIT at boot instead of on the first user (45차 §23 B2; production's prefill-warmup.py):
        one synthetic prefill per length (the served kernels specialise per M bucket) and one decode step per batch
        width through the captured graphs. Nothing is judged; the request rows are returned empty."""
        caches = self.caches
        if caches.pool.rows_in_use or any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("warmup requires empty request and state slots")
        paid = {}
        capacity = caches.pool.num_blocks * self.F.block
        try:
            for length in sorted({min(n, self.prefill_chunk or n, capacity) for n in lengths}):
                slot = caches.slots.take(0)
                caches.pool.reserve(0, length + 1)
                t0 = time.perf_counter()
                ids = torch.randint(0, min(self.F.vocab, 100_000), (length,), device=caches.device, dtype=torch.int64)
                h, aux = self._forward(Step.prefill(ids, 0, 0, slot))
                self.net.head(h[-1:]); torch.cuda.synchronize()
                paid[f"prefill/{length}"] = round(time.perf_counter() - t0, 3)
                del h, aux
                caches.pool.release(0); caches.slots.give(slot); caches.reset()
            if self.decode_graphs is not None:
                for width in widths:
                    if width > caches.pool.max_seqs:
                        break
                    rows = list(range(width))
                    slots = [caches.slots.take(r) for r in rows]
                    for r in rows:
                        caches.pool.reserve(r, 8 + self.drafter.k)
                        self.tokens[r] = [1] * 4; self.prompt_len[r] = 4; self.ctx[r] = 4; self.slot[r] = slots[r - rows[0]]
                        self.limits[r] = (8, 0.0); self.min_new[r] = 0; self._bind_options(r, None)
                    t0 = time.perf_counter()
                    try:
                        self.decode(rows, [caches.pool.row(r) for r in rows], slots)
                        torch.cuda.synchronize()
                        paid[f"decode/{width}"] = round(time.perf_counter() - t0, 3)
                    finally:
                        for r in rows:
                            self.close(r); self.forget(r)
                            caches.pool.release(r)
                        for s in slots:
                            caches.slots.give(s)
                        caches.reset()
        finally:
            self.accepted_total = self.drafted_total = 0
            self.decode_shape_counts = {}
            self.accepted_per_step = [0] * len(self.accepted_per_step)
            self.steps = 0
        return paid

    def _warmup_prefill_memory(self):
        """Exercise the largest legal prefill at both ends of the KV capacity.

        Inputs are synthetic; this qualifies memory preparation, not quality.
        Unseen tail shapes remain subject to the same allocator byte ceiling.
        """
        if self.prefill_chunk is None:
            raise ValueError("full-model memory preparation requires the scheduler's prefill chunk")
        caches = self.caches
        if caches.pool.rows_in_use or any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("memory preparation requires empty request and state slots")
        capacity = caches.pool.num_blocks * self.F.block
        length = min(self.prefill_chunk, capacity)
        slot = caches.slots.take(0)
        try:
            caches.pool.reserve(0, capacity)
            for context in sorted({0, capacity-length}):
                self.memory.checkpoint(f"prefill/{length}/{context}/before")
                ids = torch.zeros(length, device=caches.device, dtype=torch.int64)
                step = Step.prefill(ids, context, 0, slot)
                h, aux = self._forward(step)
                self.net.head(h[-1:])
                if aux is not None:
                    self.drafter.observe(caches.draft_ring(slot),
                                         torch.arange(context, context+length, device=caches.device), aux)
                del h, aux, step, ids
                self.memory.checkpoint(f"prefill/{length}/{context}/prepared")
        finally:
            caches.pool.release(0)
            caches.slots.give(slot)
            caches.reset()

    WARM_PREFILL_TOKENS = (1, 8, 64, 512)
    """Prompt widths run once before the door opens, so no request compiles a kernel.

    A kernel that has not been compiled for a token count compiles on the first
    request that brings one, inside that request's prefill, where the whole compile
    lands on its time to first token. The 2026-09-12 boot did exactly that: 43 s
    after it printed `serving`, a b12x MoE shape took 1.55 s to build, and the
    engine kept adding artifacts for minutes (boot-time study 5-g). The memory
    warmup above only ever runs the full chunk, so every prompt shorter than one
    arrived at a cold kernel.

    These four cover the short end, where first prompts live, at a few hundred
    tokens of work. They are not a guarantee: a width not on this list still
    compiles when it first arrives, and the honest fix for that is a kernel-side
    registry of the shapes each lane wants, which is vLLM's answer
    (model_executor/warmup) and is the kernel owners' to build.
    """

    def _warmup_serving_kernels(self) -> None:
        """Run each declared prompt width once, for its kernels, not for its memory."""
        caches, F = self.caches, self.F
        if caches.pool.rows_in_use or any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("kernel warmup requires empty request and state slots")
        widths = [w for w in self.WARM_PREFILL_TOKENS
                  if w <= min(self.prefill_chunk or w, caches.pool.num_blocks * F.block)]
        if not widths:
            return
        slot = caches.slots.take(0)
        try:
            caches.pool.reserve(0, max(widths))
            for width in widths:
                ids = torch.zeros(width, device=caches.device, dtype=torch.int64)
                h, aux = self._forward(Step.prefill(ids, 0, 0, slot))
                self.net.head(h[-1:])
                del h, aux, ids
        finally:
            caches.pool.release(0)
            caches.slots.give(slot)
            caches.reset()
        if self.memory is not None:
            self.memory.checkpoint(f"warm kernels {widths}")

    def close_decode(self):
        if self.sampling_graphs is not None:
            self.sampling_graphs.close()
            self.sampling_graphs = None
        if self.decode_graphs is not None:
            self.decode_graphs.graphs.close()
            self.decode_graphs = None
        if self.drafter.k and self.drafter.decode_graphs is not None:
            self.drafter.decode_graphs.proposals.close()
            self.drafter.decode_graphs.observations.close()
            self.drafter.decode_graphs = None
        if self.memory is not None:
            self.memory.close()

    # -- the runner's protocol -------------------------------------------------------
    def validate(self, ids, max_new, temperature) -> None:
        if not ids or any(type(t) is not int or not 0 <= t < self.F.vocab for t in ids):
            raise ValueError("prompt token id is outside the model vocabulary")
        if type(max_new) is not int or max_new <= 0:
            raise ValueError("generation limit must be a positive integer")
        if type(temperature) not in (int, float):
            raise ValueError("temperature must be finite and nonnegative")
        try:
            valid = isfinite(temperature) and temperature >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("temperature must be finite and nonnegative")

    def validate_options(self, options: dict) -> None:
        """The door asks before enqueueing: every option a request carries must be one this engine serves (D3)."""
        from engine.base.sampler import validate_options
        validate_options(options)
        if options.get("grammar") is not None and self.grammars is None:
            raise ValueError("structured output (response_format) is not served: no grammar compiler is bound")

    def history(self, seq: int) -> "list[int]":
        """Every token the row has seen or produced: what a re-sent chat must start with to continue it (B1)."""
        return list(self.tokens[seq])

    def logprobs(self, seq: int) -> "list | None":
        return self.lps.get(seq)

    def _bind_options(self, seq: int, options: "dict | None") -> None:
        options = dict(options or {})
        self.options[seq] = options
        self.ends[seq] = self.eos | set(options.get("stop_token_ids") or ())
        if options.get("seed") is not None:
            self.gens[seq] = torch.Generator(device=self.caches.device).manual_seed(int(options["seed"]))
        else:
            self.gens.pop(seq, None)
        if options.get("logprobs") is not None:
            self.lps[seq] = []
        else:
            self.lps.pop(seq, None)
        if options.get("grammar") is not None:
            if self.grammars is None:
                raise ValueError("structured output (response_format) is not served: no grammar compiler is bound")
            self.matchers[seq] = self.grammars.matcher(options["grammar"], self.drafter.k + 2)
        else:
            self.matchers.pop(seq, None)

    def add(self, seq: int, ids: "list[int]", max_new: "int | None" = None, temperature: "float | None" = None,
            min_new: int = 0, options: "dict | None" = None, media=None) -> None:
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        self.validate(ids, max_new, temperature)
        if type(min_new) is not int or not 0 <= min_new <= max_new:
            raise ValueError("min_tokens must be an integer between 0 and the generation limit")
        if seq in self.tokens:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        if options:
            self.validate_options(options)
        if media:
            self._bind_media(seq, list(ids), media, base=0)
        self.tokens[seq] = list(ids); self.prompt_len[seq] = len(ids)
        self.limits[seq] = (max_new, temperature)
        self.min_new[seq] = min_new
        self._bind_options(seq, options)

    def forget(self, seq: int) -> None:
        if seq in self.slot:
            raise ValueError(f"seq {seq} is still live")
        for rows in (self.tokens, self.prompt_len, self.limits, self.min_new, self.options, self.gens, self.ends, self.lps, self.matchers,
                     self.media, self.embeds, self.inflight):
            rows.pop(seq, None)                             # a row leaving does not move the others: the pipeline shrinks its view

    # -- pictures (45차 §23 A7): the door hands canvases with the positions their rows take; every rank encodes them
    # -- itself (vision.Vision, replicated) at the first prefill chunk that reaches those positions -----------------
    def _bind_media(self, seq: int, ids: "list[int]", media, base: int) -> None:
        if self.vision is None:
            raise ValueError("images are not served: no vision tower is bound")
        V = self.vision.V
        items = []
        for m in media:
            positions = [int(p) for p in m["positions"]]
            grid = tuple(int(g) for g in m["grid"])
            if (m.get("kind") not in ("image", "video") or not positions or len(grid) != 3
                    or any(b <= a for a, b in zip(positions, positions[1:])) or positions[0] < 0 or positions[-1] >= len(ids)
                    or len(positions) != V.tokens(grid) or any(ids[p] != V.image_token for p in positions)):
                raise ValueError("a media record needs a kind, a grid and increasing placeholder positions inside the prompt")
            items.append({"kind": m["kind"], "digest": str(m["digest"]), "positions": [base + p for p in positions],
                          "canvas": m.get("canvas"), "grid": grid})
        self.media.setdefault(seq, []).extend(items)

    def media_marks(self, seq: int) -> "list[tuple[int, str]]":
        """(first position, digest) of every picture in the conversation, in order (the door's continuation check)."""
        return [(m["positions"][0], m["digest"]) for m in self.media.get(seq, [])]

    def _patches(self, seq: int, lo: int, hi: int) -> tuple:
        """The rows standing inside [lo, hi) of the prompt, as (positions relative to lo, rows) pairs -- encoding a
        picture on the way if this is the first chunk to reach it, dropping it once the chunk passed its last row."""
        out = []
        done = []
        for i, m in enumerate(self.media.get(seq, [])):
            pos = m["positions"]
            if pos[-1] < lo or pos[0] >= hi:
                continue
            rows = self.embeds.setdefault(seq, {}).get(i)
            if rows is None:
                if m["canvas"] is None:
                    raise RuntimeError("a resumed conversation asked for rows it no longer carries")
                rows = self.vision.encode(m["canvas"], m["grid"])
                if rows.shape[0] != len(pos):
                    raise RuntimeError(f"the vision tower produced {rows.shape[0]} rows for {len(pos)} placeholders")
                self.embeds[seq][i] = rows
            p = torch.tensor(pos, dtype=torch.int64, device=rows.device)
            keep = (p >= lo) & (p < hi)
            out.append((p[keep] - lo, rows[keep]))
            if pos[-1] < hi:
                done.append(i)
        for i in done:                                                        # its rows are in the caches now
            self.embeds[seq].pop(i, None)
            self.media[seq][i]["canvas"] = None
        return tuple(out)

    def open(self, seq: int, slot: int) -> None:
        self.slot[seq] = slot; self.ctx[seq] = 0
        self.caches.reset_slot(slot)

    def close(self, seq: int) -> None:
        for d in (self.ctx, self.slot):
            d.pop(seq, None)

    def checkpoint(self, seq: int, position: int, snap: int) -> None:
        """The runner's prefix cache keeps this sequence's state at a chunk boundary (base/prefix.py)."""
        self.caches.checkpoint(self.slot[seq], position, snap)

    def restore(self, seq: int, position: int, snap: int) -> None:
        """A new sequence adopts a cached prefix: its rings take the boundary's state, its context starts there."""
        self.caches.restore(self.slot[seq], position, snap)
        self.ctx[seq] = position

    # -- parking (D16): the host side of a conversation travels as a record, the slot's bytes with the tier --
    def park(self, seq: int) -> dict:
        """Close the row and hand back everything the host held for it."""
        if seq not in self.slot:
            raise ValueError(f"seq {seq} is not open")
        record = {"context": self.ctx[seq], "pending": len(self.tokens[seq]) - self.ctx[seq],
                  "tokens": list(self.tokens[seq]), "prompt_len": self.prompt_len[seq],
                  "limits": [self.limits[seq][0], self.limits[seq][1]], "min_new": self.min_new.get(seq, 0),
                  "options": {k: v for k, v in self.options.get(seq, {}).items() if k != "grammar"},
                  "media": [[m["kind"], m["digest"], m["positions"][0], len(m["positions"]), list(m["grid"])]
                            for m in self.media.get(seq, [])]}         # marks only: the caches hold the pictures' effect
        self.close(seq)
        self.forget(seq)
        return record

    def resume(self, seq: int, slot: int, record: dict) -> None:
        """Reopen the row in `slot` from a record; the slot's bytes were restored by the tier, so no reset."""
        if seq in self.tokens or seq in self.slot:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        self.tokens[seq] = list(record["tokens"]); self.prompt_len[seq] = int(record["prompt_len"])
        self.limits[seq] = (int(record["limits"][0]), float(record["limits"][1]))
        self.min_new[seq] = int(record.get("min_new", 0))
        options = dict(record.get("options") or {})
        if "logit_bias" in options:
            options["logit_bias"] = {int(k): float(v) for k, v in options["logit_bias"].items()}   # JSON keys come back as text
        self._bind_options(seq, options)
        if record.get("media"):
            self.media[seq] = [{"kind": kind, "digest": str(digest), "positions": list(range(int(first), int(first) + int(count))),
                                "canvas": None, "grid": tuple(int(g) for g in grid)} for kind, digest, first, count, grid in record["media"]]
        self.slot[seq] = slot; self.ctx[seq] = int(record["context"])

    def state_bytes(self, slot: int):
        return self.caches.slot_bytes(slot)

    def extend(self, seq: int, ids: "list[int]", max_new: "int | None" = None, temperature: "float | None" = None,
               min_new: int = 0, options: "dict | None" = None, media=None) -> int:
        """A new turn: more prompt tokens on a conversation the caches still hold.
        Returns the tokens to prefill -- the last sampled token (never fed) and
        the new ones -- so `generated` counts this turn only from here on."""
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        self.validate(ids, max_new, temperature)
        if type(min_new) is not int or not 0 <= min_new <= max_new:
            raise ValueError("min_tokens must be an integer between 0 and the generation limit")
        if options:
            self.validate_options(options)
        if media:
            self._bind_media(seq, list(ids), media, base=len(self.tokens[seq]))
        self._moved()
        self.tokens[seq] += list(ids); self.prompt_len[seq] = len(self.tokens[seq])
        self.limits[seq] = (max_new, temperature)
        self.min_new[seq] = min_new
        self._bind_options(seq, options)
        return len(self.tokens[seq]) - self.ctx[seq]

    def extension_tokens(self, seq: int, ids) -> int:
        """Inspect the next turn's prefill size before reserving its budget."""
        return len(self.tokens[seq]) + len(ids) - self.ctx[seq]

    def horizon(self, seq: int) -> int:
        """The exclusive end of the row's next decode writes -- and of every step launched ahead of the host (B3)."""
        return self.ctx[seq] + (1 + self.drafter.k) * (1 + self.inflight.get(seq, 0))

    # -- decode steps ahead of the host (pipeline.py, 45차 §23 B3) ------------------------------------------------
    def _plain_ahead(self, seq: int) -> bool:
        """Greedy, or temperature / top_p only: the device can commit, observe and propose without the host."""
        opts = self.options.get(seq, {})
        if any(opts.get(k) is not None for k in ("top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty",
                                                  "logit_bias", "logprobs", "grammar", "min_p")):
            return False
        if seq in self.matchers or seq in self.gens or seq in self.lps:
            return False
        return self.min_new.get(seq, 0) <= self._generated_count(seq)

    def async_ready(self, seqs) -> bool:
        if self.pipeline is None or self.decode_graphs is None or not self.drafter.k:
            return False
        if self.pipeline.pending and any(s not in self.pipeline.batch for s in seqs):
            return False                                    # rows joined: the runner drains, then the view is rebuilt from the host
        return all(self._plain_ahead(s) for s in seqs)

    def decode_async(self, seqs, blocks, slots):
        return self.pipeline.launch(seqs, slots)

    def context(self, seq: int) -> int:
        return self.ctx[seq]

    def generated(self, seq: int) -> "list[int]":
        return self.tokens[seq][self.prompt_len[seq]:]

    def _generated_count(self, seq: int) -> int:
        return len(self.tokens[seq]) - self.prompt_len[seq]

    def _sample(self, logits: torch.Tensor, temps: "list[float]") -> torch.Tensor:
        # Temperatures already live on the host: no device predicate or random
        # draw is needed for an entirely greedy step. Such steps leave the RNG
        # untouched; stochastic/mixed steps retain the base sampler's draws.
        if all(t <= 0 for t in temps):
            return logits[:, :self.decodable].argmax(dim=-1)
        if self.decodable is not None and logits.shape[-1] > self.decodable:
            logits = logits.clone(); logits[:, self.decodable:] = float("-inf")
        t = torch.tensor(temps, dtype=torch.float32, device=logits.device)
        p = torch.full((logits.shape[0],), self.top_p, device=logits.device)
        return sample(logits, t, p, self.gen)

    def _no_end_yet(self, seq: int, picks: "list[int]", hidden: torch.Tensor) -> "list[int]":
        """OpenAI min_tokens: while fewer than `min_new` tokens are generated, an end token cannot be
        chosen -- the position takes its best other token instead (the same as masking the end tokens
        before the pick, applied only where a pick was an end token, so the fused/captured samplers
        stay as they are; `hidden` rows correspond to `picks`)."""
        need = self.min_new.get(seq, 0) - self._generated_count(seq)
        ends = self.ends.get(seq, self.eos)
        if need <= 0 or not ends:
            return picks
        fixed = list(picks)
        for i, token in enumerate(picks[:need]):
            if token in ends:
                row = self.net.head(hidden[i:i + 1])[0].clone()
                row[list(ends)] = float("-inf")
                if self.decodable is not None and row.shape[-1] > self.decodable:
                    row[self.decodable:] = float("-inf")
                fixed[i] = int(row.argmax().item())
        return fixed

    def _forward(self, step: Step):
        self.caches.prepare(step)
        if self.drafter.k:
            return self.net.forward(step, self.caches, aux_layers=self.aux_layers)
        return self.net.forward(step, self.caches), None

    def _sample_hidden(self, hidden, temps):
        if all(t <= 0 for t in temps):
            return self.net.head_tokens(hidden, self.decodable)
        return self._sample(self.net.head(hidden), temps)

    # -- picking tokens: the captured samplers for plain rows; rows with options (base/sampler.OPTION_KEYS) or a
    # -- stochastic row with drafts take the base sampler over their gathered logits, identically on every rank ------
    def _rich(self, seq: int) -> bool:
        from engine.base.sampler import needs_rich_sampler
        return needs_rich_sampler(self.options.get(seq, {}), self.limits[seq][1], bool(self.drafter.k))

    def _gather(self, local: torch.Tensor) -> torch.Tensor:
        """This rank's logits shard [rows, vp] -> every rank's whole rows [rows, vocab] fp32 (a collective: same order everywhere)."""
        return self.net.comm.all_gather(local, dim=-1).float()

    def _row_logits(self, seq: int, raw: torch.Tensor, position: int, drafts_before: "list[int]") -> torch.Tensor:
        """`raw` [vocab] processed for `seq` at this step's position: bias, penalties over the row's tokens (with the drafts
        assumed accepted before it), the decodable cut, min_tokens and the grammar's mask."""
        from engine.base.sampler import process_logits
        opts = self.options.get(seq, {})
        prompt = self.tokens[seq][: self.prompt_len[seq]]
        generated = self.tokens[seq][self.prompt_len[seq]:] + list(drafts_before)
        mask = None
        need = self.min_new.get(seq, 0) - self._generated_count(seq) - len(drafts_before)
        if need > 0 and self.ends.get(seq):
            mask = torch.ones(raw.shape[-1], dtype=torch.bool, device=raw.device)
            mask[list(self.ends[seq])] = False
        return process_logits(raw, opts, prompt, generated, self.decodable, mask)

    def _pick_rich(self, seq: int, rows: torch.Tensor, drafts: "list[int]", draft_probs: "torch.Tensor | None"):
        """One sequence's positions through the base sampler. rows: [len(drafts) + 1, vocab] fp32 raw logits.
        Returns (accepted drafts, committed tokens, per-token (id, logprob, top) or None)."""
        from engine.base.sampler import distribution, draw, speculative_pick, top_logprobs
        opts = self.options.get(seq, {})
        temperature = self.limits[seq][1]
        gen = self.gens.get(seq, self.gen)
        matcher = self.matchers.get(seq)
        masks = matcher.masks(drafts, rows.device) if matcher is not None else [None] * (len(drafts) + 1)
        processed, dists = [], []
        for i in range(min(len(masks), rows.shape[0])):
            logits = self._row_logits(seq, rows[i], i, drafts[:i])
            if masks[i] is not None:
                logits = logits.masked_fill(~masks[i], float("-inf"))
            processed.append(logits)
            dists.append(distribution(logits, temperature, opts.get("top_k"), opts.get("top_p")))
        if temperature <= 0 or draft_probs is None or not drafts:
            picks = [int(d.argmax().item()) if temperature <= 0 else draw(d, gen) for d in dists]
            accepted = 0
            for d, got in zip(drafts, picks):
                if d != got:
                    break
                accepted += 1
            new = picks[: accepted + 1]
        else:
            accepted, new = speculative_pick(torch.stack(dists), drafts[: len(dists) - 1], draft_probs, gen)
        want = opts.get("logprobs")
        lps = None
        if want is not None:
            lps = [(tok, *top_logprobs(processed[i], tok, want)) for i, tok in enumerate(new)]
        return accepted, new, lps

    def _commit(self, seq: int, accepted: int, new: "list[int]", lps, drafted: int) -> "tuple[list[int], bool]":
        """Clip to the limit and the row's end tokens, record, advance the grammar; returns (committed, done)."""
        new = new[:max(0, self.limits[seq][0] - self._generated_count(seq))]
        for i, token in enumerate(new):
            if token in self.ends.get(seq, self.eos):
                new = new[:i + 1]
                break
        if lps is not None and seq in self.lps:
            self.lps[seq] += lps[: len(new)]
        matcher = self.matchers.get(seq)
        if matcher is not None and new:
            matcher.advance(new)
        self.tokens[seq] += new
        committed = min(accepted, len(new))
        self.accepted_total += committed
        self.drafted_total += drafted
        if committed < len(self.accepted_per_step):
            self.accepted_per_step[committed] += 1   # the shape of acceptance, not only its mean: what prices spec_k
        done = any(t in self.ends.get(seq, self.eos) for t in new) or self._generated_count(seq) >= self.limits[seq][0]
        return new, done

    def _moved(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stale = True

    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int, marks=None) -> bool:
        """`marks`: {absolute position: snapshot} for the block boundaries inside this step that the prefix cache keeps
        (base/runner): the KDA states are taken by the forward at those cuts; the drafter's context ring at a mark is the
        ring before this step plus the step's positions before the mark, observed into the snapshot here."""
        self._moved()
        ids = torch.tensor(self.tokens[seq][start: start + tokens], dtype=torch.int64, device=self.caches.device)
        patches = self._patches(seq, start, start + tokens) if seq in self.media else ()
        cuts = tuple(sorted((int(p) - start, int(snap)) for p, snap in (marks or {}).items()))
        h, aux = self._forward(Step.prefill(ids, start, seq, slot, patches, cuts))
        self.ctx[seq] = start + tokens
        if aux is not None:                                                 # every prompt token is context for the drafter
            for rel, snap in cuts:                                          # the marks' rings first: the step's observe below overwrites cells
                self.caches.mark_draft(snap, slot)
                self.drafter.observe(self.caches.snapshot_draft_ring(snap), torch.arange(start, start + rel, device=ids.device), aux[:rel])
            self.drafter.observe(self.caches.draft_ring(slot), torch.arange(start, start + tokens, device=ids.device), aux)
        if self.ctx[seq] == self.prompt_len[seq]:                         # the prompt is in: the first token comes from its last position
            if self._rich(seq):
                rows = self._gather(self.net.head_local(h[-1:]))
                accepted, new, lps = self._pick_rich(seq, rows, [], None)
                self._commit(seq, 0, new[:1], lps, 0)
            else:
                first = self._sample_hidden(h[-1:], [self.limits[seq][1]])
                self.tokens[seq].append(self._no_end_yet(seq, [int(first.item())], h[-1:])[0])
        self.steps += 1
        generated = self._generated_count(seq)
        return generated > 0 and (self.tokens[seq][-1] in self.ends.get(seq, self.eos) or generated >= self.limits[seq][0])

    def decode(self, seqs, blocks, slots) -> "list[bool]":
        self._moved()
        flat, segments, drafts, draft_probs = [], [], {}, {}
        for seq, slot in zip(seqs, slots):
            ring = self.caches.draft_ring(slot) if self.drafter.k else None
            if self.drafter.k and self._rich(seq) and self.limits[seq][1] > 0:
                drafts[seq], draft_probs[seq] = self.drafter.propose_sampled(self.tokens[seq][-1], self.ctx[seq], ring, self.limits[seq][1],
                                                                             self.gens.get(seq, self.gen), self.F.vocab)
            else:
                drafts[seq] = self.drafter.propose(self.tokens[seq][-1], self.ctx[seq], ring)
                draft_probs[seq] = None
            ids = [self.tokens[seq][-1]] + drafts[seq]
            segments.append(Segment(seq, slot, self.ctx[seq], len(flat), len(ids)))
            flat.extend(ids)
        step = Step(torch.tensor(flat, dtype=torch.int64, device=self.caches.device), tuple(segments))
        rich = {s.seq: self._rich(s.seq) for s in step.segments}
        temps = [self.limits[s.seq][1] for s in step.segments for _ in range(s.length)]
        if self.decode_graphs is None:
            h, aux = self._forward(step)
            local = self.net.head_local(h)
            sampled = self._sample_hidden(h, temps).tolist() if not all(rich.values()) else None
        else:
            shape = self.decode_graphs.shape(step)                         # the sampler names itself from it too
            # Which captured graph this step ran: its sequence count is the batch the scheduler
            # actually filled, and its capacity bucket is the only production evidence for how
            # far the ladder needs to reach (decode_graphs.capacity_ladder).
            key = (shape[0], shape[2])
            self.decode_shape_counts[key] = self.decode_shape_counts.get(key, 0) + 1
            h, aux, local = self.decode_graphs.run(step, shape)
            sampled = self.sampling_graphs.run(shape, temps).tolist() if not all(rich.values()) else None
        finished = []
        for s in step.segments:
            rows = slice(s.start, s.start + s.length)
            if rich[s.seq]:
                full = self._gather(local[rows])
                accepted, new, lps = self._pick_rich(s.seq, full, drafts[s.seq], draft_probs[s.seq])
            else:
                picks = self._no_end_yet(s.seq, sampled[rows], h[rows])
                accepted = 0
                for d, got in zip(drafts[s.seq], picks):
                    if d != got:
                        break
                    accepted += 1
                new, lps = picks[: accepted + 1], None                     # the accepted drafts' confirmations, then the correction
            new, done = self._commit(s.seq, accepted, new, lps, len(drafts[s.seq]))
            committed = len(new)                                           # clipped tokens must not enter the next turn's context
            if aux is not None:
                observe = self.drafter.observe_decode if self.decode_graphs is not None else self.drafter.observe
                observe(self.caches.draft_ring(s.slot), torch.arange(s.ctx, s.ctx + committed, device=h.device), aux[s.start: s.start + committed])
            self.ctx[s.seq] += committed
            finished.append(done)
        self.steps += 1
        return finished
