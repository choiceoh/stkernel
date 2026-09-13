"""The decoder as a composition (base): a layer plan, a residual form and the features it names -- one step loop.

A model here is not a file. It is three declarations a profile makes from its checkpoint:

- the **plan**: for every layer, the token mixer it runs (full attention in some variant, linear attention in some
  variant), the channel mixer (a dense MLP, an MoE), and the features injected into the residual before it (Qwen3.8's
  hashed n-gram table, DeepSeek-V4.1's engram);
- the **residual form**: how a sublayer reads its input from the residual state and writes its output back -- plain
  pre-norm, GLM-5.3's mhc streams, Qwen3.8's gated residual streams, DeepSeek-V4.1's split-sinkhorn streams;
- the **features** the plan names, each bound to the checkpoint's weights (engine/modules: named by feature, shared by
  every profile that has it).

CHARTER D6 as corrected: implementation per profile, features shared as modules, kernels under the modules. This file
holds the part that is the same for every model -- the loop -- and nothing with a model's name in it.

ONE step function, as GLM-5.3's net: a step is segments -- (seq, ctx, start, length) -- over a flat token array; a
prefill chunk is one long segment, a decode step is short ones. Features see the flat tensors and the segments and
keep what they carry between steps in a `State`, keyed by (layer, feature, sequence). Token-wise features (norms, MLPs,
residual forms) ignore the segments; sequence features (mixers, injections that look back) walk them.

The `State` here is the reference store: plain tensors per sequence, the state after the sequence's last computed
token. The served store is base/kv's paged blocks and slots, sized from the specs the features declare
(`Composition.cache_specs` -> base/cache_spec.plan), addressed by position so a rejected draft is overwritten rather
than rolled back -- that store and the kernel lanes behind the features are the served composition's, not this file's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from engine.base.cache_spec import PagedSpec, SlotSpec

SITES = ("mixer", "mlp")            # the two sublayers of every layer, in order


@dataclass(frozen=True)
class Segment:
    seq: int
    ctx: int                        # tokens already computed for this sequence: this segment's positions are ctx..
    start: int                      # first token of the segment in the step's flat arrays
    length: int
    verify: bool = False            # provisional tokens (a drafter's): the state after each is kept until `accept`

    def __post_init__(self):
        if min(self.ctx, self.start) < 0 or self.length <= 0:
            raise ValueError(f"a segment has a nonnegative context and start and a positive length: {self}")


@dataclass(frozen=True)
class Step:
    ids: torch.Tensor               # [N] int64
    segments: "tuple[Segment, ...]"

    def __post_init__(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.int64 or not self.segments:
            raise ValueError("a step is a flat int64 token vector and at least one segment")
        at = 0
        seqs = set()
        for s in self.segments:
            if s.start != at or s.seq in seqs:
                raise ValueError("segments tile the flat tokens in order, one per sequence")
            at += s.length
            seqs.add(s.seq)
        if at != self.ids.numel():
            raise ValueError(f"segments cover {at} tokens of {self.ids.numel()}")

    def positions(self) -> torch.Tensor:
        """[N] int64 absolute positions: ctx + j within each segment."""
        return torch.cat([torch.arange(s.ctx, s.ctx + s.length, dtype=torch.int64, device=self.ids.device)
                          for s in self.segments])

    def last(self) -> torch.Tensor:
        """[segments] the flat index of each segment's last token -- where a step samples."""
        return torch.tensor([s.start + s.length - 1 for s in self.segments], dtype=torch.int64, device=self.ids.device)

    @staticmethod
    def of(chunks) -> "Step":
        """A step from (seq, ctx, ids) or (seq, ctx, ids, verify) per sequence, laid out in order."""
        segments, at, flat = [], 0, []
        for chunk in chunks:
            seq, ctx, ids = chunk[:3]
            verify = bool(chunk[3]) if len(chunk) > 3 else False
            segments.append(Segment(seq, ctx, at, ids.numel(), verify))
            flat.append(ids.reshape(-1).to(torch.int64))
            at += ids.numel()
        return Step(torch.cat(flat), tuple(segments))


class State:
    """What features carry between steps -- the reference store (see the module docstring). Two kinds of key, the two
    of base/cache_spec: a whole value per sequence (`get`/`put`: a slot -- a recurrent state, a conv history) and rows
    per token (`put_rows`/`rows`: paged -- keys and values, indexer keys). A feature owns its keys. `check` refuses a
    step that does not continue its sequences; `commit` records where they now are. base/composed.PositionStore is the
    same protocol over blocks and slots.

    A verify segment's tokens are provisional (a drafter's): a feature records the value after EACH of them
    (`put(..., at=j)`, `put_state` below) instead of the value after the last, and the step leaves the sequence where
    it was until `accept(seq, n)` keeps the first n -- the values after token n-1 become the sequence's, its rows past
    them are dropped (a served store overwrites them instead). Rolling back is choosing a position, never recomputing."""

    def __init__(self):
        self._values: dict = {}
        self._rows: dict = {}
        self._provisional: dict = {}                 # (layer, key, seq) -> {offset: value}
        self._verifying: dict = {}                   # seq -> (ctx, length) of a verify step awaiting accept
        self._step: dict = {}                        # seq -> segment of the step being run
        self.contexts: dict = {}

    def get(self, layer: int, key: str, seq: int, default=None):
        return self._values.get((layer, key, seq), default)

    def put(self, layer: int, key: str, seq: int, value, at: "int | None" = None) -> None:
        segment = self._step.get(seq)
        if segment is not None and segment.verify:
            if at is None or not 0 <= at < segment.length:
                raise ValueError(f"{key}: a verify step keeps the value after each of its {segment.length} tokens")
            self._provisional.setdefault((layer, key, seq), {})[at] = value
            return
        if at is not None:
            raise ValueError(f"{key}: only a verify step keeps values by offset")
        self._values[(layer, key, seq)] = value

    def put_rows(self, layer: int, key: str, seq: int, rows) -> None:
        """This step's rows for `seq`, appended after the rows of the positions before it."""
        held = self._rows.get((layer, key, seq))
        self._rows[(layer, key, seq)] = rows if held is None else torch.cat([held, rows])

    def rows(self, layer: int, key: str, seq: int, count: int):
        """The rows of positions [0, count): what the sequence holds so far, this step's included once put."""
        held = self._rows.get((layer, key, seq))
        if held is None or held.shape[0] < count:
            raise ValueError(f"sequence {seq} holds {0 if held is None else held.shape[0]} rows of {key}, asked {count}")
        return held[:count]

    def check(self, step: Step) -> None:
        """Refuse a step whose segments do not continue their sequences from where the state left them."""
        for s in step.segments:
            if s.seq in self._verifying:
                raise ValueError(f"sequence {s.seq} has a verify step waiting for accept")
            if self.contexts.get(s.seq, 0) != s.ctx:
                raise ValueError(f"sequence {s.seq} is at {self.contexts.get(s.seq, 0)} tokens, the step says {s.ctx}")
        self._step = {s.seq: s for s in step.segments}

    def commit(self, step: Step) -> None:
        for s in step.segments:
            if s.verify:
                self._verifying[s.seq] = (s.ctx, s.length)
            else:
                self.contexts[s.seq] = s.ctx + s.length
        self._step = {}

    def accept(self, seq: int, n: int) -> None:
        """Keep the first `n` tokens of the sequence's verify step: their values are the sequence's, its context
        ctx + n."""
        if seq not in self._verifying:
            raise ValueError(f"sequence {seq} has no verify step to accept")
        ctx, length = self._verifying[seq]
        if not 1 <= n <= length:
            raise ValueError(f"accept keeps 1..{length} tokens of sequence {seq}'s verify step, not {n}")
        del self._verifying[seq]
        for key in [k for k in self._provisional if k[2] == seq]:
            held = self._provisional.pop(key)
            if n - 1 not in held:
                raise ValueError(f"{key[1]}: layer {key[0]} kept no value after token {n - 1}")
            self._values[key] = held[n - 1]
        for key in [k for k in self._rows if k[2] == seq]:
            self._rows[key] = self._rows[key][:ctx + n]
        self.contexts[seq] = ctx + n

    def drop(self, seq: int) -> None:
        self._values = {k: v for k, v in self._values.items() if k[2] != seq}
        self._rows = {k: v for k, v in self._rows.items() if k[2] != seq}
        self._provisional = {k: v for k, v in self._provisional.items() if k[2] != seq}
        self._verifying.pop(seq, None)
        self.contexts.pop(seq, None)


def put_state(state, layer: int, key: str, segment: Segment, final, each=None) -> None:
    """The value a sequence carries past `segment`: `final`, the value after its last token -- or, when the segment is a
    verify one, `each` [length, ...], the value after every token, so the model can accept any prefix. A feature
    that cannot give `each` cannot be verified, and says so here rather than leaving a rejected draft in its state."""
    if not segment.verify:
        state.put(layer, key, segment.seq, final)
        return
    if each is None or each.shape[0] != segment.length:
        raise ValueError(f"{key}: a verify segment of {segment.length} tokens needs the value after each of them")
    for j in range(segment.length):
        state.put(layer, key, segment.seq, each[j], at=j)


class Residual(Protocol):
    """How sublayers read and write the residual state (engine/modules/residual: the forms). `enter` and `leave` see the
    step and the state: a form may keep per-sequence state of its own (Inkling's convs on the sublayer outputs) and
    declare it with `cache_specs(layers)` like a feature."""
    def open(self, x: torch.Tensor) -> torch.Tensor: ...                                    # embeddings [N, H] -> state
    def enter(self, layer: int, site: str, h: torch.Tensor, step: "Step", state: "State") -> "tuple[torch.Tensor, object]": ...
    def leave(self, layer: int, site: str, out: torch.Tensor, carry, step: "Step", state: "State") -> torch.Tensor: ...
    def close(self, h: torch.Tensor) -> torch.Tensor: ...                                    # state -> final [N, H]


class Feature(Protocol):
    """A sublayer (a mixer, a channel mixer) or an injection: (layer, input, step, state) -> output. An injection's input
    is the residual state and its output is added to it."""
    def __call__(self, layer: int, x: torch.Tensor, step: Step, state: State) -> torch.Tensor: ...


@dataclass(frozen=True)
class Layer:
    mixer: str                      # a feature name, e.g. "linear_attention", "sparse_attention"
    mlp: str                        # e.g. "moe", "dense"
    inject: "tuple[str, ...]" = ()  # features added into the residual state before the layer, in order

    def names(self) -> "tuple[str, ...]":
        return (*self.inject, self.mixer, self.mlp)


@dataclass(frozen=True)
class Plan:
    layers: "tuple[Layer, ...]"

    def __post_init__(self):
        if not self.layers or not all(isinstance(layer, Layer) and all(layer.names()) for layer in self.layers):
            raise ValueError("a plan is a nonempty sequence of layers, each naming its features")

    def layers_of(self, name: str) -> "list[int]":
        """The layers that run feature `name`."""
        return [i for i, layer in enumerate(self.layers) if name in layer.names()]


@dataclass
class Composition:
    plan: Plan
    embed: Callable                 # ids [N] -> [N, H]
    residual: Residual
    features: dict = field(default_factory=dict)
    head: Callable = None           # [M, H] -> logits [M, V]
    offset: int = 0                 # the model layer the plan's first layer is: a drafter's layers follow the target's
    fuse: Callable = None           # (embeddings [N, H], given [N, ...], step) -> residual state: a head that opens
                                    # from another model's state (an MTP head reads the target's; engine/modules/mtp)

    def __post_init__(self):
        missing = sorted({name for layer in self.plan.layers for name in layer.names()} - set(self.features))
        if missing:
            raise ValueError(f"the plan names features the composition was not given: {missing}")
        if self.head is None:
            raise ValueError("a composition needs a head")

    def forward(self, step: Step, state: State, *, logits: str = "last", hidden: bool = False, given=None):
        """Run one step: logits for each segment's last token ("last") or for every token ("all"). The state advances
        past the step's tokens (a verify segment's, once accepted). With `hidden`, also the residual state before the
        closing mix for every token -- what a drafter reads (Qwen3.8's MTP takes the multi-stream state) -- as
        (logits, hidden). `given` [N, ...]: the state a fusing head opens from (`fuse`) instead of the residual's open.
        Features and the residual see model layer `offset + i` for plan layer i."""
        if logits not in ("last", "all"):
            raise ValueError("logits are 'last' or 'all'")
        if given is not None and self.fuse is None:
            raise ValueError("this composition opens from its embeddings; it has no fuse for a given state")
        state.check(step)
        if given is None:
            h = self.residual.open(self.embed(step.ids))
        else:
            if given.shape[0] != step.ids.numel():
                raise ValueError(f"a given state holds {given.shape[0]} rows for a step of {step.ids.numel()} tokens")
            h = self.fuse(self.embed(step.ids), given, step)
        for plan_index, layer in enumerate(self.plan.layers):
            layer_index = self.offset + plan_index
            for name in layer.inject:
                h = h + self.features[name](layer_index, h, step, state)
            for site, name in zip(SITES, (layer.mixer, layer.mlp)):
                x, carry = self.residual.enter(layer_index, site, h, step, state)
                h = self.residual.leave(layer_index, site, self.features[name](layer_index, x, step, state), carry,
                                        step, state)
        out = self.residual.close(h)
        state.commit(step)
        result = self.head(out if logits == "all" else out[step.last()])
        return (result, h) if hidden else result

    def cache_specs(self) -> "tuple[list[PagedSpec], list[SlotSpec]]":
        """What the features cache, per kind: every feature that declares `cache_specs(layers)` for the layers the plan
        runs it on. base/cache_spec.plan turns these into blocks and slots."""
        paged, slots = [], []
        for _, _, spec in self._specs():
            (paged if isinstance(spec, PagedSpec) else slots).append(spec)
        return paged, slots

    def spec_layers(self) -> "dict[str, list[int]]":
        """For every keyed spec, the model layers it is kept for, in order: a store indexes a spec's per-layer regions
        by a layer's rank in this list (the plan's third GDN layer is the GDN feature's third region)."""
        return {spec.key: layers for _, layers, spec in self._specs() if spec.key}

    def _specs(self):
        for name, feature in self.features.items():
            declare = getattr(feature, "cache_specs", None)
            layers = [self.offset + i for i in self.plan.layers_of(name)]
            if declare is None or not layers:
                continue
            for spec in declare(layers):
                yield name, layers, spec
        declare = getattr(self.residual, "cache_specs", None)       # the residual form's own state, on every layer
        if declare is not None:
            layers = [self.offset + i for i in range(len(self.plan.layers))]
            for spec in declare(layers):
                yield "residual", layers, spec


__all__ = ["SITES", "Segment", "Step", "State", "put_state", "Residual", "Feature", "Layer", "Plan", "Composition"]
