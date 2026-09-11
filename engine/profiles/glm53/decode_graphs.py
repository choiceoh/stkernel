"""Fixed-shape target decode with context, request and state-slot ids on device.

Active sequence count, tokens per sequence and a declared context-capacity
bucket select a graph. Exact context lengths and physical cache ownership
are replay inputs, including rollback.
Paged and recurrent writes go directly to the arena using device slot ids.
The served ring lanes directly address both convolution and recurrent state;
only the small indexer tail needs a temporary buffer. Functional lanes also
gather convolution history and initial recurrent state.
No request may be live during capture.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from engine.base.graphs import DecodeGraphs
from engine.profiles.glm53.net import Segment


@triton.jit
def _scatter_rows(SRC, DST, INDEX, VALID, WIDTH: tl.constexpr,
                  SRC_STRIDE: tl.constexpr, DST_STRIDE: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = row < tl.load(VALID)
    dst = tl.load(INDEX + row, valid, other=0)
    value = tl.load(SRC + row * SRC_STRIDE + col, valid & (col < WIDTH), other=0)
    tl.store(DST + dst * DST_STRIDE + col, value, valid & (col < WIDTH))


def scatter_rows(src, dst, indices, valid):
    """Write only complete pools; padded rows must never alias real cache rows."""
    width = src.shape[-1]
    _scatter_rows[(src.shape[0],)](src, dst, indices, valid, width, src.stride(0),
                                  dst.stride(0), triton.next_power_of_2(width))


def complete_pools(net, layer, segment, tail, k, gate, caches):
    """Return the fixed candidate capacity after masked pool and ring writes."""
    F = net.F
    kp, d = F.kpool, F.idx_dim
    ctx, length = segment.ctx, segment.length
    lead = ctx % kp
    count = (lead + length) // kp
    max_pools = (kp - 1 + length) // kp
    relative = torch.arange(max_pools * kp, device=k.device) - lead
    current = relative.clamp(0, length - 1).long()
    previous = ((ctx + relative) % tail.shape[0]).long()
    kw = torch.where((relative < 0)[:, None], tail[previous, 0], k[current])
    gw = torch.where((relative < 0)[:, None], tail[previous, 1], gate[current])
    pk, ps = net.lanes.kpool_compress(kw.view(max_pools, kp, d),
                                     gw.view(max_pools, kp, d), net.p[f"L{layer}.idx.ape"])
    pids = ctx // kp + torch.arange(max_pools, device=k.device)
    # Padded pids at the final context boundary are not read by scatter_rows.
    pids = pids.clamp_max(caches.candidate_capacity - 1)
    slots = caches.pool_slots(layer, segment.seq, pids).long()
    scatter_rows(pk.view(torch.uint8), caches.pool_keys(layer).view(torch.uint8), slots, count)
    scatter_rows(ps.reshape(-1, 1), caches.pool_scales(layer).unsqueeze(-1), slots, count)
    caches.write_tail(layer, segment.slot, ctx, k, gate)
    return caches.candidate_capacity


@dataclass
class DeviceStep:
    ids: torch.Tensor
    contexts: torch.Tensor
    tokens: int
    captured = True

    @property
    def segments(self):
        return tuple(Segment(i, i, self.contexts[i:i+1].reshape(()), i * self.tokens, self.tokens)
                     for i in range(self.contexts.numel()))

    @property
    def positions(self):
        return (self.contexts[:, None] + torch.arange(self.tokens, device=self.ids.device)).flatten()


class GraphCaches:
    def __init__(self, real, sequence_ids, slots, capacity):
        self.real, self.sequence_ids, self.slots = real, sequence_ids, slots
        self.F, self.layout = real.F, real.layout
        self.candidate_capacity = capacity // real.F.kpool

    def gather(self):
        # Unreserved pages are masked out of attention by valid pool counts.
        # Translate them to a readable page so padded gathers stay in bounds.
        self.block_table = self.real.block_table.index_select(0, self.sequence_ids).clamp_min(0)

    def kda_history(self, layer, slot, context):
        from engine.kernels.state import kda_history
        return kda_history(self.real._fields["conv", layer], self.real._fields["rec", layer],
                           self.slots[slot:slot+1], context, self.F.conv - 1)

    def kda_ring_history(self, layer, slot, context):
        from engine.kernels.state import conv_history
        physical = self.slots[slot:slot+1]
        hist = conv_history(self.real._fields["conv", layer], physical, context, self.F.conv - 1)
        return hist, self.real._fields["rec", layer], physical

    def kda_rings(self, layer, slot):
        return self.real._fields["conv", layer], self.real._fields["rec", layer], self.slots[slot:slot+1]

    def write_conv(self, layer, slot, context, inputs):
        from engine.kernels.state import write_conv
        write_conv(inputs, self.real._fields["conv", layer], self.slots[slot:slot+1], context)

    def write_rec(self, layer, slot, context, states):
        from engine.kernels.state import write_ring
        write_ring(states, self.real._fields["rec", layer], self.slots[slot:slot+1], context)

    def tail(self, layer, slot):
        return self.real._fields["tail", layer].index_select(0, self.slots[slot:slot+1])[0]

    def write_tail(self, layer, slot, context, keys, gates):
        from engine.kernels.state import write_ring
        write_ring(torch.stack((keys, gates), dim=1), self.real._fields["tail", layer],
                   self.slots[slot:slot+1], context)

    def latent(self, layer):
        return self.real.latent(layer)

    def pool_keys(self, layer):
        return self.real.pool_keys(layer)

    def pool_scales(self, layer):
        return self.real.pool_scales(layer)

    def token_slots(self, layer, seq, positions):
        from engine.profiles.glm53.caches import Glm53Caches
        return Glm53Caches.token_slots(self, layer, seq, positions)

    def token_map(self, layer, seq):
        from engine.profiles.glm53.caches import Glm53Caches
        return Glm53Caches.token_map(self, layer, seq)

    def pool_slots(self, layer, seq, pool_ids):
        from engine.profiles.glm53.caches import Glm53Caches
        return Glm53Caches.pool_slots(self, layer, seq, pool_ids)


def capacity_ladder(pool_tokens: int, max_position: int, ceiling: "int | None") -> "list[int]":
    """The context-capacity buckets to capture, smallest first.

    A decode step runs the first bucket that covers its longest sequence, so the
    ladder must reach the longest context the door will admit -- and no further:
    every bucket above it is a graph captured for a request that cannot arrive.
    `ceiling` is that served number (adapter.max_context); None means the
    checkpoint's trained positions. The KV pool bounds both: a sequence cannot
    hold more tokens than the pool has.
    """
    served = max_position if ceiling is None else int(ceiling)
    if served <= 0 or pool_tokens <= 0 or max_position <= 0:
        raise ValueError("served ceiling, pool capacity and trained positions must be positive")
    total = min(pool_tokens, served, max_position)
    ladder = [min(4096, total)]
    while ladder[-1] < total:
        ladder.append(min(total, ladder[-1] * 2))
    return ladder


class Glm53DecodeGraphs:
    def __init__(self, net, caches, max_seqs, tokens, aux_layers=(), memory=None, ceiling=None):
        if any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("capture requires no live state slots")
        if tokens not in (1, net.F.spec_k + 1):
            raise ValueError("decode capture needs the declared target or verify width")
        if not getattr(net.comm, "graph_capture_safe", True):
            # base/comm.LocalTP crosses ranks through a host barrier: it leaves no node in
            # the graph and its peer reads would race on replay. Die, never capture (D3).
            raise ValueError(f"{type(net.comm).__name__} collectives cannot be captured: "
                             "its ranks meet on a host barrier, which no replay performs")
        self.net, self.caches, self.tokens = net, caches, tokens
        self.memory = memory
        self.aux_layers = tuple(aux_layers)
        # The ladder ends where the door stops admitting. `ceiling` is that one served
        # number (adapter.max_context, which serve.py refuses past); unset means the
        # model's trained positions. Capping it is the only lever on the graph count:
        # a bucket is captured whether or not any request will reach it, and each costs
        # a warmup pair plus a capture (their seconds are memory rows, "target/<shape>/").
        self.capacities = capacity_ladder(caches.block_table.shape[1] * net.F.block,
                                         net.F.max_position, ceiling)

        # The rank-local logits go into a buffer this class owns, one per (n, tokens), instead of
        # into an allocation each capture makes for itself. The sampler depends on the logits shape
        # alone, so sharing the buffer across a row's capacity buckets is what lets one sampler graph
        # serve all of them: 72 sampling graphs become 8 (boot-time study, 2026-09-11).
        self.logits = {}
        # Replay writes four small arrays per step. Staged through ONE pinned block so
        # each is an async copy on the caller's stream instead of a fresh CPU tensor and
        # a pageable (implicitly synchronizing) transfer. Safe to overwrite between
        # steps: the sampler's readback synchronizes the stream before the next fill.
        self.staging = torch.empty(3, max_seqs, dtype=torch.int64, pin_memory=True)
        self.staged = self.staging.numpy()          # write through numpy: no per-element torch dispatch

        def logits_for(n, t):
            key = (n, t)
            if key not in self.logits:
                self.logits[key] = torch.empty(n * t, net.vp, device=caches.device,
                                               dtype=net.p["head"].dtype)
            return self.logits[key]

        def make_inputs(n, t, capacity):
            device = caches.device
            seqs = torch.arange(n, device=device, dtype=torch.int64)
            slots = seqs + 1
            contexts = torch.zeros(n, device=device, dtype=torch.int64)
            step = DeviceStep(torch.zeros(n * t, device=device, dtype=torch.int64), contexts, t)
            return step, seqs, slots, GraphCaches(caches, seqs, slots, capacity), logits_for(n, t)

        def forward(inputs):
            step, _, _, scratch, logits = inputs
            scratch.gather()
            try:
                result = net.forward(step, scratch, aux_layers=self.aux_layers)
                h, aux = result if self.aux_layers else (result, None)
                logits.copy_(net.head_local(h))
                return h, aux, logits
            finally:
                del scratch.block_table

        try:
            self.graphs = DecodeGraphs(forward, make_inputs,
                                       [(n, tokens, capacity) for capacity in self.capacities
                                        for n in range(1, max_seqs + 1)], memory=memory, label="target",
                                       resources=net.lanes.graph_resources)
        finally:
            # Warmup and capture execute real writes, before requests exist.
            caches.reset()

    def shape(self, step):
        if any(s.length != self.tokens for s in step.segments):
            raise ValueError("decode tokens differ from the captured contract")
        end = max(s.ctx + s.length for s in step.segments)
        for capacity in self.capacities:
            if end <= capacity:
                return len(step.segments), self.tokens, capacity
        raise ValueError("decode context exceeds the captured cache capacity")

    def run(self, step, shape=None):
        """`shape` is this step's, when the caller already asked for it (the sampler needs it too)."""
        shape = self.shape(step) if shape is None else shape
        self.caches.prepare(step)
        segments = step.segments
        staged = self.staged
        for i, s in enumerate(segments):
            staged[0, i], staged[1, i], staged[2, i] = s.ctx, s.seq, s.slot

        def fill(inputs):
            target, seqs, slots, _, _ = inputs
            target.ids.copy_(step.ids, non_blocking=True)
            n = len(segments)
            target.contexts.copy_(self.staging[0, :n], non_blocking=True)
            seqs.copy_(self.staging[1, :n], non_blocking=True)
            slots.copy_(self.staging[2, :n], non_blocking=True)

        return self.graphs.run(shape, fill)


class DrafterDecodeGraphs:
    """One proposal graph and the finite accepted-prefix context updates."""
    def __init__(self, drafter, caches, memory=None):
        self.field = caches._fields["draft", -1]
        self.drafter = drafter
        device = caches.device

        def propose_inputs(n, t):
            return dict(anchor=torch.zeros(1, device=device, dtype=torch.int64),
                        position=torch.zeros((), device=device, dtype=torch.int64),
                        slot=torch.zeros(1, device=device, dtype=torch.int64))

        def propose(inputs):
            ring = self.field.index_select(0, inputs["slot"])[0]
            return drafter.propose_tensor(inputs["anchor"], inputs["position"], ring)

        def observe_inputs(n, t):
            return dict(positions=torch.arange(t, device=device, dtype=torch.int64),
                        aux=torch.zeros(t, drafter.F.hidden * len(drafter.aux_layers),
                                        device=device, dtype=torch.bfloat16),
                        slot=torch.zeros(1, device=device, dtype=torch.int64))

        def observe(inputs):
            rings = self.field.index_select(0, inputs["slot"])
            drafter.observe(rings[0], inputs["positions"], inputs["aux"])
            self.field.index_copy_(0, inputs["slot"], rings)

        try:
            self.proposals = DecodeGraphs(propose, propose_inputs, [(1, drafter.k + 1)],
                                          memory=memory, label="drafter/propose")
            self.observations = DecodeGraphs(observe, observe_inputs,
                                             [(1, t) for t in range(1, drafter.k + 2)],
                                             memory=memory, label="drafter/observe")
        except BaseException:
            if hasattr(self, "proposals"):
                self.proposals.close()
            raise
        finally:
            caches.reset()

    def slot(self, ring):
        stride = self.field.stride(0) * self.field.element_size()
        delta = ring.data_ptr() - self.field.data_ptr()
        if delta % stride or not 0 < delta // stride < self.field.shape[0]:
            raise ValueError("drafter ring must belong to a real arena state slot")
        return delta // stride

    def propose(self, anchor, position, ring):
        slot = self.slot(ring)
        def fill(inputs):
            inputs["anchor"].fill_(anchor)
            inputs["position"].fill_(position)
            inputs["slot"].fill_(slot)
        return self.proposals.run((1, self.drafter.k + 1), fill)

    def observe(self, ring, positions, aux):
        slot = self.slot(ring)
        def fill(inputs):
            inputs["positions"].copy_(positions)
            inputs["aux"].copy_(aux)
            inputs["slot"].fill_(slot)
        self.observations.run((1, positions.numel()), fill)


class SamplingGraphs:
    """Greedy and stochastic sampling bind the target graphs' output buffers.

    Choosing the declared sampling policy uses host request metadata. Greedy
    replay draws no random numbers; mixed/stochastic replay advances the
    engine's explicit generator exactly as the eager sampler does.
    """
    def __init__(self, target, generator, decodable, top_p):
        from engine.base.sampler import sample
        from engine.modules.vocab import argmax
        self.tokens = target.tokens
        # One sampler per LOGITS BUFFER. A target shape is (seqs, tokens, capacity) but the sampler
        # only ever sees (seqs * tokens, vocab), and the target graphs of one (seqs, tokens) write
        # the same static buffer -- so capturing one per capacity captured the same program nine
        # times. The identity check keeps that contract from rotting: a target that hands out a
        # different buffer per capacity must not have its samplers quietly bound to the first.
        first = {}
        for shape in target.graphs.outputs:
            key, logits = shape[:2], target.graphs.outputs[shape][2]
            if key not in first:
                first[key] = shape
            elif target.graphs.outputs[first[key]][2] is not logits:
                raise ValueError(f"target shapes {first[key]} and {shape} must share one logits buffer")
        shapes = list(first)
        # Same pinned staging as the target's replay path, for the one array this one writes.
        self.temps = torch.empty(max(n * t for n, t in shapes), dtype=torch.float32, pin_memory=True)
        self.staged = self.temps.numpy()
        saved = generator.get_state()

        def make_inputs(*shape):
            n, t = shape[:2]
            logits = target.graphs.outputs[first[(n, t)]][2]
            # Capture records the target outputs but need not initialize them.
            # Sampling warmup must see finite logits before the first request.
            logits.zero_()
            return logits, torch.ones(n*t, device=logits.device), torch.full((n*t,), top_p, device=logits.device)

        def greedy(inputs):
            return argmax(inputs[0], target.net.comm, target.net.rank * target.net.vp, decodable)

        def stochastic(inputs):
            local_logits, temps, p = inputs
            logits = target.net.comm.all_gather(local_logits, dim=-1)
            if decodable is not None and logits.shape[-1] > decodable:
                logits = logits.clone()
                logits[:, decodable:] = float("-inf")
            return sample(logits, temps, p, generator, top_p_enabled=top_p < 1.)

        try:
            memory = getattr(target, "memory", None)
            self.greedy = DecodeGraphs(greedy, make_inputs, shapes, memory=memory, label="sampling/greedy")
            self.stochastic = DecodeGraphs(stochastic, make_inputs, shapes, generators=(generator,),
                                          memory=memory, label="sampling/stochastic")
        except BaseException:
            if hasattr(self, "greedy"):
                self.greedy.close()
            raise
        finally:
            generator.set_state(saved)

    def run(self, shape, temperatures):
        """`shape` is the target's, whose first two entries name the sampler."""
        shape = tuple(shape[:2])
        if all(t <= 0 for t in temperatures):
            return self.greedy.run(shape, lambda inputs: None)
        rows = len(temperatures)
        self.staged[:rows] = temperatures
        return self.stochastic.run(shape, lambda inputs: inputs[1].copy_(self.temps[:rows], non_blocking=True))

    def close(self):
        self.greedy.close()
        self.stochastic.close()
