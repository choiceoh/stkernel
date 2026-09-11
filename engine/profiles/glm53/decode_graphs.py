"""Fixed-shape target decode with context, request and state-slot ids on device.

Active sequence count, tokens per sequence and a declared context-capacity
bucket select a graph. Exact context lengths and physical cache ownership
are replay inputs, including rollback.
Paged writes go directly to the arena; state rings are gathered by slot and
committed inside the graph. No request may be live during capture.
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
    keep = min(length, tail.shape[0])
    positions = ctx + torch.arange(length - keep, length, device=k.device)
    tail[positions % tail.shape[0], 0] = k[-keep:]
    tail[positions % tail.shape[0], 1] = gate[-keep:]
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
        self.fields = {key: value.index_select(0, self.slots)
                       for key, value in self.real._fields.items() if key[0] != "draft"}

    def commit(self):
        for key, value in self.fields.items():
            self.real._fields[key].index_copy_(0, self.slots, value)

    def kda(self, layer, slot):
        return self.fields["conv", layer][slot], self.fields["rec", layer][slot]

    def tail(self, layer, slot):
        return self.fields["tail", layer][slot]

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


class Glm53DecodeGraphs:
    def __init__(self, net, caches, max_seqs, tokens, aux_layers=()):
        if any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("capture requires no live state slots")
        if tokens not in (1, net.F.spec_k + 1):
            raise ValueError("decode capture needs the declared target or verify width")
        self.net, self.caches, self.tokens = net, caches, tokens
        self.aux_layers = tuple(aux_layers)
        total = caches.block_table.shape[1] * net.F.block
        self.capacities = [min(4096, total)]
        while self.capacities[-1] < total:
            self.capacities.append(min(total, self.capacities[-1] * 2))

        def make_inputs(n, t, capacity):
            device = caches.device
            seqs = torch.arange(n, device=device, dtype=torch.int64)
            slots = seqs + 1
            contexts = torch.zeros(n, device=device, dtype=torch.int64)
            step = DeviceStep(torch.zeros(n * t, device=device, dtype=torch.int64), contexts, t)
            return step, seqs, slots, GraphCaches(caches, seqs, slots, capacity)

        def forward(inputs):
            step, _, _, scratch = inputs
            scratch.gather()
            try:
                result = net.forward(step, scratch, aux_layers=self.aux_layers)
                h, aux = result if self.aux_layers else (result, None)
                logits = net.head(h)
                scratch.commit()
                return h, aux, logits
            finally:
                # These are graph intermediates. Retaining each shape's copy
                # would pin a separate set of recurrent states per bucket,
                # defeating the shared graph pool on the full model.
                scratch.fields.clear()
                del scratch.block_table

        try:
            self.graphs = DecodeGraphs(forward, make_inputs,
                                       [(n, tokens, capacity) for capacity in self.capacities
                                        for n in range(1, max_seqs + 1)])
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

    def run(self, step):
        shape = self.shape(step)
        self.caches.prepare(step)

        def fill(inputs):
            target, seqs, slots, _ = inputs
            target.ids.copy_(step.ids)
            target.contexts.copy_(torch.tensor([s.ctx for s in step.segments], dtype=torch.int64))
            seqs.copy_(torch.tensor([s.seq for s in step.segments], dtype=torch.int64))
            slots.copy_(torch.tensor([s.slot for s in step.segments], dtype=torch.int64))

        return self.graphs.run(shape, fill)


class DrafterDecodeGraphs:
    """One proposal graph and the finite accepted-prefix context updates."""
    def __init__(self, drafter, caches):
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
            self.proposals = DecodeGraphs(propose, propose_inputs, [(1, drafter.k + 1)])
            self.observations = DecodeGraphs(observe, observe_inputs,
                                             [(1, t) for t in range(1, drafter.k + 2)])
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
        self.tokens = target.tokens
        shapes = list(target.graphs.outputs)
        saved = generator.get_state()

        def make_inputs(*shape):
            n, t = shape[:2]
            logits = target.graphs.outputs[shape][2]
            # Capture records the target outputs but need not initialize them.
            # Sampling warmup must see finite logits before the first request.
            logits.zero_()
            return logits, torch.ones(n*t, device=logits.device), torch.full((n*t,), top_p, device=logits.device)

        def greedy(inputs):
            return inputs[0][:, :decodable].argmax(-1)

        def stochastic(inputs):
            logits, temps, p = inputs
            if decodable is not None and logits.shape[-1] > decodable:
                logits = logits.clone()
                logits[:, decodable:] = float("-inf")
            return sample(logits, temps, p, generator, top_p_enabled=top_p < 1.)

        try:
            self.greedy = DecodeGraphs(greedy, make_inputs, shapes)
            self.stochastic = DecodeGraphs(stochastic, make_inputs, shapes, generators=(generator,))
        finally:
            generator.set_state(saved)

    def run(self, shape, temperatures):
        if all(t <= 0 for t in temperatures):
            return self.greedy.run(shape, lambda inputs: None)
        return self.stochastic.run(shape, lambda inputs: inputs[1].copy_(torch.tensor(temperatures)))

    def close(self):
        self.greedy.close()
        self.stochastic.close()
