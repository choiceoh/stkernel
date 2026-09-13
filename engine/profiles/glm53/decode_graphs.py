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

from engine.base.constants import iota
from engine.base.graphs import DecodeGraphs
from engine.profiles.glm53.net import Segment


@triton.jit
def _scatter_rows(SRC, DST, INDEX, VALID, WIDTH: tl.constexpr,
                  SRC_STRIDE: tl.constexpr, DST_STRIDE: tl.constexpr,
                  BLOCK: tl.constexpr, SEG_SRC: tl.constexpr = 0, SEG_INDEX: tl.constexpr = 0):
    # Grid axis 1 is the segment of a captured decode step: program (row, seg) reads segment seg's count,
    # indices and source rows. A one-segment launch has one program there, at seg 0 with both segment
    # strides 0 -- the same loads and stores as before the axis existed.
    row = tl.program_id(0)
    seg = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = row < tl.load(VALID + seg)
    dst = tl.load(INDEX + seg * SEG_INDEX + row, valid, other=0)
    value = tl.load(SRC + seg * SEG_SRC + row * SRC_STRIDE + col, valid & (col < WIDTH), other=0)
    tl.store(DST + dst * DST_STRIDE + col, value, valid & (col < WIDTH))


def scatter_rows(src, dst, indices, valid):
    """Write only complete pools; padded rows must never alias real cache rows.

    One segment: src [P, W], indices [P], valid a device scalar count. Every segment of a captured step
    at once: src [n, P, W], indices [n, P], valid [n] -- one launch whose program (p, i) is what the
    one-segment launch's program p does for segment i."""
    width = src.shape[-1]
    if src.ndim == 3:
        n, rows = src.shape[0], src.shape[1]
        if indices.shape != (n, rows) or valid.shape != (n,) or valid.stride(0) != 1:
            raise ValueError("segment-batched scatter takes [n, P] indices and one contiguous count per segment")
        _scatter_rows[(rows, n)](src, dst, indices, valid, width, src.stride(1), dst.stride(0),
                                 triton.next_power_of_2(width), SEG_SRC=src.stride(0), SEG_INDEX=indices.stride(0))
        return
    _scatter_rows[(src.shape[0],)](src, dst, indices, valid, width, src.stride(0),
                                  dst.stride(0), triton.next_power_of_2(width))


def complete_pools(net, layer, contexts, length, tails, k, gate, caches):
    """Complete every segment's pools for this step, then return the fixed candidate capacity.

    A captured step's segments have the same length and differ only in device scalars,
    so the whole prologue -- the window that joins the tail ring's earlier tokens to
    this step's new ones, and the pooling itself -- is one set of operations over
    [segments, pools, ...] rather than one set per segment. Stacking them is the same
    work in the same order because `compress_pool_keys` runs one program per pool and
    reads each through its own strides; segments only decide which pools exist.

    The cache writes are one launch each as well (45차, the C=4 question: three launches
    a segment a layer were three a layer): `scatter_rows` and `write_ring_rows` carry the
    segment on a grid axis, and each program does what the one-segment launch's did.
    """
    F = net.F
    kp, d = F.kpool, F.idx_dim
    n, tail_width = tails.shape[0], tails.shape[1]
    max_pools = (kp - 1 + length) // kp
    lead = contexts % kp                                                  # [n] the half-built pool
    counts = (lead + length) // kp                                        # [n] complete pools this step
    relative = iota(max_pools * kp, k.device) - lead[:, None]             # [n, pools*kpool]
    current = relative.clamp(0, length - 1)
    previous = (contexts[:, None] + relative) % tail_width
    rows = iota(n, k.device)[:, None]
    earlier = (relative < 0)[..., None]                                   # before this step: from the ring
    kw = torch.where(earlier, tails[rows, previous, 0], k[rows, current])
    gw = torch.where(earlier, tails[rows, previous, 1], gate[rows, current])
    pk, ps = net.lanes.kpool_compress(kw.view(n * max_pools, kp, d),
                                      gw.view(n * max_pools, kp, d), net.p[f"L{layer}.idx.ape"])
    # Padded pids at the final context boundary are not read by scatter_rows.
    pids = (contexts[:, None] // kp + iota(max_pools, k.device)).clamp_max(caches.candidate_capacity - 1)
    slots = caches.pool_rows(layer, pids).long()                          # [n, pools]
    keys, scales = caches.pool_keys(layer).view(torch.uint8), caches.pool_scales(layer).unsqueeze(-1)
    pk8 = pk.view(torch.uint8).view(n, max_pools, -1)
    ps1 = ps.view(n, max_pools, 1)
    scatter_rows(pk8, keys, slots, counts.contiguous())
    scatter_rows(ps1, scales, slots, counts.contiguous())
    caches.write_tails(layer, contexts, k, gate)
    return caches.candidate_capacity


@dataclass
class DeviceStep:
    ids: torch.Tensor
    contexts: torch.Tensor
    tokens: int
    captured = True
    patches = ()                  # image embeddings are installed during eager prefill
    marks = ()                    # prefix checkpoints belong to prefill boundaries

    def __post_init__(self):
        # Built once. Every layer asks for this tuple -- one loop in KDA, two in sparse
        # MLA, so 45 layers ask 75 times per step -- and rebuilding it costs two view ops
        # per segment per ask. The views stay correct because replay overwrites `contexts`
        # in place rather than rebinding it, which is the same reason the captured graph
        # can record their addresses.
        self._segments = tuple(
            Segment(i, i, self.contexts[i:i+1].reshape(()), i * self.tokens, self.tokens)
            for i in range(self.contexts.numel()))

    @property
    def segments(self):
        return self._segments

    @property
    def positions(self):
        return (self.contexts[:, None] + torch.arange(self.tokens, device=self.ids.device)).flatten()

    def subset(self, start, end):
        return DeviceStep(self.ids[start * self.tokens:end * self.tokens], self.contexts[start:end], self.tokens)


class GraphCaches:
    def __init__(self, real, sequence_ids, slots, capacity):
        self.real, self.sequence_ids, self.slots = real, sequence_ids, slots
        self.F, self.layout = real.F, real.layout
        self.capacity = capacity
        self.candidate_capacity = capacity // real.F.kpool

    def gather(self):
        # Unreserved pages are masked out of attention by valid pool counts.
        # Translate them to a readable page so padded gathers stay in bounds.
        self.block_table = self.real.block_table.index_select(0, self.sequence_ids).clamp_min(0)

    def subset(self, start, end):
        child = GraphCaches(self.real, self.sequence_ids[start:end], self.slots[start:end],
                            self.capacity)
        child.block_table = self.block_table[start:end]
        return child

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

    def kda_rings_rows(self, layer):
        """Every segment's rings at once: the conv and recurrent fields whole, and the step's physical slots in
        segment order -- what net._kda hands the row lanes (one launch per kernel for the whole step)."""
        return self.real._fields["conv", layer], self.real._fields["rec", layer], self.slots

    def write_conv(self, layer, slot, context, inputs):
        from engine.kernels.state import write_conv
        write_conv(inputs, self.real._fields["conv", layer], self.slots[slot:slot+1], context)

    def write_rec(self, layer, slot, context, states):
        from engine.kernels.state import write_ring
        write_ring(states, self.real._fields["rec", layer], self.slots[slot:slot+1], context)

    def tail(self, layer, slot):
        return self.real._fields["tail", layer].index_select(0, self.slots[slot:slot+1])[0]

    def tails(self, layer):
        """Every segment's tail ring in one gather; tail(layer, i) is row i of it."""
        return self.real._fields["tail", layer].index_select(0, self.slots)

    def write_tail(self, layer, slot, context, keys, gates):
        from engine.kernels.state import write_ring
        write_ring(torch.stack((keys, gates), dim=1), self.real._fields["tail", layer],
                   self.slots[slot:slot+1], context)

    def write_tails(self, layer, contexts, keys, gates):
        """write_tail for every segment at once: row i of keys/gates [n, t, d] goes to segment i's tail ring
        from contexts[i]. One stack and one ring launch instead of one of each per segment."""
        from engine.kernels.state import write_ring_rows
        write_ring_rows(torch.stack((keys, gates), dim=2), self.real._fields["tail", layer],
                        self.slots, contexts.contiguous())

    def latent(self, layer):
        return self.real.latent(layer)

    def pool_keys(self, layer):
        return self.real.pool_keys(layer)

    def pool_scales(self, layer):
        return self.real.pool_scales(layer)

    def token_slots(self, layer, seq, positions):
        from engine.profiles.glm53.caches import Glm53Caches
        return Glm53Caches.token_slots(self, layer, seq, positions)

    def token_rows(self, layer, positions):
        """token_slots for every segment at once: row i of `positions` [rows, t] is segment i's positions, read
        against row i of the gathered block table -- what token_slots(layer, i, ...) does one row at a time."""
        F, p = self.F, self.layout
        blocks = torch.gather(self.block_table, 1, (positions // F.block).long())
        return (blocks * (p.block_bytes // F.kv_lora)
                + p.token_offsets[layer] // F.kv_lora + positions % F.block).to(blocks.dtype)

    def token_map(self, layer, seq):
        from engine.profiles.glm53.caches import Glm53Caches
        return Glm53Caches.token_map(self, layer, seq)

    def token_maps(self, layer):
        """token_map for every segment at once: the gathered block table whole, row i being what
        token_map(layer, i) returns, with the same scalars."""
        F, p = self.F, self.layout
        return self.block_table, F.block, p.block_bytes // F.kv_lora, p.token_offsets[layer] // F.kv_lora

    def pool_slots(self, layer, seq, pool_ids):
        from engine.profiles.glm53.caches import Glm53Caches
        return Glm53Caches.pool_slots(self, layer, seq, pool_ids)

    def candidate_rows(self, layer, n_cand):
        """pool_slots(layer, i, iota(n_cand)) for every segment i at once, as gather indices: a captured step's
        candidates are the bucket's whole capacity for every row (the selection masks what lies past each
        row's context), so the ids are one kept constant and the block-table read is one gather."""
        ids = iota(n_cand, self.block_table.device)[None, :].expand(self.block_table.shape[0], n_cand)
        return self.pool_rows(layer, ids).long()

    def pool_rows(self, layer, pool_ids):
        """pool_slots for every segment at once: row i of `pool_ids` reads row i of the
        gathered block table, which is what pool_slots(layer, i, ...) does one at a time."""
        F, p = self.F, self.layout
        per, record = F.block // F.kpool, F.idx_dim + 4
        blocks = torch.gather(self.block_table, 1, (pool_ids // per).long())
        return (blocks * (p.block_bytes // record)
                + p.pool_offsets[layer] // record + pool_ids % per).to(blocks.dtype)


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
    def __init__(self, net, caches, max_seqs, tokens, aux_layers=(), memory=None, ceiling=None,
                 detail=False, execution_plan=None, drafter=None):
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
        from engine.profiles.glm53.execution import ExecutionPlan, CudaStreams
        self.execution_plan = execution_plan or ExecutionPlan()
        self.append_child = None
        if self.execution_plan.decode_iterations > 1:
            from engine.kernels.bounded_graph import append_child, build
            build()
            self.append_child = append_child
        self.observations = {}
        self.streams = CudaStreams() if self.execution_plan.overlap else None
        self.observe_stream = torch.cuda.Stream() if self.execution_plan.early_observe else None
        if self.observe_stream is not None and (drafter is None or not drafter.fast_attention or not self.aux_layers):
            raise ValueError("early observation requires the native DFlash2 context projection")
        # The ladder ends where the door stops admitting. `ceiling` is that one served
        # number (adapter.max_context, which serve.py refuses past); unset means the
        # model's trained positions. Capping it is the only lever on the graph count:
        # a bucket is captured whether or not any request will reach it, and each costs
        # a warmup pass plus a capture (their seconds are one memory row, "target/<shape>").
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
                                          dtype=torch.bfloat16)
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
            prepared = None
            def observe(aux):
                nonlocal prepared
                # Both the projection and the existing fused context writer
                # retain their numerical boundaries. No tentative cache write.
                positions = step.positions.view(-1, tokens)
                valid = torch.full_like(step.contexts, tokens)
                ready = torch.cuda.Event()
                ready.record()
                positions.record_stream(self.observe_stream)
                valid.record_stream(self.observe_stream)
                aux.record_stream(self.observe_stream)
                with torch.cuda.stream(self.observe_stream):
                    self.observe_stream.wait_event(ready)
                    context = drafter._project_context(positions, aux, valid, observe=False)
                prepared = (positions, context)
            try:
                hook = observe if self.observe_stream is not None else None
                if self.execution_plan.direct_mhc:
                    from engine.profiles.glm53.direct_mhc import decode_direct
                    result = decode_direct(net, step, scratch, self.aux_layers, hook)
                elif self.streams is not None:
                    from engine.profiles.glm53.execution import decode_overlap
                    result = decode_overlap(net, step, scratch, self.execution_plan, self.streams,
                                            self.aux_layers, hook)
                else:
                    result = net.forward(step, scratch, aux_layers=self.aux_layers, aux_ready=hook)
                h, aux = result if self.aux_layers else (result, None)
                logits.copy_(net.head_local(h))
                if prepared is not None:
                    self.observations[(step.contexts.numel(), tokens, scratch.capacity)] = prepared
                return h, aux, logits
            finally:
                if self.observe_stream is not None:
                    torch.cuda.current_stream().wait_stream(self.observe_stream)
                del scratch.block_table

        try:
            # Largest shape first. The first capture sizes the pool every later one
            # shares, so the small rungs reuse it instead of making it grow -- and on
            # unified memory a pool that grows is pages mapped again. vLLM orders its
            # captures largest-first and says the same reason.
            self.graphs = DecodeGraphs(forward, make_inputs,
                                       [(n, tokens, capacity) for n in range(max_seqs, 0, -1)
                                        for capacity in reversed(self.capacities)],
                                       memory=memory, label="target",
                                       resources=net.lanes.graph_resources, detail=detail,
                                       append_child=self.append_child)
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

    def shape_for(self, n: int, end: int):
        """The graph for `n` rows whose longest context may reach `end` (a step ahead of the host rounds up)."""
        for capacity in self.capacities:
            if end <= capacity:
                return n, self.tokens, capacity
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

    def run_device(self, shape, host_step, ids, contexts, seqs, slots):
        """Replay with every input already on the device (45차 §23 B3: the step ahead of the host reads the previous
        step's commit, not the host's view). `host_step` carries the segments the block tables are prepared from --
        its contexts may lag the device's; the reservation covers the lag."""
        self.caches.prepare(host_step)
        return self.run_inputs(shape, ids, contexts, seqs, slots)

    def run_inputs(self, shape, ids, contexts, seqs, slots):
        """Device-only fill/replay; the caller already published reserved block mappings."""

        def fill(inputs):
            target, seqs_in, slots_in, _, _ = inputs
            n = contexts.numel()
            target.ids.copy_(ids)
            target.contexts.copy_(contexts)
            seqs_in.copy_(seqs[:n])
            slots_in.copy_(slots[:n])

        return self.graphs.run(shape, fill)


class DrafterDecodeGraphs:
    """One proposal graph and the finite accepted-prefix context updates, per row (the synchronous step), and the
    same over every row of a step at once (the pipeline, 45차 §23 GPU 판정 4차): one replay a step instead of one
    a row, the weights read once, the rings never copied."""
    def __init__(self, drafter, caches, memory=None, generator=None, vocab=None, prepared_context=False,
                 append_child=None):
        self.field = caches._fields["draft", -1]
        self.drafter = drafter
        device = caches.device
        rows_max = caches.pool.max_seqs
        aux_width = drafter.F.hidden * len(drafter.aux_layers)

        def rows_masked_inputs(n, t):
            return dict(slots=torch.arange(1, n + 1, device=device, dtype=torch.int64),        # distinct at capture: no two rows one slot
                        positions=torch.zeros(n, t, device=device, dtype=torch.int64),
                        aux=torch.zeros(n * t, aux_width, device=device, dtype=torch.bfloat16),
                        valid=torch.zeros(n, device=device, dtype=torch.int64))

        def rows_masked(inputs):
            drafter.observe_rows(self.field, inputs["slots"], inputs["positions"], inputs["aux"], inputs["valid"])

        def prepared_inputs(n, t):
            inputs = rows_masked_inputs(n, t)
            inputs["context"] = torch.zeros(n, t, drafter.F.layers, 2, drafter.local_kv_heads,
                                            drafter.F.head_dim, device=device, dtype=torch.bfloat16)
            return inputs

        def prepared(inputs):
            drafter.observe_prepared(self.field, inputs["slots"], inputs["positions"], inputs["context"],
                                     inputs["valid"], inputs["aux"])

        def rows_propose_inputs(n, t):
            return dict(anchors=torch.zeros(n, device=device, dtype=torch.int64),
                        positions=torch.zeros(n, device=device, dtype=torch.int64),
                        slots=torch.arange(1, n + 1, device=device, dtype=torch.int64),
                        alive=torch.zeros(n, device=device, dtype=torch.bool))          # capture feeds junk: no row counts

        def rows_propose(inputs):
            return drafter.propose_rows(self.field, inputs["slots"], inputs["anchors"], inputs["positions"], alive=inputs["alive"])

        def rows_sampled_inputs(n, t):
            inputs = rows_propose_inputs(n, t)
            inputs["temps"] = torch.ones(n, device=device, dtype=torch.float32)
            return inputs

        def rows_sampled(inputs):
            return drafter.propose_rows(self.field, inputs["slots"], inputs["anchors"], inputs["positions"],
                                        temps=inputs["temps"], generator=generator, vocab=vocab, alive=inputs["alive"])

        def propose_inputs(n, t):
            return dict(anchor=torch.zeros(1, device=device, dtype=torch.int64),
                        position=torch.zeros((), device=device, dtype=torch.int64),
                        slot=torch.zeros(1, device=device, dtype=torch.int64))

        def propose(inputs):
            ring = ((self.field,inputs["slot"]) if drafter.fast_attention
                    else self.field.index_select(0, inputs["slot"])[0])
            return drafter.propose_tensor(inputs["anchor"], inputs["position"], ring)

        def observe_inputs(n, t):
            return dict(positions=torch.arange(t, device=device, dtype=torch.int64),
                        aux=torch.zeros(t, drafter.F.hidden * len(drafter.aux_layers),
                                        device=device, dtype=torch.bfloat16),
                        slot=torch.zeros(1, device=device, dtype=torch.int64))

        def observe(inputs):
            if drafter.fast_attention:
                drafter.observe((self.field,inputs["slot"]), inputs["positions"], inputs["aux"])
                return
            rings = self.field.index_select(0, inputs["slot"])
            drafter.observe(rings[0], inputs["positions"], inputs["aux"])
            self.field.index_copy_(0, inputs["slot"], rings)

        def masked_inputs(n, t):
            return dict(positions=torch.arange(t, device=device, dtype=torch.int64),
                        aux=torch.zeros(t, drafter.F.hidden * len(drafter.aux_layers),
                                        device=device, dtype=torch.bfloat16),
                        slot=torch.zeros(1, device=device, dtype=torch.int64),
                        valid=torch.zeros((), device=device, dtype=torch.int64))

        def observe_masked(inputs):
            if drafter.fast_attention:
                drafter.observe_masked((self.field,inputs["slot"]), inputs["positions"], inputs["aux"], inputs["valid"])
                return
            rings = self.field.index_select(0, inputs["slot"])
            drafter.observe_masked(rings[0], inputs["positions"], inputs["aux"], inputs["valid"])
            self.field.index_copy_(0, inputs["slot"], rings)

        rows_shapes = [(n, drafter.k + 1) for n in range(1, rows_max + 1)]
        saved = generator.get_state() if generator is not None else None                # capture draws; the engine's stream must not move
        try:
            self.proposals = DecodeGraphs(propose, propose_inputs, [(1, drafter.k + 1)],
                                          memory=memory, label="drafter/propose")
            self.observations = DecodeGraphs(observe, observe_inputs,
                                             [(1, t) for t in range(1, drafter.k + 2)],
                                             memory=memory, label="drafter/observe")
            # the step ahead of the host observes all K+1 positions with a device count of the valid ones (B3)
            self.masked = DecodeGraphs(observe_masked, masked_inputs, [(1, drafter.k + 1)],
                                       memory=memory, label="drafter/observe_masked")
            self.rows_masked = DecodeGraphs(rows_masked, rows_masked_inputs, rows_shapes,
                                            memory=memory, label="drafter/observe_rows", append_child=append_child)
            if prepared_context:
                # Commit/calibration stays captured too. Moving FC into target
                # must not replace the remaining stage with eager dispatch.
                self.rows_prepared = DecodeGraphs(prepared, prepared_inputs, rows_shapes,
                                                   memory=memory, label="drafter/commit_prepared", append_child=append_child)
            self.rows_propose = DecodeGraphs(rows_propose, rows_propose_inputs, rows_shapes,
                                             memory=memory, label="drafter/propose_rows", append_child=append_child)
            if generator is not None and vocab is not None:
                self.rows_sampled = DecodeGraphs(rows_sampled, rows_sampled_inputs, rows_shapes, generators=(generator,),
                                                 memory=memory, label="drafter/propose_rows_sampled")
        except BaseException:
            self.close()
            raise
        finally:
            if saved is not None:
                generator.set_state(saved)
            caches.reset()

    def close(self):
        for name in ("proposals", "observations", "masked", "rows_masked", "rows_prepared", "rows_propose", "rows_sampled"):
            graphs = getattr(self, name, None)
            if graphs is not None:
                graphs.close()
                setattr(self, name, None)

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

    def observe_masked(self, ring, positions, aux, valid):
        """All K+1 positions of a step ahead of the host; `valid` (a device scalar) says how many enter the ring."""
        slot = self.slot(ring)
        def fill(inputs):
            inputs["positions"].copy_(positions)
            inputs["aux"].copy_(aux)
            inputs["slot"].fill_(slot)
            inputs["valid"].copy_(valid)
        self.masked.run((1, positions.numel()), fill)

    def propose_from(self, anchor, position, slot):
        """`propose` with the anchor, the position and the slot as device tensors (45차 §23 B3)."""
        def fill(inputs):
            inputs["anchor"].copy_(anchor.reshape(1))
            inputs["position"].copy_(position.reshape(()))
            inputs["slot"].copy_(slot.reshape(1))
        return self.proposals.run((1, self.drafter.k + 1), fill)

    # -- every row of a step at once ---------------------------------------------------------------------
    def observe_prepared_rows(self, slots, positions, context, valid, aux):
        def fill(inputs):
            for key, value in (("slots", slots), ("positions", positions), ("context", context),
                               ("valid", valid), ("aux", aux)):
                inputs[key].copy_(value)
        self.rows_prepared.run(tuple(positions.shape), fill)

    def observe_rows(self, slots, positions, aux, valid):
        """`observe_masked` for the rows [n]: positions [n, t], aux [n*t, A], valid [n] -- all device tensors."""
        def fill(inputs):
            inputs["slots"].copy_(slots)
            inputs["positions"].copy_(positions)
            inputs["aux"].copy_(aux)
            inputs["valid"].copy_(valid)
        self.rows_masked.run(tuple(positions.shape), fill)

    def propose_rows(self, anchors, positions, slots, alive):
        """Every row's greedy drafts, [n, K]; `alive` [n] marks the real rows (a calibration run sums only those)."""
        def fill(inputs):
            inputs["anchors"].copy_(anchors)
            inputs["positions"].copy_(positions)
            inputs["slots"].copy_(slots)
            inputs["alive"].copy_(alive)
        return self.rows_propose.run((anchors.numel(), self.drafter.k + 1), fill)

    def propose_rows_sampled(self, anchors, positions, slots, temps, alive):
        """Every row's drafts drawn at its temperature (0 = greedy) and the distribution they came from, as the
        candidates and their mass: [n, K], [n, K, sel_top_k], [n, K, sel_top_k]."""
        def fill(inputs):
            inputs["anchors"].copy_(anchors)
            inputs["positions"].copy_(positions)
            inputs["slots"].copy_(slots)
            inputs["temps"].copy_(temps)
            inputs["alive"].copy_(alive)
        return self.rows_sampled.run((anchors.numel(), self.drafter.k + 1), fill)


class SamplingGraphs:
    """Greedy and stochastic sampling bind the target graphs' output buffers.

    Greedy replay draws no random numbers; mixed/stochastic replay advances the
    engine's explicit generator exactly as the eager sampler does.

    Temperature, top-k and top-p all arrive as per-row arrays staged from pinned
    memory, so the captured program has no nucleus branch to be recorded with or
    without -- one capture serves every truncation a request can ask for. That is
    why a plain `temperature + top_p` row no longer needs the rich sampler
    (base/sampler.needs_rich_sampler).
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
        # Same pinned staging as the target's replay path, for the three arrays this one writes.
        width = max(n * t for n, t in shapes)
        self.default_p = top_p
        self.policy = (torch.empty(width, dtype=torch.float32, pin_memory=True),
                       torch.empty(width, dtype=torch.int32, pin_memory=True),
                       torch.empty(width, dtype=torch.float32, pin_memory=True))
        self.staged = [x.numpy() for x in self.policy]
        saved = generator.get_state()

        def make_inputs(*shape):
            n, t = shape[:2]
            logits = target.graphs.outputs[first[(n, t)]][2]
            # Capture records the target outputs but need not initialize them.
            # Sampling warmup must see finite logits before the first request.
            logits.zero_()
            return (logits, torch.ones(n*t, device=logits.device),
                    torch.zeros(n*t, dtype=torch.int32, device=logits.device),
                    torch.full((n*t,), top_p, device=logits.device))

        def greedy(inputs):
            return argmax(inputs[0], target.net.comm, target.net.rank * target.net.vp, decodable)

        def stochastic(inputs):
            local_logits, temps, k, p = inputs
            # The undecodable tail is a `valid` width the sampler stops at, not a copy of the
            # whole gathered block with minus infinity written into its end.
            return sample(target.net.comm.all_gather(local_logits, dim=-1), temps, p, generator,
                          top_k=k, valid=decodable)

        try:
            memory = getattr(target, "memory", None)
            self.greedy = DecodeGraphs(greedy, make_inputs, shapes, memory=memory, label="sampling/greedy",
                                        append_child=getattr(target, "append_child", None))
            self.stochastic = DecodeGraphs(stochastic, make_inputs, shapes, generators=(generator,),
                                          memory=memory, label="sampling/stochastic")
        except BaseException:
            if hasattr(self, "greedy"):
                self.greedy.close()
            raise
        finally:
            generator.set_state(saved)

    def run(self, shape, temperatures, top_k=None, top_p=None):
        """`shape` is the target's, whose first two entries name the sampler."""
        shape = tuple(shape[:2])
        if all(t <= 0 for t in temperatures):
            return self.greedy.run(shape, lambda inputs: None)
        rows = len(temperatures)
        self.staged[0][:rows] = temperatures
        self.staged[1][:rows] = 0 if top_k is None else top_k
        self.staged[2][:rows] = self.default_p if top_p is None else top_p

        def fill(inputs):
            for held, static in zip(self.policy, inputs[1:]):
                static.copy_(held[:rows], non_blocking=True)
        return self.stochastic.run(shape, fill)

    def close(self):
        self.greedy.close()
        self.stochastic.close()
