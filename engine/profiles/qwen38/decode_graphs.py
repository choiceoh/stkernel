"""Qwen3.8-Flash-Next's captured decode (profile): the target's decode or verify step and the MTP head's draft step, one
CUDA graph per (rows, tokens a row, context bucket), replayed after the step's ids and per-row values are written into
the graph's static inputs (base/graphs.DecodeGraphs; I1: no Python on the decode hot path).

    TargetGraphs   net.forward over a DeviceStep -> (logits [rows*t, vocab] gathered, streams [rows*t, hc*H])
    DraftGraphs    net.mtp_forward over the target's next tokens and its streams -> each row's drafts [rows, K] int64:
                   the head's greedy pick at the row's last kept position, then the chain (draft_chain): K-1 launches of
                   one position a row in the same replay, each taking the row's last pick as its token and the head's
                   own streams there as its state

A bucket is a page-table width in blocks: a step replays the smallest bucket that covers its longest row, and the
ladder doubles from 4,096 tokens to the served ceiling plus the positions a row writes (the verify width; for the draft
graphs the chain's reach past it), bounded by the pool. Rows are 1..max_seqs. A row shorter than the graph's width is
padded with its last token: the padded positions are written like a rejected draft's and overwritten by the row's next
step, so a padded row must hold the reservation of the full width (checked, and the block table published for it); a
draft row's chain positions, after its observed ones, are provisional the same way. Captures run largest first -- the first sizes the pool the rest share -- with
no request live: warmups write the caches at slots 1..rows and every cache is reset afterwards.

The two instances own separate memory pools (base/graphs' second rule). The target's outputs are the graph's own
tensors: the caller consumes the logits before the next replay, and the drafter keeps its own copy of the streams rows it
defers (adapter.ServedMTP), so no replay of either graph can overwrite what the other still reads.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from engine.base.graphs import DecodeGraphs
from engine.profiles.qwen38.net import FIRST_BUCKET, DeviceStep, Segment


def bucket_ladder(block: int, pool_blocks: int, ceiling: int, tokens: int) -> "list[int]":
    """Page-table widths in blocks to capture, smallest first -- net.bucket_blocks' rungs, which prefill addresses too,
    so both compile the paged kernels at the same widths. The last row a door admits starts at ceiling-1 and a padded
    step writes `tokens` positions from there, so the top bucket covers ceiling-1+tokens positions -- and no more than
    the pool holds. Every bucket above the ceiling would be a graph for a request that cannot arrive."""
    if block <= 0 or pool_blocks <= 0 or ceiling <= 0 or tokens <= 0:
        raise ValueError("block, pool, ceiling and width must be positive")
    top = min(pool_blocks, -(-(ceiling - 1 + tokens) // block))
    rungs, reach = [], FIRST_BUCKET
    while not rungs or rungs[-1] < top:
        blocks = min(top, -(-reach // block))
        if not rungs or blocks > rungs[-1]:
            rungs.append(blocks)
        reach *= 2
    return rungs


class _Rows:
    """The graphs' shared admission: rows, width, buckets, and the reservation and block-table publication of the
    padded positions."""

    def __init__(self, net, caches, max_seqs: int, tokens: int, ceiling: int, reach: "int | None" = None):
        F = net.F
        reach = tokens if reach is None else reach
        if max_seqs <= 0 or not 1 <= tokens <= F.spec_k + 1 or reach < tokens:
            raise ValueError(f"captured rows are 1..max_seqs of 1..{F.spec_k + 1} tokens, reaching at least as far")
        if any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("capture requires no live state slots")
        if not getattr(net.comm, "graph_capture_safe", True):
            # base/comm.LocalTP crosses ranks through a host barrier: no replay performs it (D3)
            raise ValueError(f"{type(net.comm).__name__} collectives cannot be captured")
        self.net, self.caches, self.F = net, caches, F
        self.max_seqs, self.tokens, self.reach = max_seqs, tokens, reach
        self.buckets = bucket_ladder(F.block, caches.block_table.shape[1], ceiling, reach)
        self.shapes = [(n, tokens, b) for n in range(max_seqs, 0, -1) for b in reversed(self.buckets)]

    def shape(self, rows: int, end: int) -> "tuple[int, int, int]":
        if not 1 <= rows <= self.max_seqs:
            raise ValueError(f"no captured graph for {rows} rows (1..{self.max_seqs})")
        for blocks in self.buckets:
            if end <= blocks * self.F.block:
                return rows, self.tokens, blocks
        raise ValueError(f"a decode row reaches position {end}, past the captured buckets")

    def publish(self, rows) -> None:
        """rows: (seq, slot, ctx, width) a row -- the positions it writes from ctx, within the graphs' reach. Each
        must be reserved; the row's mappings are uploaded."""
        pool = self.caches.pool
        for seq, _slot, ctx, width in rows:
            if not 1 <= width <= self.reach:
                raise ValueError(f"seq {seq}: a row writes 1..{self.reach} positions, not {width}")
            if ctx + width > pool.tokens[seq]:
                raise ValueError(f"seq {seq}: a {width}-position row at {ctx} passes its reservation of {pool.tokens[seq]}")
        self.caches.prepare(SimpleNamespace(segments=[Segment(seq, slot, ctx, 0, width)
                                                      for seq, slot, ctx, width in rows]))

    def close(self) -> None:
        self.graphs.close()


class TargetGraphs(_Rows):
    def __init__(self, net, caches, max_seqs: int, tokens: int, *, ceiling: int, memory=None, detail: bool = False):
        super().__init__(net, caches, max_seqs, tokens, ceiling)
        dev = caches.device
        self._meta_host = torch.empty(3 * max_seqs, dtype=torch.int64, pin_memory=True)
        self._meta = self._meta_host.numpy()
        self.metadata = {}

        def make_inputs(n, t, blocks):
            meta = self.metadata[n, t, blocks] = torch.empty(3, n, dtype=torch.int64, device=dev)
            contexts, seqs, slots = meta.unbind(0)
            contexts.zero_()
            torch.arange(n, out=seqs)
            torch.add(seqs, 1, out=slots)
            return DeviceStep(torch.zeros(n * t, dtype=torch.int64, device=dev), contexts, slots, seqs, t, blocks)

        def forward(step):
            hidden, streams = net.forward(step, caches, streams=True)
            return net.head(hidden), streams

        try:
            self.graphs = DecodeGraphs(forward, make_inputs, self.shapes, memory=memory, label="target",
                                       resources=net.lanes.graph_resources, detail=detail)
        finally:
            caches.reset()                          # warmups and captures wrote real caches before any request

    def admits(self, step, pool) -> bool:
        """A step the graphs serve: 1..max_seqs rows of 1..tokens tokens, each reserved for the full width."""
        return (len(step.segments) <= self.max_seqs
                and all(1 <= s.length <= self.tokens and s.ctx + self.tokens <= pool.tokens[s.seq] for s in step.segments))

    def run(self, step, known=None) -> "tuple[torch.Tensor, torch.Tensor, list | None]":
        """Replay for a served step (net.Step with state slots): (logits, streams, rows) -- the graph's own outputs,
        [rows*t, ...], and `rows` the flat output row of each of the step's tokens, or None when they are the same.
        `known`: (the step's ids as the host holds them, each row's ngram_size - 1 tokens before its context, DEAD before
        the sequence) -- what the PLE staging hashes; without it both are read back off the device."""
        segments, t = step.segments, self.tokens
        n = len(segments)
        shape = self.shape(n, max(s.ctx + t for s in segments))
        self.publish([(s.seq, s.slot, s.ctx, t) for s in segments])
        meta = self._meta
        for i, s in enumerate(segments):
            meta[i], meta[n + i], meta[2 * n + i] = s.ctx, s.seq, s.slot
        padded = any(s.length != t for s in segments)
        if padded:
            index = [s.start + min(j, s.length - 1) for s in segments for j in range(t)]
            ids = step.ids.index_select(0, torch.tensor(index, device=step.ids.device))
        else:
            ids = step.ids
        host = self._meta_host[:3 * n].view(3, n)
        net = self.net
        if net.ple_stage is not None:
            # the PLE rows of the step's n x t tokens, read off the SSD table on the host before the replay (a replay
            # reads no host value): a one-row read of the step's ids, the carried ids from the rings, the hash
            if known is None:
                staged, carried = ids.tolist(), None
            else:
                flat, carried = known
                staged = [flat[i] for i in index] if padded else list(flat)
            net.stage_ple([s.slot for s in segments], [s.ctx for s in segments], staged, t, self.caches, carried=carried)

        def fill(inputs):
            inputs.ids.copy_(ids)
            self.metadata[shape].copy_(host, non_blocking=True)
            if net.ple_stage is not None:
                net.ple_stage.upload()

        logits, streams = self.graphs.run(shape, fill)
        rows = [i * t + j for i, s in enumerate(segments) for j in range(s.length)] if padded else None
        return logits, streams, rows


def draft_chain(net, caches, step, given, last, counts, k: int) -> torch.Tensor:
    """The MTP head's picks for a draft step: the head over the observation step (`given` the target's streams at
    its rows, `last` each row's last observed row, `counts` each row's observed positions), its greedy pick at `last`,
    then k-1 chain steps of one position a row -- the head at the position after the row's last observed one, taking
    the row's pick as its token and the head's own streams there as its state (engine/modules/mtp.MTPDrafter's chain,
    run over every row at once) -- each step's pick: [rows, k] int64. The chain's rows sit in the target's blocks at
    their positions like a padded row's: provisional, overwritten by the row's next step."""
    hidden, streams = net.mtp_forward(step, given, caches, last_hidden_only=False)
    picks = [net.draft_tokens(hidden.index_select(0, last))]
    given, contexts = streams.index_select(0, last), step.contexts + counts
    for _ in range(1, k):
        chain = DeviceStep(picks[-1], contexts, step.slots, step.seqs, 1, step.blocks)
        hidden, streams = net.mtp_forward(chain, given, caches, last_hidden_only=False)
        picks.append(net.draft_tokens(hidden))
        given, contexts = streams, contexts + 1
    return torch.stack(picks, dim=1)


class DraftGraphs(_Rows):
    """The head's draft step: each row's observed positions padded to the verify width `tokens` (k+1) in one launch,
    then the chain of k-1 single-position launches, in one replay (draft_chain). A row reaches tokens+k-1 positions
    from its context at most (`extent`), and `reach` sizes the buckets for it."""

    def __init__(self, net, caches, max_seqs: int, tokens: int, *, k: int, ceiling: int, memory=None,
                 detail: bool = False):
        if not 1 <= k < tokens:
            raise ValueError("the draft graphs chain k >= 1 picks after an observation of up to k+1 positions")
        super().__init__(net, caches, max_seqs, tokens, ceiling, reach=tokens + k - 1)
        self.k = k
        F, dev = net.F, caches.device
        width = F.hc * F.hidden
        self._meta_host = torch.empty(5 * max_seqs, dtype=torch.int64, pin_memory=True)
        self._ids_host = torch.empty(max_seqs * tokens, dtype=torch.int64, pin_memory=True)
        self._meta, self._ids = self._meta_host.numpy(), self._ids_host.numpy()
        self.metadata = {}

        def make_inputs(n, t, blocks):
            meta = self.metadata[n, t, blocks] = torch.empty(5, n, dtype=torch.int64, device=dev)
            contexts, seqs, slots, last, counts = meta.unbind(0)
            contexts.zero_()
            torch.arange(n, out=seqs)
            torch.add(seqs, 1, out=slots)
            torch.add(seqs * t, t - 1, out=last)
            counts.fill_(t)
            given = torch.zeros(n * t, width, dtype=torch.bfloat16, device=dev)
            return DeviceStep(torch.zeros(n * t, dtype=torch.int64, device=dev), contexts, slots, seqs, t, blocks), \
                given, last, counts

        def forward(inputs):
            step, given, last, counts = inputs
            return draft_chain(net, caches, step, given, last, counts, k)

        try:
            self.graphs = DecodeGraphs(forward, make_inputs, self.shapes, memory=memory, label="draft",
                                       resources=net.lanes.graph_resources, detail=detail)
        finally:
            caches.reset()

    def extent(self, observed: int) -> int:
        """Positions a row of `observed` tokens writes from its context: the verify width (the observation, padded),
        then the chain's k-1 positions after the observed ones."""
        return max(self.tokens, observed + self.k - 1)

    def run(self, rows) -> "list[list[int]]":
        """rows: (seq, slot, ctx, next ids [m], given streams [m, hc*H]) with 1 <= m <= tokens -> each row's k drafts."""
        t = self.tokens
        n = len(rows)
        for _seq, _slot, _ctx, next_ids, streams in rows:
            if not 1 <= len(next_ids) <= t or streams.shape[0] != len(next_ids):
                raise ValueError(f"a draft row observes 1..{t} positions with their streams")
        widths = [self.extent(len(next_ids)) for _, _, _, next_ids, _ in rows]
        shape = self.shape(n, max(ctx + w for (_, _, ctx, _, _), w in zip(rows, widths)))
        self.publish([(seq, slot, ctx, w) for (seq, slot, ctx, _, _), w in zip(rows, widths)])
        meta, ids = self._meta, self._ids
        given = []
        for i, (seq, slot, ctx, next_ids, streams) in enumerate(rows):
            m = len(next_ids)
            meta[i], meta[n + i], meta[2 * n + i], meta[3 * n + i], meta[4 * n + i] = ctx, seq, slot, i * t + m - 1, m
            for j in range(t):
                ids[i * t + j] = next_ids[min(j, m - 1)]
            given.append(streams if m == t else torch.cat([streams, streams[-1:].expand(t - m, -1)]))
        given = torch.cat(given)
        host_meta, host_ids = self._meta_host[:5 * n].view(5, n), self._ids_host[:n * t]

        def fill(inputs):
            step, given_in, _last, _counts = inputs
            step.ids.copy_(host_ids, non_blocking=True)
            self.metadata[shape].copy_(host_meta, non_blocking=True)
            given_in.copy_(given)

        return self.graphs.run(shape, fill).tolist()


__all__ = ["bucket_ladder", "draft_chain", "TargetGraphs", "DraftGraphs"]
