"""Qwen3.8-Flash-Next's captured decode (profile): the target's decode or verify step and the MTP head's draft step, one
CUDA graph per (rows, tokens a row, context bucket), replayed after the step's ids and per-row values are written into
the graph's static inputs (base/graphs.DecodeGraphs; I1: no Python on the decode hot path).

    TargetGraphs   net.forward over a DeviceStep -> (logits [rows*t, vocab] gathered, streams [rows*t, hc*H])
    DraftGraphs    net.mtp_forward over the target's next tokens and its streams -> each row's draft [rows] int64, the
                   head's greedy pick at the row's last kept position

A bucket is a page-table width in blocks: a step replays the smallest bucket that covers its longest row, and the
ladder doubles from 4,096 tokens to the served ceiling plus the verify width, bounded by the pool. Rows are 1..max_seqs.
A row shorter than the graph's width is padded with its last token: the padded positions are written like a rejected
draft's and overwritten by the row's next step, so a padded row must hold the reservation of the full width (checked,
and the block table published for it). Captures run largest first -- the first sizes the pool the rest share -- with
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

    def __init__(self, net, caches, max_seqs: int, tokens: int, ceiling: int):
        F = net.F
        if max_seqs <= 0 or not 1 <= tokens <= F.spec_k + 1:
            raise ValueError(f"captured rows are 1..max_seqs of 1..{F.spec_k + 1} tokens")
        if any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("capture requires no live state slots")
        if not getattr(net.comm, "graph_capture_safe", True):
            # base/comm.LocalTP crosses ranks through a host barrier: no replay performs it (D3)
            raise ValueError(f"{type(net.comm).__name__} collectives cannot be captured")
        self.net, self.caches, self.F = net, caches, F
        self.max_seqs, self.tokens = max_seqs, tokens
        self.buckets = bucket_ladder(F.block, caches.block_table.shape[1], ceiling, tokens)
        self.shapes = [(n, tokens, b) for n in range(max_seqs, 0, -1) for b in reversed(self.buckets)]

    def shape(self, rows: int, end: int) -> "tuple[int, int, int]":
        if not 1 <= rows <= self.max_seqs:
            raise ValueError(f"no captured graph for {rows} rows (1..{self.max_seqs})")
        for blocks in self.buckets:
            if end <= blocks * self.F.block:
                return rows, self.tokens, blocks
        raise ValueError(f"a decode row reaches position {end}, past the captured buckets")

    def publish(self, rows) -> None:
        """rows: (seq, slot, ctx) a row. Each row's full width must be reserved; its mappings are uploaded."""
        pool, t = self.caches.pool, self.tokens
        for seq, _slot, ctx in rows:
            if ctx + t > pool.tokens[seq]:
                raise ValueError(f"seq {seq}: a {t}-token row at {ctx} passes its reservation of {pool.tokens[seq]}")
        self.caches.prepare(SimpleNamespace(segments=[Segment(seq, slot, ctx, 0, t) for seq, slot, ctx in rows]))

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

    def run(self, step) -> "tuple[torch.Tensor, torch.Tensor, list | None]":
        """Replay for a served step (net.Step with state slots): (logits, streams, rows) -- the graph's own outputs,
        [rows*t, ...], and `rows` the flat output row of each of the step's tokens, or None when they are the same."""
        segments, t = step.segments, self.tokens
        n = len(segments)
        shape = self.shape(n, max(s.ctx + t for s in segments))
        self.publish([(s.seq, s.slot, s.ctx) for s in segments])
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

        def fill(inputs):
            inputs.ids.copy_(ids)
            self.metadata[shape].copy_(host, non_blocking=True)

        logits, streams = self.graphs.run(shape, fill)
        rows = [i * t + j for i, s in enumerate(segments) for j in range(s.length)] if padded else None
        return logits, streams, rows


class DraftGraphs(_Rows):
    def __init__(self, net, caches, max_seqs: int, tokens: int, *, ceiling: int, memory=None, detail: bool = False):
        super().__init__(net, caches, max_seqs, tokens, ceiling)
        F, dev = net.F, caches.device
        width = F.hc * F.hidden
        self._meta_host = torch.empty(4 * max_seqs, dtype=torch.int64, pin_memory=True)
        self._ids_host = torch.empty(max_seqs * tokens, dtype=torch.int64, pin_memory=True)
        self._meta, self._ids = self._meta_host.numpy(), self._ids_host.numpy()
        self.metadata = {}

        def make_inputs(n, t, blocks):
            meta = self.metadata[n, t, blocks] = torch.empty(4, n, dtype=torch.int64, device=dev)
            contexts, seqs, slots, last = meta.unbind(0)
            contexts.zero_()
            torch.arange(n, out=seqs)
            torch.add(seqs, 1, out=slots)
            torch.add(seqs * t, t - 1, out=last)
            given = torch.zeros(n * t, width, dtype=torch.bfloat16, device=dev)
            return DeviceStep(torch.zeros(n * t, dtype=torch.int64, device=dev), contexts, slots, seqs, t, blocks), \
                given, last

        def forward(inputs):
            step, given, last = inputs
            hidden, _ = net.mtp_forward(step, given, caches, last_hidden_only=False)
            return net.head_tokens(hidden.index_select(0, last))

        try:
            self.graphs = DecodeGraphs(forward, make_inputs, self.shapes, memory=memory, label="draft",
                                       resources=net.lanes.graph_resources, detail=detail)
        finally:
            caches.reset()

    def run(self, rows) -> "list[int]":
        """rows: (seq, slot, ctx, next ids [m], given streams [m, hc*H]) with 1 <= m <= tokens -> each row's draft."""
        t = self.tokens
        n = len(rows)
        shape = self.shape(n, max(ctx + t for _, _, ctx, _, _ in rows))
        self.publish([(seq, slot, ctx) for seq, slot, ctx, _, _ in rows])
        meta, ids = self._meta, self._ids
        given = []
        for i, (seq, slot, ctx, next_ids, streams) in enumerate(rows):
            m = len(next_ids)
            if not 1 <= m <= t or streams.shape[0] != m:
                raise ValueError(f"a draft row observes 1..{t} positions with their streams")
            meta[i], meta[n + i], meta[2 * n + i], meta[3 * n + i] = ctx, seq, slot, i * t + m - 1
            for j in range(t):
                ids[i * t + j] = next_ids[min(j, m - 1)]
            given.append(streams if m == t else torch.cat([streams, streams[-1:].expand(t - m, -1)]))
        given = torch.cat(given)
        host_meta, host_ids = self._meta_host[:4 * n].view(4, n), self._ids_host[:n * t]

        def fill(inputs):
            step, given_in, _ = inputs
            step.ids.copy_(host_ids, non_blocking=True)
            self.metadata[shape].copy_(host_meta, non_blocking=True)
            given_in.copy_(given)

        return self.graphs.run(shape, fill).tolist()


__all__ = ["bucket_ladder", "TargetGraphs", "DraftGraphs"]
