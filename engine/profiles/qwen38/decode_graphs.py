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
    pictures = None                 # net.pictures when the net serves pictures (__init__); a text-only graph has none
    _meta_rows = 0                  # the metadata rows past the fixed ones: 1 for the rows' mRoPE deltas

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
        # a net that serves pictures turns a sequence whose prompt held one at its mRoPE positions: its rows carry the
        # sequence's delta (net.pictures, the adapter's; a sequence absent there is 0) in one more metadata row. A
        # text-only net's graphs have no such row -- the step they capture is the one they always captured.
        self.pictures = net.pictures if getattr(net, "serves_pictures", False) else None
        self._meta_rows = 0 if self.pictures is None else 1
        self.buckets = bucket_ladder(F.block, caches.block_table.shape[1], ceiling, reach)
        self.shapes = [(n, tokens, b) for n in range(max_seqs, 0, -1) for b in reversed(self.buckets)]

    def delta(self, seq: int) -> int:
        """A row's mRoPE delta: its prompt's (net.pictures), 0 for a text-only sequence."""
        layout = self.pictures.get(seq)
        return 0 if layout is None else int(layout[1])

    def shape(self, rows: int, end: int, tokens: "int | None" = None) -> "tuple[int, int, int]":
        if not 1 <= rows <= self.max_seqs:
            raise ValueError(f"no captured graph for {rows} rows (1..{self.max_seqs})")
        for blocks in self.buckets:
            if end <= blocks * self.F.block:
                return rows, self.tokens if tokens is None else tokens, blocks
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
    """`narrow_rows`: a step of that many rows or fewer whose longest row is shorter than the verify width replays a
    graph as wide as that row (every width 1..tokens is captured for those row counts) -- a drafter that proposes
    fewer than k drafts (adapter.ServedMTP `threshold`) then pays the verify step it asks for, not k+1 positions a row.
    0 (the default): one width, as before."""

    def __init__(self, net, caches, max_seqs: int, tokens: int, *, ceiling: int, memory=None, detail: bool = False,
                 narrow_rows: int = 0):
        super().__init__(net, caches, max_seqs, tokens, ceiling)
        if not 0 <= narrow_rows <= max_seqs:
            raise ValueError(f"narrow widths serve 0..{max_seqs} rows, not {narrow_rows}")
        self.narrow_rows = narrow_rows
        self.shapes = [(n, t, b) for n in range(max_seqs, 0, -1) for t in self.widths(n) for b in reversed(self.buckets)]
        dev = caches.device
        rows = 3 + self._meta_rows
        self._meta_host = torch.empty(rows * max_seqs, dtype=torch.int64, pin_memory=True)
        self._meta = self._meta_host.numpy()
        self.metadata = {}

        def make_inputs(n, t, blocks):
            meta = self.metadata[n, t, blocks] = torch.empty(rows, n, dtype=torch.int64, device=dev)
            contexts, seqs, slots, *deltas = meta.unbind(0)
            contexts.zero_()
            torch.arange(n, out=seqs)
            torch.add(seqs, 1, out=slots)
            for d in deltas:
                d.zero_()
            return DeviceStep(torch.zeros(n * t, dtype=torch.int64, device=dev), contexts, slots, seqs, t, blocks,
                              deltas[0] if deltas else None)

        def forward(step):
            hidden, streams = net.forward(step, caches, streams=True)
            return net.head(hidden), streams

        try:
            self.graphs = DecodeGraphs(forward, make_inputs, self.shapes, memory=memory, label="target",
                                       resources=net.lanes.graph_resources, detail=detail)
        finally:
            caches.reset()                          # warmups and captures wrote real caches before any request

    def widths(self, rows: int) -> "list[int]":
        """The widths captured for a step of `rows` rows, widest first."""
        return list(range(self.tokens, 0, -1)) if rows <= self.narrow_rows else [self.tokens]

    def width(self, rows: int, longest: int) -> int:
        """The narrowest captured width a step of `rows` rows whose longest row is `longest` tokens fits."""
        return min(t for t in self.widths(rows) if t >= longest)

    def admits(self, step, pool) -> bool:
        """A step the graphs serve: 1..max_seqs rows of 1..tokens tokens, each reserved for the width it replays."""
        n = len(step.segments)
        if not 1 <= n <= self.max_seqs or not all(1 <= s.length <= self.tokens for s in step.segments):
            return False
        t = self.width(n, max(s.length for s in step.segments))
        return all(s.ctx + t <= pool.tokens[s.seq] for s in step.segments)

    def run(self, step, known=None) -> "tuple[torch.Tensor, torch.Tensor, list | None, int]":
        """Replay for a served step (net.Step with state slots): (logits, streams, rows, t) -- the graph's own outputs,
        [rows*t, ...], `rows` the flat output row of each of the step's tokens (None when they are the same) and `t`
        the width replayed (`width`). `known`: (the step's ids as the host holds them, each row's ngram_size - 1 tokens
        before its context, DEAD before the sequence) -- what the PLE staging hashes; without it both are read back off
        the device."""
        segments = step.segments
        n = len(segments)
        t = self.width(n, max(s.length for s in segments))
        shape = self.shape(n, max(s.ctx + t for s in segments), t)
        self.publish([(s.seq, s.slot, s.ctx, t) for s in segments])
        meta = self._meta
        for i, s in enumerate(segments):
            meta[i], meta[n + i], meta[2 * n + i] = s.ctx, s.seq, s.slot
            if self.pictures is not None:
                meta[3 * n + i] = self.delta(s.seq)
        padded = any(s.length != t for s in segments)
        if padded:
            index = [s.start + min(j, s.length - 1) for s in segments for j in range(t)]
            ids = step.ids.index_select(0, torch.tensor(index, device=step.ids.device))
        else:
            ids = step.ids
        host = self._meta_host[:(3 + self._meta_rows) * n].view(3 + self._meta_rows, n)
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
        return logits, streams, rows, t


def draft_chain(net, caches, step, given, last, counts, k: int, *, probability: bool = False, sampled=None):
    """The MTP head's picks for a draft step: the head over the observation step (`given` the target's streams at
    its rows, `last` each row's last observed row, `counts` each row's observed positions), its greedy pick at `last`,
    then k-1 chain steps of one position a row -- the head at the position after the row's last observed one, taking
    the row's pick as its token and the head's own streams there as its state (engine/modules/mtp.MTPDrafter's chain,
    run over every row at once) -- each step's pick: [rows, k] int64. The chain's rows sit in the target's blocks at
    their positions like a padded row's: provisional, overwritten by the row's next step. `probability`: (picks, the
    head's probability of each pick [rows, k] fp32) -- net.draft_tokens', the same on every rank. `sampled`:
    (temperature [rows], top_k [rows], top_p [rows], uniforms [rows, k], candidates) -- each pick DRAWN from the head's
    distribution under the row's sampler (net.draft_sample, for block verification) and the chain continuing from the
    drawn token -> (picks, their probabilities, the candidates [rows, k, C], the distributions over them [rows, k, C])."""
    cands, dists = [], []

    def pick(hidden, depth):
        if sampled is not None:
            temperature, top_k, top_p, uniforms, candidates = sampled
            got, p, cand, dist = net.draft_sample(hidden, temperature, top_k, top_p, uniforms[:, depth],
                                                  candidates=candidates)
            cands.append(cand)
            dists.append(dist)
            return got, p
        return net.draft_tokens(hidden, probability=True) if probability else (net.draft_tokens(hidden), None)

    # past its attention the observation runs each row's last observed position only (net.mtp_forward `rows`)
    hidden, given = net.mtp_forward(step, given, caches, last_hidden_only=False, rows=last)
    first, p = pick(hidden, 0)
    picks, probs = [first], [p]
    contexts = step.contexts + counts
    for depth in range(1, k):
        chain = DeviceStep(picks[-1], contexts, step.slots, step.seqs, 1, step.blocks, getattr(step, "deltas", None))
        hidden, streams = net.mtp_forward(chain, given, caches, last_hidden_only=False)
        got, p = pick(hidden, depth)
        picks.append(got)
        probs.append(p)
        given, contexts = streams, contexts + 1
    if sampled is not None:
        return (torch.stack(picks, dim=1), torch.stack(probs, dim=1), torch.stack(cands, dim=1),
                torch.stack(dists, dim=1))
    if probability:
        return torch.stack(picks, dim=1), torch.stack(probs, dim=1)
    return torch.stack(picks, dim=1)


class DraftGraphs(_Rows):
    """The head's draft step: each row's observed positions padded to the verify width `tokens` (k+1) in one launch,
    then the chain of k-1 single-position launches, in one replay (draft_chain). A row reaches tokens+k-1 positions
    from its context at most (`extent`), and `reach` sizes the buckets for it. `candidates` > 0: the sampled chain
    (draft_chain `sampled`) -- each row's temperature, top-k, top-p and its k DRAFT uniforms are inputs of the replay
    (a greedy row passes temperature 0 and draws the argmax), and `run` also returns each row's candidates and the
    distribution over them."""

    def __init__(self, net, caches, max_seqs: int, tokens: int, *, k: int, ceiling: int, memory=None,
                 detail: bool = False, probability: bool = False, candidates: int = 0):
        if not 1 <= k < tokens:
            raise ValueError("the draft graphs chain k >= 1 picks after an observation of up to k+1 positions")
        if candidates < 0:
            raise ValueError(f"a sampled draft reads a positive number of candidates, not {candidates}")
        super().__init__(net, caches, max_seqs, tokens, ceiling, reach=tokens + k - 1)
        self.k, self.probability, self.candidates = k, probability or candidates > 0, candidates
        F, dev = net.F, caches.device
        width = F.hc * F.hidden
        rows = 5 + self._meta_rows
        self._meta_host = torch.empty(rows * max_seqs, dtype=torch.int64, pin_memory=True)
        self._ids_host = torch.empty(max_seqs * tokens, dtype=torch.int64, pin_memory=True)
        self._meta, self._ids = self._meta_host.numpy(), self._ids_host.numpy()
        self.metadata = {}

        def make_inputs(n, t, blocks):
            meta = self.metadata[n, t, blocks] = torch.empty(rows, n, dtype=torch.int64, device=dev)
            contexts, seqs, slots, last, counts, *deltas = meta.unbind(0)
            contexts.zero_()
            torch.arange(n, out=seqs)
            torch.add(seqs, 1, out=slots)
            torch.add(seqs * t, t - 1, out=last)
            counts.fill_(t)
            for d in deltas:
                d.zero_()
            given = torch.zeros(n * t, width, dtype=torch.bfloat16, device=dev)
            sampler = None
            if candidates:
                sampler = (torch.zeros(n, dtype=torch.float32, device=dev), torch.zeros(n, dtype=torch.int32, device=dev),
                           torch.ones(n, dtype=torch.float32, device=dev), torch.zeros(n, k, dtype=torch.float32, device=dev))
            return DeviceStep(torch.zeros(n * t, dtype=torch.int64, device=dev), contexts, slots, seqs, t, blocks,
                              deltas[0] if deltas else None), given, last, counts, sampler

        def forward(inputs):
            step, given, last, counts, sampler = inputs
            return draft_chain(net, caches, step, given, last, counts, k, probability=probability,
                               sampled=None if sampler is None else (*sampler, candidates))

        try:
            self.graphs = DecodeGraphs(forward, make_inputs, self.shapes, memory=memory, label="draft",
                                       resources=net.lanes.graph_resources, detail=detail)
        finally:
            caches.reset()

    def extent(self, observed: int) -> int:
        """Positions a row of `observed` tokens writes from its context: the verify width (the observation, padded),
        then the chain's k-1 positions after the observed ones."""
        return max(self.tokens, observed + self.k - 1)

    def run(self, rows, sampling=None):
        """rows: (seq, slot, ctx, next ids [m], given streams [m, hc*H]) with 1 <= m <= tokens -> each row's k drafts,
        and with `probability` (drafts, each draft's probability under the head) -- lists a row. With `candidates`:
        `sampling` a row's (temperature, top_k, top_p, its k DRAFT uniforms) or None (the argmax), and the answer
        (drafts, probabilities, candidates [n, k, C], distributions [n, k, C]) -- the last two copied off the replay's
        outputs, which the next replay overwrites."""
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
            if self.pictures is not None:
                meta[5 * n + i] = self.delta(seq)
            for j in range(t):
                ids[i * t + j] = next_ids[min(j, m - 1)]
            given.append(streams if m == t else torch.cat([streams, streams[-1:].expand(t - m, -1)]))
        given = torch.cat(given)
        host_meta, host_ids = self._meta_host[:(5 + self._meta_rows) * n].view(5 + self._meta_rows, n), self._ids_host[:n * t]

        settings = None
        if self.candidates:
            sampling = [None] * n if sampling is None else list(sampling)
            if len(sampling) != n:
                raise ValueError(f"{n} draft rows need {n} sampling settings, not {len(sampling)}")
            greedy = (0.0, 0, 1.0, [0.0] * self.k)
            chosen = [greedy if s is None else s for s in sampling]
            for s in chosen:
                if len(s[3]) != self.k:
                    raise ValueError(f"a sampled draft row draws {self.k} uniforms, not {len(s[3])}")
            settings = (torch.tensor([float(s[0]) for s in chosen], dtype=torch.float32),
                        torch.tensor([int(s[1]) for s in chosen], dtype=torch.int32),
                        torch.tensor([float(s[2]) for s in chosen], dtype=torch.float32),
                        torch.tensor([[float(u) for u in s[3]] for s in chosen], dtype=torch.float32))

        def fill(inputs):
            step, given_in, _last, _counts, sampler = inputs
            step.ids.copy_(host_ids, non_blocking=True)
            self.metadata[shape].copy_(host_meta, non_blocking=True)
            given_in.copy_(given)
            if sampler is not None:
                for into, value in zip(sampler, settings):
                    into.copy_(value)

        out = self.graphs.run(shape, fill)
        if self.candidates:
            picks, probs, cand, dist = out
            return picks.tolist(), probs.tolist(), cand[:n].clone(), dist[:n].clone()
        if self.probability:
            picks, probs = out
            return picks.tolist(), probs.tolist()
        return out.tolist()


__all__ = ["bucket_ladder", "draft_chain", "TargetGraphs", "DraftGraphs"]
