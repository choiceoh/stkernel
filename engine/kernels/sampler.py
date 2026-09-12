"""The whole batch's picks in one kernel: no sort, no multinomial, no crossing to the host.

A nucleus is a value threshold. Sorting a row finds it by putting every token in order and
walking the cumulative mass; that costs O(V log V) and, at a 154,880-token vocabulary, 2.8 ms
for the 24 rows of one speculative decode step. But the threshold is the only thing the sort is
asked for, and a threshold can be searched for directly: `tau` is the largest value whose tail
mass still reaches `top_p`, and the tail mass at a candidate value is one reduction over the row.

The search runs on the BIT PATTERN of w = exp((logit - max)/T), which lies in (0, 1]. Positive
floats are ordered by their bits, so the candidate range is the integer interval [0, 2^30), and
a four-way split resolves two bits a round: fifteen rounds land on the exact threshold -- not a
tolerance, the same token set the sort would have kept. Each round is one streaming pass over
the row, which is why the split is four and not sixteen: the passes are memory-bound at this
size (the whole batch is 15 MB and stays in L2), so extra thresholds a round are nearly free
until the accumulators spill, and four is where that stops paying.

The draw is the inverse CDF over the kept mass with ONE uniform for the row -- not V of them.
vLLM draws V exponentials a row (Gumbel-max) because it composes torch ops and torch has no
sync-free categorical draw; inside a kernel the cumulative walk is already there.

Nothing here reads a host predicate or writes one: temperature, top-k, top-p and the uniform all
arrive as per-row tensors, so one launch serves a mixed batch and the whole thing captures into
a CUDA graph.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))
MIN_KEY = tl.constexpr(-9223372036854775808)
BITS = tl.constexpr(30)        # bits(1.0f) = 0x3F800000 < 2^30: the whole range of w in (0, 1]
ROUNDS = tl.constexpr(15)      # a four-way split resolves two bits a round


@triton.jit
def _tails(lp, N, mx, inv, t1, t2, t3, BLOCK: tl.constexpr, COUNT: tl.constexpr):
    """The row's tail mass (or population) at three bit thresholds, in one streaming pass.

    Three accumulators ride the same load: the loads are what the pass costs, so asking three
    questions of each element instead of one is what turns thirty rounds into fifteen.
    """
    cols = tl.arange(0, BLOCK)
    a1 = tl.zeros([BLOCK], tl.float32)
    a2 = tl.zeros([BLOCK], tl.float32)
    a3 = tl.zeros([BLOCK], tl.float32)
    for off in range(0, N, BLOCK):
        idx = off + cols
        m = idx < N
        v = tl.load(lp + idx, mask=m, other=NEG_INF).to(tl.float32)
        w = tl.exp((v - mx) * inv)                   # (0, 1]; a masked lane is exp(-inf) = 0
        b = w.to(tl.int32, bitcast=True)
        if COUNT:
            val = tl.where(m, 1.0, 0.0)
        else:
            val = tl.where(m, w, 0.0)
        a1 += tl.where(b >= t1, val, 0.0)
        a2 += tl.where(b >= t2, val, 0.0)
        a3 += tl.where(b >= t3, val, 0.0)
    return tl.sum(a1), tl.sum(a2), tl.sum(a3)


@triton.jit
def _tail(lp, N, mx, inv, t, BLOCK: tl.constexpr):
    """The row's tail mass at one bit threshold."""
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, N, BLOCK):
        idx = off + cols
        m = idx < N
        v = tl.load(lp + idx, mask=m, other=NEG_INF).to(tl.float32)
        w = tl.exp((v - mx) * inv)
        acc += tl.where(m & (w.to(tl.int32, bitcast=True) >= t), w, 0.0)
    return tl.sum(acc)


@triton.jit
def _search(lp, N, mx, inv, target, standing, BLOCK: tl.constexpr, COUNT: tl.constexpr):
    """The largest bit threshold whose tail still reaches `target`, and that tail.

    The invariant is the ordinary one of a binary search over the integers: the answer is always
    at or above `lo`, and after the last round the step is one, so `lo` IS the answer. `standing`
    is the tail already known at zero, which is the answer when no threshold above it reaches the
    target -- a row whose whole mass barely clears `top_p` must come back with that mass, not nought.
    """
    lo = 0
    step = 1 << (BITS - 2)
    reached = standing
    for _ in range(ROUNDS):
        t1 = lo + step
        t2 = lo + 2 * step
        t3 = lo + 3 * step
        s1, s2, s3 = _tails(lp, N, mx, inv, t1, t2, t3, BLOCK, COUNT)
        lo = tl.where(s3 >= target, t3, tl.where(s2 >= target, t2, tl.where(s1 >= target, t1, lo)))
        reached = tl.where(s3 >= target, s3, tl.where(s2 >= target, s2, tl.where(s1 >= target, s1, reached)))
        step = step >> 2
    return lo, reached


@triton.jit
def _greedy(lp, N, BLOCK: tl.constexpr):
    """The row's argmax as one int64 max, ties to the smallest id -- modules/vocab.argmax's key."""
    cols = tl.arange(0, BLOCK)
    key = tl.full((), MIN_KEY, tl.int64)
    for off in range(0, N, BLOCK):
        idx = off + cols
        m = idx < N
        v = tl.load(lp + idx, mask=m, other=NEG_INF).to(tl.float32)
        bits = v.to(tl.int32, bitcast=True).to(tl.int64)
        ordered = tl.where(bits < 0, bits ^ 0x7fffffff, bits)
        key = tl.maximum(key, tl.max(tl.where(m, (ordered << 32) | (0xffffffff - idx.to(tl.int64)), MIN_KEY)))
    return 0xffffffff - (key & 0xffffffff)


@triton.jit
def _sampler(LOGITS, TEMP, TOPK, TOPP, UNIFORM, OUT, PROBS, TAU, KEPT,
             sl, sp, N, WIDTH, BLOCK: tl.constexpr, WRITE: tl.constexpr, DRAW: tl.constexpr,
             REPORT: tl.constexpr):
    row = tl.program_id(0)
    lp = LOGITS + row.to(tl.int64) * sl
    temp = tl.load(TEMP + row)
    cols = tl.arange(0, BLOCK)
    pout = PROBS + row.to(tl.int64) * sp

    if temp <= 0.0:
        # A greedy row asks one question of the row and no random number at all; the packed-key
        # pass is the whole of it, and it is kept off the stochastic path because int64 shifts
        # over a 154,880-token row are not free.
        best = _greedy(lp, N, BLOCK)
        if DRAW:
            tl.store(OUT + row, best)
        if WRITE:
            for off in range(0, WIDTH, BLOCK):
                idx = off + cols
                tl.store(pout + idx, tl.where(idx == best, 1.0, 0.0), mask=idx < WIDTH)
        if REPORT:
            tl.store(TAU + row, 0)
            tl.store(KEPT + row, 1.0)
    else:
        # -- one pass for the row's max and its softmax denominator ------------------------------
        # The max has to come first for exp not to overflow, and the denominator wants the max:
        # the online rescale (flash attention's) gets both out of a single read of the row.
        inv = 1.0 / temp
        mx = NEG_INF
        total = 0.0
        for off in range(0, N, BLOCK):
            idx = off + cols
            v = tl.load(lp + idx, mask=idx < N, other=NEG_INF).to(tl.float32)
            alive = (idx < N) & (v > NEG_INF)
            nm = tl.maximum(mx, tl.max(tl.where(alive, v, NEG_INF)))
            total = tl.where(mx > NEG_INF, total * tl.exp((mx - nm) * inv), 0.0) \
                + tl.sum(tl.where(alive, tl.exp((v - nm) * inv), 0.0))
            mx = nm

        # -- the two truncations, each a threshold search over the same bit range ----------------
        tau = 0
        kept = total
        k = tl.load(TOPK + row)
        if (k > 0) & (k < N):
            tau, _ = _search(lp, N, mx, inv, k.to(tl.float32), 0.0, BLOCK, True)
            kept = _tail(lp, N, mx, inv, tau, BLOCK)
        p = tl.load(TOPP + row)
        if p < 1.0:
            # top-p sits on top of top-k: below tau_k the tail already holds the whole kept mass,
            # so searching the full range again can only land at or above it. No second bracket.
            tau, kept = _search(lp, N, mx, inv, p * kept, kept, BLOCK, False)

        # -- the draw: one uniform walked against the kept mass, in id order ----------------------
        # A caller that only wants the distributions (the speculative path picks with them, not
        # from them) passes no uniform, and the cumulative walk is not run at all.
        run = 0.0
        pick = N + 0
        last = 0
        if DRAW:
            aim = tl.load(UNIFORM + row) * kept
        else:
            aim = 0.0
        # nothing past `valid` is written unless the distributions are wanted, and nothing past it
        # can be picked, so the last pass stops at the shorter of the two
        if WRITE:
            stop = WIDTH
        else:
            stop = N
        for off in range(0, stop, BLOCK):
            idx = off + cols
            live = idx < N
            v = tl.load(lp + idx, mask=live, other=NEG_INF).to(tl.float32)
            w = tl.exp((v - mx) * inv)
            keep = live & (w.to(tl.int32, bitcast=True) >= tau)
            wk = tl.where(keep, w, 0.0)
            if DRAW:
                pick = tl.minimum(pick, tl.min(tl.where(keep & (tl.cumsum(wk, axis=0) + run > aim), idx, N)))
                last = tl.maximum(last, tl.max(tl.where(keep, idx, 0)))
                run += tl.sum(wk)
            if WRITE:
                # a row whose every logit is minus infinity has no mass to share out; it is broken
                # upstream, and a NaN distribution is not how this should say so
                tl.store(pout + idx, tl.where(kept > 0.0, wk / kept, 0.0), mask=idx < WIDTH)
        # a uniform of 1 - 1ulp against a mass the search summed in another order can walk off the
        # end of the row; the last kept id is where that walk was headed.
        if DRAW:
            tl.store(OUT + row, tl.where(pick >= N, last, pick).to(tl.int64))
        if REPORT:
            tl.store(TAU + row, tau)
            tl.store(KEPT + row, kept)


def sample_rows(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
                uniform: "torch.Tensor | None", valid: "int | None" = None, probs: "torch.Tensor | None" = None,
                report: bool = False):
    """[M] token ids for M rows of logits, and optionally each row's truncated distribution.

    `logits` [M, V] in any float dtype (bf16 converts exactly to fp32, and reading it narrow is
    half the bytes of every pass). `temperature`, `top_k`, `top_p`, `uniform` are per row; a
    temperature at or below zero is that row's greedy pick and takes no part of the uniform.
    `valid` cuts the vocabulary to the decodable ids. `probs`, when given, is written with the
    row's sampling distribution (zero outside the nucleus, zero past `valid`, one-hot when greedy).
    `uniform` may be None for a caller that wants only `probs`; then no draw happens and the answer
    is None.
    """
    if logits.ndim != 2:
        raise ValueError("the sampler takes one block of rows")
    M, V = logits.shape
    N = V if valid is None else min(V, valid)
    if N <= 0:
        raise ValueError("no decodable token")
    if uniform is None and probs is None:
        raise ValueError("the sampler was asked for neither a pick nor a distribution")
    for name, t in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p),
                    ("uniform", temperature if uniform is None else uniform)):
        if t.shape != (M,):
            raise ValueError(f"{name} must carry one value a row")
    block = max(128, min(4096, triton.next_power_of_2(N)))
    out = torch.empty(M, dtype=torch.int64, device=logits.device)
    tau = torch.empty(M, dtype=torch.int32, device=logits.device) if report else out
    kept = torch.empty(M, dtype=torch.float32, device=logits.device) if report else out
    _sampler[(M,)](logits, temperature, top_k, top_p, temperature if uniform is None else uniform, out,
                   probs if probs is not None else logits, tau, kept,
                   logits.stride(0), probs.stride(0) if probs is not None else 0, N, V,
                   BLOCK=block, WRITE=probs is not None, DRAW=uniform is not None, REPORT=report,
                   num_warps=max(4, block // 256))
    picked = out if uniform is not None else None
    return (picked, tau, kept) if report else picked
