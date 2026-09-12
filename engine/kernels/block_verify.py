"""Block verification (Sun et al.) for a decode batch in one launch.

`base/sampler.block_verify_batch` written as torch ops issues about 180 device operations per call and, at the
shape production runs -- one sequence, K=5, V=154,880 -- spends 2,425 us of wall on 397 us of GPU work
(probes/decode_middle_cost.py, ledger 45차 §68). Most of those tensors hold five numbers; the cost is dispatch,
not arithmetic. This is the same fold kernels/decode_commit.advance did for the commit beside it.

What stays outside, and why:

  the uniforms   drawn by the caller from the engine's generator, in the same order on every rank. Moving the
                 draw in here would put rank agreement inside a kernel launch, and D12's replay with it.
  the draw       `_inverse_cdf` is a cumsum and a searchsorted. torch's cumsum over 154,880 floats is a parallel
                 scan; accumulating sequentially here would differ in the last bits, and near a boundary that
                 changes the token. Two kernels are not worth breaking bit-exactness for.

So this computes everything between: the per-position acceptance thresholds, how many drafts were accepted, and
the normalised residual the correction is drawn from.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

TINY = tl.constexpr(1e-30)   # annotation form is rejected by the JIT; this is the call form it wants


@triton.jit
def _verify(TARGET, DRAFTS, CAND, QPROB, U, ACCEPTED, AT, BEFORE, REST,
            V, sT_n, sT_p, sD, sC_n, sC_k, sU,
            K: tl.constexpr, C: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0)
    target = TARGET + n * sT_n
    ks = tl.arange(0, BK)
    cs = tl.arange(0, BC)
    live_c = cs < C

    # -- the draft's own mass, and the target on it: C numbers a position, no vocabulary pass -----------
    carried = tl.zeros([BK], tl.float32)
    running = 1.0
    for i in tl.static_range(K):
        token = tl.load(DRAFTS + n * sD + i)
        on_draft = tl.load(target + i * sT_p + token)
        cand = tl.load(CAND + n * sC_n + i * sC_k + cs, mask=live_c, other=0)
        q = tl.load(QPROB + n * sC_n + i * sC_k + cs, mask=live_c, other=0.0)
        by_draft = tl.sum(tl.where(cand == token, q, 0.0), axis=0)
        step = tl.where(by_draft > 0, on_draft / tl.maximum(by_draft, TINY), 0.0)
        running = tl.minimum(running * step, 1.0)
        carried = tl.where(ks == i, running, carried)

    # -- one pass over the vocabulary for the K-1 row sums the thresholds need --------------------------
    rows = 1 + ks                                                   # positions 1..K-1
    live_k = ks < K - 1
    tsum = tl.zeros([BK], tl.float32)
    for v0 in range(0, V, BLOCK):
        vs = v0 + tl.arange(0, BLOCK)
        alive = vs < V
        tile = tl.load(target + rows[:, None] * sT_p + vs[None, :],
                       mask=live_k[:, None] & alive[None, :], other=0.0)
        tsum += tl.sum(tile, axis=1)
    # One consumer of a loop result, then use the copy: TritonGPUOptimizeThreadLocality asserts
    # `loopResult.hasOneUse()` and a reduction read twice afterwards trips it.
    sums = tsum + 0.0

    # -- thresholds, and how far the block was accepted ------------------------------------------------
    accepted = 0
    for i in tl.static_range(K):
        a = tl.sum(tl.where(ks == i, carried, 0.0), axis=0)
        if i < K - 1:
            cand = tl.load(CAND + n * sC_n + (i + 1) * sC_k + cs, mask=live_c, other=0)
            q = tl.load(QPROB + n * sC_n + (i + 1) * sC_k + cs, mask=live_c, other=0.0)
            t_c = tl.load(target + (i + 1) * sT_p + cand, mask=live_c, other=0.0)
            on_cand = a * t_c
            whole = a * tl.sum(tl.where(ks == i, sums, 0.0), axis=0)
            mass = whole - tl.sum(tl.where(live_c, on_cand, 0.0), axis=0) \
                + tl.sum(tl.where(live_c, tl.maximum(on_cand - q, 0.0), 0.0), axis=0)
            denominator = mass + 1.0 - a
            threshold = tl.where(denominator > 0, mass / tl.maximum(denominator, TINY), 1.0)
        else:
            threshold = a                                           # the last position's threshold is its own P_K
        u = tl.load(U + n * sU + i)
        accepted = tl.where(u <= threshold, i + 1, accepted)         # the FURTHEST position that passes

    at = tl.minimum(accepted, K)
    before = tl.where(accepted > 0, tl.sum(tl.where(ks == accepted - 1, carried, 0.0), axis=0), 1.0)
    tl.store(ACCEPTED + n, accepted)
    tl.store(AT + n, at)
    tl.store(BEFORE + n, before)

    # -- the residual the correction is drawn from: before*p, less the draft where it put mass ----------
    cand = tl.load(CAND + n * sC_n + tl.minimum(at, K - 1) * sC_k + cs, mask=live_c, other=0)
    q = tl.load(QPROB + n * sC_n + tl.minimum(at, K - 1) * sC_k + cs, mask=live_c, other=0.0)
    q = tl.where(at < K, q, 0.0)
    rest = REST + n * V
    total = 0.0
    plain = 0.0
    for v0 in range(0, V, BLOCK):
        vs = v0 + tl.arange(0, BLOCK)
        alive = vs < V
        p = tl.load(target + at * sT_p + vs, mask=alive, other=0.0)
        r = before * p
        # subtract the draft at its C candidates, which is all it is not zero on
        hit = (cand[:, None] == vs[None, :]) & live_c[:, None] & alive[None, :]
        r = tl.maximum(r - tl.sum(tl.where(hit, q[:, None], 0.0), axis=0), 0.0)
        tl.store(rest + vs, r, mask=alive)
        total += tl.sum(tl.where(alive, r, 0.0), axis=0)
        plain += tl.sum(tl.where(alive, p, 0.0), axis=0)
    mass_total = total + 0.0
    mass_plain = plain + 0.0

    scale = tl.where(mass_total > 0, 1.0 / tl.maximum(mass_total, TINY), 0.0)
    fall = tl.where(mass_total > 0, 0.0, 1.0 / tl.maximum(mass_plain, TINY))
    for v0 in range(0, V, BLOCK):
        vs = v0 + tl.arange(0, BLOCK)
        alive = vs < V
        r = tl.load(rest + vs, mask=alive, other=0.0)
        p = tl.load(target + at * sT_p + vs, mask=alive, other=0.0)
        tl.store(rest + vs, r * scale + p * fall, mask=alive)


def verify_rows(target_probs, drafts, draft_cand, draft_probs, uniforms, rest=None):
    """(accepted [n], at [n], rest [n, V]) -- everything block verification does before the correction draw.

    `uniforms` [n, K] are the caller's, in the caller's order. `rest` may be a buffer to write into.
    """
    n, t, V = target_probs.shape
    K, C = draft_cand.shape[1], draft_cand.shape[2]
    if t != K + 1 or drafts.shape != (n, K) or draft_probs.shape != (n, K, C) or uniforms.shape != (n, K):
        raise ValueError("block verification wants target [n, K+1, V], drafts [n, K], candidates [n, K, C]")
    accepted = torch.empty(n, dtype=torch.int64, device=target_probs.device)
    at = torch.empty_like(accepted)
    before = torch.empty(n, dtype=torch.float32, device=target_probs.device)
    if rest is None:
        rest = torch.empty(n, V, dtype=torch.float32, device=target_probs.device)
    _verify[(n,)](target_probs, drafts, draft_cand, draft_probs, uniforms, accepted, at, before, rest,
                  V, target_probs.stride(0), target_probs.stride(1), drafts.stride(0),
                  draft_cand.stride(0), draft_cand.stride(1), uniforms.stride(0),
                  K, C, triton.next_power_of_2(K), triton.next_power_of_2(C), 4096, num_warps=8)
    return accepted, at, rest
