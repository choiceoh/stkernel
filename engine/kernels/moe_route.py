"""The softmax top-k router in one launch: probabilities, the k largest, the renormalisation, the BF16 boundary and the
EP remap.

Qwen3.8's router (engine/modules/moe.route_softmax_topk, transformers qwen4_exp) is a composition of small torch
kernels: a widening, the softmax, the top-k, a sum, a division, a rounding, a widening again for the dispatcher; then
the EP remap of a captured step (profiles/qwen38/lanes.local_routes) is eight more. A captured step pays all of them at
every MoE layer, for one to eight rows of 512 scores. This is the same arithmetic over a row in one program.

What differs from the composition, and why it is a `kernel` and not a `fold`:

    ties      the k largest are taken by k rounds of argmax, the lowest expert id first among equal probabilities.
              torch.topk leaves the order of equal elements unspecified, and BF16 logits tie often (512 draws on a
              seven-bit fraction): over 98,304 rows of random BF16 scores the expert SET was torch's in every row and
              the ORDER of tied experts inside the k differed in 14% of them -- the dispatcher then sums the same
              weighted outputs in another order. A tie at the k-th place may name another expert than torch does;
              every rank runs this launch on the same scores, so the ranks agree.
    exp       Triton's exp is not torch's: the two differ in up to a few last FP32 bits (8e-7 relative), and the
              softmax's denominator is this program's reduction, not torch's. With BF16 scores weights round to BF16
              (the reference returns them in the scores' dtype), so nearly every weight is the
              reference's bit for bit (99.9996% of them, same rows) and the rest are one BF16 step away.

    (Measured 2026-09-18 on an RTX 5050 -- sm_120, triton 3.6, torch 2.11 -- not on a GB10.)

The shared expert's gate reads the same scores row, and its sigmoid is NOT folded in here, for that second reason: it is
consumed in FP32 (kernels/moe_output.gated_sum), where no rounding absorbs the difference, and torch's own sigmoid is
the oracle's bytes (the same measurement: 32% of 2^20 values differ between tl.exp's sigmoid and torch.sigmoid).
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _softmax_topk(Scores, Ids, Weights, sS, sI, sW, FIRST, LOCAL, FOREIGN,
                  E: tl.constexpr, K: tl.constexpr, EB: tl.constexpr, KB: tl.constexpr,
                  REMAP: tl.constexpr, ROUND: tl.constexpr):
    r = tl.program_id(0)
    e = tl.arange(0, EB)
    live = e < E
    logits = tl.load(Scores + r * sS + e, live, float("-inf")).to(tl.float32)
    shifted = tl.exp(logits - tl.max(logits, 0))
    p = tl.where(live, shifted / tl.sum(shifted, 0), -1.0)
    kk = tl.arange(0, KB)
    ids = tl.zeros([KB], tl.int32)
    top = tl.zeros([KB], tl.float32)
    for j in tl.static_range(K):
        value, best = tl.max(p, 0, return_indices=True, return_indices_tie_break_left=True)
        best = best.to(tl.int32)
        top = tl.where(kk == j, value, top)
        ids = tl.where(kk == j, best, ids)
        p = tl.where(e == best, -1.0, p)
    w = top / tl.sum(top, 0)
    if ROUND:
        w = w.to(tl.bfloat16).to(tl.float32)
    if REMAP:
        local = ids - FIRST
        foreign = (local < 0) | (local >= LOCAL)
        ids = tl.where(foreign, FOREIGN, local)
        w = tl.where(foreign, 0.0, w)
    keep = kk < K
    tl.store(Ids + r * sI + kk, ids, keep)
    tl.store(Weights + r * sW + kk, w, keep)


def softmax_topk(scores, k, *, experts=None, first=None, local=None, foreign=0, exact=False):
    """scores BF16/FP32 [N, >= experts] with packed columns -> (ids int32 [N, k], weights FP32 [N, k]).

    The first `experts` columns (all by default) are the router's logits: softmax in FP32, the k largest, renormalised
    to sum to one, then the input dtype's rounding -- engine/modules/moe.route_softmax_topk with `normalize`, then
    `.float()`. FP32 logits retain FP32 weights, in eager and captured steps alike. With `first`
    and `local`, an id outside this rank's [first, first + local) becomes `foreign` (local expert 0, or the dispatcher's
    zero-weight sentinel) at weight exactly 0 and the others count from `first` (lanes.local_routes). `exact` keeps the
    weights unrounded (the tests hold the arithmetic and the rounding apart: Triton's CPU interpreter does not round
    BF16 the way a GPU does)."""
    if scores.ndim != 2 or scores.dtype not in (torch.bfloat16, torch.float32) or not scores.is_cuda or scores.stride(1) != 1:
        raise ValueError("the router takes BF16/FP32 scores [N, >= experts] with packed columns on a CUDA device")
    rows, columns = scores.shape
    experts = columns if experts is None else experts
    if type(experts) is not int or type(k) is not int or not 1 <= k <= experts <= columns or k > 64:
        raise ValueError(f"the router picks 1..64 of the row's experts, not {k} of {experts} in {columns} columns")
    remap = first is not None or local is not None
    if remap and (type(first) is not int or type(local) is not int or first < 0 or local <= 0
                  or type(foreign) is not int or not 0 <= foreign <= local):
        raise ValueError("an EP remap names the rank's first expert, its expert count and a foreign id in [0, local]")
    ids = torch.empty(rows, k, dtype=torch.int32, device=scores.device)
    weights = torch.empty(rows, k, dtype=torch.float32, device=scores.device)
    if rows:
        _softmax_topk[(rows,)](scores, ids, weights, scores.stride(0), ids.stride(0), weights.stride(0),
                               first if remap else 0, local if remap else 0, foreign,
                               E=experts, K=k, EB=tr.next_power_of_2(experts), KB=tr.next_power_of_2(k),
                               REMAP=remap, ROUND=not exact and scores.dtype == torch.bfloat16,
                               num_warps=4, enable_fp_fusion=False)
    return ids, weights


@tr.jit
def _compact_routes(Ids, Weights, Local, W, Mine, N, FIRST, LOCAL, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < N
    shifted = tl.load(Ids + i, live, 0) - FIRST
    mine = (shifted >= 0) & (shifted < LOCAL)
    tl.store(Local + i, tl.where(mine, shifted, 0), live)
    tl.store(W + i, tl.where(mine, tl.load(Weights + i, live, 0.0), 0.0), live)
    tl.store(Mine + i, mine.to(tl.int8), live)


def compact_routes(ids, weights, first, local):
    """(local ids, weights, mine) [N, k] of global routes for an eager step's compact MoE, in one launch: this rank's
    routes counted from `first`, another rank's on local expert 0 at weight exactly 0 (profiles/qwen38/lanes.
    local_routes without a sentinel), and which routes are this rank's (bool, for torch.nonzero) -- the nine small torch
    launches of the remap and the mask as one, bit for bit theirs (selects and an integer offset)."""
    if (ids.ndim != 2 or weights.shape != ids.shape or ids.dtype != torch.int32 or weights.dtype != torch.float32
            or not ids.is_contiguous() or not weights.is_contiguous() or ids.device != weights.device
            or type(first) is not int or type(local) is not int or first < 0 or local <= 0):
        raise ValueError("compact routes take int32 ids and FP32 weights [N, k], packed, and the rank's expert span")
    out = torch.empty_like(ids), torch.empty_like(weights), torch.empty(ids.shape, dtype=torch.int8, device=ids.device)
    n = ids.numel()
    if n:
        _compact_routes[(tr.cdiv(n, 1024),)](ids, weights, *out, n, first, local, BLOCK=1024, num_warps=4)
    return out[0], out[1], out[2].view(torch.bool)


@tr.jit
def _pair_rows(X, Local, W, Token, Route, Xp, Ip, Wp, sX, sXp, K, H, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(Token + p)
    tl.store(Xp + p * sXp + c, tl.load(X + t * sX + c, c < H), c < H)
    if tl.program_id(1) == 0:
        at = t * K + tl.load(Route + p)
        tl.store(Ip + p, tl.load(Local + at))
        tl.store(Wp + p, tl.load(W + at))


def pair_rows(x, local_ids, weights, token, route):
    """The compact MoE's pairs gathered in one launch: (x's rows [P, H] at `token`, the pairs' local ids [P, 1] and
    weights [P, 1] at (token, route)) -- x.index_select and two index gathers, the same bytes (copies)."""
    if (x.ndim != 2 or x.stride(1) != 1 or local_ids.ndim != 2 or weights.shape != local_ids.shape
            or not local_ids.is_contiguous() or not weights.is_contiguous() or local_ids.shape[0] != x.shape[0]
            or token.ndim != 1 or route.shape != token.shape or token.dtype != torch.int64 or route.dtype != torch.int64
            or not token.is_contiguous() or not route.is_contiguous()):
        raise ValueError("pair rows take x [N, H], packed ids and weights [N, k], and int64 (token, route) [P]")
    p, h = token.shape[0], x.shape[1]
    xp = torch.empty(p, h, dtype=x.dtype, device=x.device)
    ip = torch.empty(p, 1, dtype=local_ids.dtype, device=x.device)
    wp = torch.empty(p, 1, dtype=weights.dtype, device=x.device)
    if p:
        block = min(1024, tr.next_power_of_2(h))
        _pair_rows[(p, tr.cdiv(h, block))](x, local_ids, weights, token, route, xp, ip, wp, x.stride(0), xp.stride(0),
                                           local_ids.shape[1], h, BLOCK=block, num_warps=4)
    return xp, ip, wp


__all__ = ["softmax_topk", "compact_routes", "pair_rows"]
