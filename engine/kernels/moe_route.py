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


__all__ = ["softmax_topk"]
