"""The candidate selector's greedy walk in one launch (45차 §86).

The drafter proposes by walking K positions: at each one it reads the row of `scores` its predecessor selected,
takes the best of that position's `sel_top_k` candidates, and carries the choice forward. Written as a loop it
is five iterations of gather / argmax / gather over tensors holding sixteen numbers -- about twenty-five device
operations for a walk whose whole state is one integer.

Only the greedy walk folds. A sampled row walks over keyed uniforms (base/draws: a hash of what each draw is
for, the same on every rank), and moving that walk inside a kernel would put D12's replay in there with it --
the same line `block_verify` draws.

The walk is exact, not close: the keys it maximises over are the scores as computed, and `argmax` breaks ties
toward the lower index on both sides.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _walk(SCORES, CAND, OUT, sSn, sSk, sSp, sCn, sCk, sOn, K: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BC)
    live = c < C
    prev = 0
    for step in range(K):
        scores = tl.load(SCORES + row * sSn + step * sSk + prev * sSp + c, live, other=-float("inf"))
        best = tl.argmax(scores, 0)
        tl.store(OUT + row * sOn + step, tl.load(CAND + row * sCn + step * sCk + best))
        prev = best


def greedy_walk(scores: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """scores [n, K, predecessors, candidates], cand [n, K, candidates] -> the walk's drafts [n, K]."""
    if scores.ndim != 4 or cand.ndim != 3 or cand.shape[:2] != scores.shape[:2] or cand.shape[2] != scores.shape[3]:
        raise ValueError("the walk takes scores [n, K, prev, cand] and the candidates they name")
    if scores.shape[2] != scores.shape[3]:
        raise ValueError("a step's predecessors are the previous step's candidates")
    n, K, _, C = scores.shape
    if not scores.is_cuda:
        return _by_torch(scores, cand)
    scores, cand = scores.contiguous(), cand.contiguous()
    out = torch.empty(n, K, dtype=cand.dtype, device=cand.device)
    if n:
        _walk[(n,)](scores, cand, out, scores.stride(0), scores.stride(1), scores.stride(2),
                    cand.stride(0), cand.stride(1), out.stride(0),
                    K=K, C=C, BC=triton.next_power_of_2(C), num_warps=1)
    return out


@triton.jit
def _walk_scores(UNARY, CAND, ANCHOR, PROJ, PRED, SUCC, OUT,
                 sUn, sUk, sCn, sCk, sPn, sPk, sOn,
                 K: tl.constexpr, C: tl.constexpr, R: tl.constexpr, BC: tl.constexpr, BR: tl.constexpr):
    """The selector's scores and the walk over them, without the scores ever existing.

    A step needs one row of the predecessor codebook -- the token the last step chose -- so the [n, K, C, R]
    products the torch form materialised are, at any moment, one [R] vector against C codebook rows."""
    row = tl.program_id(0)
    c = tl.arange(0, BC)
    r = tl.arange(0, BR)
    live_c, live_r = c < C, r < R
    token = tl.load(ANCHOR + row)
    for step in tl.static_range(K):
        cand = tl.load(CAND + row * sCn + step * sCk + c, live_c, other=0)
        weight = (tl.load(PRED + token * R + r, live_r, other=0.0).to(tl.float32)
                  * tl.load(PROJ + row * sPn + step * sPk + r, live_r, other=0.0))
        succ = tl.load(SUCC + cand[:, None] * R + r[None, :], live_c[:, None] & live_r[None, :], other=0.0)
        score = tl.load(UNARY + row * sUn + step * sUk + c, live_c, other=-float("inf"))
        score += tl.sum(succ.to(tl.float32) * weight[None, :], 1)
        token = tl.load(CAND + row * sCn + step * sCk + tl.argmax(tl.where(live_c, score, -float("inf")), 0))
        tl.store(OUT + row * sOn + step, token)


def walk_scores(unary, cand, anchors, proj, predecessor, successor):
    """The selector's greedy walk straight from what it reads: unary [n, K, C] fp32, cand [n, K, C], anchors
    [n], proj [n, K, R] fp32, the two codebooks [vocab, R] -> the drafts [n, K]."""
    n, K, C = unary.shape
    R = proj.shape[-1]
    if cand.shape != unary.shape or proj.shape[:2] != (n, K) or anchors.numel() != n:
        raise ValueError("the walk takes a row of candidates and a projection for every step")
    if predecessor.shape != successor.shape or predecessor.shape[-1] != R:
        raise ValueError("both codebooks must be [vocab, rank] at the projection's rank")
    if not unary.is_cuda:
        return _scores_by_torch(unary, cand, anchors, proj, predecessor, successor)
    unary, cand, proj = unary.contiguous(), cand.contiguous(), proj.contiguous()
    out = torch.empty(n, K, dtype=cand.dtype, device=cand.device)
    if n:
        _walk_scores[(n,)](unary, cand, anchors.contiguous(), proj, predecessor, successor, out,
                           unary.stride(0), unary.stride(1), cand.stride(0), cand.stride(1),
                           proj.stride(0), proj.stride(1), out.stride(0),
                           K=K, C=C, R=R, BC=triton.next_power_of_2(C),
                           BR=triton.next_power_of_2(R), num_warps=4)
    return out


def _scores_by_torch(unary, cand, anchors, proj, predecessor, successor):
    """The form this replaces: the scores in full, then the walk over them."""
    n, K, C = unary.shape
    pred_ids = torch.cat([anchors.view(n, 1, 1).expand(n, 1, C), cand[:, :-1]], 1)
    pred = predecessor[pred_ids].float()
    succ = successor[cand].float()
    scores = unary[:, :, None, :] + torch.einsum("nkpr,nkcr->nkpc", pred * proj[:, :, None, :], succ)
    return _by_torch(scores, cand)


def _by_torch(scores, cand):
    """The loop this replaces, kept as the reference it is judged against."""
    n, K = scores.shape[:2]
    rows = torch.arange(n, device=scores.device)
    prev = torch.zeros(n, dtype=torch.int64, device=scores.device)
    out = []
    for step in range(K):
        best = scores[rows, step, prev].argmax(-1)
        out.append(cand[rows, step, best])
        prev = best
    return torch.stack(out, 1)
