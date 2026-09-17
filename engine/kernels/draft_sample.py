"""Batch conditional distributions, then walk keyed uniforms in one launch.

The score lattice and Torch softmax/sum arithmetic are unchanged. The
single-row CDF retains its serial order; multiple rows retain Torch cumsum.
Only the predecessor index is sequential; it need not launch Torch operators
for every draft position. The sparse probability returned is the one drawn.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sample_walk(SCORES, PROBS, CDF, MASS, CAND, TEMPS, UNIFORMS, OUT, SUPPORT, Q,
                 sUn: tl.constexpr, sUk: tl.constexpr,
                 K: tl.constexpr, C: tl.constexpr, BC: tl.constexpr,
                 GREEDY_ROWS: tl.constexpr, SINGLE_ROW: tl.constexpr, LAST_MASS: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BC)
    live = c < C
    stochastic = True
    if GREEDY_ROWS:
        stochastic = tl.load(TEMPS + row) > 0
    prev = 0
    for step in range(K):
        lattice = ((row*K + step)*C + prev)*C
        p = tl.load(PROBS + lattice + c, live, 0)
        if SINGLE_ROW:
            # Torch sends a single <=16-entry row through CUB's serial
            # per-thread scan. A many-row cumsum uses a different tree.
            # Preserve the former order, including uniforms at CDF cuts.
            running = 0.
            walk = tl.full((BC,), -float('inf'), tl.float32)
            for index in tl.static_range(C):
                running += tl.load(PROBS + lattice + index)
                walk = tl.where(c == index, running, walk)
            if LAST_MASS:
                total = running
            else:
                total = tl.load(MASS + (row*K + step)*C + prev)
        else:
            walk = tl.load(CDF + lattice + c, live, -float('inf'))
            total = tl.load(MASS + (row*K + step)*C + prev)
        uniform = tl.load(UNIFORMS + row*sUn + step*sUk)
        aim = uniform * total
        # searchsorted(right=True), clamped to the first final maximum so
        # u=1 and zero-probability tails keep the last candidate with mass.
        pick = tl.minimum(tl.sum((live & (walk <= aim)).to(tl.int32), 0), tl.argmax(walk, 0))
        if GREEDY_ROWS:
            scores = tl.load(SCORES + lattice + c, live, -float('inf'))
            best = tl.argmax(scores, 0)
            pick = tl.where(stochastic, pick, best)
            p = tl.where(stochastic, p, (c == best).to(tl.float32))
        offset = (row*K + step)*C
        ids = tl.load(CAND + offset + c, live, 0)
        token = tl.load(CAND + offset + pick)
        tl.store(OUT + row*K + step, token)
        tl.store(SUPPORT + offset + c, ids, live)
        tl.store(Q + offset + c, p, live)
        prev = pick


def sampled_walk(scores, cand, temps, uniforms, *, greedy_rows=True, last_mass=False):
    """scores [n,K,C,C] FP32 -> drafts, private support, sparse FP32 mass.

    The DFlash support is bounded to at most 16 candidates.
    Batched proposal uses sum(mass), matching sampler._inverse_cdf. The
    synchronous proposal historically uses the last CDF entry instead.
    Keep that distinction explicit rather than silently changing its draws.
    """
    if scores.ndim != 4 or scores.shape[2] != scores.shape[3]:
        raise ValueError('sampled walk requires [rows, steps, candidates, candidates] scores')
    n, k, c, _ = scores.shape
    if (k < 1 or not 1 <= c <= 16 or cand.shape != (n, k, c) or temps.shape != (n,)
            or uniforms.shape != (n, k)):
        raise ValueError('sampled walk requires 1..16 candidates and matching temperatures and keyed uniforms')
    if (scores.dtype != torch.float32 or cand.dtype != torch.int64 or temps.dtype != torch.float32
            or uniforms.dtype != torch.float32
            or any(x.device != scores.device for x in (cand, temps, uniforms))):
        raise ValueError('sampled walk requires FP32 scores/temperatures/uniforms and int64 candidates on one device')
    if not scores.is_cuda:
        return _by_torch(scores, cand, temps, uniforms, greedy_rows=greedy_rows, last_mass=last_mass)
    scores, cand, temps = scores.contiguous(), cand.contiguous(), temps.contiguous()
    heat = temps.clamp_min(1e-5).view(n, 1, 1, 1)
    probs = torch.softmax(scores / heat, dim=-1)
    cdf = probs if n == 1 else probs.cumsum(-1)
    mass = (probs if n == 1 else cdf[..., -1].contiguous()) if last_mass else probs.sum(-1)
    out = torch.empty((n, k), dtype=cand.dtype, device=cand.device)
    support = torch.empty_like(cand)
    q = torch.empty((n, k, c), dtype=torch.float32, device=scores.device)
    if n:
        _sample_walk[(n,)](scores, probs, cdf, mass, cand, temps, uniforms, out, support, q,
                           *uniforms.stride(), K=k, C=c, BC=triton.next_power_of_2(c),
                           GREEDY_ROWS=greedy_rows, SINGLE_ROW=n == 1, LAST_MASS=last_mass, num_warps=1)
    return out, support, q


def _by_torch(scores, cand, temps, uniforms, *, greedy_rows=True, last_mass=False):
    """The former per-position walk, including its two historical mass rules."""
    from engine.base.sampler import _inverse_cdf
    n, k, c, _ = scores.shape
    rows = torch.arange(n, device=scores.device)
    prev = torch.zeros(n, dtype=torch.int64, device=scores.device)
    heat = temps.clamp_min(1e-5).view(n, 1)
    stochastic = (temps > 0).view(n, 1)
    out, masses = [], []
    q = torch.empty((n, k, c), dtype=torch.float32, device=scores.device)
    for step in range(k):
        sel = scores[rows, step, prev]
        best = sel.argmax(-1) if greedy_rows else None
        p = torch.softmax(sel / heat, dim=-1)
        if greedy_rows:
            p = torch.where(stochastic, p, torch.zeros_like(p).scatter_(1, best[:, None], 1.))
        if last_mass:
            walk = p.cumsum(-1)
            pick = torch.searchsorted(walk.contiguous(), (uniforms[:, step]*walk[:, -1])[:, None].contiguous(), right=True).squeeze(1)
            pick = torch.minimum(pick, walk.argmax(-1))
        else:
            pick = _inverse_cdf(p, uniforms[:, step])
        if greedy_rows:
            pick = torch.where(stochastic.view(n), pick, best)
        out.append(cand[rows, step, pick])
        if greedy_rows:
            q[:, step] = p
        else:
            masses.append(p)
        prev = pick
    return torch.stack(out, 1), cand.clone(), q if greedy_rows else torch.stack(masses, 1)
