"""CPU rotation ablation on real production experts: does folding an orthogonal
Hadamard rotation into the NVFP4 quantization basis beat the in-kernel scale
search, and do the two compose?

Weight-side numbers are fully real: the experts are the production rank's own
dequantised tensors, re-quantised along their reduction dim with and without a
rotation folded into the columns (exactly the offline repack a serving build
would pay). Activation-side numbers are mechanism, not a serving estimate:
gaussian inputs are rotation-invariant and must show ~no gain (an arithmetic
sanity check of the harness itself), and outlier-shaped inputs show the
mechanism's ceiling. A real activation number needs a capture boot; this probe
only orders the arms.

Quantisation semantics follow probes/nvfp4_scale_search_review.compare_block
(the as1/as2 recipe) term for term -- the e4m3 scale ladder, the e2m1 RNE
encoder, the base code nearest_even(amax/gs/6, SCALES), even-index tie
breaks, SSE-best candidate among offsets within the radius -- and the
vectorised implementation is cross-checked against it block for block
before anything runs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from engine.modules.moe import dequant_nvfp4
from engine.modules.nvfp4_sf import unswizzle_sf
from probes.nvfp4_all_activation_projection import Rank
from probes.nvfp4_scale_search_review import SCALES, compare_block

FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
OFFSETS = (0, -1, 1, -2, 2)
SCALES_T = torch.tensor(SCALES, dtype=torch.float64)
FP4_T = torch.tensor(FP4, dtype=torch.float64)
FP4_MID = torch.tensor([(FP4[i] + FP4[i + 1]) / 2 for i in range(7)], dtype=torch.float64)


def _even_tie_index(dist, values):
    """argmin with nearest_even's tie rule: among exactly-equal distances the
    min over (distance, i & 1) keeps the EVEN table index."""
    best = dist.min(dim=-1, keepdim=True).values
    ties = dist == best
    first = ties.to(torch.int64).argmax(dim=-1)
    even = (torch.arange(values, dtype=torch.int64) & 1) == 0
    even_first = (ties & even).to(torch.int64).argmax(dim=-1)
    has_even = (ties & even).any(dim=-1)
    return torch.where(has_even, even_first, first)


def encode_levels(a, scale):
    """|values|/scale -> nearest FP4 level index, even-index ties, then restored."""
    ratio = a.double() / scale.double().unsqueeze(-1)
    dist = (ratio.unsqueeze(-1) - FP4_T).abs()
    idx = _even_tie_index(dist, 8)
    return FP4_T[idx]


def quant_matrix(x, gs, radius):
    """Vectorised per-16-block NVFP4 encode with the recipe's scale search.

    x [rows, cols] is quantised along the last dim in 16-blocks with one e4m3
    scale per block: base code = nearest_even(block_amax/gs, SCALES), then the
    SSE-best restored block among the ladder offsets within the radius."""
    rows, cols = x.shape
    blocks = x.double().reshape(-1, 16)
    block_amax = blocks.abs().amax(-1)
    dist = ((block_amax / gs / 6.0).unsqueeze(1) - SCALES_T).abs()
    base = _even_tie_index(dist, 127)
    best_sse = None
    best_restored = None
    for offset in OFFSETS:
        if abs(offset) > radius:
            continue
        code = (base + offset).clamp(0, 126)
        scale = SCALES_T[code] * gs
        restored = torch.copysign(encode_levels(blocks.abs(), scale) * scale.unsqueeze(-1), blocks)
        sse = (blocks - restored).square().sum(-1)
        if best_sse is None:
            best_sse, best_restored = sse, restored
        else:
            better = sse < best_sse                       # strict: first-best offset wins, like compare_block
            best_sse = torch.where(better, sse, best_sse)
            best_restored = torch.where(better.unsqueeze(1), restored, best_restored)
    return best_restored.float().reshape(rows, cols)


def cross_check(gs=0.07):
    """The vectorised quantiser must match compare_block exactly, block for
    block, on random, tied and outlier blocks at every radius."""
    g = torch.Generator().manual_seed(7)
    blocks = torch.cat([
        torch.randn(64, 16, generator=g),
        torch.tensor([[1.25] * 16, [0.75] * 16, [5.0] * 16] * 8),       # exact FP4 midpoints: tie rules
        (torch.randn(8, 16, generator=g) * torch.tensor([1.] * 12 + [40.] * 4)),
    ]).double()
    for radius in (0, 1, 2):
        mine = quant_matrix(blocks.float(), gs, radius).double()
        pairs = [compare_block(tuple(v), gs, radius=radius) for v in blocks.tolist()]
        theirs = torch.tensor([min(p, key=lambda d: d['sse'])['restored'] for p in pairs])
        worst = float((mine - theirs).abs().max())
        assert worst == 0.0, f'radius {radius}: vectorised quantiser differs from compare_block by {worst}'


_H = {}


def hadamard_cached(n):
    if n not in _H:
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)
        _H[n] = h
        assert (h @ h.T - torch.eye(n)).abs().max() < 1e-4
    return _H[n]


def power2_or(n, fallback):
    return n if (n & (n - 1)) == 0 else fallback


def global_scale(t):
    return float(t.abs().max()) / (448.0 * 6.0)


def activate(upgate):
    # Rank files are explicitly up|gate, as consumed by b12x, not gate|up.
    up, gate = upgate.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16().float()


def down(x, w):
    # Preserve the reference's BF16 round trip for each K128 contribution.
    return sum((x[:, k:k + 128] @ w[:, k:k + 128].T).bfloat16().float()
               for k in range(0, x.shape[1], 128))


def sse(x, reference):
    return float((x.double() - reference.double()).square().sum())


ROTATIONS = ('none', 'h128', 'hfull')
RADII = (0, 1, 2)


def rotated(t, rotation, d):
    """Rotate `t` along its last dim of width `d`: 'hfull' is the one [d, d]
    Hadamard (d a power of two), 'h128' the block-diagonal of 128-size blocks
    (applied as a reshape, never materialised)."""
    if rotation == 'none':
        return t
    assert t.shape[-1] == d and d % 128 == 0, (t.shape, d)
    if rotation == 'h128':
        lead = t.reshape(-1, d // 128, 128)
        return (lead @ hadamard_cached(128)).reshape(t.shape)
    assert power2_or(d, 128) == d, d
    return t @ hadamard_cached(d)


def cell(rank, layer, expert, samples):
    prefix = f'L{layer}.moe.'
    tensors = {}
    for name in ('w13', 'w13_sf', 'w2', 'w2_sf'):
        tensors[name], _ = rank.expert(prefix + name, expert)
    scales = {}
    for name in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale'):
        if prefix + name in rank.header:
            scales[name], _ = rank.expert(prefix + name, expert)
        else:
            scales[name] = torch.tensor(1.)
    n, k = tensors['w2'].shape[1] * 2, tensors['w13'].shape[1] * 2
    w13 = dequant_nvfp4(tensors['w13'], unswizzle_sf(tensors['w13_sf'], 2 * n, k // 16), scales['w13_alpha'])
    w2 = dequant_nvfp4(tensors['w2'], unswizzle_sf(tensors['w2_sf'], k, n // 16), scales['w2_alpha'])
    assert k == 4096 and k % 128 == 0 and n % 128 == 0, (k, n)
    rows = {}

    # ---- weight side: fully real. Requantisation SSE against the rank's own
    # dequantised weight, per rotation arm and search radius.
    weight_sse = {}
    for stage, w in (('fc1_w13', w13), ('fc2_w2', w2)):
        for rotation in ROTATIONS:
            rotated_w = rotated(w, rotation, w.shape[1])
            for radius in RADII:
                rebuilt = quant_matrix(rotated_w, global_scale(rotated_w), radius)
                weight_sse[f'{stage}.{rotation}.r{radius}'] = sse(rebuilt, rotated_w)
    rows['weight_sse'] = weight_sse

    # ---- activation side: mechanism only (module docstring).
    g = torch.Generator().manual_seed(917 + layer * 1000 + expert)
    x_gauss = torch.randn(samples, k, generator=g).bfloat16().float()
    channel = torch.exp(torch.randn(k, generator=g) * 1.2)
    channel[torch.randperm(k, generator=g)[:8]] *= 25.0
    x_outlier = (torch.randn(samples, k, generator=g) * channel).bfloat16().float()
    activation_sse = {}
    for kind, x in (('gauss', x_gauss), ('outlier', x_outlier)):
        for rotation in ROTATIONS:
            rotated_x = rotated(x, rotation, k)
            for radius in RADII:
                rebuilt = quant_matrix(rotated_x, global_scale(rotated_x), radius)
                activation_sse[f'{kind}.{rotation}.r{radius}'] = sse(rebuilt, rotated_x)
    rows['activation_sse'] = activation_sse

    # ---- composed full-MLP projection error: both stages under the SAME arm,
    # each against its own unquantised reference (the rotations cancel exactly
    # downstream, so the only difference between arms is quantisation noise).
    full = {}
    for rotation in ROTATIONS:
        w13_r = rotated(w13, rotation, k)
        w2_r = rotated(w2, rotation, n)
        x_r = rotated(x_outlier, rotation, k)
        a_r = None
        for radius in RADII:
            w13_q = quant_matrix(w13_r, global_scale(w13_r), radius)
            w2_q = quant_matrix(w2_r, global_scale(w2_r), radius)
            fc1_ref = activate(x_r @ w13_r.T)
            fc1_q = activate(quant_matrix(x_r, global_scale(x_r), radius) @ w13_q.T)
            a_r = fc1_ref if a_r is None else a_r
            a_q_r = rotated(fc1_q, rotation, n)
            a_ref_r = rotated(a_r, rotation, n)
            y_q = down(quant_matrix(a_q_r, global_scale(a_q_r), radius), w2_q)
            y_ref = down(a_ref_r, w2_r)
            full[f'{rotation}.r{radius}'] = sse(y_q, y_ref)
    rows['full_mlp_projection_sse'] = full
    rows['shape'], rows['layer'], rows['expert'] = [k, n], layer, expert
    return rows


def pooled(rows, getter, base, key):
    old = math.fsum(getter(r)[base] for r in rows)
    new = math.fsum(getter(r)[key] for r in rows)
    return dict(sse=new, vs_none_r0_pct=100 * (1 - new / old) if old else None)


def run(args):
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(7)
    cross_check(0.07)
    for gs in (0.3, 1.5):                                   # scale anchors change the ladder landing spot
        cross_check(gs)
    print(json.dumps(dict(event='cross-check-vs-compare_block', ok=True)), flush=True)

    rank = Rank(args.ranks)
    rows = []
    for layer in (3, 20, 40):
        for expert in (0, 73, 287):
            rows.append(cell(rank, layer, expert, args.samples))
            print(json.dumps(dict(event='cell', layer=layer, expert=expert)), flush=True)
    rank.file.close()

    totals = {}
    for stage in ('fc1_w13', 'fc2_w2'):
        for rotation in ROTATIONS:
            for radius in RADII:
                totals.setdefault('weight_sse', {})[f'{stage}.{rotation}.r{radius}'] = pooled(
                    rows, lambda r: r['weight_sse'], f'{stage}.none.r0', f'{stage}.{rotation}.r{radius}')
    for kind in ('gauss', 'outlier'):
        for rotation in ROTATIONS:
            for radius in RADII:
                totals.setdefault('activation_sse', {})[f'{kind}.{rotation}.r{radius}'] = pooled(
                    rows, lambda r: r['activation_sse'], f'{kind}.none.r0', f'{kind}.{rotation}.r{radius}')
    for rotation in ROTATIONS:
        for radius in RADII:
            totals.setdefault('full_mlp_projection_sse', {})[f'{rotation}.r{radius}'] = pooled(
                rows, lambda r: r['full_mlp_projection_sse'], 'none.r0', f'{rotation}.r{radius}')

    result = dict(scope=__doc__, gpu_used=False, torch=torch.__version__,
                  ranks=dict(path=str(args.ranks), header_sha256=rank.header_sha256),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  totals=totals, cells=rows)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(totals=totals)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--output', type=Path, required=True)
    run(ap.parse_args())
