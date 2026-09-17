"""CPU global-scale (alpha) re-anchoring headroom on real production experts.

The NVFP4 activation/weight search moves a block +-2 scale codes around the
ladder entry nearest to block_amax/gs/6 -- if the static global scale `gs` is
mis-anchored the whole ladder lands off-centre and no radius recovers it. This
probe measures, on the production rank's own dequantised experts, the
quantisation SSE under three anchors: the STORED checkpoint alpha, the
amax/(448*6) recomputed alpha, and the sweep-optimal alpha (log grid, +-4
octaves). The gap between stored and optimal is the re-anchoring headroom a
repack could claim; the gap between stored and recomputed says whether the
checkpoint's alpha simply drifted from the packed values (GPTQ moved the
weights after the alpha was stamped).

Weight numbers are fully real. The activation arm uses synthetic inputs and is
labelled mechanism: a static gs is re-anchored for a CORPUS, and that needs
captured activations, not this fixture.
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
from probes.nvfp4_hadamard_rotation_probe import global_scale, quant_matrix, sse


def sweep_alpha(t, radius=2, points=33):
    """SSE-optimal gs over a +-4-octave log grid around the recomputed anchor."""
    anchor = global_scale(t)
    best_gs, best_sse = anchor, None
    for i in range(points):
        gs = anchor * (2.0 ** (-4.0 + 8.0 * i / (points - 1)))
        trial = sse(quant_matrix(t, gs, radius), t)
        if best_sse is None or trial < best_sse:
            best_gs, best_sse = gs, trial
    return best_gs, best_sse


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
    rows = {}

    # ---- weight side: fully real. The production rank carries no per-tensor
    # alpha (checked: L3/20/40 moe headers lack it -- the global scale lives in
    # the SF codes / packer), so the served anchor IS the amax-recomputed one.
    # The question is its gap to the sweep optimum.
    for stage, w in (('fc1_w13', w13), ('fc2_w2', w2)):
        recomputed = global_scale(w)
        opt_gs, opt_sse = sweep_alpha(w)
        arms = {
            'recomputed': sse(quant_matrix(w, recomputed, 2), w),
            'optimal': opt_sse,
        }
        rows[stage] = dict(recomputed_alpha=recomputed, optimal_alpha=opt_gs,
                           optimal_over_recomputed=opt_gs / recomputed, sse=arms)
        rows[stage]['dims'] = list(w.shape)

    # ---- activation side: mechanism (module docstring). stored static gs vs
    # recomputed vs sweep, on the two synthetic input shapes.
    g = torch.Generator().manual_seed(917 + layer * 1000 + expert)
    inputs = {}
    inputs['gauss'] = torch.randn(samples, k, generator=g).bfloat16().float()
    channel = torch.exp(torch.randn(k, generator=g) * 1.2)
    channel[torch.randperm(k, generator=g)[:8]] *= 25.0
    inputs['outlier'] = (torch.randn(samples, k, generator=g) * channel).bfloat16().float()
    act = {}
    for kind, x in inputs.items():
        for radius in (1, 2):
            recomputed = global_scale(x)
            opt_gs, opt_sse = sweep_alpha(x, radius)
            act[f'{kind}.r{radius}'] = dict(
                recomputed=sse(quant_matrix(x, recomputed, radius), x),
                optimal=opt_sse, optimal_over_recomputed=opt_gs / recomputed)
    rows['activation_mechanism'] = act
    rows['shape'], rows['layer'], rows['expert'] = [k, n], layer, expert
    return rows


def pooled(rows, stage, key):
    return math.fsum(r[stage]['sse'][key] for r in rows)


def run(args):
    torch.set_num_threads(4)
    rank = Rank(args.ranks)
    rows = []
    for layer in (3, 20, 40):
        for expert in (0, 73, 287):
            rows.append(cell(rank, layer, expert, args.samples))
            print(json.dumps(dict(event='cell', layer=layer, expert=expert)), flush=True)
    rank.file.close()

    totals = {}
    for stage in ('fc1_w13', 'fc2_w2'):
        recomputed, optimal = (pooled(rows, stage, k) for k in ('recomputed', 'optimal'))
        totals[stage] = dict(recomputed_sse=recomputed, optimal_sse=optimal,
                             optimal_vs_recomputed_pct=100 * (1 - optimal / recomputed) if recomputed else None,
                             optimal_over_recomputed_median=sorted(
                                 r[stage]['optimal_over_recomputed'] for r in rows)[len(rows) // 2])
    for kind in ('gauss', 'outlier'):
        for radius in (1, 2):
            key = f'{kind}.r{radius}'
            recomputed = math.fsum(r['activation_mechanism'][key]['recomputed'] for r in rows)
            optimal = math.fsum(r['activation_mechanism'][key]['optimal'] for r in rows)
            totals.setdefault('activation_mechanism', {})[key] = dict(
                recomputed_sse=recomputed, optimal_sse=optimal,
                optimal_vs_recomputed_pct=100 * (1 - optimal / recomputed) if recomputed else None)

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
