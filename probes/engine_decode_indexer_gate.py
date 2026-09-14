"""FP32 head-gate + boundary proof with real weights and synthetic activations.

The synthetic pool-ranking check is a numerical sensitivity test, not live
acceptance. Every timing arm contains the same boundary and uses the same
weights; only the FP32 projection/reduction changes.
"""
import hashlib

import torch

from probes.engine_decode_fusions import _capture
from probes.engine_decode_dsa_inputs import ROWS, timings


def check(report, ranks=None, *, timing=True, layers=11, ranking=True):
    from engine.base.kernel_shape import bound
    from engine.kernels.decode_projection import indexer_boundary
    from engine.kernels.indexer_gate import IndexerHeadGate
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loader = rank_loader(path)
        names = sorted(k.removesuffix('w_heads') for k in loader.keys() if k.endswith('.idx.w_heads'))
        if len(names) != 11:
            raise RuntimeError('head-gate qualification needs all 11 DSA layers')
        weights = loader.load([n+s for n in names for s in ('w_heads', 'k_norm_w', 'k_norm_b')], device='cuda')
        cells = [tuple(weights[n+s] for s in ('w_heads', 'k_norm_w', 'k_norm_b')) for n in names]
        origin = str(path)
    else:
        cells = [(torch.randn(32, 4096, device='cuda')*.02, torch.randn(128, device='cuda'),
                  torch.randn(128, device='cuda')) for _ in range(layers)]
        origin = 'synthetic FP32 weights'
    owners = [IndexerHeadGate(w, rows=ROWS) for w, _, _ in cells]
    report('head_gate_weights', source=origin, layers=len(cells),
           sha256=[hashlib.sha256(w.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for w, _, _ in cells],
           resident_bytes=0, arithmetic='FP32 products and fixed reductions, no TF32 or atomics')
    scale = 128 ** -.5 * 32 ** -.5
    pool_size, pool_topk = bound().indexer.pool, bound().indexer.topk // bound().indexer.pool
    ranking_changes = []
    for rows in ROWS:
        for stride in (4096, 4104):
            parent = torch.randn(rows, stride, dtype=torch.bfloat16, device='cuda')
            x = parent if stride == 4096 else parent[:, 4:4100]
            q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device='cuda')
            key = torch.randn(rows, 256, dtype=torch.bfloat16, device='cuda')[:, :128]
            graphs, outputs = [], []
            def base():
                return [(None, indexer_boundary(q, key, x.float() @ w.T, nw, nb, scale, rows=ROWS))
                        for w, nw, nb in cells]
            def candidate():
                result = []
                for owner, (_, nw, nb) in zip(owners, cells):
                    partials = owner(x)
                    result.append((partials, indexer_boundary(q, key, partials, nw, nb, scale,
                                                               rows=ROWS, head_splits=16)))
                return result
            try:
                for fn in (base, candidate):
                    graph, out = _capture(fn)
                    graphs.append(graph); outputs.append(out)
                worst = 0.
                for magnitude in (0., .001, 1., 16., 1.):
                    x.normal_().mul_(magnitude); q.normal_().mul_(magnitude); key.normal_()
                    for order in ((0, 1), (1, 0)):
                        for arm in order:
                            for partial, output in outputs[arm]:
                                if partial is not None:
                                    partial.fill_(float('nan'))
                                for t in output:
                                    t.fill_(float('nan'))
                            graphs[arm].replay()
                        for (partials, actual), (_, expected), (w, _, _) in zip(outputs[1], outputs[0], cells):
                            if not partials.isfinite().all().item() or any(not t.float().isfinite().all().item() for t in actual):
                                raise RuntimeError('head gate left poisoned/nonfinite partials or outputs')
                            for got, want in zip(actual[:2], expected[:2]):
                                torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)
                            error = ((actual[2]-expected[2]).abs().amax(1) /
                                     expected[2].abs().amax(1).clamp_min(1e-12)).max().item()
                            worst = max(worst, error)
                            if error > 1e-5:
                                raise RuntimeError(f'head-gate row-relative error {error} > 1e-5')
                            # Independent FP64 reference also catches a shared FP32 comparison mistake.
                            ref64 = x.double() @ w.double().T
                            raw = partials.double().sum(1)
                            error64 = ((raw-ref64).abs().amax(1) / ref64.abs().amax(1).clamp_min(1e-12)).max().item()
                            if error64 > 1e-5:
                                raise RuntimeError(f'head-gate FP64 row-relative error {error64} > 1e-5')
                stable = [tuple(t.clone() for t in out) for _, out in outputs[1]]
                for _ in range(50):
                    graphs[1].replay()
                    for (_, actual), expected in zip(outputs[1], stable):
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)
                report('head_gate_numerics', rows=rows, input_stride=stride, layers=len(cells),
                       exact_queries_keys=True, effective_gate_row_relative_max=worst,
                       fp64_reference=True, replay_orders='BA/AB', deterministic_replays=50,
                       partial_bytes_per_call=rows*16*32*4)
                if ranking and stride == 4096:
                    # Downstream formula with 32K/128K pool counts, bounded in
                    # 1K slices to avoid a full [M,H,pools] scratch allocation.
                    from engine.modules.sparse_indexer import indexer_logits
                    for ctx in (32000, 128000):
                        keys = torch.randn(ctx//pool_size, 128, dtype=torch.bfloat16, device='cuda')
                        changed = total = 0
                        for (_, a), (_, b) in zip(outputs[1], outputs[0]):
                            logits = [torch.cat([indexer_logits(out[0].float(), chunk, out[2])
                                                for chunk in keys.split(1024)], dim=1) for out in (a, b)]
                            top = [v.topk(pool_topk, sorted=False).indices.sort(1).values for v in logits]
                            changed += int((top[0] != top[1]).any(1).sum().item())
                            total += rows
                        report('head_gate_pool_ranking', context=ctx, rows=rows, compared_queries=total,
                               changed_sets=changed, topk=pool_topk, scope='synthetic pool sensitivity; not acceptance')
                        if changed:
                            ranking_changes.append(dict(rows=rows, context=ctx, changed=changed, total=total))
                if timing and stride == 4096:
                    timings(report, 'indexer_head_gate_boundary', rows, graphs, layers=len(cells))
            finally:
                for graph in graphs:
                    graph.reset()
    if ranking_changes:
        raise RuntimeError(f'head-gate synthetic pool sets changed: {ranking_changes}')
