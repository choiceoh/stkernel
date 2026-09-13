"""Real-weight decode fusion qualification, admitted through engine_kernel_check.

The ordinary component is only a numerical/timing reference, never a second
consumer boot. Include packing, output conversion and all postops in timing.
"""
import gc
from statistics import median
import torch
import torch.nn.functional as F

from probes.engine_decode_fusions import _capture, _time
from probes.engine_decode_scatter_check import rank_path


def timing(report, name, rows, base, candidate, functions, **extra):
    graphs = [base, candidate]
    # 128 MiB exceeds GB10 L2. Events inside each captured component exclude
    # eviction bandwidth and host enqueue delays from the measured interval.
    cold = torch.empty(128 << 20, dtype=torch.uint8, device='cuda')
    evicted = []
    try:
        for fn in functions:
            start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            def run(fn=fn, start=start, end=end):
                cold.fill_(19)
                start.record()
                result = fn()
                end.record()
                return result
            evicted.append((_capture(run)[0], start, end))
        for cache in ('warm', 'evicted'):
            samples = []
            for label, i in (('B', 0), ('A', 1), ('A', 1), ('B', 0)):
                if cache == 'warm':
                    ms = _time(graphs[i], iterations=64)
                else:
                    graph, start, end = evicted[i]
                    values = []
                    for _ in range(32):
                        graph.replay()
                        end.synchronize()
                        values.append(start.elapsed_time(end))
                    ms = median(values)
                samples.append(dict(arm=label, ms=ms))
            report('decode_batch_timing', candidate=name, rows=rows, cache=cache, samples=samples,
                   eviction_bytes=cold.numel() if cache == 'evicted' else 0,
                   timing_version=2, eviction_in_timing=False,
                   scope='captured same-weight components; eviction outside events; not engine speed', **extra)
    finally:
        for graph, _, _ in evicted:
            graph.reset()


def indexer_check(report, ranks):
    from engine.kernels.decode_projection import IndexerPair, indexer_boundary, DECODE_ROWS
    from engine.kernels.glm_pointwise import layernorm
    from engine.kernels.indexer import head_gate
    from engine.kernels.kpool import fwht128_quant_fp8
    from engine.profiles.glm53.weights import rank_loader
    path = rank_path(ranks)
    loader = rank_loader(path)
    prefixes = sorted(k.removesuffix('wk') for k in loader.keys() if k.endswith('.idx.wk'))
    if len(prefixes) != 11:
        raise RuntimeError('expected all 11 real indexer layers')
    weights = loader.load([n + s for n in prefixes for s in ('wk', 'gate', 'w_heads', 'k_norm_w', 'k_norm_b')], device='cuda')
    cells = [(weights[n+'wk'], weights[n+'gate'], weights[n+'w_heads'], weights[n+'k_norm_w'], weights[n+'k_norm_b']) for n in prefixes]
    owners = [IndexerPair(a, b) for a, b, *_ in cells]
    heads = cells[0][2].shape[0]
    scale = 128 ** -.5 * heads ** -.5
    for rows in DECODE_ROWS:
        parent = torch.full((rows, 6416), float('nan'), dtype=torch.bfloat16, device='cuda')
        x = parent[:, :4096]
        q = torch.randn(rows, heads, 128, dtype=torch.bfloat16, device='cuda')
        x.normal_()
        def base():
            outputs = []
            for wk, gate, wh, nw, nb in cells:
                k = layernorm(F.linear(x, wk), nw, nb, 1e-6)
                w = x.float() @ wh.T
                g = F.linear(x, gate)
                q8, qs = fwht128_quant_fp8(q.reshape(-1, 128))
                outputs.append((q8.view_as(q), k, head_gate(w, qs.view(rows, heads), scale), g))
            return outputs
        def candidate():
            outputs = []
            for owner, (_, _, wh, nw, nb) in zip(owners, cells):
                k, gate = owner(x)
                q8, key, effective = indexer_boundary(q, k, x.float() @ wh.T, nw, nb, scale)
                outputs.append((q8, key, effective, gate))
            return outputs
        graphs, outputs = [], []
        try:
            for fn in (base, candidate):
                graph, output = _capture(fn)
                graphs.append(graph); outputs.append(output)
            maximum = [0.] * 4
            for magnitude in (0., .001, 1., 32., 1.):
                x.normal_().mul_(magnitude); q.normal_().mul_(magnitude)
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        for output in outputs[arm]:
                            for value in output:
                                value.fill_(float('nan'))
                        graphs[arm].replay()
                    for actual, expected in zip(outputs[1], outputs[0]):
                        for i, (got, want) in enumerate(zip(actual, expected)):
                            if not got.float().isfinite().all().item():
                                raise RuntimeError('indexer left a poisoned/nonfinite output')
                            error = ((got.float()-want.float()).norm() / want.float().norm().clamp_min(1e-10)).item()
                            maximum[i] = max(maximum[i], error)
                            if i in (0, 2):
                                torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)
                            elif error > .0005:
                                raise RuntimeError(f'indexer output {i} relative error {error}')
            report('decode_batch_numerics', candidate='indexer_pair_boundary', rows=rows, layers=11,
                   relative_max=maximum, exact_query_and_weights=True, poisoned_replay=True, rank_file=str(path))
            timing(report, 'indexer_pair_boundary', rows, *graphs, (base, candidate), layers=11,
                   resident_bytes=sum(o.weight.numel()*2 for o in owners))
        finally:
            for graph in graphs:
                graph.reset()
    del owners, cells, weights
    gc.collect()


def wide_check(report, ranks):
    from engine.kernels.dense import DenseLinear, W4Pack, w4_gemm, extension
    from engine.profiles.glm53.weights import rank_loader
    path = rank_path(ranks)
    loader = rank_loader(path)
    ext = extension()
    # Cover target W4 projection families once each, with actual rank weights.
    suffixes = ('.kda.in_proj', '.kda.o_proj', '.mla.qkv_a', '.mla.q_b', '.mla.o_proj',
                '.idx.wq_b', '.mlp.gate_up', '.mlp.down')
    keys = [next(k for k in sorted(loader.keys()) if k.endswith(s)) for s in suffixes]
    for key in keys:
        weight = loader.load([key], device='cuda')[key]
        layer = DenseLinear(weight, prefill=False)
        for tile, p in enumerate(layer.packs):
            packs = [W4Pack(p.data.clone(), p.scale.clone(), p.rowscale.clone(), p.rows, p.cols)
                     for _ in range(4)]
            for rows in (9, 14, 16, 21, 28, 32):
                backing = torch.full((rows, p.cols + 8), float('nan'), device='cuda', dtype=torch.bfloat16)
                x = backing[:, 4:4+p.cols]
                x.normal_()
                def candidate():
                    outputs = []
                    for pack in packs:
                        y = torch.empty(rows, pack.rows, dtype=x.dtype, device=x.device)
                        ext.run_gemm_wide_input(x, pack.data, pack.scale, y, pack.rows,
                                                1., 0, pack.rowscale.data_ptr(), 0, 0, 0)
                        outputs.append(y)
                    return outputs
                graphs, outputs = [], []
                base = lambda: [w4_gemm(x, pack) for pack in packs]
                try:
                    for fn in (base, candidate):
                        graph, output = _capture(fn)
                        graphs.append(graph); outputs.append(output)
                    for magnitude in (0., .001, 1., 50., 1.):
                        x.normal_().mul_(magnitude)
                        for order in ((0, 1), (1, 0)):
                            for arm in order:
                                for out in outputs[arm]:
                                    out.fill_(float('nan'))
                                graphs[arm].replay()
                            for got, want in zip(outputs[1], outputs[0]):
                                torch.testing.assert_close(got, want, rtol=0, atol=0)
                    report('decode_batch_numerics', candidate='wide_input', key=key, tile=tile,
                           rows=rows, n=p.rows, k=p.cols, exact=True, poisoned_replay=True, rank_file=str(path))
                    if rows in (14, 21, 28):
                        timing(report, 'wide_input', rows, *graphs, (base, candidate), key=key, tile=tile, n=p.rows, k=p.cols,
                               packs=len(packs), plan=ext.gemm2_plan(rows, p.rows, p.cols))
                finally:
                    for graph in graphs:
                        graph.reset()
            del packs
        del layer, weight
        gc.collect()
