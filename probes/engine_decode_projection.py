"""Same-rank small-projection and serial shared-expert qualification."""
import torch
import torch.nn.functional as F

from probes.engine_decode_capacity import rank_path
from probes.engine_decode_fusions import _capture, _time


def _error(actual, expected):
    if not actual.isfinite().all().item():
        raise RuntimeError('candidate left non-finite output')
    delta = actual.float() - expected.float()
    return (delta.norm() / expected.float().norm().clamp_min(1e-10)).item()


def _timing(report, name, rows, packs, graphs, **extra):
    samples = [dict(arm=arm, ms=_time(graphs[i], iterations=64))
               for arm, i in (('B', 0), ('A', 1), ('A', 1), ('B', 0))]
    report('decode_projection_timing', candidate=name, rows=rows, packs=packs,
           samples=samples, **extra,
           scope='real rank weights; captured component chain, not engine speed or acceptance')


def paired_check(report, ranks):
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_engine_decode_projection')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 2:
        raise RuntimeError('projection and route-reduction numerical gates did not pass')
    from engine.profiles.glm53.weights import rank_loader
    from engine.kernels.decode_projection import KdaPair, IndexerPair
    path = rank_path(ranks)
    loader = rank_loader(path)
    for family, suffixes, cls, expected_count in (
        ('kda_pair', ('.kda.f_b', '.kda.g_b'), KdaPair, 34),
        ('indexer_pair', ('.idx.wk', '.idx.gate'), IndexerPair, 11),
    ):
        keys = sorted(k for k in loader.keys() if k.endswith(suffixes[0]))
        if len(keys) != expected_count:
            raise RuntimeError(f'{family}: expected {expected_count} real projection pairs')
        pairs = [(key, key.removesuffix(suffixes[0]) + suffixes[1]) for key in keys]
        loaded = loader.load([key for pair in pairs for key in pair], device='cuda')
        weights = [(loaded[a], loaded[b]) for a, b in pairs]
        owners = [cls(*pair) for pair in weights]
        cases, graphs = [], []
        try:
            for rows in (1, 6, 7, 14, 21, 28):
                parent = torch.randn(rows, 6416, device='cuda', dtype=torch.bfloat16)
                if family == 'kda_pair':
                    inputs = (parent[:, 6160:6288], parent[:, 6288:6416])
                    base = lambda: [(F.linear(inputs[0], a), F.linear(inputs[1], b)) for a, b in weights]
                else:
                    inputs = (parent[:, :4096],)
                    base = lambda: [(F.linear(inputs[0], a), F.linear(inputs[0], b)) for a, b in weights]
                pair, outputs = [], []
                for fn in (base, lambda: [owner(*inputs) for owner in owners]):
                    graph, output = _capture(fn)
                    pair.append(graph); outputs.append(output); graphs.append(graph)
                errors = []
                for scale in (0., .01, 1., 16., 1.):
                    parent.normal_().mul_(scale)
                    for order in ((0, 1), (1, 0)):
                        for i in order:
                            for values in outputs[i]:
                                for value in values:
                                    value.fill_(float('nan'))
                            pair[i].replay()
                        errors.extend(_error(a, b) for ac, ex in zip(outputs[1], outputs[0]) for a, b in zip(ac, ex))
                if max(errors) > .0005:
                    raise RuntimeError(f'{family}/M{rows}: BF16 same-weight relative error {max(errors)}')
                report('decode_projection_numerics', candidate=family, rows=rows, packs=len(owners),
                       relative_max=max(errors), changed_inputs=True, parent_stride=6416, rank_file=str(path))
                cases.append((rows, pair, parent, inputs, outputs))
            for rows, pair, parent, inputs, outputs in cases:
                _timing(report, family, rows, len(owners), pair,
                        extra_weight_bytes=sum(o.weight.numel() * 2 for o in owners) if family == 'indexer_pair' else 0)
        finally:
            for graph in graphs:
                graph.reset()


def shared_check(report, ranks):
    from engine.profiles.glm53.weights import rank_loader
    from engine.kernels.dense import DenseLinear
    from engine.kernels.dense.shared_mlp import SharedMLP
    from engine.kernels.glm_pointwise import swiglu_clamped
    path = rank_path(ranks)
    loader = rank_loader(path)
    keys = sorted(k for k in loader.keys() if k.endswith('.moe.sh_gate_up'))
    if len(keys) != 42:
        raise RuntimeError('shared serial qualification requires all 42 real layers')
    pairs = [(key, key.removesuffix('sh_gate_up') + 'sh_down') for key in keys]
    loaded = loader.load([key for pair in pairs for key in pair], device='cuda')
    weights = [(DenseLinear(loaded[a], prefill=False), DenseLinear(loaded[b], prefill=False)) for a, b in pairs]
    owners = [SharedMLP(a, b, 10.) for a, b in weights]
    cases, graphs = [], []
    try:
        for rows in (14, 21, 28):
            parent = torch.randn(rows, 4112, device='cuda', dtype=torch.bfloat16)
            x = parent[:, :4096]
            def baseline():
                return [down(swiglu_clamped(*gu(x).chunk(2, -1), 10.)) for gu, down in weights]
            pair, outputs = [], []
            for fn in (baseline, lambda: [owner(x) for owner in owners]):
                graph, output = _capture(fn)
                graphs.append(graph); pair.append(graph); outputs.append(output)
            errors = []
            for scale in (0., .1, 1., 12., 1.):
                parent.normal_().mul_(scale)
                for order in ((0, 1), (1, 0)):
                    for index in order:
                        for output in outputs[index]:
                            output.fill_(float('nan'))
                        pair[index].replay()
                    errors.extend(_error(a, b) for a, b in zip(outputs[1], outputs[0]))
            if max(errors) > .002:
                raise RuntimeError(f'shared serial/M{rows}: same-pack relative error {max(errors)}')
            report('decode_projection_numerics', candidate='shared_serial', rows=rows, packs=len(owners),
                   relative_max=max(errors), changed_inputs=True, rank_file=str(path),
                   weight_packing='same RTN packs on both arms; calibration not exercised')
            cases.append((rows, pair, parent, x, outputs))
        for rows, pair, parent, x, outputs in cases:
            _timing(report, 'shared_serial', rows, len(owners), pair, overlap=False,
                    qualification='existing fusion with real weights; no new fusion claim')
    finally:
        for graph in graphs:
            graph.reset()
