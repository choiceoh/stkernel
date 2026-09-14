"""Same-weight MLA decode contractions including the consumers' layout copies."""
import hashlib

import torch

from probes.engine_decode_fusions import _capture
from probes.engine_decode_dsa_inputs import ROWS, timings


def check(report, ranks=None, *, layers=11, timing=True):
    from engine.kernels.mla.decode_absorb import DecodeAbsorb
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loader = rank_loader(path)
        names = sorted(k for k in loader.keys() if k.endswith('.mla.kv_b'))
        if len(names) != 11:
            raise RuntimeError('decode absorb gate requires all 11 real DSA weights')
        loaded = loader.load(names, device='cuda')
        weights = [loaded[n].view(16, 512, 512) for n in names]
        source = str(path)
    else:
        weights = [(torch.randn(16, 512, 512, device='cuda')*.02).bfloat16() for _ in range(layers)]
        source = 'synthetic BF16 weights'
    owners = [DecodeAbsorb(w[:, :256], w[:, 256:], rows=ROWS) for w in weights]
    report('decode_absorb_weights', source=source, layers=len(weights), resident_repack_bytes=0,
           sha256=[hashlib.sha256(w.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for w in weights])
    for m in ROWS:
        q = torch.randn(m, 16, 256, dtype=torch.bfloat16, device='cuda')
        context = torch.randn(m, 16, 512, dtype=torch.bfloat16, device='cuda')
        graphs, outputs = [], []
        def base():
            return [(torch.einsum('thd,hdc->thc', q, w[:, :256]).contiguous(),
                     torch.einsum('thc,hvc->thv', context, w[:, 256:]).reshape(m, 4096)) for w in weights]
        def candidate():
            return [(owner(q), owner(context, transpose=True).reshape(m, 4096)) for owner in owners]
        try:
            for fn in (base, candidate):
                graph, out = _capture(fn)
                graphs.append(graph); outputs.append(out)
            errors = [0., 0.]
            for factor in (0., .01, 1., -8., 1.):
                q.normal_().mul_(factor); context.normal_().mul_(factor)
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        for pair in outputs[arm]:
                            for t in pair:
                                t.fill_(float('nan'))
                        graphs[arm].replay()
                    for actual, expected in zip(outputs[1], outputs[0]):
                        for side, (got, want) in enumerate(zip(actual, expected)):
                            if not got.isfinite().all().item():
                                raise RuntimeError('decode absorb left poisoned/nonfinite output')
                            if not got.is_contiguous() or got.reshape(m, -1).data_ptr() != got.data_ptr():
                                raise RuntimeError('decode absorb consumer still needs a layout copy')
                            error = ((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12)).item()
                            errors[side] = max(errors[side], error)
                            if error > .0005:
                                raise RuntimeError(f'decode absorb side {side} relative L2 error {error} > 0.0005')
            previous = [tuple(t.clone() for t in pair) for pair in outputs[1]]
            for _ in range(10):
                graphs[1].replay()
                for actual, expected in zip(outputs[1], previous):
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
            report('decode_absorb_numerics', rows=m, layers=len(weights), relative_l2_max=errors,
                   replay_orders='BA/AB', deterministic_replays=10, token_major=True, consumer_copies=0,
                   weight_layout='original kv_b views, including output storage offset')
            if timing:
                timings(report, 'decode_absorb_with_consumer_layout', m, graphs, layers=len(weights),
                       tile_m=16 if m <= 16 else 32, baseline_includes_layout_copies=True)
        finally:
            for graph in graphs:
                graph.reset()
