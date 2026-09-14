"""Same-build C1 output/query reduction gate; no model boot or consumer verdict."""
import hashlib
import json
from pathlib import Path

import torch

from probes.engine_decode_dsa_inputs import timings
from probes.engine_decode_fusions import _capture


def check(report, ranks=None, *, timing=True):
    from engine.kernels.dense import DenseLinear, extension
    from engine.kernels.dense.query_pair import QueryPair
    keys = ('L0.kda.o_proj', 'L0.mlp.down', 'L3.mla.q_b', 'L3.idx.wq_b')
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loaded = rank_loader(path).load(keys, device='cuda')
        weights = [loaded[k] for k in keys]
        origin = str(path)
    else:
        weights = [(torch.randn(n, k, device='cuda')*.02).bfloat16()
                   for n, k in ((4096, 2048), (4096, 3072), (4096, 1536), (4096, 1536))]
        origin = 'synthetic BF16 weights'
    assert [tuple(w.shape) for w in weights] == [(4096, 2048), (4096, 3072), (4096, 1536), (4096, 1536)]
    report('forward_reduction_weights', source=origin, keys=keys,
           weight_sha256=[hashlib.sha256(w.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for w in weights],
           packing='identical RTN W4 packs; acceptance is not measured')
    owners = [DenseLinear(w, prefill=False) for w in weights]
    ext = extension()
    for owner in owners[:2]:
        p, k = owner.packs[0], owner.cols
        assert ext.gemm2_plan(8, 4096, k)[0] == 3
        for private in (False, True):
            if private:
                owner.isolate_workspace()
            parent = torch.randn(8, k+8, device='cuda', dtype=torch.bfloat16)
            x = parent[:, 4:k+4]
            guard = torch.full((2, 10, 4096), -123., device='cuda', dtype=torch.bfloat16)
            address = torch.tensor([guard[0, 1].data_ptr()], device='cuda', dtype=torch.int64)
            graphs, outputs = [], []
            try:
                # Baseline is the unchanged ordinary GEMM from this extension.
                for rows in ((), (8, 16, 24, 32)):
                    owner.decode_input_rows = rows
                    graph, out = _capture(lambda: owner(x))
                    graphs.append(graph); outputs.append(out)
                direct = _capture(lambda: owner._write_slot(x, address))[0]
                graphs.append(direct)
                for step, magnitude in enumerate((0., .001, .1, 1., 50., 0.)):
                    parent.normal_().mul_(magnitude)
                    for order in ((0, 1, 2), (2, 1, 0)):
                        guard.fill_(-123.)
                        address.fill_(guard[step % 2, 1].data_ptr())
                        for out in outputs:
                            out.fill_(float('nan'))
                        for i in order:
                            graphs[i].replay()
                        assert outputs[1].isfinite().all().item()
                        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                        torch.testing.assert_close(guard[step % 2, 1:-1], outputs[0], rtol=0, atol=0)
                        assert guard[step % 2, (0, -1)].eq(-123.).all().item()
                        assert guard[1-step % 2].eq(-123.).all().item()
                report('forward_reduction_exact', rows=8, n=4096, k=k, private_workspace=private,
                       changed_input=True, rebound_direct_address=True, split=3, replay_orders='BAD/DAB')
                if timing and not private:
                    timings(report, 'forward_cta_reduction', 8, graphs[:2], n=4096, k=k, split=3)
            finally:
                for graph in graphs:
                    graph.reset()
    pair = QueryPair(*owners[2:], rows=(8, 16, 24, 32))
    packs = [owner.packs[0] for owner in owners[2:]]
    for m in (8, 16, 32, 8):
        x = torch.randn(m, 1544, device='cuda', dtype=torch.bfloat16)[:, 4:1540]
        base_out = [torch.empty(m, 4096, device='cuda', dtype=torch.bfloat16) for _ in packs]
        def base():
            ext.run_query_pair(x, [p.data for p in packs], [p.scale for p in packs],
                               [p.rowscale for p in packs], base_out, False)
            return base_out
        graphs, outputs = [], []
        try:
            for fn in (base, lambda: pair(x)):
                graph, out = _capture(fn)
                graphs.append(graph); outputs.append(out)
            for magnitude in (0., .001, 1., 50., 0.):
                x.normal_().mul_(magnitude)
                for order in ((0, 1), (1, 0)):
                    for out in outputs:
                        for y in out:
                            y.fill_(float('nan'))
                    for i in order:
                        graphs[i].replay()
                    for a, b in zip(outputs[1], outputs[0]):
                        assert a.isfinite().all().item()
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            report('query_local_reduction_exact', rows=m, k=1536, replay_orders='BA/AB',
                   same_source_control=True, packed_input_bytes=12672 if m==8 else 50688)
            if timing and m == 8:
                timings(report, 'query_local_reduction', m, graphs, queries=2)
        finally:
            for graph in graphs:
                graph.reset()


def main(ranks=None):
    def report(event, **values):
        print(json.dumps(dict(event=event, **values)), flush=True)
    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/__init__.py',
             'engine/kernels/dense/query_pair.py', 'probes/engine_forward_reduce.py')
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files})
    torch.manual_seed(91424)
    check(report, ranks)
    report('complete', status='PASS', consumer_metrics_measured=False)
