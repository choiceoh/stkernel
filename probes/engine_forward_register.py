"""Same-build register/shared W4 pipeline comparison, without a model boot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from probes.engine_decode_dsa_inputs import timings
from probes.engine_decode_fusions import _capture


def check(report, ranks=None, *, timing=True):
    from engine.kernels.dense import DenseLinear, extension
    keys = ('L0.kda.in_proj', 'L0.mlp.gate_up', 'L3.mla.o_proj',
            'L0.kda.o_proj', 'L0.mlp.down', 'L3.mla.q_b', 'L3.idx.wq_b')
    shapes = ((6416, 4096), (6144, 4096), (4096, 4096), (4096, 2048),
              (4096, 3072), (4096, 1536), (4096, 1536))
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loaded = rank_loader(path).load(keys, device='cuda')
        weights, origin = [loaded[k] for k in keys], str(path)
    else:
        weights = [(torch.randn(n, k, device='cuda')*.02).bfloat16() for n, k in shapes]
        origin = 'synthetic BF16 weights'
    assert [tuple(w.shape) for w in weights] == list(shapes)
    report('weights', source=origin, keys=keys, shapes=shapes,
           sha256=[hashlib.sha256(w.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for w in weights],
           packing='identical RTN W4 packs; no consumer acceptance measurement')
    owners = [DenseLinear(w, prefill=False) for w in weights]
    ext = extension()
    for key, owner in zip(keys[:5], owners[:5]):
        p, n, k = owner.packs[0], owner.rows, owner.cols
        # The ordinary, private and direct entries must consume changed input
        # after the PDL wait, not the input that happened to be present at capture.
        for private in (False, True):
            if private:
                owner.isolate_workspace()
            x = torch.randn(8, k+8, device='cuda', dtype=torch.bfloat16)[:, 4:k+4]
            guard = torch.full((2, 2, 10, n), -123., device='cuda', dtype=torch.bfloat16)
            addresses = [torch.tensor([guard[i, 0, 1].data_ptr()], device='cuda', dtype=torch.int64) for i in (0, 1)]
            outputs = [torch.empty(8, n, device='cuda', dtype=torch.bfloat16) for _ in (0, 1)]
            graphs = []
            try:
                for register in (False, True):
                    y = outputs[int(register)]
                    graphs.append(_capture(lambda: ext.run_gemm_bound_input(
                        x, p.data, p.scale, y, n, p.rowscale.data_ptr(), owner.workspace, None, register))[0])
                if n == 4096:
                    for register in (False, True):
                        address = addresses[int(register)]
                        graphs.append(_capture(lambda: ext.run_gemm_bound_input(
                            x, p.data, p.scale, address, n, p.rowscale.data_ptr(), owner.workspace, address, register))[0])
                for step, magnitude in enumerate((0., .001, .1, 1., 50., 0.)):
                    x.normal_().mul_(magnitude)
                    for order in (range(len(graphs)), reversed(range(len(graphs)))):
                        guard.fill_(-123.)
                        for arm, address in enumerate(addresses):
                            address.fill_(guard[arm, step % 2, 1].data_ptr())
                        for y in outputs:
                            y.fill_(float('nan'))
                        for i in order:
                            graphs[i].replay()
                        assert outputs[1].isfinite().all().item()
                        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                        if n == 4096:
                            for arm in (0, 1):
                                torch.testing.assert_close(guard[arm, step % 2, 1:-1], outputs[0], rtol=0, atol=0)
                                assert guard[arm, step % 2, (0, -1)].eq(-123.).all().item()
                                assert guard[arm, 1-step % 2].eq(-123.).all().item()
                report('register_exact', key=key, rows=8, n=n, k=k, private_workspace=private,
                       plan=ext.gemm2_plan(8, n, k), direct_output=n == 4096,
                       input_stride=x.stride(0), replay_orders='BA/AB', rebound_descriptor=True)
                if timing and not private:
                    timings(report, 'register_w4', 8, graphs[:2], key=key, n=n, k=k)
                    if n == 4096:
                        timings(report, 'register_w4_direct', 8, graphs[2:], key=key, n=n, k=k)
            finally:
                for graph in graphs:
                    graph.reset()
    packs = [owner.packs[0] for owner in owners[5:]]
    for m in (8, 32, 8):
        x = torch.randn(m, 1544, device='cuda', dtype=torch.bfloat16)[:, 4:1540]
        outputs = [[torch.empty(m, 4096, device='cuda', dtype=torch.bfloat16) for _ in packs] for _ in (0, 1)]
        graphs = []
        try:
            for register in (False, True):
                graphs.append(_capture(lambda: ext.run_query_pair(
                    x, [p.data for p in packs], [p.scale for p in packs],
                    [p.rowscale for p in packs], outputs[int(register)], True, register))[0])
            for magnitude in (0., .001, 1., 50., 0.):
                x.normal_().mul_(magnitude)
                for order in ((0, 1), (1, 0)):
                    for arm in outputs:
                        for y in arm:
                            y.fill_(float('nan'))
                    for i in order:
                        graphs[i].replay()
                    for a, b in zip(outputs[1], outputs[0]):
                        assert a.isfinite().all().item()
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            report('register_queries_exact', rows=m, k=1536, replay_orders='BA/AB',
                   control='main CTA-local reduction, same build and packs')
            if timing:
                timings(report, 'register_w4_queries', m, graphs, queries=2)
        finally:
            for graph in graphs:
                graph.reset()


def main(ranks=None):
    def report(event, **values):
        print(json.dumps(dict(event=event, **values)), flush=True)
    root = Path(__file__).resolve().parents[1]
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in (
               'engine/kernels/dense/kernels.cu', 'probes/engine_forward_register.py')})
    torch.manual_seed(91425)
    check(report, ranks)
    report('complete', status='PASS', consumer_metrics_measured=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks')
    main(ap.parse_args().ranks)
