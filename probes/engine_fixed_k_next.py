"""Matched next-step K7 component gates; no production-speed claim.

All selectors are private compile/capture inputs. Serving retains its qualified
paths until component and consumer results are recorded.
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from probes.engine_decode_fusions import _capture, _time
from probes.engine_fixed_k_cost import timings

ROOT = Path(__file__).resolve().parents[1]


def compile_check(report):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('compile-only gate requires no GPU exposure')
    from engine.kernels import dense, mla
    for name, module in (('dense', dense.build()), ('mla', mla._build())):
        report('compile', component=name, module=module.__file__,
               binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    from engine.kernels.router_fused import build
    module = build()
    report('compile', component='router_fused', module=module.__file__,
           binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    if torch.cuda.is_initialized():
        raise RuntimeError('compile gate initialized CUDA')


def mhc_check(report, ranks):
    from engine.kernels.dense import DenseLinear, producer_pack_nbytes
    from engine.kernels.dense.mhc import MHC
    from probes.engine_mhc_c2_packed import Rows, SCALARS, load, compare
    from engine.profiles.glm53.weights import rank_loader
    path, keys, weights, coeff, digest = load(ranks)
    owner = MHC(weights)
    projection_weights = rank_loader(Path(path)).load([f'L{i}.kda.in_proj' for i in (0, 1, 2)], device='cuda')
    projections = [DenseLinear(projection_weights[f'L{i}.kda.in_proj'], prefill=False) for i in (0, 1, 2)]
    for p in projections:
        p.decode_input_rows = (8, 16)
    for rows in (8,):
        data = Rows(rows)
        for packets in (False, True):
            def call(enabled, subset=keys, project=False):
                result = []
                with patch.object(owner, 'EXPAND_FN', enabled):
                    for i, key in enumerate(subset):
                        packed = (torch.empty(producer_pack_nbytes(rows, 4096), device='cuda', dtype=torch.uint8)
                                  if rows == 8 else None)
                        values = owner(key, data.meta if packets else data.x, data.res, data.post, data.comb,
                                       *coeff[key], *SCALARS,
                                       packets=data.descriptor if packets else None, output_pack=packed)
                        result.append((*values, *([packed] if packed is not None else [])))
                        if project:
                            result[-1] += (projections[i](values[-1], producer_pack=packed),)
                return result
            graphs, outputs = zip(*[_capture(lambda enabled=e: call(enabled)) for e in (False, True)])
            try:
                for step, magnitude in enumerate((0., .001, 1., 30.)):
                    data.fill(step, magnitude)
                    for order in (graphs, graphs[::-1]):
                        for arm in outputs:
                            for values in arm:
                                for t in values:
                                    t.fill_(0xa5 if t.dtype == torch.uint8 else float('nan'))
                        for graph in order:
                            graph.replay()
                        for key, want, got in zip(keys, *outputs):
                            compare([want[:4]], [got[:4]], f'mHC coefficient reuse {key}')
                            if rows == 8 and not torch.equal(want[4], got[4]):
                                raise AssertionError(f'{key}: producer pack differs')
                report('numerics', component='mhc_expand_fn', rows=rows, packets=packets,
                       bitwise=True, coefficients=len(keys), magnitudes=4, orders='forward/reverse',
                       weights_sha256=digest)
                samples = []
                for _ in range(3):
                    for arm in (0, 1, 1, 0):
                        samples.append(dict(arm=arm, us=_time(graphs[arm], iterations=8)*1000))
                report('timing', component='mhc_model_chain', rows=rows, packets=packets,
                       boundaries=len(keys), samples=samples)
            finally:
                for graph in graphs:
                    graph.reset()
            if packets:
                subset = [f'L{i}.hc.attn_fn' for i in (0, 1, 2)]
                graphs, outputs = zip(*[_capture(lambda enabled=e: call(enabled, subset, True)) for e in (False, True)])
                try:
                    for step, magnitude in enumerate((0., .001, 1., 30.)):
                        data.fill(step, magnitude)
                        for graph in reversed(graphs):
                            graph.replay()
                        for want, got in zip(*outputs):
                            torch.testing.assert_close(got[-1], want[-1], rtol=0, atol=0)
                    report('numerics', component='mhc_kda_chain', rows=rows, bitwise=True, layers=3)
                    timings(report, 'mhc_kda_chain', graphs, rows=rows, layers=3)
                finally:
                    for graph in graphs:
                        graph.reset()


def mla_check(report):
    from probes.engine_fixed_k_cost import mla_check as check
    check(report, direct_cvt=True)


def run(output, ranks, *, compile_only=False, sections=()):
    sink = open(output, 'w') if output else None
    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink:
            sink.write(line+'\n')
            sink.flush()
    failed = []
    try:
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu_used=not compile_only,
               sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT/'engine/kernels/dense/kernels.cu', ROOT/'engine/kernels/mla/glm53_megakernel.cu',
                                  ROOT/'engine/kernels/b12x/moe_static_kernel_v4.py', ROOT/'engine/kernels/b12x/moe_dispatch.py')})
        if compile_only:
            compile_check(report)
        else:
            torch.manual_seed(91718)
            wanted = set(sections) or {'mla', 'mhc'}
            if wanted - {'mla', 'mhc'}:
                raise ValueError(f'unknown component: {wanted}')
            for name, fn in (('mla', lambda: mla_check(report)), ('mhc', lambda: mhc_check(report, ranks))):
                if name in wanted:
                    try:
                        fn()
                    except Exception as exc:
                        failed.append(name)
                        report('component_failed', component=name, error=f'{type(exc).__name__}: {exc}'[:2000])
        report('complete', passed=not failed, failed=failed, scope='components only; consumer pending')
        if failed:
            raise RuntimeError(f'component gates failed: {failed}')
    finally:
        if sink:
            sink.close()
