"""Same-build fixed-K7 gate for the selected mHC input-pack fusion.

Use the canonical engine_kernel_check lane fixed_k_compile without GPUs or
fixed_k_cost with a fleet reservation. Component timings are not tok/s proof.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def compile_check(report):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('compile gate requires CUDA_VISIBLE_DEVICES= and no CUDA context')
    from engine.kernels import dense
    module = dense.build()
    report('compile', component='dense', module=module.__file__,
           binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    if torch.cuda.is_initialized():
        raise RuntimeError('compile gate touched a GPU')


def timings(report, component, graphs, **meta):
    from probes.engine_decode_fusions import _time
    samples = []
    for _ in range(3):
        for arm in (0, 1, 1, 0):
            samples.append(dict(arm=arm, us=_time(graphs[arm], iterations=64) * 1000))
    report('timing', component=component, samples=samples, **meta)


def mhc_check(report, ranks):
    from engine.kernels.dense import producer_pack_nbytes
    from engine.kernels.dense.mhc import MHC
    from probes.engine_mhc_c2_packed import load, Rows, SCALARS, compare
    from probes.engine_decode_fusions import _capture
    path, keys, weights, coeff, digest = load(ranks)
    owner = MHC(weights)
    data = Rows(8)
    # Every real coefficient set is checked; timing uses one complete boundary.
    for key in keys:
        for packets in (False, True):
            packed = torch.empty(producer_pack_nbytes(8, 4096), dtype=torch.uint8, device='cuda')
            def call(fused):
                values = owner(key, data.meta if packets else data.x, data.res, data.post, data.comb,
                               *coeff[key], *SCALARS, packets=data.descriptor if packets else None,
                               output_pack=packed if fused else None)
                if not fused:
                    owner.ext.run_input_pack(values[-1], packed)
                return values, packed.clone()
            graphs, outputs = zip(*[_capture(lambda fused=fused: call(fused)) for fused in (False, True)])
            try:
                for step, scale in enumerate((0., .001, 1., 30.)):
                    data.fill(step, scale)
                    for graph in reversed(graphs):
                        graph.replay()
                    compare([outputs[0][0]], [outputs[1][0]], 'mHC pack fusion')
                    if not torch.equal(outputs[0][1], outputs[1][1]):
                        raise AssertionError(f'{key}: fused input pack bytes differ')
                report('numerics', component='mhc', key=key, packets=packets, bitwise=True,
                       magnitudes=4, changed_descriptor=True, rank=path, weights_sha256=digest)
                if key == keys[0]:
                    timings(report, 'mhc_pack_boundary', graphs, packets=packets)
            finally:
                for graph in graphs:
                    graph.reset()
    # Distinct real coefficients exceed L2 and expose the production weight
    # stream. Single-layer hot replay alone cannot judge this boundary.
    from probes.engine_decode_fusions import _time
    for packets in (False, True):
        pack = torch.empty(producer_pack_nbytes(8, 4096), dtype=torch.uint8, device='cuda')
        def chain(fused):
            for key in keys:
                values = owner(key, data.meta if packets else data.x, data.res, data.post, data.comb,
                               *coeff[key], *SCALARS, packets=data.descriptor if packets else None,
                               output_pack=pack if fused else None)
                if not fused:
                    owner.ext.run_input_pack(values[-1], pack)
            return values
        graphs, outputs = zip(*[_capture(lambda fused=fused: chain(fused)) for fused in (False, True)])
        try:
            samples = []
            for _ in range(3):
                for arm in (0, 1, 1, 0):
                    samples.append(dict(arm=arm, us=_time(graphs[arm], iterations=8)*1000))
            report('timing', component='mhc_model_chain', packets=packets, boundaries=len(keys),
                   samples=samples, weights_sha256=digest, scope='distinct real coefficients; component only')
        finally:
            for graph in graphs:
                graph.reset()


def run(output=None, ranks=None, *, compile_only=False):
    records = []
    def report(kind, **values):
        row = dict(kind=kind, **values)
        records.append(row)
        print(json.dumps(row), flush=True)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(''.join(json.dumps(item) + '\n' for item in records))
    report('identity', gpu_used=not compile_only, torch=torch.__version__, cuda=torch.version.cuda,
           sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT/'engine/kernels/dense/kernels.cu', ROOT/'engine/kernels/mla/glm53_megakernel.cu',
                              ROOT/'engine/kernels/b12x/moe_static_kernel_v4.py', ROOT/'engine/kernels/b12x/moe_dispatch.py')})
    if compile_only:
        compile_check(report)
    else:
        torch.manual_seed(91717)
        mhc_check(report, ranks)
    report('complete', passed=True, scope='compile only' if compile_only else 'components only; consumer pending')
