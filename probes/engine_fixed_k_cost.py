"""Same-build fixed-K7 gates: resident waves, mHC input pack and split pair MLA.

Use the canonical engine_kernel_check lane fixed_k_compile without GPUs or
fixed_k_cost with a fleet reservation. Component timings are not tok/s proof.
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def compile_check(report):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('compile gate requires CUDA_VISIBLE_DEVICES= and no CUDA context')
    from engine.kernels import dense, mla
    for name, module in (('dense', dense.build()), ('mla', mla._build())):
        report('compile', component=name, module=module.__file__,
               binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', side_effect=lambda module, name, build, **kw: build()):
            for enabled in (False, True):
                cfg = md._static_v2_decode_config(dict(md._parse_glm53_static_v2('t,r,sf6,batch'),
                                                       resident_waves=enabled), 8)
                md._get_static_kernel_v2(288, 288, 8, 4096, 512, 8, 64, config=cfg,
                                        mac_override=48, activation='swigluoai_uninterleave',
                                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                report('compile', component='moe', resident_waves=enabled)
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


def mla_check(report):
    from engine.kernels import mla
    from probes.engine_decode_fusions import _capture
    mla._build()
    cache = torch.randn(32768, 512, device='cuda').to(torch.float8_e4m3fn)
    for rows in (8, 16):
        for width in (33, 2048, 2176):
            q = torch.randn(rows, 16, 512, device='cuda', dtype=torch.bfloat16)
            slots = torch.zeros(rows, width, device='cuda', dtype=torch.int32)
            lens = torch.full((rows,), width, device='cuda', dtype=torch.int32)
            control = lambda: mla.mla_decode(q, cache, slots, lens, 512**-.5, 1., splits=mla.mla_splits(rows))
            candidate = lambda: mla.mla_decode_pair(q, cache, slots, lens, 512**-.5, 1.)
            graphs, outputs = zip(*[_capture(fn) for fn in (control, candidate)])
            try:
                for case in ('identical', 'partial', 'disjoint', 'duplicates', 'empty_tail', 'permuted', 'uneven'):
                    q.normal_()
                    slots.copy_(torch.randint(32768, (rows, width), device='cuda', dtype=torch.int32))
                    lens.fill_(width)
                    if case in ('identical', 'partial', 'duplicates'):
                        shared = width if case != 'partial' else width * 3 // 4
                        slots[1::2, :shared].copy_(slots[::2, :shared])
                    if case == 'permuted':
                        full = width // 16 * 16
                        slots[1::2, :full].copy_(slots[::2, :full].reshape(rows//2, -1, 16).flip(-1).reshape(rows//2, full))
                    if case == 'uneven':
                        slots[1::2].copy_(slots[::2])
                        lens.sub_(torch.arange(rows, device='cuda', dtype=torch.int32) % 8)
                    if case == 'duplicates':
                        slots[:, :width//2].copy_(slots[:, :1].expand(-1, width//2))
                    if case == 'empty_tail':
                        lens[::2] = 0
                        lens[1::2] = 17
                    for _ in range(3):
                        for output, graph in zip(outputs, graphs):
                            output.fill_(float('nan'))
                            graph.replay()
                        # The same independent FP32 reference catches a shared wrong selection.
                        ref = mla.mla_decode_ref(q, cache, slots, lens, 512**-.5, 1.)
                        errors = [mla._rel_err(out, ref) for out in outputs]
                        if max(errors) > .02 or any(not out.isfinite().all().item() for out in outputs):
                            raise AssertionError(f'MLA {rows=} {width=} {case=}: {errors=}')
                    report('numerics', component='mla', rows=rows, width=width, case=case, errors=errors)
                    if width == 2048:
                        timings(report, 'mla', graphs, rows=rows, width=width, case=case)
            finally:
                for graph in graphs:
                    graph.reset()


def mla_profile(report):
    """Diagnostic attribution only, after all unprofiled comparison intervals."""
    from engine.kernels import mla
    from probes.engine_decode_fusions import _capture
    q = torch.randn(8, 16, 512, device='cuda', dtype=torch.bfloat16)
    cache = torch.randn(32768, 512, device='cuda').to(torch.float8_e4m3fn)
    slots = torch.randint(32768, (8, 2048), device='cuda', dtype=torch.int32)
    lens = torch.full((8,), 2048, device='cuda', dtype=torch.int32)
    for case in ('identical', 'partial', 'disjoint'):
        slots.random_(32768)
        shared = {'identical': 2048, 'partial': 1536, 'disjoint': 0}[case]
        slots[1::2, :shared].copy_(slots[::2, :shared])
        graph, output = _capture(lambda: mla.mla_decode_pair(q, cache, slots, lens, 512**-.5, 1.))
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize()
            for event in prof.key_averages():
                total = getattr(event, 'self_device_time_total', 0.) or 0.
                if total > 0:
                    report('profile', component='mla', case=case, kernel=event.key,
                           calls=event.count, total_us=total, diagnostic_only=True)
        finally:
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
        mla_check(report)
        from probes.engine_decode_scatter_check import moe_check
        moe_check(report, ranks, 'moe_resident_waves')
        mla_profile(report)
    report('complete', passed=True, scope='compile only' if compile_only else 'components only; consumer pending')
