"""Native compile, numerical and real-weight cost gates for FC2 scale search."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint8, Uint64
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from flashinfer.cute_dsl.fp4_common import quantize_block_fp4_fast
from engine.kernels.b12x.fp4_quant import max_abs_16
from engine.kernels.b12x.fp4_scale_search import quantize_block_fp4_search


class Pack:
    def __init__(self, radius):
        self.radius = radius

    @cute.jit
    def __call__(self, x: cute.Tensor, gs: cute.Tensor, out: cute.Tensor,
                 scales: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, gs, out, scales).launch(
            grid=(cute.ceil_div(x.shape[0], 128), 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, gs: cute.Tensor, out: cute.Tensor, scales: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = bid * 128 + tid
        if row < x.shape[0]:
            values = cute.make_rmem_tensor((16,), Float32)
            for j in cutlass.range_constexpr(16):
                values[j] = x[row, j]
            mx = max_abs_16(values)
            if cutlass.const_expr(self.radius < 0):
                packed, scale = quantize_block_fp4_fast(values, mx, gs[row])
            else:
                packed, scale = quantize_block_fp4_search(values, mx, gs[row], self.radius)
            out[row], scales[row] = packed, scale


def save_binary(fn, path, report, **meta):
    path.with_suffix('.ptx').write_text(fn.__ptx__)
    path.with_suffix('.cubin').write_bytes(fn.__cubin__)
    result = subprocess.run(['cuobjdump', '-res-usage', str(path.with_suffix('.cubin'))],
                            capture_output=True, text=True, check=True)
    report('compile', cubin_bytes=len(fn.__cubin__), resources=result.stdout, **meta)


def compile_check(report, dump):
    if os.environ.get('ST_PROBE_NO_GPU') != '1' or torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError('compile-only gate requires no GPU exposure')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    fake = lambda dtype, shape: cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16)
    args = (fake(Float32, (256, 16)), fake(Float32, (256,)),
            fake(Uint64, (256,)), fake(Uint8, (256,)))
    for radius in (-1, 0, 1, 2):
        fn = cute.compile(Pack(radius), *args, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                          options='--enable-tvm-ffi --keep-ptx --keep-cubin')
        save_binary(fn, dump / f'quant-{radius}', report, component='quant', radius=radius)
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        original_compile = cute.compile
        def keep(*args, **kwargs):
            kwargs['options'] = kwargs.get('options', '') + ' --keep-ptx --keep-cubin'
            return original_compile(*args, **kwargs)
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', side_effect=lambda module, name, build, **kw: build()), \
                patch.object(cute, 'compile', side_effect=keep):
            assert md._parse_glm53_static_v2('t,r,sf6,batch')['fc2_scale_search'] == 0
            for invalid in ('ss0', 'ss3', 'ss-1'):
                try:
                    md._parse_glm53_static_v2('t,r,sf6,batch,' + invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'accepted invalid search radius {invalid}')
            for rows in (8, 16, 32):
                before = len(md._STATIC_V2_KERNEL_CACHE)
                for radius in (0, 1, 2):
                    spec = 't,r,sf6,batch' + (f',ss{radius}' if radius else '')
                    cfg = md._parse_glm53_static_v2(spec)
                    fn, _ = md._get_static_kernel_v2(
                        288, 288, rows, 4096, 512, 8, rows * 8, config=cfg,
                        mac_override=48, w13_chunk=256, activation='swigluoai_uninterleave',
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                    save_binary(fn, dump / f'moe-m{rows}-s{radius}', report,
                                component='moe', rows=rows, radius=radius)
                assert len(md._STATIC_V2_KERNEL_CACHE) == before + 3, 'search cache aliases baseline'
    assert not torch.cuda.is_initialized()


def decode(packed, scales, gs):
    # Independent torch decoder, including the sign bit and FP8 scale bytes.
    nibbles = ((packed.view(torch.int64)[:, None] >>
                (4 * torch.arange(16, device=packed.device))) & 15)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         dtype=torch.float64, device=packed.device)
    return table[nibbles] * scales.view(torch.float8_e4m3fn).double()[:, None] * gs.double()[:, None]


def quant_check(report):
    from probes.engine_fp4_instructions import adversarial_inputs
    x, gs = adversarial_inputs()
    # Numerical SSE checks on realistic normal inputs and activation tails.
    torch.manual_seed(917)
    gate, up = [torch.randn(8192, 16, device='cuda') * 3 for _ in range(2)]
    real = torch.cat((torch.randn_like(gate), torch.nn.functional.silu(gate) * up))
    x = torch.cat((x, real))
    gs = torch.cat((gs, torch.exp2(torch.randint(-8, 8, (real.shape[0],), device='cuda').float())))
    compiled = {}
    for radius in (-1, 0, 1, 2):
        out = torch.empty(x.shape[0], dtype=torch.uint64, device='cuda')
        scale = torch.empty(x.shape[0], dtype=torch.uint8, device='cuda')
        args = tuple(from_dlpack(t, assumed_align=16) for t in (x, gs, out, scale))
        fn = cute.compile(Pack(radius), *args, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                          options='--enable-tvm-ffi')
        compiled[radius] = (fn, out, scale)
    for mode in ('normal', 'zero', 'negative', 'infinite', 'nan'):
        original_gs = gs.clone()
        if mode != 'normal':
            gs.fill_({'zero': 0., 'negative': -1., 'infinite': float('inf'), 'nan': float('nan')}[mode])
        for radius, (fn, out, scale) in compiled.items():
            fn(x, gs, out, scale)
        base_out, base_sf = compiled[-1][1:]
        for radius, (fn, out, scale) in compiled.items():
            if radius <= 0 or mode != 'normal':
                assert torch.equal(base_out.view(torch.uint8), out.view(torch.uint8))
                assert torch.equal(base_sf, scale)
                report('quant_parity', radius=radius, mode=mode, blocks=len(x), passed=True)
                continue
            finite = torch.isfinite(x).all(-1)
            expected_base = decode(base_out, base_sf, gs)
            expected = decode(out, scale, gs)
            base_error = (expected_base - x.double()).square().sum(-1)
            error = (expected - x.double()).square().sum(-1)
            worse = finite & (error > base_error * (1 + 2e-5) + 1e-300)
            if bool(worse.any()):
                ids = worse.nonzero().flatten()[:8]
                report('quant_failure', radius=radius, inputs=x[ids].tolist(), gs=gs[ids].tolist(),
                       base=base_error[ids].tolist(), candidate=error[ids].tolist())
                raise AssertionError(f'{int(worse.sum())} blocks have worse reconstruction SSE')
            assert torch.equal(out.view(torch.int64)[~finite], base_out.view(torch.int64)[~finite])
            assert torch.equal(scale[~finite], base_sf[~finite])
            subset = slice(len(x) - len(real), len(x))
            report('quant_numeric', radius=radius, checked_finite_blocks=int(finite.sum()),
                   changed_scales=int((scale != base_sf).sum()),
                   synthetic_mse_reduction_pct=100 * float(1 - error[subset].sum() / base_error[subset].sum()),
                   worsened_blocks=int(worse.sum()))
            expected_out, expected_sf = out.clone(), scale.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(x, gs, out, scale)
            out.fill_(42); scale.fill_(42)
            graph.replay()
            assert torch.equal(out.view(torch.uint8), expected_out.view(torch.uint8))
            assert torch.equal(scale, expected_sf)
            graph.reset()
        gs.copy_(original_gs)


def moe_check(report, ranks):
    from probes import engine_moe_c2_cells as cells
    from engine.profiles.glm53.weights import rank_loader
    from engine.profiles.glm53.lanes import served
    from engine.kernels.b12x import moe_dispatch as md
    loader = rank_loader(Path(ranks))
    chunk = md._w13_tile_chunk()
    layer = cells.Layer(loader, set(loader.keys()), 3, served(moe_static='t,r,sf6,batch,q0'), (chunk,))
    report('weights', **layer.identity)
    spread = cells.calibrate([layer], report)
    base = md._parse_glm53_static_v2('t,r,sf6,batch')
    arms = [(label, chunk, dict(base, fc2_scale_search=radius))
            for label, radius in (('base', 0), ('ss1', 1), ('ss2', 2), ('repeat', 0))]
    for rows in (8, 16):
        fx = cells.Fixtures([layer], rows)
        fixture = (f'c{rows // 8}_requests', rows, rows // 8, spread)
        fx.load(fixture, 917)
        graphs, accs = cells.capture_arms([layer], fx, arms)
        try:
            for seed in (917, 918, 919):
                fx.load(fixture, seed)
                for label in graphs:
                    accs[label][0].fill_(float('nan'))
                    graphs[label].replay()
                    assert bool(torch.isfinite(accs[label][0]).all()), label
                reference = accs['base'][0].double()
                noise = cells.fp32_noise(accs['repeat'][0], accs['base'][0])
                assert noise['fp32_max_ulps'] <= cells.MAX_ULPS
                for label in ('ss1', 'ss2'):
                    candidate = accs[label][0].double()
                    report('moe_difference_not_quality', rows=rows, seed=seed, arm=label,
                           relative_l2=float((candidate-reference).norm() / reference.norm()),
                           max_abs=float((candidate-reference).abs().max()), control_noise=noise)
            fx.load(fixture, 917)
            for label in ('ss1', 'ss2'):
                cells.bracket(report, graphs, 'base', label, brackets=4, rows=rows,
                              fixture=fixture[0], layer=3)
            fx.routes[0].zero_()
            for label, graph in graphs.items():
                accs[label][0].fill_(float('nan'))
                graph.replay()
                assert int(torch.count_nonzero(accs[label][0])) == 0, label
            report('zero_routes', rows=rows, passed=True)
        finally:
            for graph in graphs.values():
                graph.reset()


def run(output, ranks, *, compile_only=False, quant_only=False):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump = output.parent / 'scale-search-native'
    dump.mkdir(exist_ok=True)
    with output.open('w') as sink:
        def report(event, **values):
            line = json.dumps(dict(event=event, **values))
            print(line, flush=True)
            sink.write(line + '\n'); sink.flush()
        paths = ['engine/kernels/b12x/fp4_scale_search.py', 'engine/kernels/b12x/moe_static_kernel_v4.py',
                 'engine/kernels/b12x/moe_dispatch.py', 'probes/engine_fp4_scale_search.py']
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu_used=not compile_only,
               source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
        if compile_only:
            compile_check(report, dump)
        else:
            torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
            report('device', name=torch.cuda.get_device_name())
            quant_check(report)
            if not quant_only:
                moe_check(report, ranks)
            report('memory', peak_bytes=torch.cuda.max_memory_allocated())
        report('verdict', passed=True)
