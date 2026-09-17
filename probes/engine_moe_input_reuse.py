"""Reuse token input quantization while preserving every expert's scale contract."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch

def compile_check(report):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('compile-only gate requires no GPU exposure')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', side_effect=lambda module, name, build, **kw: build()):
            for rows in (8, 16):
                for mode in (0, 1, 2, 3):
                    cfg = dict(md._parse_glm53_static_v2('t,r,sf6,batch'), input_vec16=True, input_reuse=mode)
                    md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, rows*8, config=cfg,
                                            mac_override=48, w13_chunk=256,
                                            activation='swigluoai_uninterleave',
                                            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                    report('compile', rows=rows, input_reuse=mode)
    if torch.cuda.is_initialized():
        raise RuntimeError('compile gate initialized CUDA')


def moe_frontend_check(report, ranks):
    """Compare registered route bytes before blaming downstream MMA arithmetic."""
    from probes import engine_moe_c2_cells as cells
    from engine.profiles.glm53.weights import rank_loader
    from engine.profiles.glm53.lanes import served
    from engine.kernels.b12x import moe_dispatch as md
    loader = rank_loader(Path(ranks))
    layer = cells.Layer(loader, set(loader.keys()), 3,
                        served(moe_static='t,r,sf6,batch,q0'), (256,))
    get_kernel = md._get_static_kernel_v2
    captured = []
    class Observe:
        def __init__(self, kernel): self.kernel = kernel
        def __getattr__(self, name): return getattr(self.kernel, name)
        def __call__(self, *args):
            args[5].fill_(0xa5)
            args[6].fill_(0xa5)
            result = self.kernel(*args)
            captured.append(args)
            return result
    def observe(*args, **kwargs):
        kernel, mac = get_kernel(*args, **kwargs)
        return Observe(kernel), mac
    def snapshot(args):
        active = int(args[14].item())
        counts, experts, tokens = (args[i].cpu() for i in (13, 15, 22))
        rows = tokens.shape[1]
        packed = args[5].view(288, rows, 2048).cpu()
        scale = args[6].view(288, -1).cpu()
        block = torch.arange(256)
        result = {}
        for local in range(active):
            for row in range(int(counts[local])):
                token = int(tokens[local, row])
                offsets = ((row // 128) * 32768 + (block // 4) * 512
                           + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4)
                result[(int(experts[local]), token)] = (packed[local, row].clone(), scale[local, offsets].clone())
        return result
    original = layer.q13
    for scale_case in ('uniform', 'uniform_unique', 'uniform_shared', 'uniform_nonunit', 'mixed', 'distinct', 'zero'):
        values = (torch.ones(288, device='cuda') if scale_case in ('uniform', 'uniform_unique', 'uniform_shared') else
                  torch.full((288,), .37, device='cuda') if scale_case == 'uniform_nonunit' else
                  torch.arange(288, device='cuda').remainder(4).float() * .31 if scale_case == 'mixed' else
                  torch.linspace(.01, 1.7, 288, device='cuda') if scale_case == 'distinct' else
                  torch.zeros(288, device='cuda'))
        layer.q13 = values
        for rows in (8, 16):
            fx = cells.Fixtures([layer], rows)
            fx.load(('independent', rows, rows, 1.), 41)
            if scale_case == 'uniform_unique':
                fx.ids[0].copy_(torch.arange(288 - rows * 8, 288, device='cuda').view(rows, 8))
            elif scale_case == 'uniform_shared':
                fx.ids[0].copy_(torch.arange(8, device='cuda').expand(rows, 8))
            snapshots = {}
            for mode in (0, 1, 2, 3):
                cfg = dict(md._parse_glm53_static_v2('t,r,sf6,batch'), input_vec16=True, input_reuse=mode)
                with patch.object(md, '_STATIC_V2_OVERRIDE', cfg), patch.object(md, '_get_static_kernel_v2', observe):
                    layer.moe(256, fx.x, fx.ids[0], fx.routes[0])
                torch.cuda.synchronize()
                snapshots[mode] = snapshot(captured[-1])
                captured.clear()
            for mode in (1, 2, 3):
                base, got = snapshots[0], snapshots[mode]
                if base.keys() != got.keys():
                    raise AssertionError(f'{scale_case} rows={rows} mode={mode}: routed rows changed')
                changes = {str(key): [int((a != b).sum()) for a, b in zip(base[key], got[key])]
                           for key in base if any(not torch.equal(a, b) for a,b in zip(base[key], got[key]))}
                report('frontend_bytes', rows=rows, scale_case=scale_case, mode=mode,
                       routes=len(base), unique_experts=len({expert for expert, _ in base}),
                       mismatched_routes=changes)
                if changes:
                    raise AssertionError(f'{scale_case} rows={rows} mode={mode}: packed inputs changed')
    layer.q13 = original


def moe_check(report, ranks):
    from probes import engine_moe_c2_cells as cells
    from engine.profiles.glm53.weights import rank_loader
    from engine.profiles.glm53.lanes import served
    from engine.kernels.b12x import moe_dispatch as md
    path = Path(ranks)
    if path.suffix != '.safetensors':
        path = path / 'rank0of4.safetensors'
    loader = rank_loader(path)
    lane = served(moe_static='t,r,sf6,batch,q0')
    chunk = md._w13_tile_chunk()
    layers = [cells.Layer(loader, set(loader.keys()), i, lane, (chunk,)) for i in (3, 4, 5)]
    report('weights', component='moe', layers=[layer.identity for layer in layers],
           allocated_bytes=torch.cuda.memory_allocated())
    spread = cells.calibrate(layers, report)
    base = md._parse_glm53_static_v2('t,r,sf6,batch')
    candidates = ['reuse', 'striped', 'routes']
    arms = [(name, chunk, dict(base, input_vec16=True, input_reuse=mode))
            for name, mode in (('base', 0), ('reuse', 1), ('striped', 2), ('routes', 3), ('repeat', 0))]
    failures = []
    for rows in (8, 16):
        fixtures = [(f'c{rows//8}_requests', rows, rows//8, spread),
                    (f'm{rows}_independent', rows, rows, 1.), (f'm{rows}_shared', rows, 1, 0.)]
        fx = cells.Fixtures(layers, rows)
        fx.load(fixtures[0], 1)
        for scope, group in (('single', layers[:1]), ('chain', layers)):
            graphs, accs = cells.capture_arms(group, fx, arms)
            try:
                failed = cells.exact_arms(report, group, fx, fixtures, graphs, accs,
                                          'base', 'repeat', candidates, scope=scope)
                failures.extend(failed)
                if not failed:
                    uniques = fx.load(fixtures[0], 7)[:len(group)]
                    for candidate in candidates:
                        cells.bracket(report, graphs, 'base', candidate, brackets=5,
                                      fixture=fixtures[0][0], rows=rows, scope=scope,
                                      layers=len(group), unique_experts=uniques)
            finally:
                for graph in graphs.values():
                    graph.reset()
        cells.stamp_cells(report, layers, arms[:-1], spread, rows=rows)
    if failures:
        raise RuntimeError(f'MoE input reuse failed: {failures}')



def run(output, ranks, *, compile_only=False):
    root = Path(__file__).resolve().parents[1]
    sink = open(output, 'w') if output else None
    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink:
            sink.write(line+'\n'); sink.flush()
    try:
        sources = ['engine/kernels/b12x/moe_static_kernel_v4.py', 'engine/kernels/b12x/moe_dispatch.py']
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu_used=not compile_only,
               sources={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources})
        if compile_only:
            compile_check(report)
        else:
            torch.manual_seed(91731)
            path = Path(ranks)
            if path.is_dir():
                ranks = str(path / 'rank0of4.safetensors')
            report('device', name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
            moe_frontend_check(report, ranks)
            moe_check(report, ranks)
        report('complete', passed=True, scope='component numerics and latency; consumer pending')
    except Exception as exc:
        report('complete', passed=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if sink: sink.close()
