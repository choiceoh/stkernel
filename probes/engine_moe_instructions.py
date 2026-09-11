"""Frozen static versus max3 instruction change on actual TP=4 L3 rank weights.

Uses the current declared stock lane with FP32 scatter, identical prepared weights, and independent
compiled kernels. AB/BA cycles balance launch order. Kernel-cache bypass is
local to this probe so a baseline can never load a candidate artifact.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from engine.profiles.glm53.lanes import served, MOE_STATIC_STOCK
from engine.profiles.glm53.weights import rank_loader
from engine.kernels.b12x import moe_dispatch as md
from engine_fp4_instructions import paired_timing


def baseline_class(directory):
    def load(file, name):
        spec = importlib.util.spec_from_file_location('engine.kernels.b12x.'+name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return (load(directory/'moe_static_kernel.py', '_instructions_baseline_static').MoEStaticKernel,
            load(directory/'moe_micro_kernel.py', '_instructions_baseline_micro').MoEMicroKernel)


def relative(a, b):
    return ((a.float()-b.float()).abs().max()/b.float().abs().max().clamp_min(1e-8)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--baseline-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, nargs='+', default=[1, 4, 6, 12])
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    keys = ['L3.moe.'+key for key in ('w13', 'w13_sf', 'w2', 'w2_sf')]
    tensors = rank_loader(args.rank).load(keys, device='cuda')
    weights = [tensors[k] for k in keys]
    baseline, candidate = baseline_class(args.baseline_dir), (md.MoEStaticKernel, md.MoEMicroKernel)
    lane = served(moe_static=MOE_STATIC_STOCK)
    lane.moe_prepare(*weights, 8, 10.)
    caches = [{}, {}]
    micro_caches = [{}, {}]
    kernels = []
    original_compile = md.cute.compile
    def compile_with_dump(*a, **kw):
        kw['options'] = kw.get('options', '')+' --keep-ptx --keep-cubin'
        return original_compile(*a, **kw)
    def builder(module, name, build, **kw):
        with patch.object(md.cute, 'compile', compile_with_dump):
            compiled = build()
        kernels.append(dict(name=name, variant=active_variant[0]))
        dump = args.out.parent/'compiled'
        dump.mkdir(exist_ok=True)
        ptx = getattr(compiled, '__ptx__', None)
        cubin = getattr(compiled, '__cubin__', None)
        if isinstance(ptx, str):
            (dump/f'{active_variant[0]}-{name}.ptx').write_text(ptx)
        if isinstance(cubin, bytes):
            (dump/f'{active_variant[0]}-{name}.cubin').write_bytes(cubin)
        return compiled
    active_variant = [0]
    cases = []
    trash = torch.empty(64*1024**2//4, device='cuda')
    with patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
        for tokens in args.tokens:
            for routing in ('shared', 'mixed'):
                torch.manual_seed(932+tokens)
                x = torch.randn(tokens, 4096, device='cuda', dtype=torch.bfloat16)*.5
                if routing == 'shared':
                    sel = torch.arange(8, device='cuda', dtype=torch.int32).repeat(tokens, 1)
                else:
                    sel = (torch.arange(tokens*8, device='cuda', dtype=torch.int32).reshape(tokens, 8)*5)%288
                route = torch.rand(tokens, 8, device='cuda')
                route[0, 0] = 0.
                route /= route.sum(-1, keepdim=True)
                def call():
                    return lane.moe(x, sel, route, *weights, 10.)
                graphs, repeats, outputs = {}, [], []
                for variant, cls in enumerate((baseline, candidate)):
                    active_variant[0] = variant
                    with patch.object(md, 'MoEStaticKernel', cls[0]), \
                         patch.object(md, 'MoEMicroKernel', cls[1]), \
                         patch.object(md, '_MICRO_KERNEL_CACHE', micro_caches[variant]), \
                         patch.object(md, '_STATIC_KERNEL_CACHE', caches[variant]):
                        call()
                        runs = torch.stack([call().clone() for _ in range(16)])
                        outputs.append(runs[0])
                        repeats.append(relative(runs, runs[0]))
                        for count in (1, 8):
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(count):
                                    output = call()
                            output.fill_(float('nan'))
                            graph.replay()
                            assert torch.isfinite(output).all(), 'capture must execute'
                            graphs[variant, count] = (graph, output)
                error = relative(outputs[1], outputs[0])
                # FP32 atomic scatter is order-dependent. Report baseline and
                # candidate repeat spread instead of claiming bitwise MoE equality.
                assert error <= .001 and max(repeats) <= .001, (error, repeats)
                samples = {regime: [[], []] for regime in ('warm', 'evicted')}
                for round in range(12):
                    for regime, count in (('warm', 8), ('evicted', 1)):
                        for variant in ((0, 1) if round%2 == 0 else (1, 0)):
                            graph, _ = graphs[variant, count]
                            if regime == 'warm':
                                graph.replay()
                            else:
                                trash.zero_()
                            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            start.record(); graph.replay(); end.record(); end.synchronize()
                            samples[regime][variant].append(start.elapsed_time(end)*1000/count)
                graph_errors = []
                for (variant, count), (graph, output) in graphs.items():
                    err = relative(output, outputs[variant])
                    assert err <= .001, err
                    graph_errors.append(err)
                    graph.reset()
                row = dict(tokens=tokens, routing=routing, relative_max=error,
                           bits_exact=torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8)),
                           repeat_relative=repeats, graph_relative=graph_errors,
                           samples_us=samples, paired={k: paired_timing(v) for k,v in samples.items()})
                cases.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}), flush=True)
    files = [args.baseline_dir/'moe_static_kernel.py', args.baseline_dir/'moe_micro_kernel.py',
             Path('engine/kernels/b12x/moe_static_kernel.py'), Path('engine/kernels/b12x/moe_micro_kernel.py'),
             Path('engine/kernels/b12x/fp4_quant.py')]
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, lane=MOE_STATIC_STOCK,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  kernels=kernels, cases=cases, peak_allocated=torch.cuda.max_memory_allocated())
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
