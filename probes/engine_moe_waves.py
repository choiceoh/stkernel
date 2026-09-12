"""Judge v4 resident-wave scheduling on the same real TP4 weight pack.

Invoked by the admitted engine_kernel_check lane. This is a kernel experiment;
it cannot enable a serving spec or supply an engine throughput verdict.
"""
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch


def relative(actual, expected):
    return ((actual.float() - expected.float()).abs().max()
            / expected.float().abs().max().clamp_min(1e-8)).item()


def check(report, ranks):
    from engine.kernels.b12x import moe_dispatch as md
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_decode_fusions import _capture

    if not ranks:
        raise ValueError('the wave probe requires the actual consumer --ranks directory')
    path = Path(ranks) / 'rank0of4.safetensors'
    keys = ['L3.moe.' + name for name in ('w13', 'w13_sf', 'w2', 'w2_sf')]
    loaded = rank_loader(path).load(keys, device='cuda')
    weights = [loaded[key] for key in keys]
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*weights, 8, 10.)
    config = md._parse_glm53_static_v2('t,r,sf6')
    torch.manual_seed(91615)
    x = torch.randn(7, 4096, device='cuda', dtype=torch.bfloat16) * .5
    sel = torch.arange(56, device='cuda', dtype=torch.int32).reshape(7, 8)
    route = torch.rand(7, 8, device='cuda')
    route /= route.sum(-1, keepdim=True)
    trash = torch.empty(64 * 1024**2 // 4, device='cuda')
    graphs, outputs, resources = [], [], []
    try:
        for enabled in (False, True):
            with patch.object(md, '_STATIC_V2_OVERRIDE', dict(config, even=enabled)):
                graph, output = _capture(lambda: lane.moe(x, sel, route, *weights, 10.))
                graphs.append(graph)
                outputs.append(output)
                resources.extend(lane.graph_resources())

        # The same graphs must re-read routing, active expert totals and input
        # values on every replay. M=7/top-k=8 gives 8..56 unique experts.
        for unique in (8, 16, 28, 40, 56, 8):
            sel.copy_((torch.arange(56, device='cuda').reshape(7, 8) % unique).int())
            x.normal_().mul_(.5)
            route.uniform_(.05, 1.)
            route[0, 0] = 0.
            route.div_(route.sum(-1, keepdim=True))
            reference = None
            spreads, errors = [0., 0.], []
            for repeat in range(16):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    outputs[index].fill_(float('nan'))
                    graphs[index].replay()
                if not all(torch.isfinite(value).all().item() for value in outputs):
                    raise RuntimeError('wave graph failed to overwrite every output')
                if reference is None:
                    reference = [value.clone() for value in outputs]
                spreads = [max(previous, relative(value, initial)) for previous, value, initial
                           in zip(spreads, outputs, reference)]
                errors.append(relative(outputs[1], outputs[0]))
            # FP32 atomic scatter can change summation order. Preserve the
            # native MoE repeat/graph tolerance and report the actual spread.
            if max(*spreads, *errors) > .001:
                raise RuntimeError(f'wave numerical difference: U={unique}, spread={spreads}, error={max(errors)}')
            report('moe_waves_numerics', rows=7, unique_experts=unique,
                   repeat_relative=spreads, relative_max=max(errors), repeats=16)

        # All numerical cases precede timing. Evicted samples model a new
        # layer's pack; warm samples expose any extra scheduling overhead.
        for unique in (8, 16, 28, 40, 56):
            sel.copy_((torch.arange(56, device='cuda').reshape(7, 8) % unique).int())
            samples = {name: [] for name in ('warm', 'evicted')}
            for regime in samples:
                for _ in range(4):
                    for label, index in (('B', 0), ('A', 1), ('A', 1), ('B', 0)):
                        if regime == 'evicted':
                            trash.zero_()
                        else:
                            graphs[index].replay()
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record(); graphs[index].replay(); end.record(); end.synchronize()
                        samples[regime].append(dict(arm=label, us=start.elapsed_time(end) * 1000))
            report('moe_waves_timing', rows=7, unique_experts=unique, samples=samples,
                   scope='same L3 TP4 pack and source; FP32 scatter; default off; not engine throughput')
        source = Path(md.__file__)
        report('moe_waves_complete', passed=True, gpu=torch.cuda.get_device_name(),
               dispatcher_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               max_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        for graph in graphs:
            graph.reset()
