"""Bounded same-pack MoE and router candidates; no serving dispatch changes."""
from pathlib import Path
from unittest.mock import patch

import torch

from probes.engine_decode_fusions import _capture, _time
from probes.engine_moe_waves import relative


def rank_path(ranks):
    from engine.profiles.glm53 import facts
    if not ranks:
        raise ValueError('capacity probes require the exact consumer --ranks')
    root = Path(ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent / root
    return root / 'rank0of4.safetensors'


def moe_check(report, ranks, lane_name):
    from engine.kernels.b12x import moe_dispatch as md
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.weights import rank_loader
    path = rank_path(ranks)
    keys = ['L3.moe.' + name for name in ('w13', 'w13_sf', 'w2', 'w2_sf')]
    loaded = rank_loader(path).load(keys, device='cuda')
    weights = [loaded[key] for key in keys]
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*weights, 8, 10.)
    base = md._parse_glm53_static_v2('t,r,sf6')
    if lane_name == 'moe_batch':
        row_cases, candidate = (14, 21, 28), dict(base, probe_batch_reform=True)
    elif lane_name == 'moe_stage_fc1':
        row_cases, candidate = (7,), dict(base, fc1=3, fc2=1)
    elif lane_name == 'moe_stage_fc2':
        row_cases, candidate = (7,), dict(base, fc1=1, fc2=3)
    else:
        raise ValueError(lane_name)
    torch.manual_seed(91713)
    trash = torch.empty(64 * 1024**2 // 4, device='cuda')
    cases, graphs, resources = [], [], []
    try:
        # Every numerical routing/shape precedes every performance sample.
        for rows in row_cases:
            x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16) * .5
            linear = torch.arange(rows * 8, device='cuda').reshape(rows, 8)
            sel = (linear % 8).int()
            route = torch.ones(rows, 8, device='cuda') / 8
            pair, outputs = [], []
            for config in (base, candidate):
                with patch.object(md, '_STATIC_V2_OVERRIDE', config):
                    graph, output = _capture(lambda: lane.moe(x, sel, route, *weights, 10.))
                    pair.append(graph); outputs.append(output)
                    graphs.append(graph)
                    resources.extend(lane.graph_resources())
            unique_cases = sorted({8, 16, min(32, rows * 8), min(40, rows * 8),
                                   min(56, rows * 8), min(112, rows * 8), rows * 8})
            for unique in unique_cases + [8]:
                sel.copy_((linear % unique).int())
                x.normal_().mul_(.5)
                route.uniform_(.05, 1.)
                route[0, 0] = 0.
                route.div_(route.sum(-1, keepdim=True))
                initial, spreads, errors = None, [0., 0.], []
                for repeat in range(8):
                    for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        outputs[index].fill_(float('nan'))
                        pair[index].replay()
                    if not all(value.isfinite().all().item() for value in outputs):
                        raise RuntimeError(f'{lane_name} did not overwrite every output')
                    if initial is None:
                        initial = [value.clone() for value in outputs]
                    spreads = [max(prev, relative(value, first)) for prev, value, first
                               in zip(spreads, outputs, initial)]
                    errors.append(relative(outputs[1], outputs[0]))
                if max(*spreads, *errors) > .001:
                    raise RuntimeError(f'{lane_name}: M{rows}/U{unique}: {spreads=} {max(errors)=}')
                report('capacity_moe_numerics', candidate=lane_name, rows=rows,
                       unique_experts=unique, max_expert_rows=(rows*8+unique-1)//unique,
                       relative_max=max(errors), repeat_relative=spreads,
                       changed_routing_replay=True, tile_m=16,
                       fc1_stages=candidate['fc1'], fc2_stages=candidate['fc2'])
            route.zero_()
            for graph in pair:
                graph.replay()
            if any(value.count_nonzero().item() for value in outputs):
                raise RuntimeError(f'{lane_name}: zero routes retained old output')
            route.fill_(1./8)
            # Captured input pointers remain owned through all later timings.
            cases.append((rows, unique_cases, linear, sel, pair, x, route, outputs))
        for rows, unique_cases, linear, sel, pair, x, route, outputs in cases:
            for unique in unique_cases:
                sel.copy_((linear % unique).int())
                samples = {name: [] for name in ('warm', 'evicted')}
                for regime in samples:
                    for _ in range(4):
                        for arm, index in (('B', 0), ('A', 1), ('A', 1), ('B', 0)):
                            if regime == 'evicted':
                                trash.zero_()
                            else:
                                pair[index].replay()
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            start.record(); pair[index].replay(); end.record(); end.synchronize()
                            samples[regime].append(dict(arm=arm, us=start.elapsed_time(end)*1000))
                report('capacity_moe_timing', candidate=lane_name, rows=rows,
                       unique_experts=unique, samples=samples,
                       scope='same real L3 TP4 pack, component only, no collective or engine speed')
        report('capacity_moe_complete', candidate=lane_name, passed=True,
               gpu=torch.cuda.get_device_name(), rank_file=str(path),
               max_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        for graph in graphs:
            graph.reset()


def router_check(report, ranks):
    from safetensors import safe_open
    from engine.kernels.glm_pointwise import router_logits, route_weights
    path = rank_path(ranks)
    weights = []
    with safe_open(str(path), framework='pt', device='cpu') as source:
        for key in sorted(key for key in source.keys() if key.endswith('.moe.gate')):
            gate = source.get_tensor(key).cuda()
            bias = source.get_tensor(key.removesuffix('gate')+'bias').cuda()
            weights.append((gate, gate.float(), bias))
    if len(weights) != 42:
        raise RuntimeError('router probe must cover all 42 actual layers')
    torch.manual_seed(91714)
    cases, graphs = [], []
    try:
        for rows in (14, 21, 28):
            x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
            error = 0.
            for magnitude in (.01, .1, 1., 10.):
                for correlated in (False, True):
                    x.normal_().mul_(magnitude)
                    if correlated:
                        for lo in range(0, rows, 7):
                            x[lo+1:lo+7].mul_(.02).add_(x[lo:lo+1])
                    for gate, fp32, bias in weights:
                        base, cand = x.float() @ fp32.T, router_logits(x, gate)
                        torch.testing.assert_close(cand, base, rtol=5e-5, atol=3e-4)
                        error = max(error, (cand-base).abs().max().item())
                        ids, values = route_weights(cand, bias, 8, 2.5)
                        ref_ids, ref_values = route_weights(base, bias, 8, 2.5)
                        torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
                        torch.testing.assert_close(values, ref_values, rtol=5e-5, atol=3e-6)
            pair, outputs = [], []
            for tensorcore in (False, True):
                def run():
                    return [route_weights(router_logits(x, gate) if tensorcore else x.float() @ fp32.T,
                                          bias, 8, 2.5) for gate, fp32, bias in weights]
                graph, output = _capture(run)
                graphs.append(graph); pair.append(graph); outputs.append(output)
            for _ in range(3):
                x.normal_()
                for graph in pair:
                    graph.replay()
                for (ids, values), (ref_ids, ref_values) in zip(outputs[1], outputs[0]):
                    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
                    torch.testing.assert_close(values, ref_values, rtol=5e-5, atol=3e-6)
            report('capacity_router_numerics', rows=rows, layers=42, selected_ids_exact=True,
                   max_logit_error=error, graph_replay=True, rank_file=str(path))
            cases.append((rows, pair, x))
        for rows, pair, _ in cases:
            measurements = [dict(arm=arm, ms=_time(pair[index], iterations=64))
                            for _ in range(2)
                            for arm, index in (('B', 0), ('A', 1), ('A', 1), ('B', 0))]
            report('capacity_router_timing', rows=rows, layers=42, measurements=measurements,
                   scope='complete 42-router projection and selection, component only')
        report('capacity_router_complete', passed=True, gpu=torch.cuda.get_device_name())
    finally:
        for graph in graphs:
            graph.reset()
