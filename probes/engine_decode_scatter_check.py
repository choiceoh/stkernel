"""Bounded same-pack MoE and router candidates; no serving dispatch changes."""
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch

import torch

from probes.engine_decode_fusions import _capture, _time
def relative(actual, expected):
    return ((actual.float() - expected.float()).abs().max()
            / expected.float().abs().max().clamp_min(1e-8)).item()


def rank_path(ranks):
    from engine.profiles.glm53 import facts
    if not ranks:
        raise ValueError('capacity probes require the exact consumer --ranks')
    root = Path(ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent / root
    return root / 'rank0of4.safetensors'


def moe_check(report, ranks, lane_name):
    if lane_name == 'moe_route_scatter':
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName(
            'tests.test_engine_decode_projection.ProjectionTests.test_route_reduction_layout_and_replayed_zero_overwrite')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or result.skipped or result.testsRun != 1:
            raise RuntimeError('route reduction layout/FTZ gate did not pass')
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
    if lane_name in ('moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct'):
        row_cases = (7, 14, 21, 28)
        candidate = dict(base,
                         probe_route_scatter=lane_name != 'moe_direct_scatter',
                         probe_direct_scatter=lane_name != 'moe_route_scatter')
    else:
        raise ValueError(lane_name)
    torch.manual_seed(91713)
    expert_order = torch.randperm(weights[0].shape[0], device='cuda')
    trash = torch.empty(64 * 1024**2 // 4, device='cuda')
    cases, graphs, resources, scatter_owners = [], [], [], []
    try:
        # Every numerical routing/shape precedes every performance sample.
        for rows in row_cases:
            x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16) * .5
            linear = torch.arange(rows * 8, device='cuda').reshape(rows, 8)
            sel = expert_order[linear % 8].int()
            route = torch.ones(rows, 8, device='cuda') / 8
            pair, outputs = [], []
            for config, pack in ((base, weights), (candidate, weights)):
                from probes.engine_moe_scatter import route_scatter_owner
                owner = route_scatter_owner(md, scatter_owners) if config.get('probe_route_scatter') else nullcontext()
                with patch.object(md, '_STATIC_V2_OVERRIDE', config), owner:
                    graph, output = _capture(lambda: lane.moe(x, sel, route, *pack, 10.))
                    pair.append(graph); outputs.append(output)
                    graphs.append(graph)
                    resources.extend(lane.graph_resources())
            unique_cases = sorted({8, 16, min(32, rows * 8), min(40, rows * 8),
                                   min(56, rows * 8), min(112, rows * 8), rows * 8})
            for unique in unique_cases + [8]:
                sel.copy_(expert_order[linear % unique].int())
                x.normal_().mul_(.5)
                route.uniform_(.05, 1.)
                route[0, 0] = 0.
                route.div_(route.sum(-1, keepdim=True))
                initial, spreads, errors = None, [0., 0.], []
                for repeat in range(8):
                    # Every route/part must overwrite its own cell on changed
                    # routing, including zero weights and partially filled tiles.
                    for owner in scatter_owners:
                        owner.scratch.fill_(float('nan'))
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
                report('decode_scatter_numerics', candidate=lane_name, rows=rows,
                       unique_experts=unique, max_expert_rows=(rows*8+unique-1)//unique,
                       relative_max=max(errors), repeat_relative=spreads,
                       changed_routing_replay=True, tile_m=16 if rows == 7 else 32,
                       fc1_stages=candidate['fc1'], fc2_stages=candidate['fc2'])
            route.zero_()
            for owner in scatter_owners:
                owner.scratch.fill_(float('nan'))
            for graph in pair:
                graph.replay()
            if any(value.count_nonzero().item() for value in outputs):
                raise RuntimeError(f'{lane_name}: zero routes retained old output')
            route.fill_(1./8)
            # Captured input pointers remain owned through all later timings.
            cases.append((rows, unique_cases, linear, sel, pair, x, route, outputs))
        for rows, unique_cases, linear, sel, pair, x, route, outputs in cases:
            for unique in unique_cases:
                sel.copy_(expert_order[linear % unique].int())
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
                report('decode_scatter_timing', candidate=lane_name, rows=rows,
                       unique_experts=unique, samples=samples,
                       scope='same real L3 TP4 pack, component only, no collective or engine speed')
        report('decode_scatter_complete', candidate=lane_name, passed=True,
               gpu=torch.cuda.get_device_name(), rank_file=str(path),
               route_scratch_bytes=sum(o.scratch.numel() * o.scratch.element_size() for o in scatter_owners),
               max_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        for graph in graphs:
            graph.reset()
