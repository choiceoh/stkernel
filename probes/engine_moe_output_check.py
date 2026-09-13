"""Same-rank MoE output qualification, including real shared-expert overlap."""
from functools import partial
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest


def check(report, ranks):
    from probes.engine_decode_bundle import require_current_probe
    require_current_probe()
    import torch
    from engine.kernels.dense import DenseLinear
    from engine.kernels.dense.shared_mlp import SharedMLP, SharedOverlap
    from engine.kernels.glm_pointwise import swiglu_clamped
    from engine.kernels.moe_output import combine
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.net import Glm53Net
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_decode_batch import timing
    from probes.engine_decode_fusions import _capture
    from probes.engine_decode_scatter_check import rank_path

    suite = unittest.defaultTestLoader.loadTestsFromNames((
        'tests.test_engine_moe_output', 'tests.test_engine_moe_output_transport'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('MoE output arithmetic/transport qualification failed or skipped')
    report('moe_output_unit', tests=result.testsRun, passed=True, proxy='CPU; NIC pending')

    path, prefix = rank_path(ranks), 'L3.moe.'
    loader = rank_loader(path)
    suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'sh_gate_up', 'sh_down')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    keys = set(loader.keys())
    modelopt = all(prefix+s in keys for s in scale_names)
    if not modelopt and any(prefix+s in keys for s in scale_names):
        raise RuntimeError('partial ModelOpt scale contract')
    names = suffixes + (scale_names if modelopt else ())
    loaded = loader.load([prefix+s for s in names], device='cuda')
    # Bind identity before the native lane permutes weight storage in place.
    hashes = {key: hashlib.sha256(value.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
              for key, value in loaded.items()}
    weights = [loaded[prefix+s] for s in suffixes[:4]]
    scales = (ModelOptScales.bind(*(loaded[prefix+s] for s in scale_names),
                                 experts=288, device=weights[0].device) if modelopt else None)
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*weights, 8, 10., scales=scales)
    linears = {prefix+s: DenseLinear(loaded[prefix+s], prefill=False) for s in suffixes[4:]}
    shared = SharedMLP(linears[prefix+'sh_gate_up'], linears[prefix+'sh_down'], 10.)
    overlap = SharedOverlap(weights[0].device)
    expert = partial(lane.moe, w13=weights[0], w13_sf=weights[1], w2=weights[2],
                     w2_sf=weights[3], limit=10., scales=scales)
    torch.manual_seed(91308)
    order = torch.randperm(288, device='cuda')
    report('moe_output_identity', rank_file=str(path), source_sha256=hashes,
           scales='ModelOpt' if modelopt else 'folded', rows=[8, 16, 24, 32],
           shared_overlap_max_rows=8, gpu=torch.cuda.get_device_name())
    cases, all_graphs, owners = [], [], []
    try:
        # Complete all numerical cells before any timing.
        for rows in (8, 16, 24, 32):
            x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
            linear = torch.arange(rows*8, device='cuda').reshape(rows, 8)
            ids = order[linear % 8].int()
            routes = torch.full((rows, 8), 1./8, device='cuda')
            net = NS(F=NS(spec_k=7, swiglu_limit=10.), p={}, shared_overlap=overlap,
                     shared_mlp={3: shared}, route=lambda *a, ids=ids, routes=routes: (ids, routes),
                     _experts={3: expert}, linear=lambda value, name: linears[name](value),
                     _activation=swiglu_clamped, comm=NS(all_reduce=lambda value: value))
            run = MethodType(Glm53Net._moe, net)
            def baseline(run=run, x=x):
                return run(3, x)
            def candidate(run=run, x=x):
                return run(3, x, finalize=combine)
            def exact(acc, partial):
                return combine(acc, partial), acc.bfloat16()+partial
            exact_graph, exact_outputs = _capture(lambda run=run, x=x: run(3, x, finalize=exact))
            all_graphs.append(exact_graph)
            captured = [_capture(fn) for fn in (baseline, candidate)]
            graphs = [pair[0] for pair in captured]
            outputs = [pair[1] for pair in captured]
            all_graphs.extend(graphs)
            owners.extend(lane.graph_resources())
            unique_cases = sorted({8, 32, rows*8})
            for unique in unique_cases:
                ids.copy_(order[linear % unique].int())
                for repeat in range(4):
                    x.normal_().mul_(.5)
                    untouched = x.clone()
                    routes.uniform_(.05, 1.); routes[0, 0] = 0.
                    routes.div_(routes.sum(-1, keepdim=True))
                    for out in exact_outputs:
                        out.fill_(float('nan'))
                    exact_graph.replay()
                    a, b = exact_outputs
                    if not torch.isfinite(a).all() or not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                        raise RuntimeError(f'M{rows}/U{unique}: same-accumulator finalization differs')
                    snapshots = []
                    for graph, out in zip(graphs, outputs):
                        out.fill_(float('nan')); graph.replay(); snapshots.append(out.clone())
                    if not all(torch.isfinite(v).all() for v in snapshots):
                        raise RuntimeError('MoE packet producer left unwritten values')
                    error = ((snapshots[1].float()-snapshots[0].float()).abs().max()
                             / snapshots[0].float().abs().max().clamp_min(1e-8)).item()
                    # Existing FP32 atomic scatter may vary across launches.
                    if error > .001 or not torch.equal(x, untouched):
                        raise RuntimeError(f'M{rows}/U{unique}: {error=} or input metadata was written')
                report('moe_output_numerics', rows=rows, unique_experts=unique,
                       same_accumulator_byte_exact=True, cross_launch_relative_max=error,
                       unchanged_input=True, graph_replay=True)
            routes.zero_()
            exact_graph.replay()
            if not torch.equal(exact_outputs[0].view(torch.uint8), exact_outputs[1].view(torch.uint8)):
                raise RuntimeError('zero routed weights retained a stale accumulator')
            routes.fill_(1./8)
            cases.append((rows, unique_cases, ids, linear, graphs, baseline, candidate, x, routes, net, outputs))
        for rows, uniques, ids, linear, graphs, base, cand, *_ in cases:
            for unique in uniques:
                ids.copy_(order[linear % unique].int())
                timing(report, 'moe_output_tensor', rows, *graphs, (base, cand), unique_experts=unique,
                       includes='real routed and shared experts, overlap/join, output cast/add',
                       excludes='router selection, packet exchange/NIC, remaining model layers')
        report('moe_output_complete', passed=True, default_enabled=True,
               max_allocated_bytes=torch.cuda.max_memory_allocated(),
               scope='component only; TP4 onepass decode/acceptance pending')
    finally:
        for graph in all_graphs:
            graph.reset()
