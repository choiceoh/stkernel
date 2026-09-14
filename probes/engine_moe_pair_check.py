"""C2 MoE tile comparison with real ModelOpt scales and the served FFN consumer.

The same process, weight pack, inputs and captured output finalizer are used
by both arms. Timing follows all numerical checks. Identity collectives and
synthetic activations make this a component gate, not a serving-speed result.
"""
from functools import partial
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch


def check(emit, ranks, *, output=None, shared_mode='ordinary', direct_scatter_only=False):
    import torch
    from engine.kernels.b12x import moe_dispatch as md
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
    from probes.engine_decode_scatter_check import rank_path, relative

    if shared_mode not in ('ordinary', 'serial', 'overlap'):
        raise ValueError('unknown C2 shared-expert comparison')
    records, graphs, owners, cases = [], [], [], []
    artifact = dict(passed=False, records=records,
                    scope='same-runtime component gate; no NIC, full model, tok/s or acceptance')
    output = Path(output or '/cache/c2-moe.json')

    def report(name, **values):
        records.append(dict(lane=name, shared_mode=shared_mode, **values))
        emit(name, shared_mode=shared_mode, **values)

    try:
        path, prefix = rank_path(ranks), 'L3.moe.'
        loader = rank_loader(path)
        suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'sh_gate_up', 'sh_down', 'gate', 'bias')
        scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
        keys = set(loader.keys())
        modelopt = all(prefix+s in keys for s in scale_names)
        if not modelopt and any(prefix+s in keys for s in scale_names):
            raise RuntimeError('partial ModelOpt scale contract')
        loaded = loader.load([prefix+s for s in suffixes + (scale_names if modelopt else ())], device='cuda')
        hashes = {key: hashlib.sha256(value.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                  for key, value in loaded.items()}
        weights = [loaded[prefix+s] for s in suffixes[:4]]
        scales = (ModelOptScales.bind(*(loaded[prefix+s] for s in scale_names),
                                     experts=288, device=weights[0].device) if modelopt else None)
        lane = served(moe_static='t,r,sf6')
        lane.moe_prepare(*weights, 8, 10., scales=scales)
        linears = {prefix+s: DenseLinear(loaded[prefix+s], prefill=False) for s in ('sh_gate_up', 'sh_down')}
        shared = SharedMLP(linears[prefix+'sh_gate_up'], linears[prefix+'sh_down'], 10.)
        overlap = SharedOverlap(weights[0].device)
        expert = partial(lane.moe, w13=weights[0], w13_sf=weights[1], w2=weights[2],
                         w2_sf=weights[3], limit=10., scales=scales)
        configs = [md._parse_glm53_static_v2(recipe) for recipe in ('t,r,sf6', 't,r,sf6,batch')]
        if direct_scatter_only:
            configs[0] = dict(configs[1], c2_direct_scatter=False)
        root = Path(__file__).resolve().parents[1]
        sources = ('engine/kernels/b12x/moe_dispatch.py', 'engine/kernels/b12x/moe_static_kernel_v4.py',
                   'engine/kernels/b12x/moe_static_common.py', 'engine/kernels/b12x/moe_static_kernel_v5.py',
                   'engine/kernels/moe_output.py', 'engine/profiles/glm53/net.py',
                   'engine/profiles/glm53/lanes.py', 'engine/kernels/dense/shared_mlp.py',
                   'engine/profiles/glm53/modelopt_scales.py', 'probes/engine_moe_pair_check.py',
                   'probes/engine_decode_fusions.py', 'probes/engine_decode_batch.py')
        report('moe_pair_identity', rank_file=str(path), weights_sha256=hashes,
               sources_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               scales='ModelOpt' if modelopt else 'folded', rows=[8, 16], seed=91416,
               direct_scatter_only=direct_scatter_only,
               configs_m16=[md._static_v2_decode_config(c, 16) for c in configs],
               candidate_default_enabled=False, includes='routed+shared experts, output cast/add; C1 keeps its shared overlap policy')
        torch.manual_seed(91416)
        order = torch.randperm(288, device='cuda')
        for rows in (8, 16):
            x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
            linear = torch.arange(rows*8, device='cuda').reshape(rows, 8)
            ids = order[linear % 8].int()
            routes = torch.full((rows, 8), 1./8, device='cuda')
            net = NS(F=NS(spec_k=7, swiglu_limit=10.), p={}, shared_overlap=overlap,
                     shared_mlp={3: shared}, route=lambda *a, ids=ids, routes=routes: (ids, routes),
                     _experts={3: expert}, linear=lambda value, name: linears[name](value),
                     _activation=swiglu_clamped, comm=NS(all_reduce=lambda value: value))
            run = MethodType(Glm53Net._moe, net)
            functions, pair, outputs = [], [], []
            for arm, config in enumerate(configs):
                def fn(config=config, run=run, x=x, arm=arm, ids=ids, routes=routes):
                    with patch.object(md, '_STATIC_V2_OVERRIDE', config):
                        if arm and x.shape[0] == 16 and shared_mode != 'ordinary':
                            def routed(consume):
                                return expert(x, ids, routes, finalize=consume)
                            if shared_mode == 'overlap':
                                return overlap(shared, x, routed, finish=combine)
                            return routed(lambda acc: combine(acc, shared(x)))
                        return run(3, x, finalize=combine)
                graph, result = _capture(fn)
                functions.append(fn); pair.append(graph); outputs.append(result)
                graphs.append(graph); owners.extend(lane.graph_resources())
            # Legal unique top8 routes, including all tokens sharing the same
            # experts, plus duplicate-route stress for the multi-M16 tile loop.
            uniques = sorted({min(u, rows*8) for u in (1, 2, 8, 16, 32, 56, 80, 128)})
            for unique in uniques + [8]:
                ids.copy_(order[linear % unique].int())
                worst, spreads, first = 0., [0., 0.], None
                for replay in range(8):
                    if replay % 2 == 0:
                        x.normal_().mul_(.5 if replay < 4 else 2.)
                        routes.uniform_(.05, 1.); routes[0, 0] = 0.
                        routes.div_(routes.sum(-1, keepdim=True))
                        first = None
                    untouched = (x.clone(), ids.clone(), routes.clone())
                    snapshots = [None, None]
                    for arm in ((0, 1) if replay % 2 == 0 else (1, 0)):
                        outputs[arm].fill_(float('nan'))
                        pair[arm].replay()
                        snapshots[arm] = outputs[arm].clone()
                    if not all(value.isfinite().all().item() for value in snapshots):
                        raise RuntimeError(f'M{rows}/U{unique}: nonfinite or unwritten output')
                    if not all(torch.equal(a, b) for a, b in zip(untouched, (x, ids, routes))):
                        raise RuntimeError('MoE wrote input/routing metadata')
                    worst = max(worst, relative(snapshots[1], snapshots[0]))
                    if first is None:
                        first = snapshots
                    spreads = [max(prev, relative(value, ref)) for prev, value, ref in zip(spreads, snapshots, first)]
                report('moe_pair_numerics', rows=rows, unique_experts=unique,
                       max_expert_rows=(rows*8+unique-1)//unique, relative_max=worst,
                       repeat_relative=spreads, changed_input_and_route=True, poisoned_output=True)
                if max(worst, *spreads) > .001:
                    raise RuntimeError(f'M{rows}/U{unique}: {worst=}, {spreads=}; exceeds the existing 0.001 gate')
            # A zero-weight routed MoE must leave exactly the shared path in
            # both arms, also after the prior nonzero accumulator contents.
            routes.zero_()
            zero = []
            for graph, result in zip(pair, outputs):
                graph.replay(); zero.append(result.clone())
            torch.testing.assert_close(zero[0], zero[1], rtol=0, atol=0)
            routes.fill_(1./8)
            fixtures = []
            for unique in (u for u in uniques if u >= 8):
                fixtures.append(('moe_pair_ffn', x.clone(), order[linear % unique].int(), routes.clone(),
                                 dict(unique_experts=unique, base_tile_m=16 if rows == 8 or direct_scatter_only else 32,
                                      candidate_tile_m=16)))
            # The actual L3 router on identical activations supplements the
            # explicit occupancy cases. This still is not a model trajectory.
            from engine.kernels.glm_pointwise import router_logits, route_weights
            for correlated in (False, True):
                x.normal_().mul_(.5)
                if correlated:
                    x[1:].mul_(.05).add_(x[:1])
                picked, weight = route_weights(router_logits(x, loaded[prefix+'gate']),
                                               loaded[prefix+'bias'], 8, 2.5)
                ids.copy_(picked); routes.copy_(weight)
                snapshots = []
                for graph, result in zip(pair, outputs):
                    graph.replay(); snapshots.append(result.clone())
                error = relative(snapshots[1], snapshots[0])
                if not all(v.isfinite().all().item() for v in snapshots) or error > .001:
                    raise RuntimeError('real-router component exceeds the numerical gate')
                extra = dict(unique_experts=int(ids.unique().numel()), correlated=correlated,
                             router_in_timing=False)
                report('moe_pair_real_router_numerics', rows=rows, relative_max=error, **extra)
                fixtures.append(('moe_pair_real_router_ffn', x.clone(), ids.clone(), routes.clone(), extra))
            cases.append((rows, x, ids, routes, pair, functions, fixtures))
        # All numerical cells, including the real router, precede timings.
        for rows, x, ids, routes, pair, functions, fixtures in cases:
            for name, values, selected, weights_for_rows, extra in fixtures:
                x.copy_(values); ids.copy_(selected); routes.copy_(weights_for_rows)
                timing(report, name, rows, *pair, functions, inside_events=True, **extra)
        artifact['passed'] = True
        report('moe_pair_complete', passed=True, max_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        artifact['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for graph in graphs:
            graph.reset()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2)+'\n')
