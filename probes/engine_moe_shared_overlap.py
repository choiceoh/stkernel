"""C=2 shared expert beside the routed kernel: the served serial finalizer against SharedOverlap at sixteen rows.

Both arms run Glm53Net._moe in one process on one real rank weight pack (L3), the served lane (sixteen rows take the
batch tile) and the production output cast/add (moe_output.combine). They differ only in the same-build control
`c2_overlap=False`, under which sixteen rows keep the shared chain after the routed kernel. Eight rows (C=1) are the
same code in both arms: the noise floor. A diagnostic third arm at sixteen rows runs the fused shared MLP after the
routed kernel on the same stream ("fused serial" in measurements/glm53_c2_moe_20260914).

Numerics come first and compare bytes. The fused shared MLP (C=1's form) is checked against the served serial chain
-- its output and its BF16 activation -- over a scale sweep; then the consumer at changed inputs and routes, poisoned
outputs, reversed replay order, duplicate routes, zero routed weights and the real router. Timing follows: B/A/A/B
brackets, warm and evicted (128 MiB outside the events), events inside the captured graph, router outside. No model
boot and identity collectives: kernel evidence beside production, not a consumer speed claim (D17).
"""
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from types import MethodType, SimpleNamespace as NS

import torch

from probes.engine_decode_fusions import _capture, capture_stream

ROOT = Path(__file__).resolve().parents[1]
FILES = ('engine/profiles/glm53/net.py', 'engine/kernels/dense/shared_mlp.py', 'engine/kernels/dense/kernels.cu',
         'engine/kernels/dense/__init__.py', 'engine/kernels/glm_pointwise.py', 'engine/kernels/moe_output.py',
         'engine/profiles/glm53/lanes.py', 'engine/kernels/b12x/moe_dispatch.py',
         'engine/kernels/b12x/moe_static_kernel_v4.py', 'engine/kernels/b12x/moe_static_kernel_v5.py',
         'probes/engine_moe_shared_overlap.py', 'probes/engine_decode_fusions.py', 'probes/engine_kernel_check.py')
LAYER, PREFIX, SEED = 3, 'L3.moe.', 91515
SCALES = (0., 1e-3, .02, .1, .5, 1., 2., 4., 16., 64., 256.)
UNIQUES = (1, 2, 8, 16, 20, 32, 56, 80, 104, 128)
REPLAYS = 32                      # per timing entry: the entry is their median


def bits(value):
    return value.view(torch.int16)


def differ(actual, expected):
    """(elements whose bytes differ, elements whose values differ, max relative difference).

    The gate is zero bytes. Values separate a sign-of-zero difference from arithmetic."""
    count = int((bits(actual) != bits(expected)).sum())
    if not count:
        return 0, 0, 0.
    values = int((actual.float() != expected.float()).sum())
    scale = expected.float().abs().max().clamp_min(1e-8)
    return count, values, float(((actual.float() - expected.float()).abs().max() / scale).item())


class Recorded:
    """The real SharedOverlap, counting the rows it joins: which arm dispatched where, at capture."""

    def __init__(self, owner):
        self.owner, self.calls = owner, []

    def __call__(self, shared, x, routed, **kwargs):
        self.calls.append(x.shape[0])
        return self.owner(shared, x, routed, **kwargs)


def load(report, ranks):
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.dense import DenseLinear
    from engine.kernels.dense.shared_mlp import SharedMLP
    from engine.profiles.glm53.lanes import MOE_STATIC_PRODUCTION, parse_moe_static, served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_decode_scatter_check import rank_path
    path = rank_path(ranks)
    loader = rank_loader(path)
    suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'sh_gate_up', 'sh_down', 'gate', 'bias')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    keys = set(loader.keys())
    modelopt = all(PREFIX+s in keys for s in scale_names)
    if not modelopt and any(PREFIX+s in keys for s in scale_names):
        raise RuntimeError('partial ModelOpt scale contract')
    loaded = loader.load([PREFIX+s for s in suffixes + (scale_names if modelopt else ())], device='cuda')
    # Bind identity before the served lane lays the expert bytes out tile-major in place.
    hashes = {key: hashlib.sha256(value.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
              for key, value in loaded.items()}
    weights = [loaded[PREFIX+s] for s in suffixes[:4]]
    scales = (ModelOptScales.bind(*(loaded[PREFIX+s] for s in scale_names), experts=288, device=weights[0].device)
              if modelopt else None)
    lane = served()                                     # the production recipe: 16 rows take the C=2 batch tile
    lane.moe_prepare(*weights, 8, 10., scales=scales)
    linears = {PREFIX+s: DenseLinear(loaded[PREFIX+s], prefill=False) for s in ('sh_gate_up', 'sh_down')}
    for layer in linears.values():
        layer.decode_input_rows = (8, 16)               # what prepare_decode_projections binds for MAX_SEQS=2
    shared = SharedMLP(linears[PREFIX+'sh_gate_up'], linears[PREFIX+'sh_down'], 10.)
    expert = partial(lane.moe, w13=weights[0], w13_sf=weights[1], w2=weights[2], w2_sf=weights[3],
                     limit=10., scales=scales)
    spec, _ = parse_moe_static(MOE_STATIC_PRODUCTION)
    config = md._parse_glm53_static_v2(spec)
    report('identity', rank_file=str(path), weights_sha256=hashes, scales='ModelOpt' if modelopt else 'folded',
           sources_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES},
           torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), seed=SEED,
           moe_static=MOE_STATIC_PRODUCTION, decode_configs={m: md._static_v2_decode_config(config, m) for m in (8, 16)},
           shapes={k: list(v.shape) for k, v in loaded.items()},
           arms=dict(B='served serial shared finalizer (c2_overlap=False)', A='SharedOverlap (default)',
                     F='diagnostic: fused shared MLP after the routed kernel, same stream'),
           scope='same-runtime component gate beside production; no NIC, full model, tok/s or acceptance')
    return NS(lane=lane, linears=linears, shared=shared, expert=expert, gate=loaded[PREFIX+'gate'],
              bias=loaded[PREFIX+'bias'])


def shared_audit(report, parts, rows, draws):
    """Fused shared MLP against the served serial chain, byte for byte, eager and captured.

    Eager draws alternate a down-projection observer: with it the epilogue also stores its BF16 activation, which
    is compared with the Triton SwiGLU of the serial gate/up rows. Coverage counts the distinct BF16 gate and up
    values the sweep produced (65,536 bit patterns each)."""
    from engine.kernels.glm_pointwise import swiglu_clamped
    gu, down, shared = parts.linears[PREFIX+'sh_gate_up'], parts.linears[PREFIX+'sh_down'], parts.shared
    x = torch.empty(rows, 4096, device='cuda', dtype=torch.bfloat16)
    seen = {name: torch.zeros(1 << 16, device='cuda', dtype=torch.bool) for name in ('gate', 'up')}
    stored = []
    totals = dict(draws=0, output_elements=0, output_mismatched=0, output_value_mismatched=0,
                  activation_elements=0, activation_mismatched=0, activation_value_mismatched=0)
    worst = 0.
    for scale in SCALES:
        # bytes and values, for the output and the activation
        mismatched = torch.zeros(4, device='cuda', dtype=torch.int64)
        for outliers in (False, True):
            for draw in range(draws):
                x.normal_().mul_(scale)
                if outliers:
                    x[:, ::509].mul_(64.)               # a few massive channels, as residual streams carry
                observed = draw % 2 == 0
                down.observer = (lambda value, rows_ok: stored.append(value)) if observed else None
                try:
                    fused = shared(x)
                finally:
                    down.observer = None
                gate, up = gu(x, observe=False).chunk(2, -1)
                activation = swiglu_clamped(gate, up, 10.)
                serial = down(activation, observe=False)
                mismatched[0] += (bits(fused) != bits(serial)).sum()
                mismatched[1] += (fused.float() != serial.float()).sum()
                if observed:
                    epilogue = stored.pop()
                    mismatched[2] += (bits(epilogue) != bits(activation)).sum()
                    mismatched[3] += (epilogue.float() != activation.float()).sum()
                    totals['activation_elements'] += activation.numel()
                for name, value in (('gate', gate), ('up', up)):
                    seen[name][value.reshape(-1).view(torch.int16).long() & 0xffff] = True
                totals['draws'] += 1
                totals['output_elements'] += serial.numel()
                if draw == 0:
                    worst = max(worst, differ(fused, serial)[2])
        output, output_values, act, act_values = mismatched.tolist()
        totals['output_mismatched'] += output
        totals['output_value_mismatched'] += output_values
        totals['activation_mismatched'] += act
        totals['activation_value_mismatched'] += act_values
        report('shared_audit_scale', rows=rows, scale=scale, draws=2 * draws, output_mismatched=output,
               output_value_mismatched=output_values, activation_mismatched=act,
               activation_value_mismatched=act_values)
    # The served form is captured: both chains in graphs, replayed over the sweep.
    graphs = []
    try:
        fused_graph, fused_out = _capture(lambda: shared(x))
        graphs.append(fused_graph)
        serial_graph, serial_out = _capture(lambda: down(swiglu_clamped(*gu(x, observe=False).chunk(2, -1), 10.),
                                                         observe=False))
        graphs.append(serial_graph)
        captured = torch.zeros(2, device='cuda', dtype=torch.int64)
        replays = 0
        for scale in SCALES:
            for replay in range(8):
                x.normal_().mul_(scale)
                for out, graph in ((fused_out, fused_graph), (serial_out, serial_graph))[::1 if replay % 2 else -1]:
                    out.fill_(float('nan'))
                    graph.replay()
                captured[0] += (bits(fused_out) != bits(serial_out)).sum()
                captured[1] += (fused_out.float() != serial_out.float()).sum()
                replays += 1
        captured, captured_values = captured.tolist()
    finally:
        for graph in graphs:
            graph.reset()
    coverage = {name: int(value.sum()) for name, value in seen.items()}
    exact = totals['output_mismatched'] == 0 and totals['activation_mismatched'] == 0 and captured == 0
    report('shared_audit', rows=rows, exact=exact, captured_replays=replays, captured_mismatched=captured,
           captured_value_mismatched=captured_values, first_draw_relative_max=worst, distinct_bf16=coverage,
           **totals)
    return exact


def consumer(report, parts, overlap, rows, order):
    """Glm53Net._moe arms on shared static inputs; returns the case for timing after all numerics."""
    from engine.kernels.glm_pointwise import router_logits, route_weights, swiglu_clamped
    from engine.kernels.moe_output import combine
    from engine.profiles.glm53.net import Glm53Net
    x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
    slots = torch.arange(rows * 8, device='cuda').reshape(rows, 8)
    ids = order[slots % 8].int()
    routes = torch.full((rows, 8), 1. / 8, device='cuda')
    recorded, linear_calls = Recorded(overlap), []
    def linear(value, name):
        linear_calls.append(name)
        return parts.linears[name](value)
    net = NS(F=NS(spec_k=7, swiglu_limit=10.), p={}, shared_overlap=recorded, shared_mlp={LAYER: parts.shared},
             route=lambda *args: (ids, routes), _experts={LAYER: parts.expert}, linear=linear,
             _activation=swiglu_clamped, comm=NS(all_reduce=lambda value: value))
    run = MethodType(Glm53Net._moe, net)
    fns = {'B': lambda: run(LAYER, x, finalize=combine, c2_overlap=False),
           'A': lambda: run(LAYER, x, finalize=combine)}
    if rows == 16:
        fns['F'] = lambda: parts.expert(x, ids, routes, finalize=lambda acc: combine(acc, parts.shared(x)))
    graphs, outputs, dispatch, owners = {}, {}, {}, []
    for arm, fn in fns.items():
        before = len(recorded.calls), len(linear_calls)
        graphs[arm], outputs[arm] = _capture(fn)
        dispatch[arm] = dict(overlap=len(recorded.calls) - before[0], serial_linears=len(linear_calls) - before[1])
        owners.extend(parts.lane.graph_resources())
    # Warm pass plus capture: two FFN invocations an arm.
    expected = ({'B': dict(overlap=0, serial_linears=4), 'A': dict(overlap=2, serial_linears=0),
                 'F': dict(overlap=0, serial_linears=0)} if rows == 16 else
                {'B': dict(overlap=2, serial_linears=0), 'A': dict(overlap=2, serial_linears=0)})
    report('consumer_dispatch', rows=rows, dispatch=dispatch, expected=expected)
    if dispatch != expected:
        raise RuntimeError(f'M{rows}: arms did not dispatch as declared: {dispatch}')
    labels = list(fns)
    tally = dict(cells=0, mismatched_cells=0, mismatched_elements=0, value_mismatched_elements=0, relative_max=0.,
                 repeat_mismatched_cells=0)

    def replay(order_labels):
        snapshots = {}
        for arm in order_labels:
            outputs[arm].fill_(float('nan'))
            graphs[arm].replay()
            snapshots[arm] = outputs[arm].clone()
        if not all(bool(value.isfinite().all()) for value in snapshots.values()):
            raise RuntimeError(f'M{rows}: nonfinite or unwritten FFN output')
        return snapshots

    def cell(name, reverse=False, repeat=None, **meta):
        untouched = (x.clone(), ids.clone(), routes.clone())
        snapshots = replay(labels[::-1] if reverse else labels)
        if not all(torch.equal(a, b) for a, b in zip(untouched, (x, ids, routes))):
            raise RuntimeError('the FFN wrote its input or routing metadata')
        row = dict(rows=rows, cell=name, **meta)
        for arm in labels[1:]:
            count, values, relative = differ(snapshots[arm], snapshots['B'])
            row[f'{arm}_mismatched'] = count
            row[f'{arm}_value_mismatched'] = values
            row[f'{arm}_relative_max'] = relative
            if arm == 'A':
                tally['mismatched_cells'] += bool(count)
                tally['mismatched_elements'] += count
                tally['value_mismatched_elements'] += values
                tally['relative_max'] = max(tally['relative_max'], relative)
        if repeat is not None:
            spread = {arm: differ(snapshots[arm], repeat[arm])[0] for arm in labels}
            row['repeat_mismatched'] = spread
            tally['repeat_mismatched_cells'] += any(spread.values())
        tally['cells'] += 1
        report('consumer_cell', **row)
        return snapshots

    for unique in (u for u in UNIQUES if u <= rows * 8):
        ids.copy_(order[slots % unique].int())
        for draw in range(4):
            x.normal_().mul_((.5, 2., .05, 8.)[draw])
            routes.uniform_(.05, 1.)
            routes[0, 0] = 0.
            routes.div_(routes.sum(-1, keepdim=True))
            first = cell('unique', unique_experts=unique, draw=draw)
            cell('unique_repeat', reverse=True, repeat=first, unique_experts=unique, draw=draw)
    from_router = lambda: route_weights(router_logits(x, parts.gate), parts.bias, 8, 2.5)
    for scale in (1e-3, .05, 1., 4., 16.):
        x.normal_().mul_(scale)
        picked, weight = from_router()
        ids.copy_(picked); routes.copy_(weight)
        cell('router_scale', scale=scale, unique_experts=int(ids.unique().numel()))
    routes.zero_()
    cell('zero_routed_weights')
    fixtures = []
    for groups in ((0, 1, 2) if rows == 16 else (0, 1)):
        for draw in range(6):
            x.normal_().mul_(.5)
            if groups:
                width = rows // groups
                for start in range(0, rows, width):
                    x[start+1:start+width].mul_(.05).add_(x[start:start+1])
            picked, weight = from_router()
            ids.copy_(picked); routes.copy_(weight)
            unique = int(ids.unique().numel())
            cell('real_router', request_groups=groups, draw=draw, unique_experts=unique)
            if draw == 0:
                fixtures.append((dict(request_groups=groups, unique_experts=unique,
                                      label=('independent' if not groups else
                                             f'{groups} group(s) of {rows // groups} related rows')),
                                 x.clone(), ids.clone(), routes.clone()))
    exact = tally['mismatched_cells'] == 0
    report('consumer_numerics', rows=rows, exact=exact, **tally)
    return NS(rows=rows, x=x, ids=ids, routes=routes, fns=fns, labels=labels, graphs=graphs, owners=owners,
              lane=parts.lane, fixtures=fixtures, exact=exact)


def timing(report, case, samples):
    """B/A/A/B (B/A/F/F/A/B at sixteen rows) brackets per fixture; an entry is the median of REPLAYS replays."""
    cold = torch.empty(128 << 20, dtype=torch.uint8, device='cuda')
    order = ('B', 'A', 'F', 'F', 'A', 'B') if 'F' in case.labels else ('B', 'A', 'A', 'B')
    for meta, values, selected, weights in case.fixtures:
        case.x.copy_(values); case.ids.copy_(selected); case.routes.copy_(weights)
        captured = {}
        try:
            for arm in case.labels:
                for cache in ('warm', 'evicted'):
                    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                    def run(fn=case.fns[arm], start=start, end=end, evict=cache == 'evicted'):
                        if evict:
                            cold.fill_(19)                  # inside the graph, outside the events
                        start.record()
                        result = fn()
                        end.record()
                        return result
                    captured[arm, cache] = (_capture(run)[0], start, end)
                    case.owners.extend(case.lane.graph_resources())   # CUDA holds their raw pointers
            for cache in ('warm', 'evicted'):
                entries = []
                for bracket in range(samples):
                    for arm in order:
                        graph, start, end = captured[arm, cache]
                        values_ms = []
                        for _ in range(REPLAYS):
                            graph.replay()
                            end.synchronize()
                            values_ms.append(start.elapsed_time(end))
                        entries.append(dict(bracket=bracket, arm=arm, median_ms=median(values_ms),
                                            min_ms=min(values_ms), mean_ms=mean(values_ms)))
                summary = {}
                for arm in case.labels:
                    medians = [e['median_ms'] for e in entries if e['arm'] == arm]
                    summary[arm] = dict(mean_ms=mean(medians), min_ms=min(medians), entries=len(medians),
                                        fastest_replay_ms=min(e['min_ms'] for e in entries if e['arm'] == arm))
                change = {arm: dict(mean=summary[arm]['mean_ms'] / summary['B']['mean_ms'] - 1.,
                                    min=summary[arm]['min_ms'] / summary['B']['min_ms'] - 1.)
                          for arm in case.labels if arm != 'B'}
                report('timing', rows=case.rows, cache=cache, samples=samples, replays=REPLAYS, order=list(order),
                       entries=entries, summary=summary, change=change, eviction_bytes=cold.numel() if cache == 'evicted' else 0,
                       includes='routed and shared experts, overlap/join, output cast/add',
                       excludes='router selection, packet exchange/NIC, other layers', **meta)
        finally:
            for graph, _, _ in captured.values():
                graph.reset()


def main(ranks, *, samples=None, output=None):
    from engine.kernels.dense.shared_mlp import SharedOverlap
    records = []
    artifact = dict(passed=False, exact=False, records=records,
                    scope='same-runtime component gate beside production; no NIC, full model, tok/s or acceptance')
    output = Path(output or '/cache/c2-shared-overlap.json')
    samples = int(samples or 4)

    def report(name, **values):
        records.append(dict(lane=name, **values))
        print(json.dumps(records[-1]), flush=True)

    graphs = []
    try:
        if torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError('requires GB10')
        parts = load(report, ranks)
        # The 09-14 overlap v2 failed when a fresh capture stream per cell cycled PyTorch's stream pool onto the
        # live SharedOverlap stream. _capture now reuses one stream a device; take it before the owner's and refuse
        # an alias outright rather than meeting it as a parent-stream error mid-run.
        parent = capture_stream()
        overlap = SharedOverlap(parent.device)
        if overlap.stream == parent or overlap.stream == torch.cuda.current_stream():
            raise RuntimeError('the SharedOverlap stream aliases a capture or current stream')
        report('streams', capture=parent.cuda_stream, overlap=overlap.stream.cuda_stream,
               current=torch.cuda.current_stream().cuda_stream, distinct=True)
        torch.manual_seed(SEED)
        audits = {rows: shared_audit(report, parts, rows, draws=64) for rows in (8, 16)}
        order = torch.randperm(288, device='cuda')
        cases = []
        for rows in (8, 16):
            case = consumer(report, parts, overlap, rows, order)
            cases.append(case)
            graphs.extend(case.graphs.values())
        exact = all(audits.values()) and all(case.exact for case in cases)
        artifact['exact'] = exact
        report('numerics_complete', exact=exact, shared_audit=audits, consumer={c.rows: c.exact for c in cases},
               overlap_rows=sorted(overlap.rows))
        # All numerical cells precede every timing, whatever they found: a rejection needs its numbers too.
        for case in cases:
            timing(report, case, samples)
        artifact['passed'] = exact
        report('complete', passed=exact, max_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        artifact['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for graph in graphs:
            graph.reset()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2) + '\n')
