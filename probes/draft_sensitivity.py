"""Replay captured real C1 draft proposals, changing ONE reader on every TP rank.

No engine boot, target-body weights, or calibration cache mutation. Requires the
capture made by glm53.draft_replay. Native GPU work still needs fleet admission.
CPU tests exercise contracts; this CLI deliberately refuses CPU timing as cost.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.profiles.glm53.draft_replay import agreed, sha256


def prefix_bounds(draft, labels):
    """An unseen continuation is censored, never padded with baseline guesses."""
    if len(labels) > len(draft):
        raise ValueError('labels exceed the draft horizon')
    for i, (a, b) in enumerate(zip(draft, labels)):
        if a != b:
            return i, i
    return len(labels), len(draft)


def summarize(cases, baseline, candidate, timing=None):
    if not cases or len(cases) != len(baseline) or len(cases) != len(candidate):
        raise ValueError('paired nonempty cases are required')
    b, c = [], []
    for case, old, new in zip(cases, baseline, candidate):
        if (case['temperature'] != 0 or case['label_source'] != 'committed_greedy_continuation'
                or case['k'] != cases[0]['k']
                or len(old) != case['k'] or len(new) != case['k']):
            raise ValueError('only matching-width committed greedy labels are valid')
        if old != case['baseline_drafts']:
            raise ValueError(f'baseline replay drift: {case["case_id"]}; no sensitivity verdict')
        b.append(prefix_bounds(old, case['target']))
        c.append(prefix_bounds(new, case['target']))
    n = len(cases)
    avg = lambda bounds, at: sum(x[at] for x in bounds) / n
    bmean, cmean = [avg(b, i) for i in (0, 1)], [avg(c, i) for i in (0, 1)]
    gain = [cmean[0] - bmean[1], cmean[1] - bmean[0]]
    ratio = [(1 + cmean[0]) / (1 + bmean[1]), (1 + cmean[1]) / (1 + bmean[0])]
    k = cases[0]['k']
    result = dict(cases=n, requests=len({x.get('request_key',x['case_id']) for x in cases}),
        labels_complete=sum(len(x['target']) == x['k'] for x in cases),
        baseline_prefix_bounds=bmean, candidate_prefix_bounds=cmean, prefix_gain_bounds=gain,
        unclipped_tokens_per_step_ratio_bounds=ratio,
        max_extra_step_fraction_bounds=[r - 1 for r in ratio],
        prefix_survival_baseline=[[sum(lo >= i for lo, hi in b)/n, sum(hi >= i for lo, hi in b)/n] for i in range(1,k+1)],
        prefix_survival_candidate=[[sum(lo >= i for lo, hi in c)/n, sum(hi >= i for lo, hi in c)/n] for i in range(1,k+1)],
        changed_draft_positions=sum(sum(a != z for a,z in zip(x,y)) for x,y in zip(baseline,candidate)),
        live_acceptance_measured=False, live_toks_measured=False)
    if timing is not None:
        if len(timing) != n or any(not all(math.isfinite(t[x]) and t[x] > 0 for x in ('baseline_us','candidate_us')) for t in timing):
            raise ValueError('one positive native proposal timing pair per case is required')
        delta = statistics.mean(t['candidate_us'] - t['baseline_us'] for t in timing)
        result.update(proposal_delta_us=delta, timing_scope='native TP block/head/selector graph; precomputed embedding; not full decode step',
            extra_proposal_us_per_extra_token=(delta / gain[0] if gain[0] > 0 and delta >= 0 else None))
    return result


def cost_screen(results):
    """Keep tradeoffs visible; timing medians cannot establish a deployment winner."""
    def dominates(a,b):
        benefit = a['prefix_gain_bounds'][0] >= b['prefix_gain_bounds'][1]
        time = a['proposal_delta_us'] <= b['proposal_delta_us']
        memory = max(a['active_weight_byte_delta_per_rank']) <= max(b['active_weight_byte_delta_per_rank'])
        strict = (a['prefix_gain_bounds'][0] > b['prefix_gain_bounds'][1]
                  or a['proposal_delta_us'] < b['proposal_delta_us']
                  or max(a['active_weight_byte_delta_per_rank']) < max(b['active_weight_byte_delta_per_rank']))
        return benefit and time and memory and strict
    dominated = {r['reader']:[a['reader'] for a in results if a is not r and dominates(a,r)] for r in results}
    return dict(screening_only=True, timing_uncertainty_not_resolved=True,
                non_dominated=[n for n,v in dominated.items() if not v], dominated_by=dominated,
                positive_prefix_gain=[r['reader'] for r in results if r['prefix_gain_bounds'][0] > 0],
                deployment_winner=None)


def restore_fp8(state, device):
    if state is None:
        return None
    from engine.kernels.dense import FP8Linear
    obj = FP8Linear.__new__(FP8Linear)
    obj.rows, obj.cols, obj.name = state['rows'], state['cols'], 'captured'
    obj.observer, obj.executed, obj.calibrated = None, False, True
    obj.weight = state['q'].to(device), state['scale'].to(device)
    return obj


def restore_dense(state, device):
    from engine.kernels.dense import DenseLinear, W4Pack
    obj = DenseLinear.__new__(DenseLinear)
    obj.rows, obj.cols, obj.name = state['rows'], state['cols'], 'captured'
    obj.decode_precision = state['decode_precision']
    obj.decode_input_rows = tuple(state['decode_input_rows'])
    obj.smooth = None if state['smooth'] is None else state['smooth'].to(device)
    obj.packs = tuple(W4Pack(p['data'].to(device),p['scale'].to(device),p['rowscale'].to(device),
                            p['rows'],p['cols'],p['calibrated']) for p in state['packs'])
    obj.fp8, obj.decode_fp8 = restore_fp8(state['fp8'], device), restore_fp8(state['decode_fp8'], device)
    obj.observer, obj.workspace, obj.executed = None, None, 0
    obj.bound_input_executed, obj.producer_pack_executed = set(), set()
    obj.calibrated = all(p.calibrated for p in obj.packs)
    return obj


def load_state(directory, comm, device):
    from engine.profiles.glm53.drafter import Drafter, DrafterFacts
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['version'] != 1 or sha256(directory / 'state.pt') != manifest['state_sha256']:
        raise ValueError('capture state digest mismatch')
    state = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True, mmap=True)
    if state['version'] != 1 or (state['rank'],state['world']) != (comm.rank,comm.world_size):
        raise ValueError('capture rank/world mismatch; isolated rank is not an acceptance replay')
    for name, digest in state['source_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'capture source mismatch: {name}')
    if state['torch'] != str(torch.__version__):
        raise ValueError('capture PyTorch version mismatch')
    if str(device).startswith('cuda'):
        from engine.base.kernel_shape import bind, bind_drafter, from_dict, Drafter as Geometry
        if tuple(state['kernel_shape']['device']['capability']) != torch.cuda.get_device_capability():
            raise ValueError('capture GPU architecture mismatch')
        bind(from_dict(state['kernel_shape']))
        bind_drafter(Geometry(**{k:state['facts'][k] for k in ('head_dim','kv_heads','layers','window')}))
    head = restore_fp8(state['head'], device)
    target = SimpleNamespace(comm=comm, rank=comm.rank, vp=state['vp'], head_local=head)
    facts = DrafterFacts(**dict(state['facts'], target_layers=tuple(state['facts']['target_layers'])))
    d = Drafter(facts, target, state['decodable'])
    d.p = {k:v.to(device) for k,v in state['p'].items()}
    d.dense = {k:restore_dense(v, device) for k,v in state['dense'].items()}
    d.fast_attention, d.local_heads, d.local_kv_heads = True, state['local_heads'], state['local_kv_heads']
    d.selector_alpha = tuple(state['selector_alpha'])
    d.tuning = replace(d.tuning, selector_projection_fp32=state['selector_projection_fp32'])
    return d, state, manifest


def source_weight(model, name, rank, world, smooth):
    if not name.startswith('layers.'):
        raise ValueError('fixed-ring replay cannot change FC/context readers')
    pre = '.'.join(name.split('.')[:2]) + '.'
    suffix = name[len(pre):]
    def weight(key):
        w = model.get_tensor(pre + key)
        if smooth is not None:
            w = (w.float() * smooth.cpu()).to(w.dtype)
        return w
    if suffix in ('self_attn.qkv', 'mlp.gate_up'):
        keys = [f'self_attn.{s}_proj.weight' for s in ('q','k','v')] if suffix.startswith('self_attn') else [f'mlp.{s}_proj.weight' for s in ('gate','up')]
        return torch.cat([weight(k).chunk(world,0)[rank] for k in keys])
    if suffix in ('self_attn.o_proj.weight','mlp.down_proj.weight'):
        return weight(suffix).chunk(world,1)[rank].contiguous()
    if suffix not in ('attention_conv.kernel_projection.weight','mlp_conv.kernel_projection.weight'):
        raise ValueError(f'unsupported proposal reader: {name}')
    return weight(suffix)


@contextmanager
def replace_reader(d, name, reader):
    old = d.dense[name]
    d.dense[name] = reader
    try:
        yield
    finally:
        d.dense[name] = old


def candidate_reader(weight, precision, device):
    if precision == 'bf16':
        w = weight.to(device)
        return lambda x, *a, **kw: torch.nn.functional.linear(x, w)
    from engine.kernels.dense.packing import fp8_rtn
    q, scale = fp8_rtn(weight)
    return restore_fp8(dict(rows=weight.shape[0], cols=weight.shape[1], q=q, scale=scale), device)


def active_weight_bytes(state):
    if state['decode_precision'] == 'fp8':
        lane = state['decode_fp8'] or state['fp8']
        return sum(lane[k].numel() * lane[k].element_size() for k in ('q','scale'))
    return sum(p[k].numel() * p[k].element_size() for p in state['packs'] for k in ('data','scale','rowscale'))


def capture_graph(call):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    return graph, out


def paired_time(graphs, comm, rounds):
    samples = [[], []]
    # Every round is B/A/A/B; max across ranks for each interval, then median.
    for _ in range(rounds):
        for arm in (0,1,1,0):
            comm.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graphs[arm].replay(); end.record(); end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000)
    peers = comm.gather_objects(samples)
    slowest = [[max(p[a][i] for p in peers) for i in range(len(samples[a]))] for a in (0,1)]
    return dict(baseline_us=statistics.median(slowest[0]), candidate_us=statistics.median(slowest[1]),
                baseline_samples_us=slowest[0], candidate_samples_us=slowest[1])


def case_files(directory, manifest):
    cases = []
    for path in sorted(directory.glob('case-*.pt')):
        labels = path.with_suffix('.json')
        if not labels.exists():
            raise ValueError(f'unfinished capture labels: {path.name}')
        meta = json.loads(labels.read_text())
        if sha256(path) != meta['snapshot_sha256']:
            raise ValueError('case snapshot digest mismatch')
        case = torch.load(path, map_location='cpu', weights_only=True)
        if case['version'] != 1 or case['state_sha256'] != manifest['state_sha256'] or case['case_id'] != meta['case_id']:
            raise ValueError('case belongs to a different capture state')
        case.update(meta)
        cases.append(case)
    if not cases:
        raise ValueError('no real replay cases; selector/Hessian dumps are insufficient')
    return cases


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True, help='root containing rankN/')
    ap.add_argument('--checkpoint', type=Path, required=True, help='original drafter model.safetensors')
    ap.add_argument('--reader', nargs='+', required=True, help='explicit reader names or all; one changed per arm')
    ap.add_argument('--precision', choices=('fp8-rtn','bf16'), default='fp8-rtn')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=10)
    args = ap.parse_args()
    if args.rounds < 1 or not torch.cuda.is_available():
        raise ValueError('native replay requires GPU and positive timing rounds; never use CPU time as cost')
    from engine.base.comm import Comm
    from safetensors import safe_open
    comm = Comm.init()
    try:
        directory = args.capture / f'rank{comm.rank}'
        d, state, manifest = agreed(comm,lambda: load_state(directory, comm, 'cuda'))
        transports = comm.gather_objects(state['transport'])
        if any(t != transports[0] for t in transports):
            raise ValueError('ranks captured different collective transports')
        if state['transport'] is not None:
            comm.prepare_oneshot(**state['transport'])
        def prepare_cases():
            if sha256(args.checkpoint) != state['checkpoint_sha256']:
                raise ValueError('candidate checkpoint differs from the captured checkpoint')
            cases = case_files(directory, manifest)
            for case in cases:
                if (case['k'] != d.k or case['ids'].tolist() != [case['anchor']] + [d.F.mask_id]*d.k
                        or tuple(case['embedding'].shape) != (d.k+1,d.F.hidden)
                        or case['temperature'] != 0 or case['label_source'] != 'committed_greedy_continuation'):
                    raise ValueError('case does not fit the prepared C1 greedy proposal')
            return cases
        cases = agreed(comm,prepare_cases)
        shared = [(c['case_id'],c['anchor'],c['position'],c['baseline_drafts'],c['target'],c['label_source']) for c in cases]
        peers = comm.gather_objects(shared)
        if any(p != shared for p in peers):
            raise ValueError('ranks disagree on cases/labels')
        readers = sorted(d.dense) if args.reader == ['all'] else args.reader
        if len(set(readers)) != len(readers) or any(n not in d.dense for n in readers):
            raise ValueError('readers must be unique prepared proposal reader names')
        plans = comm.gather_objects((readers,args.precision,args.rounds))
        if any(p != plans[0] for p in plans):
            raise ValueError('ranks requested different replay arms')
        report = dict(version=1, capture=str(args.capture), precision=args.precision,
            transport=state['transport'],
            rank_state_sha256=comm.gather_objects(manifest['state_sha256']),
            replay_source_sha256=sha256(__file__), results=[],
            scope='fixed-state C1 greedy proposal sensitivity; no live tok/s or output quality verdict')
        with safe_open(args.checkpoint, framework='pt', device='cpu') as model:
            for name in readers:
                def prepare_candidate():
                    weight = source_weight(model,name,comm.rank,comm.world_size,d.dense[name].smooth)
                    return weight, candidate_reader(weight,args.precision,'cuda')
                weight, candidate = agreed(comm,prepare_candidate)
                candidate_bytes = (weight.numel()*weight.element_size() if args.precision == 'bf16'
                                   else sum(t.numel()*t.element_size() for t in candidate.weight))
                byte_delta = comm.gather_objects(candidate_bytes-active_weight_bytes(state['dense'][name]))
                baseline, changed, timing = [], [], []
                for case in cases:
                    ring = case['ring'].to('cuda')
                    ids, embedding = case['ids'].to('cuda'), case['embedding'].to('cuda')
                    # Proposal always consumes the same anchor + masks; no candidate may reuse final hidden.
                    def embed(token_ids):
                        # Tensor equality cannot be read on host inside a CUDA graph.
                        # IDs are constructed by the unchanged production input builder.
                        if token_ids.shape != ids.shape:
                            raise ValueError('replay embedding row shape changed')
                        return embedding
                    d.target.embed = embed
                    anchor = ids[:1]
                    call = lambda: d.propose_tensor(anchor,case['position'],ring)
                    before = ring.clone()
                    gb, ob = capture_graph(call)
                    baseline.append(ob.tolist())
                    if baseline[-1] != case['baseline_drafts']:
                        raise ValueError(f'baseline replay drift: {case["case_id"]}; refusing ranking')
                    with replace_reader(d,name,candidate):
                        gc, oc = capture_graph(call)
                    changed.append(oc.tolist())
                    pair = paired_time((gb,gc),comm,args.rounds)
                    if not torch.equal(before,ring) or ob.tolist() != baseline[-1] or oc.tolist() != changed[-1]:
                        raise ValueError('replay mutated context or proposals are unstable')
                    timing.append(pair)
                    gb.reset(); gc.reset()
                result = dict(reader=name, **summarize(cases,baseline,changed,timing),
                    proposals=changed, timing=timing, active_weight_byte_delta_per_rank=byte_delta,
                    resident_memory_note='active reader payload only; replay holds both arms; not serving peak memory')
                report['results'].append(result)
                report['cost_screen'] = cost_screen(report['results'])
                def write_report():
                    if comm.rank == 0:
                        args.output.parent.mkdir(parents=True,exist_ok=True)
                        args.output.write_text(json.dumps(report,indent=2)+'\n')
                        print(json.dumps({k:v for k,v in result.items() if k not in ('proposals','timing')}),flush=True)
                agreed(comm,write_report)
                del candidate, weight
    finally:
        comm.close()


if __name__ == '__main__':
    main()
