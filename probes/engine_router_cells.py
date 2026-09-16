"""The decode router's launch fold as a same-build cell: seven launches a layer against one (2026-09-17).

Real rank gates (the resident FP32 copies net._router_weights holds since #1014) and biases of the 42 routed
layers, synthetic BF16 rows drawn as moe_c2_cells draws them (a request base plus SPREAD x noise, the spread
its 2026-09-17 calibration chose so that 8 rows read ~42 experts a layer). No experts are loaded, so the
probe needs ~0.3 GiB beside production. No model boot, no answer, acceptance or step/s verdict.

Arms, each a captured graph over the same x:
  served    glm_pointwise.router_logits (x.to(float) + IEEE FP32 cuBLAS sgemm + its split-K reduce) then
            route_weights (_scores, torch.topk sorted, _weights): the seven launches a layer the step serves
  served_b  a second capture of the same chain: the noise floor, and the control's own determinism
  fused     engine/kernels/router_fused.cu: the same products and formulas in one launch, another add order

Scope `single` is layer 3 (warm = the gate sits in L2, the launch floor); `chain` runs all 42 routed layers
in model order with their own gates (198 MB a replay, more than L2 holds, so even its warm replays stream
the gates as the step does). Timing is B/A/A/B brackets, warm (64 replays a sample) and evicted (a 128 MiB
flush before every replay, outside the events).

Gates and verdicts:
  - logits: fused against served, ulps of max(|value|, tensor RMS); bound 64 like the MoE cells' add-order
    gate. Beyond it the kernel is wrong, not differently ordered.
  - selection from its OWN logits: the kernel's ids must be torch.topk's set over sigmoid(fused logits) + bias
    and its weights that reference's within a few ulps. This isolates phase 2 from the add order; a mismatch
    fails the cell.
  - flip rate against served (reported, never a gate): rows whose top-8 SET differs, rows whose order
    differs. These are the serving-numerics change a bracket would have to accept; synthetic rows only
    indicate the order of magnitude.
Events: identity, weights, exact, timing, router_verdict, complete (jsonl).
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes.engine_decode_fusions import _capture  # noqa: E402
from probes.engine_moe_c2_cells import bracket, fp32_noise, grouped  # noqa: E402

RANKS = '/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors'
LAYERS = tuple(range(3, 45))               # the 42 routed layers
SPREAD = 1.2498550415039062                # moe_c2_cells calibration, 2026-09-17: 8 rows read 41.9 experts a layer
FIXTURES = (('c2_two_requests', 16, 2), ('c1_one_request', 8, 1))
SEEDS = 12                                 # x draws per fixture and scope for the exactness/flip statistics
EXPERTS, HIDDEN, TOPK, SCALE = 288, 4096, 8, 2.5
GATE_BYTES = EXPERTS * HIDDEN * 4
MAX_ULPS = 64                              # an add order moves a logit by a few ulps of its scale, never 1e4
OWN_WEIGHT_MAX_ULPS = 4                    # phase 2 against torch on the kernel's own logits: sum order only
ARMS = ('served', 'served_b', 'fused')


def _sha(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class Router:
    """One routed layer's resident FP32 gate and bias, and its two routes."""

    def __init__(self, loader, L):
        got = loader.load([f'L{L}.moe.gate', f'L{L}.moe.bias'], device='cuda')
        self.L = L
        self.gate = got[f'L{L}.moe.gate'].float().contiguous()
        self.bias = got[f'L{L}.moe.bias'].float().contiguous()
        self.identity = dict(layer=L, gate_sha256=_sha(self.gate), bias_sha256=_sha(self.bias))

    def served(self, x):
        from engine.kernels.glm_pointwise import route_weights, router_logits
        logits = router_logits(x, self.gate)
        ids, weights = route_weights(logits, self.bias, TOPK, SCALE)
        return logits, ids, weights

    def fused(self, x, logits):
        from engine.kernels.router_fused import route
        ids, weights = route(x, self.gate, self.bias, TOPK, SCALE, logits=logits)
        return logits, ids, weights


def capture(group, x, rows):
    """Graphs and their (logits, ids, weights) per layer for every arm over the static x."""
    graphs, outs = {}, {}
    fused_logits = [torch.empty(rows, EXPERTS, device='cuda', dtype=torch.float32) for _ in group]
    try:
        for label in ARMS:
            def run(label=label):
                return [(r.fused(x, fused_logits[n]) if label == 'fused' else r.served(x))
                        for n, r in enumerate(group)]
            graphs[label], outs[label] = _capture(run)
    except BaseException:
        for graph in graphs.values():
            graph.reset()
        raise
    return graphs, outs


def _reference_selection(logits, bias):
    """The engine's unbound reference (net._select_routes) over the given logits."""
    s = torch.sigmoid(logits)
    sel = (s + bias).topk(TOPK, dim=-1).indices
    w = s.gather(-1, sel)
    return sel.to(torch.int32), w / (w.sum(-1, keepdim=True) + 1e-20) * SCALE


def _merge_max(total, key, value):
    total[key] = max(total.get(key, 0.), float(value))


def _aligned(ids_from, weights_from, ids_to):
    """weights_from re-read in ids_to's column order (rows whose sets match)."""
    full = torch.zeros(ids_from.shape[0], EXPERTS, device=ids_from.device, dtype=torch.float32)
    full.scatter_(1, ids_from.long(), weights_from)
    return full.gather(1, ids_to.long())


def judge(report, group, x, graphs, outs, *, rows, groups, scope, fixture):
    """Replay every arm in both orders over SEEDS draws of x; gate the fused logits and its own selection,
    count the flips against the served chain. Returns (passed, stats)."""
    stats = dict(rows=0, set_mismatch_rows=0, order_mismatch_rows=0, weight_rows_compared=0, own_set_mismatch_rows=0,
                 own_order_mismatch_rows=0, control_self_diff=0)
    for seed in range(SEEDS):
        x.copy_(grouped(rows, groups, SPREAD, 5000 + 100 * seed + rows))
        for order in (ARMS, ARMS[::-1]):
            for label in order:
                graphs[label].replay()
            torch.cuda.synchronize()
            for n in range(len(group)):
                ls, is_, ws = outs['served'][n]
                lb, ib, wb = outs['served_b'][n]
                lf, if_, wf = outs['fused'][n]
                stats['control_self_diff'] += int((ls != lb).sum() + (is_ != ib).sum() + (ws != wb).sum())
                noise = fp32_noise(lf, ls)
                _merge_max(stats, 'logits_max_ulps', noise['fp32_max_ulps'])
                _merge_max(stats, 'logits_max_abs', noise['fp32_max_abs'])
                # the kernel's selection over its own logits, against torch
                ro, wo = _reference_selection(lf, group[n].bias)
                own_set = (ro.sort(1).values == if_.sort(1).values).all(1)
                stats['own_set_mismatch_rows'] += int(rows - own_set.sum())
                stats['own_order_mismatch_rows'] += int(rows - (ro == if_).all(1).sum())
                if own_set.any():
                    own = fp32_noise(_aligned(if_, wf, ro)[own_set], wo[own_set])
                    _merge_max(stats, 'own_weights_max_ulps', own['fp32_max_ulps'])
                # the flips against the served chain, and the weights where the set survived
                set_ok = (is_.sort(1).values == if_.sort(1).values).all(1)
                stats['rows'] += rows
                stats['set_mismatch_rows'] += int(rows - set_ok.sum())
                stats['order_mismatch_rows'] += int(rows - (is_ == if_).all(1).sum())
                if set_ok.any():
                    stats['weight_rows_compared'] += int(set_ok.sum())
                    w = fp32_noise(_aligned(if_, wf, is_)[set_ok], ws[set_ok])
                    _merge_max(stats, 'weights_max_ulps', w['fp32_max_ulps'])
                    _merge_max(stats, 'weights_max_abs', w['fp32_max_abs'])
    stats.setdefault('own_weights_max_ulps', 0.)
    stats.setdefault('weights_max_ulps', 0.)
    stats.setdefault('weights_max_abs', 0.)
    passed = (stats['logits_max_ulps'] <= MAX_ULPS and stats['own_set_mismatch_rows'] == 0
              and stats['own_weights_max_ulps'] <= OWN_WEIGHT_MAX_ULPS)
    report('exact', fixture=fixture, rows=rows, scope=scope, layers=[r.L for r in group], seeds=SEEDS,
           replay_orders='forward/reverse', logits_max_ulps_gate=MAX_ULPS, own_weights_max_ulps_gate=OWN_WEIGHT_MAX_ULPS,
           set_flip_pct=100. * stats['set_mismatch_rows'] / stats['rows'],
           order_flip_pct=100. * stats['order_mismatch_rows'] / stats['rows'], passed=passed, **stats)
    return passed, stats


def main(ranks=None, *, samples=None, output=None):
    brackets = int(samples) if samples else 2
    sink = open(output, 'w') if output else None

    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink is not None:
            sink.write(line + '\n')
            sink.flush()

    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/router_fused.cu', 'engine/kernels/router_fused.py', 'engine/kernels/glm_pointwise.py',
             'engine/kernels/router_fp32.cpp', 'engine/kernels/router_fp32.py', 'probes/engine_router_cells.py')
    failures, verdicts = [], []
    try:
        from engine.profiles.glm53.weights import rank_loader
        path = Path(ranks or RANKS)
        if path.suffix != '.safetensors':
            path = path / 'rank3of4.safetensors'
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               rank_file=str(path), layers=LAYERS, fixtures=[f[0] for f in FIXTURES], spread=SPREAD, seeds=SEEDS,
               brackets=brackets, source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
               scope='captured same-build router cells beside production; no answer, acceptance or step/s verdict')
        loader = rank_loader(path)
        routers = [Router(loader, L) for L in LAYERS]
        report('weights', layers=[r.identity for r in routers], gate_bytes_per_layer=GATE_BYTES,
               allocated_bytes=torch.cuda.memory_allocated())
        for fixture, rows, groups in FIXTURES:
            x = torch.empty(rows, HIDDEN, device='cuda', dtype=torch.bfloat16)
            x.copy_(grouped(rows, groups, SPREAD, 1))
            for scope, group in (('single', routers[:1]), ('chain', routers)):
                graphs, outs = capture(group, x, rows)
                try:
                    passed, stats = judge(report, group, x, graphs, outs, rows=rows, groups=groups, scope=scope,
                                          fixture=fixture)
                    x.copy_(grouped(rows, groups, SPREAD, 1))
                    meta = dict(fixture=fixture, rows=rows, scope=scope, layers=len(group),
                                launches=dict(served=7 * len(group), fused=len(group)),
                                gate_bytes=GATE_BYTES * len(group))
                    floor = bracket(report, graphs, 'served', 'served_b', brackets=brackets, **meta)
                    if not passed:
                        raise RuntimeError(f'{fixture} {scope}: the fused router failed its gates ({stats})')
                    res = bracket(report, graphs, 'served', 'fused', brackets=brackets, **meta)
                    verdicts.append(dict(
                        fixture=fixture, rows=rows, scope=scope, layers=len(group),
                        floor_pct={c: floor[c]['mean_change_pct'] for c in floor},
                        change_pct={c: res[c]['mean_change_pct'] for c in res},
                        served_us={c: res[c]['control_us']['mean'] for c in res},
                        fused_us={c: res[c]['candidate_us']['mean'] for c in res},
                        saved_us_per_layer={c: (res[c]['control_us']['mean'] - res[c]['candidate_us']['mean']) / len(group)
                                            for c in res},
                        set_flip_pct=100. * stats['set_mismatch_rows'] / stats['rows'],
                        order_flip_pct=100. * stats['order_mismatch_rows'] / stats['rows'],
                        logits_max_ulps=stats['logits_max_ulps'], weights_max_ulps=stats['weights_max_ulps']))
                except Exception as exc:  # the other cells' evidence is kept; the run still fails
                    failures.append(f'{fixture}:{scope}')
                    report('component_failed', fixture=fixture, scope=scope, error=f'{type(exc).__name__}: {exc}'[:2000])
                finally:
                    for graph in graphs.values():
                        graph.reset()
        report('router_verdict', cells=verdicts,
               step_ms_saved_chain_evicted={f'{v["fixture"]}': v['saved_us_per_layer']['evicted'] * 42 / 1000.
                                            for v in verdicts if v['scope'] == 'chain'})
        report('complete', status='FAIL' if failures else 'PASS', failed=failures,
               max_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if sink is not None:
            sink.close()
    if failures:
        raise RuntimeError(f'router_cells failed: {failures}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
