"""Private incident capture; never enabled outside the diagnostic branch.

Keep real operands and pre-write states so an offline CPU recurrence can judge
the serving kernels independently of prefill/decode weight precision.
"""
from dataclasses import replace
from pathlib import Path

import torch


def _copy(value):
    return None if value is None else value.detach().clone()


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    return value


def capture(net, call, L, x, step, caches, reduce=None, **kwargs):
    identity = dict(getattr(net, 'incident_step_identity', {}))
    root = getattr(net, 'incident_audit_root', None)
    enabled = (root is not None and L < 2 and identity.get('mode') == 5
               and (identity.get('prefill') or identity.get('generation') in (1, 124, 125)))
    if not enabled:
        return call(L, x, step, caches, reduce, **kwargs)
    if getattr(step, 'captured', False) or len(step.segments) != 1:
        raise ValueError('incident operand capture requires one eager segment')
    old = net.lanes
    s = step.segments[0]
    unit = old.kda_chunk_tokens
    start = ((s.length - 1) // unit) * unit if s.length > net.rec_ring else 0
    record = dict(identity=identity, layer=L, rank=net.rank, context=s.ctx,
                  tokens=s.length, start=start, input_ids=_copy(step.ids[start:]),
                  linear=[], conv=None, recurrence=None, norm=None)

    def conv_prefill(inp, weight, state):
        y, final = old.conv_prefill(inp, weight, state)
        history = inp[start - (weight.shape[1] - 1):start].T if start else state
        record['conv'] = dict(x=_copy(inp[start:]), weight=_copy(weight),
                              initial=_copy(history), actual=_copy(y[start:]))
        return y, final

    def conv_ring(inp, weight, ring, slot, context):
        pos = context + torch.arange(-(weight.shape[1] - 1), 0, device=inp.device)
        history = ring[slot, :, pos.clamp_min(0) % ring.shape[-1]].masked_fill((pos < 0)[None], 0)
        record['conv'] = dict(x=_copy(inp), weight=_copy(weight), initial=_copy(history))
        y = old.conv_ring(inp, weight, ring, slot, context)
        record['conv']['actual'] = _copy(y)
        return y

    def recurrence(q, k, v, raw, beta, a_log, bias, initial, bound, actual, final):
        record['recurrence'] = dict(q=_copy(q[:, start:]), k=_copy(k[:, start:]),
            v=_copy(v[:, start:]), raw=_copy(raw[:, start:]), beta=_copy(beta[:, start:]),
            a_log=_copy(a_log), bias=_copy(bias), initial=_copy(initial), bound=bound,
            actual=_copy(actual[:, start:]), final=_copy(final))

    def chunk(q, k, v, raw, beta, a_log, bias, state0, bound, states_at=None):
        original = list(states_at or [])
        marks = sorted(set(original + ([start // unit] if start else [])))
        result = old.kda_chunk(q, k, v, raw, beta, a_log, bias, state0, bound,
                               states_at=marks or None)
        o, state = result[:2]
        states = result[2] if marks else None
        initial = states[marks.index(start // unit)][None] if start else state0
        recurrence(q, k, v, raw, beta, a_log, bias, initial, bound, o, state)
        if original:
            return o, state, torch.stack([states[marks.index(mark)] for mark in original])
        return o, state

    def ring(q, k, v, raw, beta, a_log, bias, states, slot, context, bound):
        initial = _copy(states[slot, (context - 1) % states.shape[1]][None]) if context else None
        o = old.kda_recurrent_ring(q, k, v, raw, beta, a_log, bias, states, slot, context, bound)
        final = states[slot, (context + q.shape[1] - 1) % states.shape[1]][None]
        recurrence(q, k, v, raw, beta, a_log, bias, initial, bound, o, final)
        return o

    def norm(core, gate, weight, eps, **options):
        out = old.kda_output_norm(core, gate, weight, eps, **options)
        record['norm'] = dict(core=_copy(core[start:]), gate=_copy(gate[start:]),
                              weight=_copy(weight), eps=eps, actual=_copy(out[start:]))
        return out

    previous_linear = net.__dict__.get('linear')
    had_linear = 'linear' in net.__dict__
    linear = net.linear

    def capture_linear(inp, name, **options):
        out = linear(inp, name, **options)
        record['linear'].append(dict(name=name, input=_copy(inp[start:]), output=_copy(out[start:])))
        return out

    net.lanes = replace(old, conv_prefill=conv_prefill, conv_ring=conv_ring,
                        kda_chunk=chunk, kda_recurrent_ring=ring, kda_output_norm=norm)
    net.linear = capture_linear
    try:
        result = call(L, x, step, caches, reduce, **kwargs)
    finally:
        net.lanes = old
        if had_linear:
            net.linear = previous_linear
        else:
            del net.linear
    if any(record[key] is None for key in ('conv', 'recurrence', 'norm')):
        raise RuntimeError('incident capture did not observe every expected KDA stage')
    directory = Path(root).parent / 'incident-kda-operands'
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"rank{net.rank}-L{L}-admit{identity['admission']}-gen{identity['generation']}-ctx{s.ctx}"
    torch.save(_cpu(record), directory / (stem + '.pt'))
    weight_path = directory / f'rank{net.rank}-L{L}-weights.pt'
    if not weight_path.exists():
        weights = {}
        for suffix in ('in_proj', 'f_b', 'g_b', 'o_proj'):
            name = f'L{L}.kda.{suffix}'
            dense = net.dense.get(name)
            if dense is None:
                weights[name] = dict(bf16=net.p[name])
            else:
                weights[name] = dict(packs=[vars(p) for p in dense.packs],
                    fp8=dense.fp8.weight if dense.fp8 is not None else None,
                    smooth=dense.smooth, decode_precision=dense.decode_precision)
        torch.save(_cpu(weights), weight_path)
    return result
