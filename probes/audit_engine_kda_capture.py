"""CPU-only arithmetic audit of private, actual-request KDA operand captures."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get('ST_INCIDENT_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from engine.modules.causal_conv import causal_conv1d
from engine.modules.linear_attention import gated_delta_rule, kda_gate, kda_output_norm


def diff(actual, expected):
    a, b = actual.float(), expected.float()
    error = a - b
    return dict(shape=list(a.shape), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                relative_l2=float(error.norm() / b.norm().clamp_min(1e-30)),
                max_abs=float(error.abs().max()), rms=float(error.square().mean().sqrt()),
                reference_rms=float(b.square().mean().sqrt()), equal=bool(torch.equal(a, b)))


def load_packing():
    spec = importlib.util.spec_from_file_location('incident_dense_packing', ROOT / 'engine/kernels/dense/packing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense_reference(inp, weight, decode, packing):
    if 'bf16' in weight:
        return (inp.float() @ weight['bf16'].float().T).to(inp.dtype)
    if decode and weight['decode_precision'] == 'w4':
        parts, at = [], 0
        for pack in weight['packs']:
            q, scales = pack['data'], pack['scale']
            # The CPU reference accepts row-major weights, including 16-row CTA tiles.
            rows = q.shape[0] * q.shape[2]
            q = q.permute(0, 2, 1, 3).contiguous().reshape(rows, -1)
            scales = scales.permute(0, 2, 1, 3).contiguous().reshape(rows, -1)
            w = packing.mk_w4_dequant_rowmajor(q, scales, rgs=pack['rowscale'])[:pack['rows']]
            x = packing._mk_quant_x_ref(inp[:, at:at + pack['cols']])
            parts.append((x @ w.T).to(inp.dtype).float())
            at += pack['cols']
        return sum(parts).to(inp.dtype)
    q, scales = weight['fp8']
    w = q.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    x = inp.float().reshape(inp.shape[0], -1, 128)
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448)))
    x = ((x / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]).reshape_as(inp)
    return (x @ w.T).to(inp.dtype)


def audit(path, weights, packing):
    item = torch.load(path, map_location='cpu', weights_only=True)
    c, r, n = (item[key] for key in ('conv', 'recurrence', 'norm'))
    conv, _ = causal_conv1d(c['x'], c['weight'], initial_state=c['initial'])
    gate = kda_gate(r['raw'], r['a_log'], r['bias'], r['bound'])
    out, final = gated_delta_rule(r['q'], r['k'], r['v'], gate, torch.sigmoid(r['beta'].float()),
                                 r['initial'], decay_per_channel=True)
    norm = kda_output_norm(n['core'], n['gate'], n['weight'], n['eps'])
    row = {key: item[key] for key in ('identity', 'rank', 'layer', 'context', 'tokens', 'start')}
    row.update(conv=diff(c['actual'], conv), recurrence_output=diff(r['actual'], out),
               recurrence_final=diff(r['final'], final), output_norm=diff(n['actual'], norm))
    joined = torch.cat([r[key][0].flatten(1) for key in ('q', 'k', 'v')], -1)
    row['conv_to_recurrence'] = diff(joined, c['actual'])
    row['recurrence_to_norm'] = diff(n['core'], r['actual'][0])
    row['linear'] = []
    for linear in item['linear']:
        reference = dense_reference(linear['input'], weights[linear['name']], item['tokens'] <= 32, packing)
        row['linear'].append(dict(name=linear['name'], **diff(linear['output'], reference[:, :linear['output'].shape[1]])))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    packing, weights, rows, records = load_packing(), {}, [], []
    for path in sorted(args.directory.glob('rank*-L*-admit*.pt')):
        key = '-'.join(path.name.split('-')[:2])
        if key not in weights:
            weights[key] = torch.load(args.directory / (key + '-weights.pt'), map_location='cpu', weights_only=True)
        row = audit(path, weights[key], packing)
        rows.append(row)
        records.append(torch.load(path, map_location='cpu', weights_only=True))
        print(json.dumps(row), flush=True)
    if not rows:
        raise SystemExit('no operand captures found')
    carry = []
    for current in records:
        if current['start'] != 0:
            continue
        for previous in records:
            if (previous['identity']['admission'] == current['identity']['admission']
                    and previous['rank'] == current['rank'] and previous['layer'] == current['layer']
                    and previous['context'] + previous['tokens'] == current['context']):
                history = min(3, previous['conv']['x'].shape[0])
                carry.append(dict(rank=current['rank'], layer=current['layer'],
                    admission=current['identity']['admission'], context=current['context'],
                    state=diff(current['recurrence']['initial'], previous['recurrence']['final']),
                    conv_history=diff(current['conv']['initial'][:, -history:], previous['conv']['x'][-history:].T)))
    args.out.write_text(json.dumps(dict(torch_version=torch.__version__, cases=rows, state_carry=carry,
        scope='Same captured operands and incoming state; the final prefill kernel chunk only, not the entire historical state.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
