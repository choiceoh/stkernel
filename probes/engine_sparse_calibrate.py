"""L3/rank0 calibration and held-out evaluation of legal sparse FP4 weights.

Uses real prefix activations from engine_sparse_capture.py. Expert choice is
by training-set route frequency; method/damping choice is by validation error.
Test prompts are not used for either decision. No production adoption occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_moe_real_check import hardware_quant
from probes.engine_sparse_nvfp4 import Library, Projection, benchmark, dequant, qualify
from probes.engine_sparse_nvfp4_prune import quantize32, read_experts
from probes.engine_sparse_recovery import (choose_validation, hessian, magnitude,
                                          metrics, sparsegpt_pair, wanda_pair)


def inputs16(x):
    packed, sf = hardware_quant(x.contiguous(), torch.ones((), device=x.device))
    return dequant(packed.view(torch.uint8), sf.view(torch.uint8))


def inputs32(x):
    packed, sf = quantize32(x)
    return dequant(packed, sf)


def limit_rows(indices, cap):
    if len(indices) <= cap:
        return indices
    # Deterministic coverage across prompts, not a prefix-only subsample.
    take = torch.linspace(0, len(indices)-1, cap).round().long()
    return indices[take]


def split_rows(payload, info, expert, cap):
    selected = (payload['selected'] == expert).any(-1)
    result, counts = {}, {}
    for split in ('train', 'validation', 'test'):
        prompts = [i for i, p in enumerate(info['prompts']) if p['split'] == split]
        mask = torch.isin(payload['prompt_index'], torch.tensor(prompts)) & selected
        index = mask.nonzero().flatten()
        counts[split] = len(index)
        result[split] = limit_rows(index, cap if split == 'train' else 512)
    return result, counts


@torch.inference_mode()
def fit_projection(weight, inputs, damping, library, timing):
    x16 = {s: inputs16(x) for s, x in inputs.items()}
    x32 = {s: inputs32(x) for s, x in inputs.items()}
    targets = {s: x @ weight.T for s, x in x16.items()}
    moment = hessian(x32['train'])
    candidates = dict(magnitude=magnitude(weight), wanda=wanda_pair(weight, x32['train']))
    build_seconds = {}
    for damp in damping:
        name = f'pair_sparsegpt_damp_{damp:g}'
        start = time.monotonic()
        candidates[name] = sparsegpt_pair(weight, moment, damp)
        torch.cuda.synchronize()
        build_seconds[name] = time.monotonic()-start
    summaries = {}
    for name, (reconstructed, _, _) in candidates.items():
        summaries[name] = {s: metrics(x32[s] @ reconstructed.T, targets[s])
                           for s in ('train', 'validation')}
    selected = choose_validation(summaries)
    # Selection is final before reading any test-output metric.
    for name, (reconstructed, _, _) in candidates.items():
        summaries[name]['test'] = metrics(x32['test'] @ reconstructed.T, targets['test'])
    reconstructed, packed, sf = candidates[selected]
    xp, xs = quantize32(inputs['test'][:min(len(inputs['test']), 128)])
    projection = Projection(library, packed[None], xp[None], sf[None], xs[None])
    try:
        correctness = qualify(projection)
        measured = benchmark(projection, 12) if timing else None
    finally:
        projection.close()
    result = dict(selected=selected, candidates=summaries, build_seconds=build_seconds,
                  activation_only={s: metrics(x32[s] @ weight.T, targets[s]) for s in x32},
                  kernel_correctness=correctness, timing=measured,
                  packed_sha256=hashlib.sha256(packed.cpu().numpy().tobytes()).hexdigest(),
                  scale_sha256=hashlib.sha256(sf.cpu().numpy().tobytes()).hexdigest())
    return result, (reconstructed, packed, sf), (x16, x32, targets)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--experts', type=int, default=4)
    ap.add_argument('--train-cap', type=int, default=2048)
    ap.add_argument('--damping', type=float, nargs='+', default=[.01, .1])
    ap.add_argument('--timing', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.experts <= 8 or not 128 <= args.train_cap <= 4096:
        ap.error('bounded pilot: 1..8 experts; 128..4096 training rows each')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((3 << 30)/torch.cuda.get_device_properties(0).total_memory)
    info = json.loads(args.capture.with_suffix('.json').read_text())
    digest = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    if digest != info['tensor_file_sha256'] or info['layer'] != 3:
        raise ValueError('capture hash or layer mismatch')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    train_prompts = [i for i, p in enumerate(info['prompts']) if p['split'] == 'train']
    train_rows = torch.isin(payload['prompt_index'], torch.tensor(train_prompts))
    frequency = torch.bincount(payload['selected'][train_rows].long().flatten(), minlength=288)
    experts = frequency.argsort(descending=True, stable=True)[:args.experts].tolist()
    report = dict(scope=__doc__, capture_sha256=digest, layer=3, rank=0, experts=experts,
                  train_route_counts=frequency.tolist(), training_cap=args.train_cap,
                  expert_choice='descending training route count; expert id tie-break',
                  method_choice='minimum validation projection relative L2; no test selection',
                  scope_limit='short prefill prompts; one layer; one TP rank; selected experts only; no model quality score',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), Path(__file__).with_name('engine_sparse_recovery.py')]},
                  production_adopted=False, cases=[])
    library = Library(args.library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    exports = {}
    for expert in experts:
        indices, counts = split_rows(payload, info, expert, args.train_cap)
        if counts['train'] < 128 or min(counts['validation'], counts['test']) < 16:
            report['cases'].append(dict(expert=expert, counts=counts, skipped='insufficient routed coverage'))
            continue
        hidden = {s: payload['x'][idx].cuda() for s, idx in indices.items()}
        original, hashes = {}, {}
        for name in ('w13', 'w2'):
            packed, sf, digest = read_experts(args.rank, 3, name, [expert])
            original[name] = dequant(packed, sf)[0]
            hashes.update(digest)
        row = dict(expert=expert, counts=counts, fitted_rows={s: len(x) for s, x in hidden.items()},
                   tensor_sha256=hashes, projections={})
        result13, chosen13, baseline13 = fit_projection(original['w13'], hidden,
                                                        args.damping, library, args.timing)
        row['projections']['w13'] = result13
        mid = {}
        for split, fc1 in baseline13[2].items():
            up, gate = fc1.chunk(2, -1)
            mid[split] = swiglu_clamped(gate, up, 10.)
        result2, chosen2, baseline2 = fit_projection(original['w2'], mid,
                                                     args.damping, library, args.timing)
        row['projections']['w2'] = result2
        row['expert_chain'] = {}
        for split in ('train', 'validation', 'test'):
            up, gate = (baseline13[1][split] @ chosen13[0].T).chunk(2, -1)
            changed_mid = swiglu_clamped(gate, up, 10.)
            candidate = inputs32(changed_mid) @ chosen2[0].T
            reference = baseline2[2][split]
            row['expert_chain'][split] = metrics(candidate, reference)
        for name, chosen in [('w13', chosen13), ('w2', chosen2)]:
            exports[f'e{expert}.{name}.packed'] = chosen[1].cpu()
            exports[f'e{expert}.{name}.sf'] = chosen[2].cpu()
        report['cases'].append(row)
        report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(expert=expert, counts=counts,
                              projections={n: dict(selected=r['selected'],
                                magnitude_test=r['candidates']['magnitude']['test']['relative_l2'],
                                selected_test=r['candidates'][r['selected']]['test']['relative_l2'])
                                for n, r in row['projections'].items()}, chain=row['expert_chain']['test'])), flush=True)
    torch.save(exports, args.out.with_suffix('.weights.pt'))
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
