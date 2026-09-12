"""Evaluate frozen sparse/residual choices on a fresh prompt-only holdout.

No fitting or candidate selection is permitted here. Measures L3/rank0
projections and selected experts' weighted output, not full-model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_sparse_calibrate import inputs16, inputs32
from probes.engine_sparse_nvfp4 import dequant
from probes.engine_sparse_nvfp4_prune import read_experts
from probes.engine_sparse_recovery import magnitude, metrics
from probes.engine_sparse_residual import apply_residual


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--training-capture', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--residual', type=Path, required=True)
    ap.add_argument('--block-fit', type=Path)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((3 << 30)/torch.cuda.get_device_properties(0).total_memory)
    info = json.loads(args.capture.with_suffix('.json').read_text())
    prior = json.loads(args.training_capture.with_suffix('.json').read_text())
    new_ids = {p['id'] for p in info['prompts']}
    old_ids = {p['id'] for p in prior['prompts']}
    if new_ids & old_ids or any(p['split'] != 'test' for p in info['prompts']):
        raise ValueError('final prompts must be disjoint and all test-only')
    if {tuple(p['token_ids']) for p in info['prompts']} & {tuple(p['token_ids']) for p in prior['prompts']}:
        raise ValueError('tokenized prompt overlap')
    if file_hash(args.capture) != info['tensor_file_sha256']:
        raise ValueError('holdout tensor hash mismatch')
    recovery = json.loads(args.recovery.read_text())
    residual = json.loads(args.residual.read_text())
    if recovery['capture_sha256'] != prior['tensor_file_sha256'] or residual['recovery_sha256'] != file_hash(args.recovery):
        raise ValueError('frozen artifacts refer to different calibration data')
    packed = torch.load(args.recovery.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    factors = torch.load(args.residual.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    block_weights = None
    if args.block_fit:
        block_info = json.loads(args.block_fit.read_text())
        if block_info['recovery_sha256'] != file_hash(args.recovery):
            raise ValueError('block reconstruction used different starting weights')
        block_weights = torch.load(args.block_fit.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    report = dict(scope=__doc__, experts=recovery['experts'], layer=3, rank=0,
                  final_prompts=len(info['prompts']), tokens=len(payload['x']),
                  choices_frozen_before_final_capture=True, production_adopted=False,
                  artifact_sha256={p.name: file_hash(p) for p in [args.capture, args.recovery,
                      args.residual, args.recovery.with_suffix('.weights.pt'), args.residual.with_suffix('.weights.pt')]},
                  source_sha256=file_hash(__file__), cases=[])
    variants = ('original', 'magnitude', 'calibrated', 'residual')
    if block_weights is not None:
        variants += ('joint_reconstruction',)
        report['artifact_sha256'].update({p.name: file_hash(p) for p in
                                         [args.block_fit, args.block_fit.with_suffix('.weights.pt')]})
    totals = {v: torch.zeros(len(payload['x']), 4096, device='cuda') for v in variants}
    route_count = 0
    for case in recovery['cases']:
        if 'skipped' in case:
            continue
        expert = case['expert']
        selected = payload['selected'] == expert
        indices = selected.any(-1).nonzero().flatten()
        if len(indices) == 0:
            report['cases'].append(dict(expert=expert, rows=0))
            continue
        route_count += len(indices)
        x = payload['x'][indices].cuda()
        coefficient = (payload['coefficient'][indices] * selected[indices]).sum(-1).cuda()
        originals, corrected, mag, residual_factors, joint = {}, {}, {}, {}, {}
        for name in ('w13', 'w2'):
            raw, sf, hashes = read_experts(args.rank, 3, name, [expert])
            for key, value in hashes.items():
                if case['tensor_sha256'][key] != value:
                    raise ValueError('original expert weight changed')
            originals[name] = dequant(raw, sf)[0]
            pk, sc = (packed[f'e{expert}.{name}.{suffix}'] for suffix in ('packed', 'sf'))
            if hashlib.sha256(pk.numpy().tobytes()).hexdigest() != case['projections'][name]['packed_sha256']:
                raise ValueError('frozen sparse weight changed')
            if hashlib.sha256(sc.numpy().tobytes()).hexdigest() != case['projections'][name]['scale_sha256']:
                raise ValueError('frozen sparse scale changed')
            corrected[name] = dequant(pk.cuda(), sc.cuda())
            mag[name] = magnitude(originals[name])[0]
            residual_factors[name] = tuple(factors[f'e{expert}.{name}.{side}'].cuda() for side in ('B', 'A'))
            if block_weights is not None:
                joint[name] = dequant(*(block_weights[f'e{expert}.{name}.{suffix}'].cuda()
                                        for suffix in ('packed', 'sf')))
        row = dict(expert=expert, rows=len(indices), projections={}, expert_chain={})
        baseline_inputs = x
        for name in ('w13', 'w2'):
            x16, x32 = inputs16(baseline_inputs), inputs32(baseline_inputs)
            ref = x16 @ originals[name].T
            sparse_out = x32 @ corrected[name].T
            row['projections'][name] = dict(
                activation_only=metrics(x32 @ originals[name].T, ref),
                magnitude=metrics(x32 @ mag[name].T, ref),
                calibrated=metrics(sparse_out, ref),
                residual=metrics(sparse_out+apply_residual(baseline_inputs, *residual_factors[name]), ref))
            if joint:
                row['projections'][name]['joint_reconstruction'] = metrics(x32 @ joint[name].T, ref)
            if name == 'w13':
                up, gate = ref.chunk(2, -1)
                baseline_inputs = swiglu_clamped(gate, up, 10.)
        outputs = {}
        for variant in variants:
            weight = originals if variant == 'original' else mag if variant == 'magnitude' else corrected
            if variant == 'joint_reconstruction':
                weight = joint
            quant = inputs16 if variant == 'original' else inputs32
            fc1 = quant(x) @ weight['w13'].T
            if variant == 'residual':
                fc1 += apply_residual(x, *residual_factors['w13'])
            up, gate = fc1.chunk(2, -1)
            mid = swiglu_clamped(gate, up, 10.)
            fc2 = quant(mid) @ weight['w2'].T
            if variant == 'residual':
                fc2 += apply_residual(mid, *residual_factors['w2'])
            outputs[variant] = fc2
            totals[variant].index_add_(0, indices.cuda(), fc2 * coefficient[:, None])
        for variant in variants[1:]:
            row['expert_chain'][variant] = metrics(outputs[variant], outputs['original'])
        report['cases'].append(row)
        print(json.dumps(dict(expert=expert, rows=len(indices), chain=row['expert_chain'])), flush=True)
    active = torch.isin(payload['selected'], torch.tensor(recovery['experts'])).any(-1).cuda()
    report['selected_route_fraction'] = route_count / payload['selected'].numel()
    report['tokens_with_selected_route'] = int(active.sum())
    report['weighted_selected_expert_sum'] = {v: metrics(totals[v][active], totals['original'][active])
                                             for v in variants[1:]}
    report['per_prompt'] = []
    for i, prompt in enumerate(info['prompts']):
        mask = (payload['prompt_index'] == i).cuda() & active
        if mask.any():
            report['per_prompt'].append(dict(id=prompt['id'],
                variants={v: metrics(totals[v][mask], totals['original'][mask]) for v in variants[1:]}))
    report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
