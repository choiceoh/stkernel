"""Fit the down-projection residual on the deployed sparse expert's inputs.

Keep the sparse matrices and first-projection residual fixed. Refit only the
existing down-projection factors against the original complete expert output,
including the upstream pruning, SwiGLU and activation-quantization errors.
Factor shapes and inference operations are unchanged. This is a calibration
experiment on one layer/rank, not evidence of end-to-end model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_sparse_calibrate import inputs16, inputs32, split_rows
from probes.engine_sparse_nvfp4 import dequant
from probes.engine_sparse_nvfp4_prune import read_experts
from probes.engine_sparse_recovery import metrics
from probes.engine_sparse_residual import apply_residual, fit_lowrank


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def fit_deployed_residual(inputs, error, coefficient, rank, damping, conditioned):
    """Optionally fit router-weighted error with per-feature RMS ridge scaling.

    The coefficient multiplies both sides of the regression, so the loss is
    weighted by its square, matching the expert's routed output contribution.
    All transforms are absorbed into A; inference still uses BF16 B(A(x)).
    """
    if coefficient.shape != (len(inputs),) or not torch.isfinite(coefficient).all():
        raise ValueError('one finite routing coefficient per input required')
    if (coefficient < 0).any() or coefficient.square().sum() == 0:
        raise ValueError('nonnegative, nonzero routing coefficients required')
    x, y = inputs.float(), error.float()
    if conditioned:
        multiplier = coefficient.float() / coefficient.float().square().mean().sqrt()
        scale = (x * multiplier[:, None]).square().mean(0).sqrt()
        scale = scale.clamp_min(scale.square().mean().sqrt().clamp_min(1e-12) * 1e-3)
        b, a = fit_lowrank(x / scale * multiplier[:, None],
                          y * multiplier[:, None], rank, damping)
        # One final BF16 rounding after folding the feature transform into A.
        a = (a.float() / scale).to(torch.bfloat16).contiguous()
        return b, a
    return fit_lowrank(x, y, rank, damping)


def choose_deployed(scores):
    return min(scores, key=lambda name: (scores[name]['validation']['weighted']['relative_l2'],
                                         name != 'independent', name))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--residual', type=Path, required=True)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((4 << 30)/torch.cuda.get_device_properties(0).total_memory)
    info = json.loads(args.capture.with_suffix('.json').read_text())
    recovery = json.loads(args.recovery.read_text())
    previous = json.loads(args.residual.read_text())
    capture_hash, recovery_hash = file_hash(args.capture), file_hash(args.recovery)
    if capture_hash != info['tensor_file_sha256'] or capture_hash != recovery['capture_sha256']:
        raise ValueError('capture identity mismatch')
    if previous['recovery_sha256'] != recovery_hash:
        raise ValueError('independent factors use different sparse weights')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    packed = torch.load(args.recovery.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    factors = torch.load(args.residual.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    report = dict(scope=__doc__, source_sha256=file_hash(__file__), capture_sha256=capture_hash,
                  recovery_sha256=recovery_hash, previous_residual_sha256=file_hash(args.residual),
                  previous_weights_sha256=file_hash(args.residual.with_suffix('.weights.pt')),
                  selection='lowest validation routed-contribution relative L2; independent baseline eligible',
                  candidates='independent; deployed-input ridge damping .01/.1/1, with or without router/RMS conditioning',
                  production_adopted=False, cases=[])
    exports = {k: v.clone() for k, v in factors.items()}
    for case in recovery['cases']:
        if 'skipped' in case:
            continue
        expert = case['expert']
        indices, counts = split_rows(payload, info, expert, recovery['training_cap'])
        hidden = {s: payload['x'][idx].cuda() for s, idx in indices.items()}
        coefficients = {s: (payload['coefficient'][idx] *
                         (payload['selected'][idx] == expert)).sum(-1).cuda() for s, idx in indices.items()}
        original, sparse = {}, {}
        for name in ('w13', 'w2'):
            raw, sf, hashes = read_experts(args.rank, 3, name, [expert])
            if any(case['tensor_sha256'][k] != v for k, v in hashes.items()):
                raise ValueError('original expert weight changed')
            original[name] = dequant(raw, sf)[0]
            sparse[name] = dequant(*(packed[f'e{expert}.{name}.{suffix}'].cuda() for suffix in ('packed', 'sf')))
        first = tuple(factors[f'e{expert}.w13.{side}'].cuda() for side in ('B', 'A'))
        old = tuple(factors[f'e{expert}.w2.{side}'].cuda() for side in ('B', 'A'))
        targets, deployed, base = {}, {}, {}
        for split, x in hidden.items():
            original_fc1 = inputs16(x) @ original['w13'].T
            up, gate = original_fc1.chunk(2, -1)
            targets[split] = inputs16(swiglu_clamped(gate, up, 10.)) @ original['w2'].T
            sparse_fc1 = inputs32(x) @ sparse['w13'].T + apply_residual(x, *first)
            up, gate = sparse_fc1.chunk(2, -1)
            deployed[split] = swiglu_clamped(gate, up, 10.)
            base[split] = inputs32(deployed[split]) @ sparse['w2'].T
        rank = old[1].shape[0]
        candidates = {'independent': old}
        for conditioned in (False, True):
            for damping in (.01, .1, 1.):
                key = f'deployed_{"router_rms" if conditioned else "plain"}_{damping:g}'
                candidates[key] = fit_deployed_residual(deployed['train'], targets['train']-base['train'],
                                                       coefficients['train'], rank, damping, conditioned)
        def score(factor, split):
            output = base[split] + apply_residual(deployed[split], *factor)
            coeff = coefficients[split][:, None]
            return dict(unweighted=metrics(output, targets[split]),
                        weighted=metrics(output*coeff, targets[split]*coeff))
        scores = {key: {s: score(factor, s) for s in ('train', 'validation')}
                  for key, factor in candidates.items()}
        chosen = choose_deployed(scores)
        # All fitting and selection are complete before reporting diagnostic test scores.
        for key, factor in candidates.items():
            scores[key]['test'] = score(factor, 'test')
        for side, tensor in zip(('B', 'A'), candidates[chosen]):
            old_tensor = factors[f'e{expert}.w2.{side}']
            if tensor.shape != old_tensor.shape or tensor.dtype != old_tensor.dtype:
                raise AssertionError('factor budget changed')
            exports[f'e{expert}.w2.{side}'] = tensor.cpu()
        row = dict(expert=expert, counts=counts, rank=rank, selected=chosen, candidates=scores,
                   factor_bytes=sum(exports[f'e{expert}.{name}.{side}'].numel()*2
                                    for name in ('w13', 'w2') for side in ('B', 'A')))
        report['cases'].append(row)
        print(json.dumps(dict(expert=expert, selected=chosen,
                              validation_before=scores['independent']['validation'],
                              validation_after=scores[chosen]['validation'],
                              diagnostic_test=scores[chosen]['test'])), flush=True)
    report['factor_bytes'] = sum(t.numel()*t.element_size() for t in exports.values())
    assert report['factor_bytes'] == sum(t.numel()*t.element_size() for t in factors.values())
    report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exports, args.out.with_suffix('.weights.pt'))
    report['export_sha256'] = file_hash(args.out.with_suffix('.weights.pt'))
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
