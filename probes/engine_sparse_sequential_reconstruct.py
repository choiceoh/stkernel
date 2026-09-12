"""Reselect down-projection sparse pairs using the complete expert's target.

Given the actual sparse first projection, solve a ridge-anchored dense target
for the second projection, then apply pair-constrained SparseGPT. This changes
the sparse weights and mask without adding inference factors. Validation
selects from the unchanged baseline and three anchored reconstructions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_sparse_calibrate import inputs16, inputs32, split_rows
from probes.engine_sparse_nvfp4 import Library, Projection, dequant, qualify
from probes.engine_sparse_nvfp4_prune import quantize32, read_experts
from probes.engine_sparse_recovery import hessian, metrics, sparsegpt_pair


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def anchored_target(weight, inputs, target, damping):
    """Minimize ||X W.T-Y||^2/n + lambda ||W-W_original||^2."""
    if not 0 < damping <= 1 or inputs.ndim != 2 or target.shape != (len(inputs), len(weight)):
        raise ValueError('invalid anchored regression dimensions/damping')
    if not len(inputs) or not all(torch.isfinite(t).all() for t in (weight, inputs, target)):
        raise ValueError('nonempty finite regression inputs required')
    x = inputs.float()
    h = hessian(x)
    h.diagonal().add_(damping * h.diagonal().mean().clamp_min(1e-12))
    cross = (target.float()-x @ weight.T).T @ x / len(x)
    correction = torch.cholesky_solve(cross.T, torch.linalg.cholesky(h)).T
    return weight + correction


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((4 << 30)/torch.cuda.get_device_properties(0).total_memory)
    info = json.loads(args.capture.with_suffix('.json').read_text())
    recovery = json.loads(args.recovery.read_text())
    if file_hash(args.capture) != recovery['capture_sha256']:
        raise ValueError('capture identity mismatch')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    packed = torch.load(args.recovery.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    exports = {k: v.clone() for k, v in packed.items()}
    report = dict(scope=__doc__, source_sha256=file_hash(__file__),
                  recovery_sha256=file_hash(args.recovery), capture_sha256=recovery['capture_sha256'],
                  selection='minimum validation whole-expert relative L2; unchanged baseline wins ties',
                  production_adopted=False, cases=[])
    library = Library(args.library)
    for case in recovery['cases']:
        if 'skipped' in case:
            continue
        expert = case['expert']
        indices, counts = split_rows(payload, info, expert, recovery['training_cap'])
        hidden = {s: payload['x'][idx].cuda() for s, idx in indices.items()}
        original, sparse = {}, {}
        for name in ('w13', 'w2'):
            raw, sf, hashes = read_experts(args.rank, 3, name, [expert])
            if any(case['tensor_sha256'][k] != v for k, v in hashes.items()):
                raise ValueError('original expert weight changed')
            original[name] = dequant(raw, sf)[0]
            sparse[name] = dequant(*(packed[f'e{expert}.{name}.{suffix}'].cuda() for suffix in ('packed', 'sf')))
        inputs, targets, middle = {}, {}, {}
        for split, x in hidden.items():
            up, gate = (inputs16(x) @ original['w13'].T).chunk(2, -1)
            targets[split] = inputs16(swiglu_clamped(gate, up, 10.)) @ original['w2'].T
            up, gate = (inputs32(x) @ sparse['w13'].T).chunk(2, -1)
            middle[split] = swiglu_clamped(gate, up, 10.)
            inputs[split] = inputs32(middle[split])
        moment = hessian(inputs['train'])
        candidates = {'independent': (sparse['w2'], *(packed[f'e{expert}.w2.{s}'].cuda() for s in ('packed', 'sf')))}
        for damping in (.01, .1, 1.):
            target = anchored_target(original['w2'], inputs['train'], targets['train'], damping)
            candidates[f'anchored_{damping:g}'] = sparsegpt_pair(target, moment, .1)
        scores = {k: {s: metrics(inputs[s] @ value[0].T, targets[s]) for s in ('train', 'validation')}
                  for k, value in candidates.items()}
        chosen = min(scores, key=lambda k: (scores[k]['validation']['relative_l2'], k != 'independent', k))
        for key, value in candidates.items():
            scores[key]['test'] = metrics(inputs['test'] @ value[0].T, targets['test'])
        _, pk, sf = candidates[chosen]
        exports[f'e{expert}.w2.packed'], exports[f'e{expert}.w2.sf'] = pk.cpu(), sf.cpu()
        # Quantize actual deployed inputs and validate the chosen export natively.
        xp, xs = quantize32(middle['test'][:8])
        projection = Projection(library, pk[None], xp[None], sf[None], xs[None])
        try:
            correctness = qualify(projection)
        finally:
            projection.close()
        old_pairs = (packed[f'e{expert}.w2.packed'].cuda() & 0x77).ne(0)
        new_pairs = (pk & 0x77).ne(0)
        row = dict(expert=expert, counts=counts, selected=chosen, candidates=scores,
                   pair_mask_changed_fraction=(old_pairs != new_pairs).float().mean().item(),
                   kernel_correctness=correctness)
        report['cases'].append(row)
        print(json.dumps(dict(expert=expert, selected=chosen, before=scores['independent']['validation'],
                              after=scores[chosen]['validation'], test=scores[chosen]['test'])), flush=True)
    report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
    torch.save(exports, args.out.with_suffix('.weights.pt'))
    report['export_sha256'] = file_hash(args.out.with_suffix('.weights.pt'))
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
