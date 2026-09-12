"""Ridge-calibrated low-rank residual for the sparse FP4 pilot.

EoRA-style activation-aware compensation, implemented as reduced-rank ridge
regression of the measured projection error. This is not stock EoRA: it also
fits the K16-to-K32 input-quantization difference, and stores BF16 factors.
Factors remain separate from sparse weights; merging would destroy sparsity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_sparse_calibrate import inputs16, inputs32, split_rows
from probes.engine_sparse_nvfp4 import dequant
from probes.engine_sparse_nvfp4_prune import read_experts
from probes.engine_sparse_recovery import metrics


@torch.inference_mode()
def fit_lowrank(inputs, residual, max_rank=64, damping=.1):
    """Return B,A such that X @ A.T @ B.T approximates residual [N,O]."""
    if inputs.ndim != 2 or residual.ndim != 2 or len(inputs) != len(residual) or not len(inputs):
        raise ValueError('matching nonempty input and residual matrices required')
    if not 0 < damping <= 1 or not 1 <= max_rank <= min(inputs.shape[1], residual.shape[1]):
        raise ValueError('invalid damping or rank')
    if not torch.isfinite(inputs).all() or not torch.isfinite(residual).all():
        raise ValueError('finite calibration data required')
    x, y = inputs.float(), residual.float()
    h = x.T @ x / len(x)
    h.diagonal().add_(damping * h.diagonal().mean().clamp_min(1e-12))
    lower = torch.linalg.cholesky(h)
    cross = y.T @ x / len(x)
    whitened = torch.linalg.solve_triangular(lower, cross.T, upper=False).T
    left, singular, right = torch.linalg.svd(whitened, full_matrices=False)
    b = left[:, :max_rank] * singular[:max_rank]
    a = torch.linalg.solve_triangular(lower.T, right[:max_rank].T, upper=True).T
    return b.to(torch.bfloat16).contiguous(), a.to(torch.bfloat16).contiguous()


def apply_residual(x, b, a):
    return F.linear(F.linear(x.to(torch.bfloat16), a), b).float()


def choose_rank(scores):
    # Deterministic smallest rank wins a tie. Test scores do not enter here.
    return min(scores, key=lambda rank: (scores[rank]['validation']['relative_l2'], rank))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ranks', type=int, nargs='+', default=[16, 32, 64])
    args = ap.parse_args()
    if not args.ranks or min(args.ranks) < 1 or max(args.ranks) > 128:
        ap.error('residual ranks must be 1..128')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((4 << 30)/torch.cuda.get_device_properties(0).total_memory)
    capture_info = json.loads(args.capture.with_suffix('.json').read_text())
    recovery = json.loads(args.recovery.read_text())
    capture_hash = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    if capture_hash != capture_info['tensor_file_sha256'] or capture_hash != recovery['capture_sha256']:
        raise ValueError('capture/recovery identity mismatch')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    packed = torch.load(args.recovery.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    report = dict(scope=__doc__, capture_sha256=capture_hash,
                  recovery_sha256=hashlib.sha256(args.recovery.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ranks=args.ranks, damping=.1, rank_selection='validation projection relative L2',
                  production_adopted=False, cases=[])
    exports = {}
    for case in recovery['cases']:
        if 'skipped' in case:
            continue
        expert = case['expert']
        indices, counts = split_rows(payload, capture_info, expert, recovery['training_cap'])
        hidden = {s: payload['x'][idx].cuda() for s, idx in indices.items()}
        originals, sparse = {}, {}
        for name in ('w13', 'w2'):
            p, sf, _ = read_experts(args.rank, 3, name, [expert])
            originals[name] = dequant(p, sf)[0]
            sparse[name] = dequant(packed[f'e{expert}.{name}.packed'].cuda(),
                                   packed[f'e{expert}.{name}.sf'].cuda())
        row = dict(expert=expert, counts=counts, projections={})
        factors, targets, current_inputs = {}, {}, hidden
        for name in ('w13', 'w2'):
            start = time.monotonic()
            targets[name] = {s: inputs16(x) @ originals[name].T for s, x in current_inputs.items()}
            sparse_outputs = {s: inputs32(x) @ sparse[name].T for s, x in current_inputs.items()}
            b, a = fit_lowrank(current_inputs['train'], targets[name]['train']-sparse_outputs['train'],
                               max(args.ranks), .1)
            scores = {}
            for rank in args.ranks:
                bb, aa = b[:, :rank].contiguous(), a[:rank].contiguous()
                scores[rank] = {s: metrics(sparse_outputs[s]+apply_residual(current_inputs[s], bb, aa),
                                          targets[name][s]) for s in ('train', 'validation')}
            selected = choose_rank(scores)
            factors[name] = (b[:, :selected].contiguous(), a[:selected].contiguous())
            for rank in args.ranks:
                scores[rank]['test'] = metrics(sparse_outputs['test'] +
                    apply_residual(current_inputs['test'], b[:, :rank].contiguous(), a[:rank].contiguous()),
                    targets[name]['test'])
            row['projections'][name] = dict(selected_rank=selected, candidates=scores,
                                            fit_seconds=time.monotonic()-start,
                                            factor_bytes=sum(t.numel()*t.element_size() for t in factors[name]))
            exports[f'e{expert}.{name}.B'] = factors[name][0].cpu()
            exports[f'e{expert}.{name}.A'] = factors[name][1].cpu()
            if name == 'w13':
                current_inputs = {s: swiglu_clamped(y.chunk(2, -1)[1], y.chunk(2, -1)[0], 10.)
                                  for s, y in targets[name].items()}
        row['expert_chain'] = {}
        for split in ('train', 'validation', 'test'):
            fc1 = inputs32(hidden[split]) @ sparse['w13'].T + apply_residual(hidden[split], *factors['w13'])
            up, gate = fc1.chunk(2, -1)
            mid = swiglu_clamped(gate, up, 10.)
            fc2 = inputs32(mid) @ sparse['w2'].T + apply_residual(mid, *factors['w2'])
            row['expert_chain'][split] = metrics(fc2, targets['w2'][split])
        report['cases'].append(row)
        report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        torch.save(exports, args.out.with_suffix('.weights.pt'))
        print(json.dumps(dict(expert=expert, selected_ranks={n:r['selected_rank'] for n,r in row['projections'].items()},
                              projection_test={n:r['candidates'][r['selected_rank']]['test']['relative_l2']
                                               for n,r in row['projections'].items()},
                              chain_test=row['expert_chain']['test'])), flush=True)


if __name__ == '__main__':
    main()
