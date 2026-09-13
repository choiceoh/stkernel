#!/usr/bin/env python3
"""CPU-only fitting of small draft controls. Output is an explicit, unqualified profile.

selector reads recorded selector_calibration JSONL, split by request group.
packing reads a torch weights/statistics/held-out-input bundle (see the guide).
fc-bias reads per-rank native/reference FC output pairs and fits a mean correction.
Neither command contacts a server, reserves GPUs, or promotes a default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fit_selector(records, grid=(0., .5, .75, 1., 1.25)):
    import math
    from engine.profiles.glm53.draft_tuning import number
    grid = sorted(set([1.] + [number(a, 0, 2, 'alpha grid') for a in grid]))
    groups, samples, width = {}, [], None
    projection_modes = set()
    for row in records:
        if row.get('kind') != 'draft_selector' or row.get('rank', 0) != 0 or row.get('policy_modified', False):
            continue
        projection = row.get('selector_projection_fp32', False)
        if type(projection) is not bool:
            raise ValueError('selector projection precision must be bool')
        projection_modes.add(projection)
        if len(projection_modes) != 1:
            raise ValueError('selector fitting cannot mix projection precision modes')
        group, split = row.get('sample_group'), row.get('split')
        if not isinstance(group, str) or not group or split not in ('train', 'validation'):
            raise ValueError('selector rows require sample_group and explicit train/validation split')
        if group in groups and groups[group] != split:
            raise ValueError('one request group cannot occur in both train and validation')
        groups[group] = split
        k = row['draft_width']
        if type(k) is not int or k <= 0 or (width is not None and width != k):
            raise ValueError('selector records must use the same positive draft width')
        width = k
        target, cand, unary, edge = (row[key] for key in ('target', 'candidates', 'unary', 'edge'))
        if not 0 < len(target) <= k or not len(target) == len(cand) == len(unary) == len(edge):
            raise ValueError('selector records must contain only the observed prefix and first mismatch')
        for step, (label, ids, u, e) in enumerate(zip(target, cand, unary, edge)):
            if (not ids or len(ids) != len(set(ids)) or not len(ids) == len(u) == len(e)
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in u + e)):
                raise ValueError('selector candidates must be unique with finite score components')
            samples.append((split, step, label, ids, u, e))
    if not samples or set(groups.values()) != {'train', 'validation'}:
        raise ValueError('independent train and validation request groups are required')
    def accuracy(rows, alpha):
        return sum(ids[max(range(len(ids)), key=lambda i: u[i] + alpha * e[i])] == y
                   for _, _, y, ids, u, e in rows)
    selected, report = [], []
    for step in range(width):
        train = [r for r in samples if r[0] == 'train' and r[1] == step]
        valid = [r for r in samples if r[0] == 'validation' and r[1] == step]
        alpha = max(grid, key=lambda a: (accuracy(train, a), a == 1., -abs(a - 1.))) if train else 1.
        base, candidate = accuracy(valid, 1.), accuracy(valid, alpha)
        # Training chooses; held-out data can only veto, never choose another grid point.
        if not valid or candidate <= base:
            alpha = 1.
        selected.append(alpha)
        report.append(dict(position=step, train_rows=len(train), validation_rows=len(valid),
            validation_baseline=base, validation_selected=accuracy(valid, alpha),
            validation_covered=sum(r[2] in r[3] for r in valid), alpha=alpha))
    return dict(version=1, selector_alpha=selected, selector_projection_fp32=projection_modes.pop(),
                evidence=dict(kind='held_out_position_agreement', live_acceptance=False, positions=report))


def tag_selector(records, groups):
    """Attach an explicit request-family split to unmodified serving JSONL rows."""
    tagged = []
    for row in records:
        if row.get('kind') != 'draft_selector' or row.get('rank', 0) != 0 or row.get('policy_modified', False):
            continue
        key = f"{row['request_token']}:{row['seq']}"
        if key not in groups:
            raise ValueError(f'missing request-family split for {key}')
        tagged.append(dict(row, sample_group=groups[key]['sample_group'], split=groups[key]['split']))
    return tagged


def _w4_reference(weight, hessian, damping):
    import torch
    from engine.kernels.dense.packing import (_E2M1_GRID, _E2M1_MIDS, _w4_row_shift,
        _w4_gptq_codes, mk_w4_dequant_rowmajor, gptq_factor)
    rows, cols = weight.shape
    need, shift, _ = _w4_row_shift(weight, (rows + 127) // 128 * 128, cols // 16, False)
    factor = gptq_factor(hessian, percdamp=damping, act_order=True, factor_device='cpu')
    codes, scales = _w4_gptq_codes(weight, shift, need, hessian,
        torch.tensor(_E2M1_MIDS), torch.tensor(_E2M1_GRID), act_order=True, factor=factor)
    nibbles = codes[:, ::2] | (codes[:, 1::2] << 4)
    return mk_w4_dequant_rowmajor(nibbles, scales, wgs=float(torch.exp2(-shift[0])))[:rows]


def fit_packing(bundle, alphas=(.25, .5, .75), dampings=(.005, .01, .02)):
    """Fit one draft norm group against held-out inputs, using the actual W4/A8 reference.

    `validation` is unsmoothed BF16 norm output. `weight_peaks` must cover ALL
    full source readers of that norm, including consumers on other TP ranks;
    `weights` contains the packed reader shards evaluated by this bundle.
    """
    import torch
    from engine.kernels.dense.smoothing import scales, fold, smooth_weight, smooth_hessian
    from engine.kernels.dense.packing import _mk_quant_x_ref
    from engine.profiles.glm53.draft_tuning import number
    identifiers = [bundle[f'{part}_ids'] for part in ('train', 'selection', 'validation')]
    if (any(not ids or any(not isinstance(i, str) or not i for i in ids) for ids in identifiers)
            or any(set(identifiers[i]) & set(identifiers[j]) for i in range(3) for j in range(i))):
        raise ValueError('packing fit requires disjoint calibration, selection and validation request ids')
    h, peaks, wp, norm, fit, audit, weights = (bundle[k] for k in
        ('H', 'amax', 'weight_peaks', 'norm_weight', 'selection', 'validation', 'weights'))
    cols = norm.numel()
    if (cols <= 0 or cols % 128 or h.shape != (cols, cols) or peaks.shape != (cols,)
            or wp.shape != (cols,) or norm.shape != (cols,)
            or any(x.ndim != 2 or x.shape[1] != cols or x.shape[0] < 1 or x.dtype != torch.bfloat16 for x in (fit, audit))
            or norm.dtype != torch.bfloat16 or not weights
            or any(w.ndim != 2 or w.shape[1] != cols or w.dtype != torch.bfloat16 for w in weights.values())
            or any(t.device.type != 'cpu' or not bool(torch.isfinite(t).all())
                   for t in (h, peaks, wp, norm, fit, audit, *weights.values()))):
        raise ValueError('packing fit needs finite CPU statistics, BF16 norm/validation/reader weights, K aligned to 128')
    if len(identifiers[1]) != len(fit) or len(identifiers[2]) != len(audit):
        raise ValueError('each held-out activation row needs its request-family id')
    norm_key = bundle['norm_key']
    import re
    match = re.fullmatch(r'(layers\.[0-9]+)\.(input_layernorm|post_attention_layernorm)\.weight', norm_key)
    reader = (match[1] + ('.self_attn.qkv' if match[2] == 'input_layernorm' else '.mlp.gate_up')) if match else None
    if set(weights) != {reader}:
        raise ValueError('packing bundle must evaluate the fused reader of its draft norm group')
    if any(bool((w.float().abs().amax(0) > wp).any()) for w in weights.values()):
        raise ValueError('weight_peaks must cover every full source reader, including the evaluated shard')
    if (bool((peaks < 0).any()) or bool((wp < 0).any()) or bool((h.diagonal() < 0).any())
            or not bool((h.diagonal() > 0).any()) or not bool((peaks > 0).any())):
        raise ValueError('packing statistics require nonnegative peaks and nonempty Hessian signal')
    alphas = sorted(set([.5] + [number(a, 0, 1, 'smoothing grid') for a in alphas]))
    dampings = sorted(set([.01] + [number(d, 0, 1, 'damping grid', positive=True) for d in dampings]))
    # Whole request families, never adjacent rows of one prompt, are held out.
    def loss(rows, factors, prepared):
        qx = _mk_quant_x_ref((rows.float() / factors).to(torch.bfloat16))
        error, energy = 0., 0.
        for name, weight in weights.items():
            target = rows.float() @ weight.float().T
            actual = (qx @ prepared[name].T).bfloat16().float()
            error += float((actual - target).double().square().sum())
            energy += float(target.double().square().sum())
        return error / max(energy, 1e-30)
    best, baseline, candidates = None, None, []
    for alpha in alphas:
        factors = fold(norm.clone(), scales(peaks, [wp.reshape(1, -1)], alpha=alpha))
        hs = smooth_hessian(h, factors)
        for damping in dampings:
            prepared = {name: _w4_reference(smooth_weight(w, factors), hs, damping) for name, w in weights.items()}
            item = dict(alpha=alpha, damping=damping, fit_error=loss(fit, factors, prepared),
                        audit_error=loss(audit, factors, prepared))
            if alpha == .5 and damping == .01:
                baseline = item
            key = (item['fit_error'], (alpha, damping) != (.5, .01))
            if best is None or key < best[0]:
                best = key, item
            candidates.append(item)
    selected = best[1] if best[1]['audit_error'] < baseline['audit_error'] else baseline
    return dict(version=1, smoothing_alpha={bundle['norm_key']: selected['alpha']},
        gptq_damping={key: selected['damping'] for key in weights},
        evidence=dict(kind='held_out_W4_A8_reconstruction', live_acceptance=False,
            baseline=baseline, selected=selected, candidates=candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('selector', 'packing', 'fc-bias'))
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--groups', type=Path, help='selector only: recording-token:seq to request-family/split JSON')
    parser.add_argument('--peer-bundle', type=Path, action='append', default=[],
                        help='fc-bias only: another TP rank bundle (repeat for every peer)')
    args = parser.parse_args()
    if args.peer_bundle and args.mode != 'fc-bias':
        parser.error('--peer-bundle applies only to fc-bias')
    if args.mode == 'selector':
        with args.input.open() as source:
            records = [json.loads(line) for line in source if line.strip()]
        if args.groups:
            records = tag_selector(records, json.loads(args.groups.read_text()))
        result = fit_selector(records)
    else:
        if args.groups:
            parser.error('--groups applies only to selector records')
        import torch
        bundle = torch.load(args.input, map_location='cpu', mmap=True, weights_only=True)
        if args.mode == 'fc-bias':
            from bench.draft_fc_bias import fit_fc_bias_ranks
            peers = [torch.load(path, map_location='cpu', mmap=True, weights_only=True) for path in args.peer_bundle]
            result = fit_fc_bias_ranks([bundle] + peers)
        else:
            result = fit_packing(bundle)
    with args.input.open('rb') as source:
        result['evidence']['input_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    if args.groups:
        result['evidence']['groups_sha256'] = hashlib.sha256(args.groups.read_bytes()).hexdigest()
    if args.peer_bundle:
        result['evidence']['peer_sha256'] = []
        for path in args.peer_bundle:
            with path.open('rb') as source:
                result['evidence']['peer_sha256'].append(hashlib.file_digest(source, 'sha256').hexdigest())
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
