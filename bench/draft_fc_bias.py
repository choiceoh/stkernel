"""Explicit FC-pair collection and CPU fitting; never started by serving.

Collection uses an already prepared drafter with retained BF16 sources and
caller-supplied committed-decode inputs. It does not boot or reserve hardware.
Fit with `bench/draft_tune.py fc-bias pairs.pt --out bias.json`.
"""
import torch

from engine.profiles.glm53.draft_fc_bias import decode_reader, reader_identity, validate_entry


@torch.inference_mode()
def collect_fc_pairs(drafter, batches, *, max_rows=4096):
    """Each batch: aux [M,K], keep [M] bool, ids [M] request families, split.

    Preserve the native batch shape when executing the reader, but retain only
    committed rows. Preparation must retain fc.weight (no consume/compact).
    """
    layer = drafter.dense['fc.weight']
    source, gamma = drafter.p['fc.weight'], drafter.p['hidden_norm.weight']
    if source is None:
        raise ValueError('FC pair collection requires retained BF16 source weights')
    if type(max_rows) is not int or not 1 <= max_rows <= 65536:
        raise ValueError('FC pair collection needs an explicit bounded row budget')
    reader = decode_reader(layer)
    if getattr(reader, 'observer', None) is not None:
        raise ValueError('FC pair collection needs an observer-free prepared reader')
    sha = reader_identity(layer, source, gamma, drafter.F.rms_eps)
    rows = {part: dict(reference=[], actual=[], ids=[]) for part in ('train', 'validation')}
    count = 0
    for batch in batches:
        x, keep, ids, split = (batch[k] for k in ('aux', 'keep', 'ids', 'split'))
        if (split not in rows or x.ndim != 2 or not 1 <= len(x) <= 32 or x.shape[1] != layer.cols
                or x.dtype != torch.bfloat16 or x.device != source.device or keep.shape != (len(x),)
                or keep.dtype != torch.bool or keep.device != x.device or len(ids) != len(x)
                or any(not isinstance(i, str) or not i for i in ids)):
            raise ValueError('FC pairs require decode batches, committed-row masks, family ids and an explicit split')
        mask = keep.cpu()
        kept = int(mask.sum())
        if count + kept > max_rows:
            raise ValueError('FC pair collection exceeded its row budget')
        if not kept:
            continue
        if not bool(torch.isfinite(x[keep]).all()):
            raise ValueError('FC pair inputs must be finite on committed rows')
        # The reader sees the original M, including ghost/rejected rows, just
        # as serving does. Neither dense calibration nor bias is invoked here.
        actual = reader(x)
        reference = torch.nn.functional.linear(x, source)
        row = rows[split]
        row['actual'].append(actual[keep].cpu())
        row['reference'].append(reference[keep].cpu())
        row['ids'].extend(i for i, use in zip(ids, mask.tolist()) if use)
        count += kept
    if any(not row['ids'] for row in rows.values()):
        raise ValueError('FC pairs need committed rows in both train and validation')
    for row in rows.values():
        row['actual'], row['reference'] = torch.cat(row['actual']), torch.cat(row['reference'])
    return dict(version=1, kind='decode_fc_output_pairs', rank=drafter.target.comm.rank,
                world_size=drafter.target.comm.world_size, reader_sha256=sha, norm_weight=gamma.cpu().clone(),
                eps=drafter.F.rms_eps, **rows)


def fit_fc_bias(bundle):
    """Train a mean residual; held-out FC and normalized errors can only veto it."""
    gamma, eps = bundle['norm_weight'], bundle['eps']
    rank, world = bundle['rank'], bundle['world_size']
    if (type(bundle.get('version')) is not int or bundle['version'] != 1
            or bundle.get('kind') != 'decode_fc_output_pairs'
            or type(rank) is not int or type(world) is not int or not 0 <= rank < world <= 100
            or gamma.ndim != 1 or gamma.numel() == 0 or gamma.dtype != torch.bfloat16
            or gamma.device.type != 'cpu' or not bool(torch.isfinite(gamma).all())
            or type(eps) not in (float, int) or not 0 < eps < 1):
        raise ValueError('FC fit requires finite BF16 normalization and a decode FC pair bundle')
    groups = []
    for part in ('train', 'validation'):
        row = bundle[part]
        ref, actual, ids = (row[k] for k in ('reference', 'actual', 'ids'))
        if (ref.ndim != 2 or ref.shape[1] != len(gamma) or len(ref) == 0 or ref.shape != actual.shape
                or any(t.dtype != torch.bfloat16 or t.device.type != 'cpu' or not bool(torch.isfinite(t).all())
                       for t in (ref, actual)) or len(ids) != len(ref)
                or any(not isinstance(i, str) or not i for i in ids)):
            raise ValueError('FC fit requires finite paired BF16 rows and a family id per row')
        groups.append(set(ids))
    if groups[0] & groups[1]:
        raise ValueError('FC train and validation request families must be disjoint')
    train, valid = bundle['train'], bundle['validation']
    bias = (train['reference'].double() - train['actual'].double()).mean(0).float()
    profile = validate_entry(dict(reader_sha256=bundle['reader_sha256'], values=bias.tolist()))
    def norm(x, *, bias=None):
        xf = x.float() if bias is None else x.float() + bias
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * gamma
    def error(actual, reference):
        return float((actual.double() - reference.double()).square().sum()
                     / reference.double().square().sum().clamp_min(1e-30))
    ref, actual = valid['reference'], valid['actual']
    normalized = norm(ref)
    baseline = dict(fc_error=error(actual, ref), norm_error=error(norm(actual), normalized))
    candidate = dict(fc_error=error(actual.float() + bias, ref),
                     norm_error=error(norm(actual, bias=bias), normalized))
    accepted = all(candidate[k] < baseline[k] for k in baseline)
    return dict(version=1, fc_bias={str(rank): profile} if accepted else {}, evidence=dict(
        kind='held_out_decode_FC_bias', live_acceptance=False, selected=accepted,
        train_rows=len(train['ids']), validation_rows=len(valid['ids']),
        train_families=len(groups[0]), validation_families=len(groups[1]),
        baseline=baseline, candidate=candidate))


def fit_fc_bias_ranks(bundles):
    """A single shared profile, with corrections fitted to each rank's own pack."""
    if not bundles:
        raise ValueError('FC fit needs all TP rank bundles')
    results = [fit_fc_bias(bundle) for bundle in bundles]
    world = bundles[0]['world_size']
    if (any(b['world_size'] != world for b in bundles) or len(bundles) != world
            or {b['rank'] for b in bundles} != set(range(world))):
        raise ValueError('FC fit needs exactly one bundle for every TP rank')
    # A request family must not leak across the split through another rank.
    train = {i for b in bundles for i in b['train']['ids']}
    valid = {i for b in bundles for i in b['validation']['ids']}
    if train & valid:
        raise ValueError('FC train and validation families must be disjoint across TP ranks')
    selected = all(r['evidence']['selected'] for r in results)
    return dict(version=1, fc_bias={k: v for r in results for k, v in r['fc_bias'].items()} if selected else {},
                evidence=dict(kind='held_out_decode_FC_bias', live_acceptance=False, selected=selected,
                    ranks={str(b['rank']): r['evidence'] for b, r in zip(bundles, results)}))
