"""Preparation-only identity and binding for a decode FC output correction.

The default FC reader is FP8 for every decode batch width. W4's row-dependent
fallback needs separate calibration and is deliberately not covered by this
profile. No hashing, host transfer or reference GEMM occurs during serving.
"""
from functools import cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re

import torch


AUTO_BIAS_FILE = 'draft-fc-bias.json'
MAX_AUTO_BYTES = 8 * 1024 * 1024


def validate_entry(value):
    if not isinstance(value, dict):
        raise ValueError('fc_bias must be an object')
    if set(value) != {'reader_sha256', 'values'}:
        raise ValueError('fc_bias requires reader_sha256 and values')
    sha, values = value['reader_sha256'], value['values']
    if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('FC bias reader_sha256 must be a SHA256 digest')
    if (not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 65536
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4e38 for v in values)):
        raise ValueError('FC bias values must be a nonempty finite FP32 vector')
    return dict(reader_sha256=sha, values=tuple(float(v) for v in values))


def validate_profile(value):
    if (not isinstance(value, dict) or any(not isinstance(k, str) or not re.fullmatch('0|[1-9][0-9]?', k)
                                          for k in value)):
        raise ValueError('fc_bias must map TP rank numbers to reader corrections')
    return {rank: validate_entry(entry) for rank, entry in value.items()}


def load_auto(path, facts, comm):
    """Read the CPU fitter's shared artifact; absent/stale caches never block boot.

    Agree presence and bytes before any rank may hash its prepared GPU reader.
    Only the correction is imported, never other tuning flags from a cache file.
    """
    profile, digest, error = {}, None, None
    try:
        with Path(path).open('rb') as source:
            raw = source.read(MAX_AUTO_BYTES + 1)
        if len(raw) > MAX_AUTO_BYTES:
            raise ValueError('FC bias artifact exceeds the size limit')
        value = json.loads(raw)
        if (not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1
                or value.keys() - {'version', 'fc_bias', 'evidence'}):
            raise ValueError('automatic FC bias requires a version 1 fc-bias fitter artifact')
        profile = validate_profile(value.get('fc_bias', {}))
        ranks = {str(rank) for rank in range(comm.world_size)}
        if set(profile) != ranks or any(len(entry['values']) != facts.hidden for entry in profile.values()):
            raise ValueError('FC bias artifact must cover every TP rank with the correct hidden width')
        evidence = value.get('evidence', {})
        if (evidence.get('kind') != 'held_out_decode_FC_bias' or evidence.get('selected') is not True
                or set(evidence.get('ranks', {})) != ranks):
            raise ValueError('FC bias artifact needs held-out fitting evidence for every TP rank')
        for report in evidence['ranks'].values():
            if (report.get('selected') is not True
                    or any(type(report.get(key)) is not int or report[key] < 1
                           for key in ('train_rows', 'validation_rows', 'train_families', 'validation_families'))):
                raise ValueError('FC bias artifact needs selected fits with nonempty held-out families')
            for key in ('fc_error', 'norm_error'):
                before, after = report['baseline'][key], report['candidate'][key]
                if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (before, after))
                        or not 0 <= after < before):
                    raise ValueError('FC bias artifact must improve held-out FC and normalized errors')
        digest = hashlib.sha256(raw).hexdigest()
    except FileNotFoundError:
        error = 'missing'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:256]
    comm.wait_prepared('draft-fc-bias-auto')
    reports = comm.gather_objects(dict(digest=digest, error=error))
    if all(r['error'] == 'missing' for r in reports):
        return {}, 'missing', None
    errors = [f'rank {i}: {r["error"]}' for i, r in enumerate(reports) if r['error']]
    if errors or len({r['digest'] for r in reports}) != 1:
        return {}, 'skipped: ' + '; '.join(errors or ['different artifact digests']), None
    return profile, 'pending', digest


def decode_reader(layer):
    if layer is None or getattr(layer, 'decode_precision', None) != 'fp8':
        raise ValueError('FC bias requires the fixed FP8 decode reader')
    reader = getattr(layer, 'decode_fp8', None)
    reader = reader if reader is not None else layer.fp8
    if reader is None:
        raise ValueError('FC bias requires a prepared FP8 pack')
    return reader


@cache
def _implementation():
    root = Path(__file__).resolve().parents[2] / 'kernels'
    sha = hashlib.sha256()
    for path in ('dense/__init__.py', 'dense/fp8.py', 'deep_gemm.py', 'common/norm_rope.py'):
        sha.update(path.encode())
        sha.update((root / path).read_bytes())
    try:
        deepgemm = version('deep-gemm')
    except PackageNotFoundError:
        deepgemm = 'unversioned'
    return dict(contract='decode-fc-fp8-ue8m0-128-bf16-out-bias-f32-v1',
                source=sha.hexdigest(), torch=str(torch.__version__), cuda=torch.version.cuda, deepgemm=deepgemm)


def reader_identity(layer, source_weight, norm_weight, eps):
    """Pin the exact source, executed pack, normalization and reader implementation.

    Transfer at most 128 weight rows at once; the 160 MiB BF16 FC never needs
    a second whole host copy. Called only by explicit collection or binding.
    """
    reader = decode_reader(layer)
    q, scale = reader.weight
    metadata = dict(_implementation(), eps=float(eps), rows=layer.rows, cols=layer.cols,
                    separate_decode=getattr(layer, 'decode_fp8', None) is not None)
    if q.is_cuda:
        metadata['device'] = torch.cuda.get_device_name(q.device)
    else:
        metadata['device'] = q.device.type
    sha = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name, tensor in (('source', source_weight), ('q', q), ('scale', scale), ('norm', norm_weight)):
        sha.update(json.dumps((name, list(tensor.shape), str(tensor.dtype))).encode())
        for chunk in tensor.detach().split(128):
            sha.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def prepare_bias(drafter):
    profile = drafter.tuning.fc_bias
    drafter.fc_bias_status = drafter.tuning.fc_bias_status
    if not profile:
        return None
    automatic = drafter.tuning.fc_bias_source == 'auto'
    error, result = None, None
    comm = drafter.target.comm
    try:
        profile = validate_profile(profile)
        if set(profile) != {str(rank) for rank in range(comm.world_size)}:
            raise ValueError('FC bias must cover every TP rank')
        profile = profile[str(comm.rank)]
        if len(profile['values']) != drafter.F.hidden:
            raise ValueError('FC bias width must match the draft hidden dimension')
        source = drafter.p['fc.weight']
        actual = reader_identity(drafter.dense.get('fc.weight'), source,
                                 drafter.p['hidden_norm.weight'], drafter.F.rms_eps)
        if actual != profile['reader_sha256']:
            raise ValueError('FC bias reader identity mismatch; fit against the executed decode pack')
        result = torch.tensor(profile['values'], dtype=torch.float32, device=source.device)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    comm.wait_prepared('draft-fc-bias')
    errors = comm.gather_objects(error)
    if any(errors):
        reason = '; '.join(f'rank {i}: {e}' for i, e in enumerate(errors) if e)
        drafter.fc_bias_status = 'skipped: ' + reason
        if automatic:
            return None
        raise ValueError('draft FC bias preparation failed: ' + reason)
    drafter.fc_bias_status = 'applied-auto' if automatic else 'applied-profile'
    return result
