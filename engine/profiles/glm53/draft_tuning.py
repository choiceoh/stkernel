"""Draft-only controls, including default FP32 projection and fitted-bias discovery."""
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path


def number(value, low, high, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not low <= value <= high or (positive and value <= 0)):
        raise ValueError(f'{name} must be finite in {"(" if positive else "["}{low}, {high}]')
    return float(value)


@dataclass(frozen=True)
class DraftTuning:
    selector_alpha: tuple = ()
    smoothing_alpha: dict = field(default_factory=dict)
    gptq_damping: dict = field(default_factory=dict)
    request_boundaries: bool = False
    trace_every: int = 0
    digest: str = 'selector-fp32-fc-bias-auto-v1'
    selector_projection_fp32: bool = True
    fc_bias: dict = field(default_factory=dict)
    fc_bias_auto: bool = True
    fc_bias_source: str = 'none'
    fc_bias_status: str = 'unavailable'

    def alphas(self, k):
        if not self.selector_alpha:
            return (1.,) * k
        if len(self.selector_alpha) not in (1, k):
            raise ValueError(f'selector_alpha needs one value or exactly {k} positions')
        return self.selector_alpha * k if len(self.selector_alpha) == 1 else self.selector_alpha

    @classmethod
    def from_dict(cls, value):
        fields = {'version', 'selector_alpha', 'smoothing_alpha', 'gptq_damping',
                  'request_boundaries', 'trace_every', 'evidence', 'selector_projection_fp32', 'fc_bias',
                  'fc_bias_auto'}
        if (not isinstance(value, dict) or type(value.get('version')) is not int
                or value['version'] != 1 or value.keys() - fields):
            raise ValueError('draft tuning requires version 1 and known fields')
        alpha = value.get('selector_alpha', [])
        if not isinstance(alpha, list):
            raise ValueError('selector_alpha must be a list')
        alpha = tuple(number(x, 0, 2, 'selector_alpha') for x in alpha)
        maps = []
        for key, default, positive in (('smoothing_alpha', {}, False), ('gptq_damping', {}, True)):
            mapping = value.get(key, default)
            if not isinstance(mapping, dict) or any(not isinstance(k, str) for k in mapping):
                raise ValueError(f'{key} must map draft reader names to numbers')
            maps.append({k: number(v, 0, 1, key, positive=positive) for k, v in mapping.items()})
        boundary, trace = value.get('request_boundaries', False), value.get('trace_every', 0)
        if type(boundary) is not bool or type(trace) is not int or not 0 <= trace <= 1_000_000:
            raise ValueError('request_boundaries must be bool; trace_every must be a nonnegative bounded integer')
        projection = value.get('selector_projection_fp32', True)
        auto = value.get('fc_bias_auto', True)
        if type(projection) is not bool or type(auto) is not bool:
            raise ValueError('selector_projection_fp32 and fc_bias_auto must be bool')
        from .draft_fc_bias import validate_profile
        bias = validate_profile(value.get('fc_bias', {}))
        effective = dict(value, selector_projection_fp32=projection, fc_bias_auto=auto)
        digest = hashlib.sha256(json.dumps(effective, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return cls(alpha, *maps, boundary, trace, digest, projection, bias, auto,
                   'profile' if bias else 'none', 'pending' if bias else ('unavailable' if auto else 'disabled'))

    def validate(self, facts, dense_names):
        self.alphas(facts.k)
        if any(len(entry['values']) != facts.hidden for entry in self.fc_bias.values()):
            raise ValueError('FC bias width must match the draft hidden dimension')
        norms = {f'layers.{layer}.{norm}.weight' for layer in range(facts.layers)
                 for norm in ('input_layernorm', 'post_attention_layernorm')}
        if self.smoothing_alpha.keys() - norms or self.gptq_damping.keys() - set(dense_names):
            raise ValueError('draft tuning names must belong to the prepared draft norms/readers')


def load_agreed(path, facts, shapes, comm, *, fc_bias_path=None):
    # Every native draft rank participates, including an empty profile. A
    # missing path on one peer must not skip the other peers' preparation vote.
    tuning, error = DraftTuning(), None
    try:
        if path:
            tuning = DraftTuning.from_dict(json.loads(Path(path).read_text()))
            tuning.validate(facts, shapes)
            if tuning.fc_bias and set(tuning.fc_bias) != {str(rank) for rank in range(comm.world_size)}:
                raise ValueError('FC bias must cover every TP rank')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    discover_bias = tuning.fc_bias_auto and not tuning.fc_bias and fc_bias_path is not None
    comm.wait_prepared('draft-tuning')
    reports = comm.gather_objects(dict(digest=tuning.digest, error=error, discover_bias=discover_bias))
    errors = [f'rank {i}: {r["error"]}' for i, r in enumerate(reports) if r['error']]
    if errors or len({(r['digest'], r.get('discover_bias', False)) for r in reports}) != 1:
        raise ValueError('draft tuning must agree across ranks: ' + '; '.join(
            errors or ['different profile digests or FC cache discovery']))
    if discover_bias:
        from .draft_fc_bias import load_auto
        bias, status, artifact_digest = load_auto(fc_bias_path, facts, comm)
        digest = hashlib.sha256(f'{tuning.digest}:fc-bias-auto:{artifact_digest or status}'.encode()).hexdigest()
        tuning = replace(tuning, fc_bias=bias, fc_bias_source='auto', fc_bias_status=status, digest=digest)
    return tuning


def prepare_store(tuning, store, policy, facts, comm):
    """Refuse a tuned reader that cannot use its calibration; agree errors before allocation."""
    from .drafter import store_name
    from .draft_policy import decode_name
    explicit_bias = tuning.fc_bias and tuning.fc_bias_source != 'auto'
    if not tuning.smoothing_alpha and not tuning.gptq_damping and not tuning.trace_every and not explicit_bias:
        return
    error = None
    try:
        import torch
        if tuning.trace_every and not policy.diagnostics:
            raise ValueError('selector trace requires draft diagnostics')
        if explicit_bias and policy.fc_precision != 'fp8':
            raise ValueError('FC bias requires the fixed FP8 decode reader')
        for norm in tuning.smoothing_alpha:
            prefix = norm.rsplit('.', 2)[0] + '.'
            reader = prefix + ('self_attn.qkv' if norm.endswith('.input_layernorm.weight') else 'mlp.gate_up')
            peaks = store.amax(store_name(reader, facts))
            if (peaks is None or peaks.shape != (facts.hidden,) or not bool(torch.isfinite(peaks).all())
                    or bool((peaks < 0).any()) or not bool((peaks > 0).any())):
                raise ValueError(f'smoothing tuning requires valid channel peaks for {reader}')
        for reader in tuning.gptq_damping:
            name = store_name(reader, facts)
            required = decode_name(name) if reader == 'fc.weight' and policy.fc_calibration == 'decode' else name
            if not store.calibrated(required):
                raise ValueError(f'GPTQ damping tuning requires calibration for {required}')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    comm.wait_prepared('draft-tuning-calibration')
    errors = comm.gather_objects(error)
    if any(errors):
        raise ValueError('draft tuning calibration failed: ' + '; '.join(f'rank {i}: {e}' for i, e in enumerate(errors) if e))
    for reader, damping in tuning.gptq_damping.items():
        name = store_name(reader, facts)
        store.gptq_damping[name] = damping
        if reader == 'fc.weight':
            store.gptq_damping[decode_name(name)] = damping
