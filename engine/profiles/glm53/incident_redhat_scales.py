"""Private same-boot control for the Red Hat importer's lossy scale folding.

Keep packed FP4 weights, routing and dense calibration fixed. Separate controls
restore source weight scales alone, then the calibrated input global scales too.
Sidecars are source-verified offline and must match this boot's folded scales.
"""
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path('/home/choiceoh/glm53-logs/incident-redhat-lossless-0918')
REVISION = 'c245560b6d7e62c329cd3042343b358a4279affd'


def load_layer(net, layer):
    """Read and validate before the ordinary SF6 owner consumes raw scale bytes."""
    if not getattr(net, 'incident_redhat_prepare', False) or net.modelopt or not net.F.is_moe(layer):
        return None
    records = getattr(net, '_incident_redhat_records', None)
    if records is None:
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        if (manifest['source']['revision'] != REVISION or manifest['layers'] != 42
                or manifest['experts_per_layer'] != 288 or manifest['ranks'] != 4):
            raise ValueError('unexpected Red Hat scale source or geometry')
        records = {row['layer']: row for row in manifest['files'] if row['rank'] == net.rank}
        if set(records) != set(range(3, 45)):
            raise ValueError('incomplete rank scale sidecars')
        net._incident_redhat_records = records
    receipt = records[layer]
    path = ROOT / receipt['file']
    if (path.stat().st_size != receipt['bytes']
            or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']):
        raise ValueError(f'corrupt Red Hat scale sidecar: rank={net.rank} layer={layer}')
    from safetensors.torch import load_file
    values = load_file(str(path), device='cpu')
    if set(values) != {'w13_sf', 'w2_sf', 'w13_alpha', 'w2_alpha', 'a13_scale', 'a2_scale'}:
        raise ValueError('unexpected scale sidecar fields')
    prefix = f'L{layer}.moe.'
    device = net.p[prefix + 'w13'].device
    for name in ('w13_sf', 'w2_sf'):
        old = net.p[prefix + name]
        original = values[name]
        if old.shape != original.shape or original.dtype != torch.uint8:
            raise ValueError(f'scale geometry mismatch: {prefix + name}')
        current_hash = hashlib.sha256(old.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        if current_hash != receipt['folded_sha256'][name]:
            raise ValueError(f'folded source does not match the boot: {prefix + name}')
        values[name] = original.view(torch.float8_e4m3fn).to(device)
    for name in ('w13_alpha', 'w2_alpha', 'a13_scale', 'a2_scale'):
        if values[name].shape != (288,) or values[name].dtype != torch.float32:
            raise ValueError('expected per-expert FP32 global multipliers')
        values[name] = values[name].to(device)
    return values


def bind_layer(net, layer, values):
    """Prepare two scale controls over the identical tile-major FP4 weights."""
    if values is None:
        return
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    prefix = f'L{layer}.moe.'
    first, second = net.p[prefix + 'w13'], net.p[prefix + 'w2']
    one = torch.ones(first.shape[0], dtype=torch.float32, device=first.device)
    # Preparation may consume the raw SF storage. Clone before either call.
    owners = [('weight', values['w13_sf'], values['w2_sf'], one, one),
              ('calibrated', values['w13_sf'].clone(), values['w2_sf'].clone(),
               values['a13_scale'], values['a2_scale'])]
    variants = {}
    for name, sf13, sf2, input13, input2 in owners:
        scales = ModelOptScales.bind(values['w13_alpha'], input13, values['w2_alpha'], input2,
                                     experts=first.shape[0], device=first.device)
        args = dict(w13=first, w13_sf=sf13, w2=second, w2_sf=sf2,
                    limit=net.F.swiglu_limit, scales=scales)
        views = net.lanes.moe_prepare(first, sf13, second, sf2,
                                      net.F.topk_experts, net.F.swiglu_limit, scales=scales)
        record = dict(expert=partial(net.lanes.moe, **args), views=views, scales=scales)
        if net.lanes.moe_packets is not None and net.lanes.moe_packets_supported is not None:
            record['packet'] = partial(net.lanes.moe_packets, **args)
            record['packet_supported'] = partial(net.lanes.moe_packets_supported, **args)
        variants[name] = record
    if not hasattr(net, '_incident_redhat_layers'):
        net._incident_redhat_layers = {}
    net._incident_redhat_layers[layer] = variants
    print(f'[incident-redhat] rank={net.rank} layer={layer} source_scales=verified '
          f'variants=weight,calibrated packed_fp4=shared', flush=True)


@contextmanager
def select(net, variant):
    """Swap prepared owners for one isolated eager request forward, then restore."""
    audit = getattr(net, 'incident_audit_root', None)
    previous = {}
    try:
        net.incident_audit_root = None
        if variant is not None:
            if variant not in ('weight', 'calibrated'):
                raise ValueError('unknown Red Hat scale control')
            records = {layer: variants[variant] for layer, variants in net._incident_redhat_layers.items()}
            if set(records) != {layer for layer in net.layers if net.F.is_moe(layer)}:
                raise ValueError('lossless scale owners are incomplete')
            for attr, field in (('_experts', 'expert'), ('_expert_views', 'views'),
                                ('_quant_scales', 'scales'), ('_packet_experts', 'packet'),
                                ('_packet_capabilities', 'packet_supported')):
                previous[attr] = getattr(net, attr)
                replacement = dict(previous[attr])
                replacement.update({layer: record[field] for layer, record in records.items() if field in record})
                setattr(net, attr, replacement)
        yield
    finally:
        for attr, value in previous.items():
            setattr(net, attr, value)
        net.incident_audit_root = audit
