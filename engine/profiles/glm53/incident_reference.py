"""Private dense-weight control for the unresolved generation incident."""
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def original_fp32_constants(net, *, router, kda):
    """Private isolated control; restore checkpoint constants before each forward."""
    weights = getattr(net, '_incident_original_fp32', None)
    if weights is None:
        import hashlib
        import json
        root = Path('/home/choiceoh/glm53-logs/incident-original-fp32-0918')
        manifest = json.loads((root / 'manifest.json').read_text())
        name = f'original-fp32-rank{net.rank}.pt'
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError('original FP32 constants do not match their receipt')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        if len(weights) != 110:
            raise ValueError('expected 42 router biases and 68 KDA constants')
        for name, value in weights.items():
            if value.dtype != torch.float32 or value.shape != net.p[name].shape:
                raise ValueError(f'invalid original constant: {name}')
            if not torch.equal(net.p[name].cpu(), value.bfloat16().float()):
                raise ValueError(f'control source does not match rounded serving constant: {name}')
        weights = {name: value.to(net.p['norm'].device) for name, value in weights.items()}
        net._incident_original_fp32 = weights
        print(f'[incident-reference] rank={net.rank} original_fp32_constants={len(weights)}', flush=True)
    selected = {name: value for name, value in weights.items()
                if (router and name.endswith('.moe.bias'))
                or (kda and name.endswith(('.kda.A_log', '.kda.dt_bias')))}
    previous = {name: net.p[name] for name in selected}
    previous_fused = net._router_fused_bias
    try:
        net.p.update(selected)
        if router:
            net._router_fused_bias = {L: weights[f'L{L}.moe.bias']
                                     for L in previous_fused}
        yield
    finally:
        net.p.update(previous)
        net._router_fused_bias = previous_fused


@contextmanager
def bf16_dense(net):
    """Original BF16 dense weights, ordinary TP, native attention and NVFP4 experts.

    The verified vocabulary head retains its serving implementation. This is
    not an all-BF16 model and must not be described as one.
    """
    weights = getattr(net, '_incident_bf16_dense', None)
    if weights is None:
        from safetensors import safe_open
        path = Path(net.incident_rank_file)
        weights = {}
        with safe_open(path, framework='pt', device='cpu') as source:
            for name, layer in net.dense.items():
                if name == 'head':
                    continue
                weight = source.get_tensor(name).to(net.p['norm'].device)
                smooth = getattr(layer, 'smooth', None)
                if smooth is not None:
                    weight = (weight.float() * smooth.float()).to(weight.dtype)
                weights[name] = weight.contiguous()
        net._incident_bf16_dense = weights
        print(f'[incident-reference] rank={net.rank} bf16_dense_readers={len(weights)} '
              f'bytes={sum(w.numel()*w.element_size() for w in weights.values())}', flush=True)
    old_dense = net.dense
    old_weights = {name: net.p[name] for name in weights}
    old_overlap = net.shared_overlap
    old_transport = net.prefill_transport
    old_packets = net.prefill_ffn_packets
    try:
        net.p.update(weights)
        net.dense = {'head': old_dense['head']}
        net.shared_overlap = None
        net.prefill_transport = None
        net.prefill_ffn_packets = False
        yield
    finally:
        net.p.update(old_weights)
        net.dense = old_dense
        net.shared_overlap = old_overlap
        net.prefill_transport = old_transport
        net.prefill_ffn_packets = old_packets
