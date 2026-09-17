"""Private dense-weight control for the unresolved generation incident."""
from contextlib import contextmanager
from pathlib import Path

import torch


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
