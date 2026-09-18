"""Private eager-only comparison of native and checkpoint BF16 vocabulary heads."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as functional


def checkpoint_weight(net, device):
    weight = getattr(net, '_incident_head_weight', None)
    if weight is None:
        from safetensors import safe_open
        with safe_open(net.incident_rank_file, framework='pt', device='cpu') as source:
            weight = source.get_tensor('head').contiguous()
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (net.vp, net.F.hidden):
            raise ValueError('incident head must be the original BF16 vocabulary shard')
        digest = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
        net._incident_head_weight_sha256 = digest
        weight = weight.to(device)
        net._incident_head_weight = weight
        print(f'[incident-head-reference] rank={net.rank} shape={tuple(weight.shape)} '
              f'dtype={weight.dtype} sha256={digest}', flush=True)
    if weight.device != device:
        raise ValueError('incident head device changed')
    return weight


def project(net, hidden, *, mode, admission, generation, prefix_sha256):
    if mode not in (20, 28):
        return net.head_local(hidden)
    if hidden.ndim != 2 or hidden.shape[0] != 1 or hidden.dtype != torch.bfloat16:
        raise ValueError('incident head requires one eager BF16 hidden row')
    root = getattr(net, 'incident_audit_root', None)
    record = root is not None and (generation < 32 or generation % 32 == 0)
    native = net.head_local(hidden) if mode == 20 or record else None
    reference = (functional.linear(hidden, checkpoint_weight(net, hidden.device))
                 if mode == 28 or record else None)
    if record:
        root = Path(root).parent / 'incident-head'
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'admit{admission}-gen{generation}-rank{net.rank}.pt'
        payload = dict(admission=admission, generation=generation, rank=net.rank,
                       mode=mode, prefix_sha256=prefix_sha256,
                       weight_sha256=net._incident_head_weight_sha256,
                       hidden=hidden.detach().cpu(), native=native.detach().cpu(),
                       bf16=reference.detach().cpu())
        with path.open('xb') as output:
            torch.save(payload, output)
    return reference if mode == 28 else native
