"""FP32 KDA tree oracle: ancestor factors, no full state per speculative node."""
from dataclasses import dataclass
import math

import torch

from engine.modules.linear_attention import kda_gate, l2norm
from engine.modules.speculative_tree import Tree


class Topology:
    """One device upload for all KDA layers, including convolution ancestry."""
    def __init__(self, tree, device, *, taps=4):
        if type(taps) is not int or taps < 1:
            raise ValueError("tree convolution needs positive taps")
        self.tree, self.device, self.taps = tree, torch.device(device), taps
        self.width = max(1, max(tree.depths))
        paths = [tree.path(i) for i in range(len(tree.tokens))]
        self.paths = torch.tensor([p[:-1]+(-1,)*(self.width-len(p)+1) for p in paths],
                                  dtype=torch.int32, device=device)
        self.device = self.paths.device
        self.depths = torch.tensor(tree.depths, dtype=torch.int32, device=device)
        self.parents = torch.tensor(tree.parents, dtype=torch.int32, device=device)
        self.order = torch.tensor(tree.preorder, dtype=torch.int32, device=device)
        self.conv = torch.tensor([tuple(p[j] if j >= 0 else j for j in range(len(p)-taps, len(p)))
                                  for p in paths], dtype=torch.int32, device=device)


@dataclass
class Factors:
    tree: Tree
    initial: torch.Tensor             # one owned FP32 [H,K,V] state
    key: torch.Tensor                 # [nodes,H,K], FP32
    decay: torch.Tensor               # [nodes,H,K], FP32, channelwise
    update: torch.Tensor              # [nodes,H,V], FP32

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.initial, self.key, self.decay, self.update))

    def state(self, node):
        """Materialize just this root-to-node path, preserving multiply rounding."""
        state = self.initial.clone()
        for i in self.tree.path(node):
            state = state * self.decay[i, :, :, None]
            state = state + self.key[i, :, :, None] * self.update[i, :, None, :]
        return state


def conv(tree, raw, weight, history, *, topology=None, context=None):
    """Depthwise conv reads only ancestors, including the actual prefix taps."""
    ring = context is not None
    if (raw.ndim != 2 or weight.ndim != 2 or weight.shape[0] != raw.shape[1] or raw.shape[0] != len(tree.tokens)
            or history.ndim != 2 or history.shape[0] != raw.shape[1]
            or (ring and (type(context) is not int or context < 0 or history.shape[1] < max(1, weight.shape[1]-1)))
            or (not ring and history.shape[1] != weight.shape[1]-1)):
        raise ValueError("tree conv needs one raw row per node and a prefix history")
    topology = topology or Topology(tree, raw.device, taps=weight.shape[1])
    if topology.tree != tree or topology.device != raw.device or topology.taps != weight.shape[1]:
        raise ValueError("tree conv topology must match the tree, device and taps")
    if raw.is_cuda:
        from engine.kernels.kda.tree import conv as native
        return native(raw, weight, history, topology, context=context)
    if ring:
        positions = context+torch.arange(1-weight.shape[1], 0, device=raw.device)
        history = history[:, positions.clamp_min(0) % history.shape[1]].masked_fill((positions < 0)[None, :], 0)
    padded = torch.cat((history.float().T, raw.float()))
    taps = padded[(weight.shape[1]-1+topology.conv).long()]
    output = torch.zeros_like(raw, dtype=torch.float32)
    for tap in range(weight.shape[1]):
        output += weight[:, tap].float()*taps[:, tap]
    return torch.nn.functional.silu(output).to(raw.dtype)


def verify(tree, q, k, v, g_raw, beta_raw, a_log, dt_bias, initial, lower_bound, *, topology=None):
    """[nodes,H,D] inputs; state and all saved factors stay FP32."""
    n = len(tree.tokens)
    if (q.ndim != 3 or len(q) != n or min(q.shape) <= 0 or k.shape != q.shape or g_raw.shape != q.shape
            or v.ndim != 3 or v.shape[:2] != q.shape[:2] or v.shape[2] <= 0 or beta_raw.shape != q.shape[:2]
            or a_log.shape != (q.shape[1],) or dt_bias.numel() != q.shape[1]*q.shape[2]
            or not math.isfinite(lower_bound) or lower_bound >= 0
            or initial.shape != (q.shape[1], q.shape[2], v.shape[2]) or initial.dtype != torch.float32):
        raise ValueError("tree KDA geometry or FP32 state contract violated")
    if any(t.device != q.device for t in (k, v, g_raw, beta_raw, a_log, dt_bias, initial)):
        raise ValueError("tree KDA tensors must share a device")
    if q.is_cuda:
        from engine.kernels.kda.tree import verify as native
        return native(tree, q, k, v, g_raw, beta_raw, a_log, dt_bias, initial, lower_bound, topology=topology)
    query, key = l2norm(q.float()), l2norm(k.float())
    decay = kda_gate(g_raw, a_log, dt_bias, lower_bound).exp()
    beta = beta_raw.float().sigmoid()
    factors = Factors(tree, initial.clone(), key, decay, torch.empty_like(v, dtype=torch.float32))
    output = torch.empty_like(v, dtype=q.dtype)
    previous, state = -1, factors.initial.clone()
    for node in tree.preorder:
        parent = tree.parents[node]
        if parent != previous:
            state = factors.initial.clone() if parent == -1 else factors.state(parent)
        state = state * decay[node, :, :, None]
        pred = torch.einsum("hk,hkv->hv", key[node], state)
        update = (v[node].float() - pred) * beta[node, :, None]
        factors.update[node] = update
        state = state + key[node, :, :, None] * update[:, None, :]
        output[node] = (torch.einsum("hk,hkv->hv", query[node], state) * q.shape[-1] ** -.5).to(q.dtype)
        previous = node
    return output, factors
