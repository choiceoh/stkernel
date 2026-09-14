"""FP32 KDA tree oracle: ancestor factors, no full state per speculative node."""
from dataclasses import dataclass

import torch

from engine.modules.linear_attention import kda_gate, l2norm
from engine.modules.speculative_tree import Tree


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


def conv(tree, raw, weight, history):
    """Depthwise conv reads only ancestors, including the actual prefix taps."""
    from engine.modules.causal_conv import causal_conv1d
    if raw.ndim != 2 or raw.shape[0] != len(tree.tokens) or history.shape != (raw.shape[1], weight.shape[1] - 1):
        raise ValueError("tree conv needs one raw row per node and a prefix history")
    rows = []
    for node in range(len(tree.tokens)):
        # The ordinary conv is the independent arithmetic implementation.
        path = tree.path(node)
        out, _ = causal_conv1d(raw[list(path)], weight, initial_state=history, activation="silu")
        rows.append(out[-1])
    return torch.stack(rows)


def verify(tree, q, k, v, g_raw, beta_raw, a_log, dt_bias, initial, lower_bound):
    """[nodes,H,D] inputs; state and all saved factors stay FP32."""
    n = len(tree.tokens)
    if (q.ndim != 3 or len(q) != n or k.shape != q.shape or g_raw.shape != q.shape
            or v.shape[:2] != q.shape[:2] or beta_raw.shape != q.shape[:2]
            or initial.shape != (q.shape[1], q.shape[2], v.shape[2]) or initial.dtype != torch.float32):
        raise ValueError("tree KDA geometry or FP32 state contract violated")
    if any(t.device != q.device for t in (k, v, g_raw, beta_raw, a_log, dt_bias, initial)):
        raise ValueError("tree KDA tensors must share a device")
    if q.is_cuda:
        from engine.kernels.kda.tree import verify as native
        return native(tree, q, k, v, g_raw, beta_raw, a_log, dt_bias, initial, lower_bound)
    query, key = l2norm(q.float()), l2norm(k.float())
    decay = kda_gate(g_raw, a_log, dt_bias, lower_bound).exp()
    beta = beta_raw.float().sigmoid()
    factors = Factors(tree, initial.clone(), key, decay, torch.empty_like(v, dtype=torch.float32))
    output = []
    for node, parent in enumerate(tree.parents):
        state = factors.initial.clone() if parent == -1 else factors.state(parent)
        state = state * decay[node, :, :, None]
        pred = torch.einsum("hk,hkv->hv", key[node], state)
        update = (v[node].float() - pred) * beta[node, :, None]
        factors.update[node] = update
        state = state + key[node, :, :, None] * update[:, None, :]
        output.append(torch.einsum("hk,hkv->hv", query[node], state) * q.shape[-1] ** -.5)
    return torch.stack(output).to(q.dtype), factors
