"""The short causal conv in front of a delta-rule layer (module): GLM's KDA
runs one on q, k and v (kernel 4, silu, no bias); Qwen's GDN runs one on the
merged qkv (kernel 4). Torch reference, judged by the served Triton op
(vllm/model_executor/layers/mamba/ops/causal_conv1d.py) in probes/conv_check.py.

Depthwise, causal, with the previous kernel-1 inputs carried as state:

    y[t, c] = act( bias[c] + sum_{i<k} w[c, i] * x[t - (k-1) + i, c] )

where x[t<0] comes from `initial_state[c, :]` (the last k-1 tokens of the
previous chunk) and the final state is the last k-1 inputs of this chunk --
which is exactly what base/kv's slot holds for the layer (shapes.py:
"conv history 3, carried in the slot, never re-read from tokens").
"""
from __future__ import annotations

import torch


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: "torch.Tensor | None" = None,
                  initial_state: "torch.Tensor | None" = None, activation: "str | None" = "silu"):
    """x [T, C], weight [C, K], initial_state [C, K-1] -> (y [T, C], final_state [C, K-1])."""
    t, c = x.shape; k = weight.shape[1]
    xf = x.float().T                                                  # [C, T]
    hist = torch.zeros(c, k - 1, device=x.device) if initial_state is None else initial_state.float()
    padded = torch.cat([hist, xf], dim=1)                             # [C, K-1+T]
    y = torch.zeros(c, t, device=x.device, dtype=torch.float32)
    for i in range(k):
        y += weight[:, i:i + 1].float() * padded[:, i:i + t]
    if bias is not None:
        y += bias.float()[:, None]
    if activation == "silu":
        y = torch.nn.functional.silu(y)
    return y.T.to(x.dtype), padded[:, t:].contiguous()                 # last K-1 inputs


def causal_conv1d_update(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor,
                         bias: "torch.Tensor | None" = None, activation: "str | None" = "silu"):
    """One token: x [C], state [C, K-1] (shifted in place semantics returned) -> (y [C], new state)."""
    y, new_state = causal_conv1d(x[None], weight, bias, state, activation)
    return y[0], new_state


def _selfcheck() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, C, K = 37, 64, 4
    x = torch.randn(T, C, device=dev, dtype=torch.bfloat16); w = torch.randn(C, K, device=dev)
    y_full, s_full = causal_conv1d(x, w, None, None)
    # chunk 20 + 17 with carried state == straight run
    y1, s1 = causal_conv1d(x[:20], w, None, None); y2, s2 = causal_conv1d(x[20:], w, None, s1)
    assert torch.allclose(torch.cat([y1, y2]).float(), y_full.float(), atol=1e-2) and torch.equal(s2, s_full)
    # token-by-token update from the chunk's state == the rest of the run
    st = s1; ys = []
    for i in range(20, T):
        yi, st = causal_conv1d_update(x[i], st, w); ys.append(yi)
    assert torch.allclose(torch.stack(ys).float(), y_full[20:].float(), atol=1e-2)
    # vs torch's own conv1d (groups=C, left-pad K-1)
    ref = torch.nn.functional.conv1d(torch.nn.functional.pad(x.float().T[None], (K - 1, 0)), w[:, None, :], groups=C)[0].T
    # the module returns x.dtype (bf16 here): compare at bf16's own resolution
    assert torch.allclose(y_full.float(), torch.nn.functional.silu(ref).to(x.dtype).float(), atol=2e-2, rtol=2e-2)
    print("  causal_conv: chunked == straight, update == chunk tail, == torch conv1d(groups=C) OK")


if __name__ == "__main__":
    _selfcheck()
