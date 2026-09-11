"""Hold modules/causal_conv to the served Triton op inside the glm53 image."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.causal_conv import causal_conv1d, causal_conv1d_update


def main() -> int:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update as served_update
    torch.manual_seed(0); dev = "cuda"
    T, C, K = 96, 512, 4
    x = torch.randn(T, C, device=dev, dtype=torch.bfloat16); w = (torch.randn(C, K, device=dev) * 0.3).float()
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
    init = torch.randn(C, K - 1, device=dev, dtype=torch.bfloat16)
    ours, s_ours = causal_conv1d(x, w, None, init)
    # Slot 0 is NULL_BLOCK_ID to these kernels: a sequence whose cache index is
    # 0 is SKIPPED (the docstring's `[null_block_id, 1, 20, null_block_id]`).
    # Three diagnostics read a silent no-op as a layout bug before this was
    # found. The served runner never hands out slot 0; neither may ours.
    SLOT = 1
    conv_states = torch.zeros(2, C, K - 1, device=dev, dtype=torch.bfloat16); conv_states[SLOT] = init
    served = causal_conv1d_fn(x.T, w, None, conv_states,
                              torch.tensor([0, T], device=dev, dtype=torch.int32),
                              cache_indices=torch.tensor([SLOT], device=dev, dtype=torch.int32),
                              has_initial_state=torch.tensor([True], device=dev), activation="silu")
    served = served.T if served.shape[0] == C else served
    r_y = rel(ours, served); r_s = rel(s_ours, conv_states[SLOT])
    print(f"  prefill: y rel {r_y:.2e}, final state (written into conv_states) rel {r_s:.2e}")
    xt = torch.randn(1, C, device=dev, dtype=torch.bfloat16)
    st = conv_states.clone()
    y_ours, s_next = causal_conv1d_update(xt[0], st[SLOT].clone(), w)
    y_srv = served_update(xt.clone(), st, w, None, activation="silu",
                          conv_state_indices=torch.tensor([SLOT], device=dev, dtype=torch.int32))
    r_u = rel(y_ours, y_srv.reshape(-1)); r_us = rel(s_next, st[SLOT])
    print(f"  update:  y rel {r_u:.2e}, state rel {r_us:.2e}")
    ok = max(r_y, r_s, r_u, r_us) < 2e-2
    print("\n  " + ("causal conv reference == served op" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
