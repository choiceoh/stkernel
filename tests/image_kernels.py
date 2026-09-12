"""Whether this host can run a kernel suite at all, and why not when it cannot.

A suite that compiles against the kernels lives half in the ST image: flashinfer, tilelang,
cutlass, deep_gemm and ninja are installed there and not on a plain host. Guarded only by
`torch.cuda.is_available()`, which is true on a fleet node, those suites did not skip -- they
ERRORED, twenty-six of them, on an import that was never going to succeed. An error that means
"not here" is the worst kind: it hides whether anything is actually wrong, which is how the
suites stayed red without anyone seeing it (45차 §48, §49).
"""
from __future__ import annotations

import importlib.util

PACKAGES = ("flashinfer", "tilelang", "cutlass", "deep_gemm", "ninja")

MISSING = tuple(m for m in PACKAGES if importlib.util.find_spec(m) is None)
PRESENT = not MISSING
REASON = "requires the ST image's kernel packages (" + ", ".join(MISSING) + " missing)" if MISSING else ""


def fused_sdpa(head_dim: int) -> bool:
    """Whether this host's torch offers a backend the drafter's row path pins to, at this head size.

    `drafter._attn_rows` asks for cuDNN or memory-efficient attention and refuses the math
    fallback on CUDA on purpose -- D3, written in the line above the call. Where torch
    runtime-disables both for the shape in question, as this host's build does at a head
    dimension of four, the suite cannot run here: that is a skip, not three errors that read
    like a bug in the drafter.

    The flags say nothing -- `cudnn_sdp_enabled()` and the rest all read True here while the
    call is refused -- because the refusal is decided per call, per shape, per architecture.
    So this asks, with the shape it is about to use.
    """
    import torch
    if not torch.cuda.is_available():
        return True                                  # CPU takes `nullcontext` and the math path
    import torch.nn.functional as Fn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.zeros(1, 1, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.bool)
    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            Fn.scaled_dot_product_attention(q, q, q, attn_mask=mask, scale=1.0)
        return True
    except RuntimeError:
        return False
