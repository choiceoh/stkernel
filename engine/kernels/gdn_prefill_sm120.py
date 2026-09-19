"""GatedDeltaNet chunked prefill on the seed image's FlashInfer SM120 kernel (kernel adapter; engine/SM121_INTAKE.md
U13).

`flashinfer.gdn_prefill.chunk_gated_delta_rule` -- its SM120 CuTe-DSL delta rule, in the image
(measurements/sm121_inventory_20260919) -- behind the served GDN chunk lane's contract
(`engine/profiles/qwen38/lanes.served().gdn_chunk`: q, k [1, T, Hk, D], v [1, T, HV, D], the natural-log decay fp32
[1, T, HV], beta [1, T, HV] after its sigmoid, state0 [1, HV, K, V] fp32 or None, `states_at` 64-token chunk indices;
returns o [1, T, HV, D], the final state [1, HV, K, V] fp32 and, with `states_at`, the states at those chunk starts
[n, HV, K, V]).

Two things the call needs that the kernel's own signature does not say:

  q/k L2 norm   The image's build ignores `use_qk_l2norm_in_kernel` on the native prefill path (flashinfer#5255, open on
                2026-09-17): unnormalised q/k grow the recurrence to NaN -- every element of every variant
                sm121-gdndiag-0919d tried. So q and k are normalised here, by the served chunk kernel's own code
                (kda/chunk_decay: the strided pinned kernel where it takes the cell, l2norm_fwd otherwise), and the flag
                goes down False.
  boundaries    `states_at` becomes the kernel's state checkpoints every N tokens, N the largest multiple of 64 dividing
                every mark (so a prompt's block boundaries cost a checkpoint a block, not one every 64 tokens), and
                the named ones are picked out. The CP path takes no checkpoints, so a call with marks turns it off.

The layouts: the kernel keeps the state [HV, V, K], the engine [HV, K, V] -- transposed on the way in and out, as the
served lane does for its KDA kernel. `admits` is the shape the lane takes (D3: the caller chooses by it): at least
MIN_TOKENS tokens -- below that the served kernel is as fast or faster (sm121-gdnnorm-0919f: 128 tokens 322 against 122
us median, 1,024 tokens 232 against 388, 8,192 tokens 994 against 4,245) -- and the kernel's head width. `qualify`
holds it to the served kernel before a boot serves it.
"""
from __future__ import annotations

import math

import torch

MIN_TOKENS = 1024                   # sm121-gdnnorm-0919f: the first length measured faster than the served kernel
HEAD_DIM = 128                      # the SM120 delta rule's head width
CHUNK = 64                          # the served lane's kernel chunk, the unit `states_at` counts in
BAND = 2.0 ** -6                    # largest error over the largest value, output and state, against the served kernel


def admits(tokens: int, head_dim: int) -> bool:
    """Whether the lane takes a prefill segment: MIN_TOKENS tokens or more at the kernel's head width. Host integers
    only: every rank chooses alike, before any launch."""
    return tokens >= MIN_TOKENS and head_dim == HEAD_DIM


def _normalized(q, k):
    """q, k [1, T, H, D] L2-normalised per head, dense: the served chunk kernel's own normalisation
    (kda/chunk_decay.chunk_kda_with_decay, `use_qk_l2norm_in_kernel`)."""
    from engine.kernels.kda.kda import _glm53_qk_l2norm_strided
    from engine.kernels.kda.l2norm import l2norm_fwd
    normalized = _glm53_qk_l2norm_strided(q, k)
    if normalized is not None:
        return normalized
    return l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())


def checkpoint_every(states_at) -> int:
    """The checkpoint interval (tokens) that lands on every named 64-token chunk start: the largest multiple of 64
    dividing each mark."""
    marks = [CHUNK * int(c) for c in states_at]
    if not marks or any(m <= 0 for m in marks):
        raise ValueError("states_at names chunk starts after the first token (positive 64-token chunk indices)")
    return math.gcd(*marks)


def chunk(q, k, v, decay, beta, state0, states_at=None):
    """The served GDN chunk lane's call, on FlashInfer's SM120 kernel (the module docstring). Refuses a segment
    `admits` does not take."""
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("the GDN chunk lane takes one sequence: q, k [1, T, Hk, D] and v [1, T, HV, D]")
    t, hv, d = v.shape[1], v.shape[2], v.shape[3]
    if not admits(t, q.shape[3]) or d != HEAD_DIM:
        raise ValueError(f"the SM120 GDN prefill takes {MIN_TOKENS}+ tokens at head width {HEAD_DIM}; asked {t} tokens "
                         f"at {q.shape[3]}: the caller chooses by `admits`")
    if tuple(decay.shape) != (1, t, hv) or tuple(beta.shape) != (1, t, hv):
        raise ValueError("decay and beta are one value a value head a token [1, T, HV]")
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    qn, kn = _normalized(q, k)
    device = q.device
    initial = (torch.zeros(1, hv, d, d, dtype=torch.float32, device=device) if state0 is None
               else state0.float().transpose(-1, -2).contiguous())
    final = torch.empty_like(initial)
    args = dict(q=qn.reshape(t, q.shape[2], d), k=kn.reshape(t, k.shape[2], d), v=v[0].contiguous(),
                g=decay[0].float().exp().contiguous(), beta=beta[0].float().contiguous(), scale=d ** -0.5,
                initial_state=initial, output_final_state=True,
                cu_seqlens=torch.arange(0, 2 * t, t, dtype=torch.int64, device=device),     # [0, T], no host copy
                use_qk_l2norm_in_kernel=False, output_state=final)
    picks = None
    if states_at:
        every = checkpoint_every(states_at)
        count = t // every
        checkpoints = torch.empty(count, hv, d, d, dtype=torch.float32, device=device)
        args.update(state_checkpoints=checkpoints, checkpoint_every_n_tokens=every, use_cp=False,
                    checkpoint_cu_starts=torch.arange(0, 2 * count, count, dtype=torch.int64, device=device))
        picks = [CHUNK * int(c) // every - 1 for c in states_at]          # checkpoint j holds the state after (j+1)*N
        if any(not 0 <= p < count for p in picks):
            raise ValueError(f"a mark past the segment's {t} tokens: {list(states_at)}")
    o, state = chunk_gated_delta_rule(**args)
    o = o.view(1, t, hv, d)
    state = state.transpose(-1, -2).contiguous()
    if picks is None:
        return o, state
    return o, state, checkpoints[picks].transpose(-1, -2).contiguous()


def qualify(device, *, k_heads: int, v_heads: int, dim: int, seed: int = 0) -> dict:
    """The lane a boot serves held to the served chunk kernel it replaces (D3): MIN_TOKENS tokens with a carried state
    and two marks, the same inputs through `chunk` and through kda/chunk_decay.chunk_kda_with_decay, the output, the
    final state and the marked states each within BAND. Raises RuntimeError outside it; returns the errors."""
    from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
    from engine.kernels.kda.index import single_sequence_bounds
    gen = torch.Generator().manual_seed(seed)
    t = MIN_TOKENS

    def rand(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype).to(device)
    q, k, v = rand(1, t, k_heads, dim), rand(1, t, k_heads, dim), rand(1, t, v_heads, dim)
    a = torch.randn(1, t, v_heads, generator=gen) * 2
    decay = (-torch.nn.functional.softplus(a - 1.0)).float().to(device)              # exp(decay) in about (0.007, 1)
    beta = torch.sigmoid(torch.randn(1, t, v_heads, generator=gen)).to(torch.bfloat16).to(device)
    state0 = rand(1, v_heads, dim, dim, scale=0.5, dtype=torch.float32)
    marks = [4, 8]                                                                    # 256 and 512 tokens in
    o, state, states = chunk(q, k, v, decay, beta, state0, states_at=marks)
    ref_o, ref_state, ref_states = chunk_kda_with_decay(
        q, k, v, decay, beta, scale=dim ** -0.5, initial_state=state0.transpose(-1, -2).contiguous(),
        output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=single_sequence_bounds(t, device),
        out=torch.empty_like(v), states_at=marks)

    def error(got, want):
        got, want = got.float(), want.float()
        return float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))
    errors = {"o": error(o, ref_o), "state": error(state, ref_state.transpose(-1, -2)),
              "states": error(states, ref_states.transpose(-1, -2))}
    finite = all(bool(torch.isfinite(x).all()) for x in (o, state, states))
    if not finite or any(e > BAND for e in errors.values()):
        raise RuntimeError(f"gdn_prefill_sm120.qualify: against the served chunk kernel {errors} (band {BAND:.3g}), "
                           f"finite {finite}")
    return {"tokens": t, **{name: round(e, 6) for name, e in errors.items()}}


__all__ = ["MIN_TOKENS", "HEAD_DIM", "BAND", "admits", "checkpoint_every", "chunk", "qualify"]
