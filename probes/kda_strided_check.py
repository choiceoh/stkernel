#!/usr/bin/env python3
"""Arming gate for VLLM_GLM53_KDA_STRIDED (vLLM #55736), on GPU, bit-for-bit.

The decode path hands ``fused_recurrent_kda`` column slices of the merged
q|k|v conv output and of the fused qkvbfg_a projection.  Before this knob each
slice was made contiguous first -- 3-5 copy kernels per KDA layer per step.
With the knob the recurrent kernel reads them through explicit per-token
strides instead.

Claimed equivalence: for the SAME values, feeding the kernel a token-strided
view must produce the SAME bytes as feeding it a contiguous copy.  A stride is
addressing, not arithmetic, so anything short of bit-equality means the stride
arithmetic is wrong.  This probe is that gate; the profile keeps the knob at 0
until it passes.  No timing here -- this is correctness, not a speed claim.

The contiguous arm IS the knob-at-0 arm: with VLLM_GLM53_KDA_STRIDED unset,
_glm53_kda_input makes every input contiguous and hands the same kernel the
same H*K / HV*V / HV strides, so comparing the two arms in one process is
exactly the on/off comparison, without a second boot.

Both kda.py and fused_recurrent.py have to be mounted: the first launches the
kernel the second defines, and mounting one alone gives a patched caller a
stock kernel (`Keyword argument stride_q_token was specified but
unrecognised`).  probes/run_mk_probe.sh carries both.

Run only in an idle fleet window, in a fresh container with composed sources:
  bash probes/run_mk_probe.sh probes/kda_strided_check.py | tee /tmp/kdastrided.log
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

os.environ.setdefault("VLLM_GLM53_KDA_STRIDED", "1")

import torch  # noqa: E402

# GLM-5.3-Flash KDA at TP=4: 64 heads / 4 ranks = 16, head_dim 128.
H = 16
D = 128


def _fill(shape, gen, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32) * 0.5).to(dtype)


def make_case(num_seqs, spec_len, gen, headwise_beta=False):
    """One decode/verify batch: T tokens over num_seqs sequences.

    Returns (contiguous_inputs, strided_inputs, shared_args).  The strided
    inputs are views into a merged [T, 3*H*D] buffer holding the SAME values,
    i.e. exactly the layout the merged conv output has in production.
    """
    T = num_seqs * spec_len
    q = _fill((T, H * D), gen)
    k = _fill((T, H * D), gen)
    v = _fill((T, H * D), gen)
    g = _fill((T, H * D), gen)
    beta = _fill((T, H * D if headwise_beta else H), gen)

    merged = torch.empty((T, 3 * H * D), device="cuda", dtype=torch.bfloat16)
    merged[:, 0 * H * D : 1 * H * D] = q
    merged[:, 1 * H * D : 2 * H * D] = k
    merged[:, 2 * H * D : 3 * H * D] = v
    qs, ks, vs = (
        merged[:, i * H * D : (i + 1) * H * D] for i in range(3)
    )
    gmerged = torch.empty((T, 2 * H * D), device="cuda", dtype=torch.bfloat16)
    gmerged[:, :H * D] = g
    gs = gmerged[:, : H * D]
    bwidth = H * D if headwise_beta else H
    bmerged = torch.empty((T, 3 * bwidth), device="cuda", dtype=torch.bfloat16)
    bmerged[:, bwidth : 2 * bwidth] = beta
    bs = bmerged[:, bwidth : 2 * bwidth]

    def as4(x):
        return x.reshape(1, T, H, D)

    def view4(x):
        return x.view(T, H, D).unsqueeze(0)

    def as3(x):
        return x.reshape(1, T, H)

    def view3(x):
        return x.unsqueeze(0)

    contig = dict(
        q=as4(q).contiguous(),
        k=as4(k).contiguous(),
        v=as4(v).contiguous(),
        g=as4(g).contiguous(),
        beta=(as4(beta) if headwise_beta else as3(beta)).contiguous(),
    )
    strided = dict(
        q=view4(qs),
        k=view4(ks),
        v=view4(vs),
        g=view4(gs),
        beta=(view4(bs) if headwise_beta else view3(bs)),
    )
    for name in contig:
        a, b = contig[name], strided[name]
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert torch.equal(a, b), f"{name}: the two layouts do not hold the same values"
        assert a.is_contiguous(), f"{name}: contiguous arm is not contiguous"
    # A single-token batch makes the column slice trivially contiguous (one
    # row), so it cannot exercise the stride path -- it is still worth running
    # as the degenerate shape, just not as evidence.
    if T > 1:
        assert not strided["q"].is_contiguous(), "strided arm collapsed to contiguous"
    return contig, strided, T


def run(mod, inputs, *, spec_len, T, state, accepted, indices, spec):
    st = state.clone()
    out, final = mod.fused_recurrent_kda(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        initial_state=st,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.arange(
            0, T + 1, spec_len, device="cuda", dtype=torch.int32
        ),
        ssm_state_indices=indices,
        num_accepted_tokens=accepted if spec else None,
        sigmoid_beta=True,
        inplace_final_state=True,
    )
    return out, st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm.third_party.flash_linear_attention.ops import kda  # noqa: E402

    if not hasattr(kda, "kda_strided_inputs_enabled"):
        print("ABORT: deployed kda.py has no kda_strided_inputs_enabled -- "
              "the composed overlay is older than this probe")
        return 2
    print(f"knob VLLM_GLM53_KDA_STRIDED -> {kda.kda_strided_inputs_enabled()}")

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    cases = list(
        itertools.product(
            (1, 2, 4),        # num_seqs (MAX_SEQS=4)
            (1, 6, 8),        # spec_len: plain decode, K=5 verify, K=7 verify
            (False, True),    # scalar / headwise beta
            (False, True),    # non-spec / spec-decode slot indexing
        )
    )
    bad = 0
    for num_seqs, spec_len, headwise, spec in cases:
        contig, strided, T = make_case(num_seqs, spec_len, gen, headwise)
        nslots = spec_len + 1
        state = torch.randn(
            num_seqs * nslots + 1, H, D, D, device="cuda", dtype=torch.float32
        )
        # Slot 0 is NULL_BLOCK_ID; real slots start at 1.
        indices = (
            torch.arange(
                1, num_seqs * nslots + 1, device="cuda", dtype=torch.int32
            ).reshape(num_seqs, nslots)
        )
        accepted = torch.full(
            (num_seqs,), spec_len, device="cuda", dtype=torch.int32
        )
        kw = dict(
            spec_len=spec_len, T=T, state=state,
            accepted=accepted, indices=indices, spec=spec,
        )
        o_c, s_c = run(kda, contig, **kw)
        o_s, s_s = run(kda, strided, **kw)
        ok_o = torch.equal(o_c, o_s)
        ok_s = torch.equal(s_c, s_s)
        tag = (f"seqs={num_seqs} len={spec_len} "
               f"beta={'headwise' if headwise else 'scalar'} "
               f"{'spec' if spec else 'plain'}")
        if ok_o and ok_s:
            print(f"  OK   {tag}")
        else:
            bad += 1
            dmax = (o_c.float() - o_s.float()).abs().max().item()
            smax = (s_c - s_s).abs().max().item()
            print(f"  FAIL {tag}  out_equal={ok_o} state_equal={ok_s} "
                  f"max|dout|={dmax:.3e} max|dstate|={smax:.3e}")
    print(f"\n{len(cases) - bad}/{len(cases)} cases bit-identical")
    if bad:
        print("GATE: FAIL -- keep VLLM_GLM53_KDA_STRIDED=0")
        return 1
    print("GATE: PASS -- strided views produce the same bytes as contiguous copies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
