#!/usr/bin/env python3
"""Strided Q/K l2norm gate: candidate vs stock, bit-for-bit, on GPU.

VLLM_GLM53_KDA_PREFILL_QK_NORM replaces ``l2norm_fwd(q.contiguous())`` with
a strided/channel-major kernel over the original views. Its own source
comment requires GPU bit-equality ("Source-equivalent arithmetic is not a
GPU bit-equality result"): the fused tile layout can change the generated
reduction schedule. This probe is that missing gate -- every accepted
layout, length, and value regime must match the stock contiguous arm in
raw BF16 bytes, and the guard must still decline invalid inputs.

Run only in an idle fleet window, in a fresh container with composed sources:
  bash probes/run_mk_probe.sh probes/qk_norm_strided_check.py | tee /tmp/qknorm.log

No timing here: this is the arming gate for the knob, not a speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# The knob is latched when the deployed kda module imports, so arm (or
# accept a forwarded knob) before build_pkg() pulls it in.
os.environ.setdefault("VLLM_GLM53_KDA_PREFILL_QK_NORM", "1")

import torch  # noqa: E402

from kda_prefill_bench import H, K, build_pkg  # noqa: E402

LAYOUTS = ("conv", "qkv", "mixed")


def strided_pair(total, layout, fill, device="cuda"):
    """Return (q, k) views of shape [1, T, H, K] with the given fill."""
    if layout == "conv":
        # Channel-major conv output: token axis is the contiguous one, so
        # stride(1) == 1 selects the channel-major kernel (production path).
        base = fill(1, H, K, total)
        return base.permute(0, 3, 1, 2), base.permute(0, 3, 1, 2).clone()
    if layout == "qkv":
        # Token-major split of a merged QKV projection: stride(1) is the
        # merged width, selecting the strided-row kernel.
        base = fill(1, total, 2, H, K)
        return base[:, :, 0], base[:, :, 1]
    # mixed: q and k arrive from different producers with different strides.
    q_base = fill(1, total, 3, H, K)
    k_base = fill(1, H, K, total)
    return q_base[:, :, 1], k_base.permute(0, 3, 1, 2)


def regimes(device="cuda"):
    g = torch.Generator(device=device).manual_seed(3)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device,
                           dtype=torch.float32).mul(0.8).to(torch.bfloat16)

    def fill_with(value):
        def fill(*shape):
            return torch.full(shape, value, device=device,
                              dtype=torch.bfloat16)
        return fill

    return [
        ("randn", randn),
        ("zeros", fill_with(0.0)),
        # Squares overflow FP32 to inf: rsqrt(inf) = 0 and the normalized
        # output collapses to 0 on both arms, or the bits must still agree.
        ("huge", fill_with(6.0e19)),
        # Squares flush to zero in FP32: rsqrt(eps) scales both arms alike.
        ("tiny", fill_with(1.0e-25)),
        ("minus_one", fill_with(-1.0)),
    ]


def bit_exact(got, expected, label):
    if got.dtype != expected.dtype or got.shape != expected.shape:
        raise AssertionError(f"{label}: shape/dtype drift {got} vs {expected}")
    if not torch.equal(got.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8)):
        delta = (got.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label}: not bit-exact (max_abs={delta})")


def run_case(kda, total, layout, fill_name, fill):
    q, k = strided_pair(total, layout, fill)
    q_ref_input, k_ref_input = q.clone(), k.clone()
    normalized = kda._glm53_qk_l2norm_strided(q, k)
    if normalized is None:
        raise AssertionError(
            f"T={total} layout={layout}: guard declined an accepted fixture "
            "(knob armed + pinned source) -- the probe would test nothing")
    q_out, k_out = normalized
    if not (q_out.is_contiguous() and k_out.is_contiguous()):
        raise AssertionError(f"T={total} layout={layout}: outputs must be dense")
    # The stock arm runs the exact serving fallback on fresh contiguous
    # copies, so a candidate that mutated its inputs cannot hide behind it.
    ref_q = kda.l2norm_fwd(q_ref_input.contiguous())
    ref_k = kda.l2norm_fwd(k_ref_input.contiguous())
    bit_exact(q_out, ref_q, f"T={total} {layout}/{fill_name} q")
    bit_exact(k_out, ref_k, f"T={total} {layout}/{fill_name} k")
    bit_exact(q, q_ref_input, f"T={total} {layout}/{fill_name} q input")
    bit_exact(k, k_ref_input, f"T={total} {layout}/{fill_name} k input")
    return {"T": total, "layout": layout, "regime": fill_name,
            "q_stride": list(q.stride()), "branch":
                "channel_major" if q.stride(1) == 1 else "strided",
            "bit_exact": True}


def run_decline_cases(kda):
    """Invalid fixtures must fall back to stock (None), never run strided."""
    q = torch.randn(1, 128, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 128, H, K, device="cuda", dtype=torch.bfloat16)
    cases = {
        "fp32_dtype": (q.float(), k.float()),
        "wrong_shape": (q[:, :, :, :K - 1], k[:, :, :, :K - 1]),
        "shape_mismatch": (q, k[:, :64]),
        "five_dim": (q.unsqueeze(0), k.unsqueeze(0)),
        "head_count": (q[:, :, :8], k[:, :, :8]),
    }
    for name, (bad_q, bad_k) in cases.items():
        if kda._glm53_qk_l2norm_strided(bad_q, bad_k) is not None:
            raise AssertionError(f"guard accepted invalid fixture: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="1,31,32,33,63,127,128,129,8185,8192")
    parser.add_argument("--json", type=Path,
                        help="optional JSON path inside the container; use a writable host bind to retain it")
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or any(length <= 0 for length in lengths):
        parser.error("lengths must be positive")
    torch.manual_seed(0)
    kda = build_pkg()["kda"]
    if not getattr(kda, "_GLM53_KDA_QK_L2NORM_STRIDED", False):
        raise RuntimeError(
            "VLLM_GLM53_KDA_PREFILL_QK_NORM did not arm; a forwarded value "
            "wins, so check the shell that launched run_mk_probe.sh")
    if not kda._glm53_l2norm_source_matches(kda.l2norm_fwd):
        raise RuntimeError("pinned l2norm source drifted; probe refuses to run")
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "knob": "VLLM_GLM53_KDA_PREFILL_QK_NORM=1",
              "sources": {"kda": hashlib.sha256(
                  Path(kda.__file__).read_bytes()).hexdigest()},
              "cases": []}
    print(json.dumps({key: value for key, value in report.items()
                      if key != "cases"}), flush=True)
    for fill_name, fill in regimes():
        for total in lengths:
            for layout in LAYOUTS:
                row = run_case(kda, total, layout, fill_name, fill)
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
    run_decline_cases(kda)
    print(json.dumps({"decline_cases": "OK"}), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: strided QK l2norm bit-exact against stock on every "
          "layout/length/regime; guard declines invalid fixtures")


if __name__ == "__main__":
    main()
