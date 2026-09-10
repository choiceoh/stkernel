#!/usr/bin/env python3
"""Is casting E8M0 block scales to float32 exact? No GPU, no weights.

`dsv41_scales` converts V4.1's dense-linear scales because Triton cannot name
`float8_e8m0fnu`:

    KeyError: 'float8_e8m0fnu'   (triton/_utils.py, canonicalize_dtype)

The conversion is only admissible if it is LOSSLESS. E8M0 is a bare 8-bit
exponent, so every representable value is a power of two and float32 holds all
of them -- but "so it must be exact" is an argument, not a check. This walks
the ENTIRE 8-bit domain, all 256 codes, and requires a round trip through
float32 to return the identical byte.

It also checks the scoping, which is the part that can be wrong while every
number is right: MXFP4 wants its expert scales in e8m0, so a module that looks
like a MoE must be left alone. A blanket conversion boots and then feeds the
MXFP4 kernel a layout it does not expect.

    python3 probes/dsv41_scales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_vllm"))

from dsv41_scales import E8M0, _is_moe, convert_module      # noqa: E402


class Fake(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(
            torch.ones(2, 2, dtype=dtype), requires_grad=False)
        self.weight_scale_inv.some_loader_metadata = "carry me"


class FusedMoE(Fake):
    pass


def main() -> int:
    if E8M0 is None:
        print("  this torch has no float8_e8m0fnu; nothing to check")
        return 1
    ok = True

    # -- 1. every one of the 256 codes survives the round trip -------------
    codes = torch.arange(256, dtype=torch.uint8)
    e8 = codes.view(torch.float8_e8m0fnu)
    back = e8.to(torch.float32).to(torch.float8_e8m0fnu).view(torch.uint8)
    # code 255 is E8M0's NaN; it is not a scale any checkpoint writes, and
    # NaN != NaN makes a byte comparison the only meaningful one
    same = int((back == codes).sum())
    ok &= same == 256
    print(f"  round trip   {same}/256 codes identical")
    finite = e8.to(torch.float32)
    powers = finite[torch.isfinite(finite) & (finite > 0)]
    exact = bool(torch.all(torch.log2(powers) == torch.round(torch.log2(powers))))
    print(f"  values       {len(powers)} finite positives, all powers of two: "
          f"{'yes' if exact else 'NO'}")
    ok &= exact

    # -- 2. conversion replaces the tensor and keeps loader metadata -------
    m = Fake(torch.float8_e8m0fnu)
    n = convert_module(m)
    kept = getattr(m.weight_scale_inv, "some_loader_metadata", None)
    print(f"  convert      {n} tensor(s); dtype now "
          f"{m.weight_scale_inv.dtype}; loader metadata "
          f"{'kept' if kept == 'carry me' else 'LOST'}")
    ok &= n == 1 and m.weight_scale_inv.dtype is torch.float32
    ok &= kept == "carry me"
    # the fp8 WEIGHT must not be touched -- only the scale
    ok &= m.weight.dtype is torch.float8_e4m3fn
    print(f"  weight       untouched: {m.weight.dtype}")

    # -- 3. a module that already holds fp32 is a no-op --------------------
    m2 = Fake(torch.float32)
    print(f"  idempotent   second pass converts {convert_module(m2)} (want 0)")
    ok &= convert_module(m2) == 0

    # -- 4. scoping: MoE modules keep e8m0 --------------------------------
    moe = FusedMoE(torch.float8_e8m0fnu)
    skipped = _is_moe(moe)
    print(f"  scoping      {type(moe).__name__} recognised as MoE: "
          f"{'yes' if skipped else 'NO -- its MXFP4 scales would be converted'}")
    ok &= skipped
    ok &= not _is_moe(Fake(torch.float32))
    print(f"               {type(m).__name__} recognised as MoE: "
          f"{'yes -- WRONG' if _is_moe(m) else 'no'}")

    print("\n" + ("SCALES PASS" if ok else "SCALES FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
