#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile b12x static-lane specs to their .o on the CPU -- no GPU, ~3 s a
spec -- so a kernel edit is checked before it waits for a fleet window.

The cute-dsl compiler only needs the target name (CUTE_DSL_ARCH=sm_121a, set
by the runner's MK_PROBE_NO_GPU=1) and a few device queries, which this
script answers for a GB10 (cc 12.1, 48 SMs) without a CUDA context. What it
proves: the spec parses, the kernel traces, the TMA descriptors and smem
layouts are consistent, ptxas accepts the code. What it does not prove:
numerics, timing (the probe on a quiet GPU).

    MK_PROBE_NO_GPU=1 bash probes/run_mk_probe.sh probes/b12x_static_compile_check.py --specs "u|t|v,t|t,s"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402

# the device queries the dispatcher makes before compiling: answered for a
# GB10 so no CUDA context is created (the host may be serving)
torch.cuda.is_available = lambda: True  # type: ignore[assignment]
torch.cuda.get_device_capability = lambda *a, **k: (12, 1)  # type: ignore[assignment]

E, TOPK, HID, INTER = 288, 8, 4096, 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="u|t", help="'|'-separated static v2 specs")
    ap.add_argument("--m", type=int, default=8, help="decode rows (the m of the kernel name)")
    args = ap.parse_args()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    md.get_num_sm = lambda dev=None: 48  # type: ignore[assignment]
    md.get_max_active_clusters = lambda n=1: 48  # type: ignore[assignment]
    max_rows = md._align_up(args.m * TOPK, 128)
    ok = True
    for spec in [p.strip() for p in args.specs.split("|") if p.strip()]:
        cfg = md._parse_glm53_static_v2(spec, probe=True)
        if cfg is None:
            print(f"[{spec}] parse -> stock (nothing to compile)")
            continue
        md._STATIC_V2_KERNEL_CACHE.clear()
        t0 = time.time()
        try:
            compiled, mac = md._get_static_kernel_v2(
                E, E, args.m, HID, INTER, TOPK, max_rows,
                config=cfg, mac_override=48,
                activation="swigluoai_uninterleave", swiglu_alpha=1.0,
                swiglu_beta=0.0, swiglu_limit=10.0,
            )
        except Exception as exc:  # noqa: BLE001 -- report every spec
            ok = False
            print(f"[{spec}] COMPILE FAIL after {time.time() - t0:.1f} s: "
                  f"{type(exc).__name__}: {str(exc)[:1200]}")
            continue
        names = [k[0] + ":" + "/".join(str(x) for x in k[1:6]) for k in md._STATIC_V2_KERNEL_CACHE]
        print(f"[{spec}] compiled in {time.time() - t0:.1f} s: {names} mac={mac}")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
