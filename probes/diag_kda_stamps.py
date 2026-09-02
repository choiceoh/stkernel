#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MK-KDA phase-stamp diagnosis (srv4 scratch container, never serving).

Builds the extension with -DMK_PHASE_TS=1 (VLLM_GLM53_MK_PHASE_TS=1 must be
set BEFORE the extension compiles), runs the acc=3 fixture with drain=True
and prints the per-phase end stamps, the surviving barrier's wait, and the
span. 11차: phases 1-3 hand work to their consumers through the
g_mk_kda_* readiness counters, so a phase's end stamp includes its own
producers' waits and the old bar1/bar2/bar3 waits no longer exist.

    python3 /repo/probes/diag_kda_stamps.py [--acc 3] [--reps 5]

Stamp slots (glm53_megakernel.cu, g_mk_kda_ts[block][16]) -- the slot
semantics belong to whatever kernel is compiled, check the MK_KDA_TS
markers in the .cu before reading this table; the shipped (barrier)
order: 0 kernel entry | 1 p0 in_proj done | 2 bar1 out | 3 gates done |
  4 (conv start) | 5 conv done | 6 bar2 out | 7 delta done | 8 bar3 out |
  9 norm done | 10 bar4 out | 11 p5 o_proj done
"""
from __future__ import annotations

import argparse
import statistics
import sys

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import torch  # noqa: E402

NB = 48  # grid cap; the probe refuses anything else


def _med(xs):
    return statistics.median(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    from vllm.model_executor.layers import glm53_megakernel as mk

    torch.cuda.init()
    ext = mk._build()
    major, minor, sms, smem = ext.probe_device()
    print(f"device cc={major}.{minor} sms={sms} smem_optin={smem}")
    assert (major, minor, sms) == (12, 1, 48), "not a GB10"

    fx = mk._KdaFixture(acc=args.acc)
    got = fx.mk_run(drain=True)  # warm the build/JIT paths
    torch.cuda.synchronize()

    spans = []
    for rep in range(args.reps):
        fx.mk_run(drain=True)
        torch.cuda.synchronize()
        kda = ext.read_kda_ts()  # [NB*16], cleared on read

        def s(slot):
            return [kda[b * 16 + slot] for b in range(NB) if kda[b * 16 + slot]]

        if len(s(0)) != NB:
            print(f"rep {rep}: incomplete stamps -- skip")
            continue
        entry = min(s(0))
        span = (max(s(11)) - entry) / 1e3
        spans.append(span)
        print(f"--- rep {rep}  span {span:.1f} us")
        for label, slot in (("p0 in_proj", 1), ("bar1 out", 2),
                            ("gates", 3), ("conv", 5), ("bar2 out", 6),
                            ("delta", 7), ("bar3 out", 8), ("norm", 9),
                            ("bar4 out", 10), ("p5 o_proj", 11)):
            v = s(slot)
            print(f"  {label:<12} med {(_med(v) - entry) / 1e3:8.1f}"
                  f"  slowest {(max(v) - entry) / 1e3:8.1f}")
        # barrier waits per block: exit - entry around each barrier. The
        # medians are mostly IDLE blocks absorbing (the producing phase's
        # arrival spread is the critical-path part) -- 11차 closed the
        # "remove the barriers" question with these numbers, see
        # MEASUREMENTS.md before acting on a large med here
        for bar, sin, sout in (("bar1", 1, 2), ("bar2", 5, 6),
                               ("bar3", 7, 8), ("bar4", 9, 10)):
            ent, out = s(sin), s(sout)
            waits = [(o - e) / 1e3 for e, o in zip(ent, out) if e and o]
            if waits:
                print(f"  {bar} wait: med {_med(waits):6.1f}  max {max(waits):6.1f}")

    if spans:
        print(f"\nspan over {len(spans)} reps: med {_med(spans):.1f} "
              f"min {min(spans):.1f} max {max(spans):.1f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
