# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_timeline.py <profiler .pt.trace.json.gz> [step index, default: middle]
"""One step's timeline: streams, the MoE router gate's overlap with other
streams, gaps > 150 us and kernels > 300 us, and the drafter tail region."""
from __future__ import annotations

import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_common import cut_steps, load_kernel_events, stream_of  # noqa: E402


def main(path: str, which: int | None) -> int:
    ev = load_kernel_events(path)
    anchor, starts = cut_steps(ev)
    i = len(starts) // 2 if which is None else which
    if not (0 <= i < len(starts) - 1):
        print(f"step index {i} out of range (0..{len(starts) - 2})")
        return 1
    seg = ev[starts[i]:starts[i + 1]]
    t0 = seg[0]["ts"]
    print(f"step {i} of {len(starts) - 1} (anchor {anchor}); streams:",
          dict(collections.Counter(stream_of(e) for e in seg)))
    gates = [e for e in seg if e["name"].startswith("_deneb_gate_partial")]
    if gates:
        by_stream = collections.defaultdict(list)
        for e in seg:
            by_stream[stream_of(e)].append(e)
        ov = collections.Counter()
        ovt = 0.0
        for g in gates:
            gs, ge = g["ts"], g["ts"] + g["dur"]
            for st, evs in by_stream.items():
                if st == stream_of(g):
                    continue
                for e in evs:
                    s, t = e["ts"], e["ts"] + e["dur"]
                    if s < ge and t > gs:
                        ov[e["name"][:70]] += 1
                        ovt += min(ge, t) - max(gs, s)
        print(f"gate kernels {len(gates)}, median dur {statistics.median(g['dur'] for g in gates):.1f} us, "
              f"stream {stream_of(gates[0])}; other-stream overlap total {ovt:.0f} us;",
              ov.most_common(5))
    else:
        print("no _deneb_gate_partial kernels (moe_gate_sm121 off)")
    print("=== timeline (gaps > 150 us, kernels > 300 us) ===")
    prev_end = t0
    for e in seg:
        gap = e["ts"] - prev_end
        if gap > 150:
            print(f"  [{(prev_end - t0)/1000:6.2f} ms] GAP {gap/1000:.2f} ms")
        if e["dur"] > 300:
            print(f"  [{(e['ts'] - t0)/1000:6.2f} ms] {e['dur']:.0f} us  {e['name'][:90]}  stream={stream_of(e)}")
        prev_end = max(prev_end, e["ts"] + e["dur"])
    print(f"step span {(prev_end - t0)/1000:.2f} ms")
    fc = [e for e in seg if "deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 7u" in e["name"]]
    if fc:
        after = [e for e in seg if e["ts"] >= fc[0]["ts"] - 400]
        busy = sum(e["dur"] for e in after)
        print(f"drafter tail: {len(after)} kernels from {(after[0]['ts'] - t0)/1000:.2f} ms to step end, busy {busy/1000:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None))
