# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_timeline.py <profiler .pt.trace.json.gz>  (rank-0 decode trace; steps are cut at _gather_block_tables_kernel)
import gzip, json, sys, collections, statistics
p = sys.argv[1]
with gzip.open(p, "rt") as fh:
    tr = json.load(fh)
ev = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
starts = [i for i, e in enumerate(ev) if e["name"].startswith("_gather_block_tables_kernel")]
a, b = starts[10], starts[11]
seg = ev[a:b]
t0 = seg[0]["ts"]
streams = collections.Counter(e.get("args", {}).get("stream") for e in seg)
print("streams in step:", dict(streams))
# 1) gate kernel overlap with other streams
gates = [e for e in seg if e["name"].startswith("_deneb_gate_partial")]
ov = collections.Counter(); ovt = 0.0
for g in gates:
    gs, ge = g["ts"], g["ts"] + g["dur"]
    for e in seg:
        if e is g or e.get("args", {}).get("stream") == g.get("args", {}).get("stream"):
            continue
        s, t = e["ts"], e["ts"] + e["dur"]
        if s < ge and t > gs:
            ov[e["name"][:70]] += 1; ovt += min(ge, t) - max(gs, s)
print(f"gate kernels {len(gates)}, median dur {statistics.median(g['dur'] for g in gates):.1f} us, stream {gates[0].get('args',{}).get('stream')}")
print(f"other-stream overlap total {ovt:.0f} us over {len(gates)} gates; overlapping kernels:", ov.most_common(6))
# 2) timeline: big gaps and long kernels
print("=== timeline (gaps > 150 us, kernels > 300 us) ===")
prev_end = t0
for e in seg:
    gap = e["ts"] - prev_end
    if gap > 150:
        print(f"  [{(prev_end - t0)/1000:6.2f} ms] GAP {gap/1000:.2f} ms")
    if e["dur"] > 300:
        print(f"  [{(e['ts'] - t0)/1000:6.2f} ms] {e['dur']:.0f} us  {e['name'][:90]}  stream={e.get('args',{}).get('stream')}")
    prev_end = max(prev_end, e["ts"] + e["dur"])
print(f"step span {(prev_end - t0)/1000:.2f} ms")
# 3) drafter region: kernels after the last k_oneshot... find the fc kernel (M=7 deep_gemm) and sum kernels from 400us before it to the next step start
fc = [e for e in seg if "deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 7u" in e["name"]]
if fc:
    f = fc[0]; fs = f["ts"]
    after = [e for e in seg if e["ts"] >= fs - 400]
    busy = sum(e["dur"] for e in after)
    span = (after[-1]["ts"] + after[-1]["dur"]) - after[0]["ts"]
    print(f"drafter-region: {len(after)} kernels from {(after[0]['ts']-t0)/1000:.2f} ms to step end, busy {busy/1000:.2f} ms, span {span/1000:.2f} ms")
    names = collections.Counter(e["name"][:60] for e in after)
    print("  top names:", names.most_common(8))
