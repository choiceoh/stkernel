"""Kernel time per lane in onepass diagnostic decode traces, C=1 vs C=4 (same boot), all ranks averaged."""
import collections
import glob
import gzip
import json
import re
import sys

LANES = (
    ("TP communication", r"oneshot|k_publish|k_reserve|nccl|osar"),
    ("MoE experts", r"moe|b12x"),
    ("mHC", r"mhc|hc_prenorm"),
    ("LM head / drafter fc", r"deep_gemm.*gemm_1d1d|gemm_1d1d"),
    ("dense GEMM", r"mk_gemm|mk_input_pack|mk_query_pair|cutlass"),
    ("MLA / DSA", r"mk_mla|mqa_logits|topk|bitonicsort|_pool_slots|_attend|kpool|indexer"),
    ("KDA", r"fused_recurrent|_kda|_single_conv|conv1d"),
    ("norm / elementwise / copy", r"norm|elementwise|vectorized|copy|cat|fill|index|gather|scatter|reduce|memcpy|memset"),
)


def lane_of(name):
    low = name.lower()
    for lane, pattern in LANES:
        if re.search(pattern, low):
            return lane
    return "other (ST glue)"


def short(name):
    s = re.sub(r"^void\s+|\(anonymous namespace\)::", "", name.strip())
    s = re.split(r"[<(]", s, maxsplit=1)[0]
    return s.split("::")[-1][:48] or name[:48]


def load(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def phase_stats(run_dir, phase):
    lanes = collections.defaultdict(lambda: [0.0, 0])
    kernels = collections.defaultdict(lambda: [0.0, 0])
    steps = 0
    for path in sorted(glob.glob(f"{run_dir}/{phase}/rank-*/decode-*.trace.json.gz")):
        events = load(path)
        steps += 1
        for e in events:
            if e.get("cat") not in ("kernel", "Kernel"):
                continue
            ms = e.get("dur", 0) / 1000.0
            lane = lane_of(e.get("name", ""))
            lanes[lane][0] += ms
            lanes[lane][1] += 1
            k = kernels[short(e.get("name", ""))]
            k[0] += ms
            k[1] += 1
    return steps, lanes, kernels


run_dir = sys.argv[1]
c1_phase, c4_phase = sys.argv[2], sys.argv[3]
s1, l1, k1 = phase_stats(run_dir, c1_phase)
s4, l4, k4 = phase_stats(run_dir, c4_phase)
print(f"steps (rank-files): {c1_phase} {s1}, {c4_phase} {s4}")
t1 = sum(v[0] for v in l1.values()) / s1
t4 = sum(v[0] for v in l4.values()) / s4
print(f"kernel ms/step: C1 {t1:.2f}  C4 {t4:.2f}  x{t4/t1:.2f}")
print("| lane | C1 ms | C4 ms | C4-C1 ms | x | C1 launches | C4 launches |")
for lane in sorted(set(l1) | set(l4), key=lambda x: -(l4.get(x, [0])[0] / s4 - l1.get(x, [0])[0] / s1)):
    a, b = l1.get(lane, [0, 0]), l4.get(lane, [0, 0])
    am, bm = a[0] / s1, b[0] / s4
    print(f"| {lane} | {am:.2f} | {bm:.2f} | {bm-am:+.2f} | x{(bm/am if am else float('nan')):.2f} | {a[1]/s1:.0f} | {b[1]/s4:.0f} |")
print("\nbiggest C4-C1 kernel deltas (ms/step)")
names = set(k1) | set(k4)
deltas = sorted(((k4.get(n, [0, 0])[0] / s4 - k1.get(n, [0, 0])[0] / s1, n) for n in names), reverse=True)
for d, n in deltas[:22]:
    a, b = k1.get(n, [0, 0]), k4.get(n, [0, 0])
    print(f"  {d:+6.2f}  C1 {a[0]/s1:6.2f} ({a[1]/s1:4.0f})  C4 {b[0]/s4:6.2f} ({b[1]/s4:4.0f})  [{lane_of(n)[:10]}] {n}")
