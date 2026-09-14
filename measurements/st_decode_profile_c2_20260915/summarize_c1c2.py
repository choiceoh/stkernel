"""C=1 vs C=2 per-step kernel time from c2_profile.py output (door profile tables).

Per window: kernel us_per_step is per burst; steps per burst = (drafted / 7 / concurrency) / bursts.
Lanes follow measurements/st_decode_profile_20260914/summarize.py so the table lines up with #963.
"""
import collections
import json
import re
import sys

LANES = (
    ("TP communication", r"oneshot|k_publish|k_reserve|nccl|osar"),
    ("MoE experts", r"moe|b12x"),
    ("mHC", r"mhc|hc_prenorm"),
    ("LM head / drafter fc", r"deep_gemm.*gemm_1d1d|gemm_1d1d"),
    ("dense GEMM", r"mk_gemm|mk_input_pack|mk_query_pair|mk_wide_input|cutlass"),
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
    s = re.sub(r"\s+", " ", s)
    return s[:110]


data = json.load(open(sys.argv[1]))
per = {}
for run in data["runs"]:
    kernels = run.get("kernels") or []
    prof = run["profiled"]
    c = run["concurrency"]
    if not kernels or not prof.get("iterations") or (run.get("device_us_per_step") or 0) < 100000:
        print(f"C={c} #{run['repeat']}: empty table ({run.get('device_us_per_step')} us/burst), skipped")
        continue
    bursts = prof.get("st:step_seconds_count") or 0
    steps_per_burst = prof["iterations"] / c / bursts
    plain = run["unprofiled"]
    print(f"C={c} #{run['repeat']}: steps/burst {steps_per_burst:.2f}, step ms profiled {prof.get('step_ms'):.2f} "
          f"plain {plain.get('step_ms') or float('nan'):.2f}, accept {prof.get('acceptance'):.3f}, kernels {len(kernels)}")
    acc = per.setdefault(c, dict(n=0, kern=collections.defaultdict(lambda: [0.0, 0.0]), step_ms=[], plain_ms=[]))
    acc["n"] += 1
    acc["step_ms"].append(prof.get("step_ms"))
    acc["plain_ms"].append(plain.get("step_ms"))
    for k in kernels:
        e = acc["kern"][k["kernel"]]
        e[0] += k["us_per_step"] / steps_per_burst / 1000.0
        e[1] += k["calls"] / (prof["iterations"] / c)

if 1 not in per or 2 not in per:
    sys.exit("need both C=1 and C=2 tables")
tables = {}
for c, acc in per.items():
    lanes = collections.defaultdict(lambda: [0.0, 0.0])
    for name, (ms, calls) in acc["kern"].items():
        lanes[lane_of(name)][0] += ms / acc["n"]
        lanes[lane_of(name)][1] += calls / acc["n"]
    tables[c] = lanes
t1 = sum(v[0] for v in tables[1].values())
t2 = sum(v[0] for v in tables[2].values())
print(f"\nkernel ms/step (profiler): C1 {t1:.2f}  C2 {t2:.2f}  x{t2/t1:.2f}")
print("| lane | C1 ms | C2 ms | C2-C1 | x | C1 launches | C2 launches |\n|---|---|---|---|---|---|---|")
for lane in sorted(set(tables[1]) | set(tables[2]), key=lambda l: -(tables[2][l][0] - tables[1][l][0])):
    a, b = tables[1][lane], tables[2][lane]
    print(f"| {lane} | {a[0]:.2f} | {b[0]:.2f} | {b[0]-a[0]:+.2f} | x{(b[0]/a[0] if a[0] else float('nan')):.2f} | {a[1]:.0f} | {b[1]:.0f} |")
print("\nlargest C2-C1 kernel deltas (ms/step, launches/step)")
names = set(per[1]["kern"]) | set(per[2]["kern"])
rows = []
for n in names:
    a = [x / per[1]["n"] for x in per[1]["kern"].get(n, [0.0, 0.0])]
    b = [x / per[2]["n"] for x in per[2]["kern"].get(n, [0.0, 0.0])]
    rows.append((b[0] - a[0], a, b, n))
for d, a, b, n in sorted(rows, reverse=True)[:30]:
    print(f"  {d:+6.2f}  C1 {a[0]:6.2f} ({a[1]:5.1f})  C2 {b[0]:6.2f} ({b[1]:5.1f})  [{lane_of(n)[:10]}] {short(n)}")
