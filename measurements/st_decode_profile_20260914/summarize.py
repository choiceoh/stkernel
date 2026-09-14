"""The C=1 decode profile record: lanes and kernels per iteration from decode-profile-<label>.json.

Only runs whose table carries real device time count (a profiler run that captured nothing is listed and dropped).
Times are kernel self device time per iteration under the profiler; kernels on different streams overlap, so the
lane sum can exceed the wall. Launches are kernel calls per iteration.

    python3 summarize.py decode-profile-main956b.json [--all]
"""
import json
import re
import sys

LANES = (
    ("TP communication", r"oneshot|k_publish|k_reserve|nccl|osar"),    # before MoE: k_oneshot_moe_packets is a collective
    ("MoE experts", r"moe|b12x"),
    ("mHC", r"mhc|hc_prenorm"),
    ("LM head / drafter fc (deep_gemm fp8xfp4)", r"deep_gemm.*gemm_1d1d"),
    ("dense GEMM", r"mk_gemm|mk_input_pack|mk_query_pair|cutlass"),
    ("MLA / DSA", r"mk_mla|mqa_logits|topk|bitonicsort|_pool_slots|_attend|kpool|indexer"),
    ("KDA", r"fused_recurrent|_kda|_single_conv|conv1d"),
    ("norm / elementwise / copy", r"norm|elementwise|vectorized|copy|cat|fill|index|gather|scatter|reduce|memcpy|memset"),
)


def short(name):
    """`void (anonymous namespace)::mk_gemm2_kernel<1, ...>(Ctx)` -> `mk_gemm2_kernel`."""
    s = re.sub(r"^void\s+|\(anonymous namespace\)::", "", name.strip())
    s = re.split(r"[<(]", s, maxsplit=1)[0]
    return s.split("::")[-1][:44] or name[:44]


def lane_of(name):
    low = name.lower()
    for lane, pattern in LANES:
        if re.search(pattern, low):
            return lane
    return "other (ST glue kernels)"


def main():
    data = json.load(open(sys.argv[1]))
    show_all = "--all" in sys.argv
    good = []
    for r in data["runs"]:
        ok = r["concurrency"] == 1 and (r.get("device_us_per_step") or 0) > 100000
        p, u = r["profiled"], r["unprofiled"]
        print(f"C={r['concurrency']} #{r['repeat']}: device {r.get('device_us_per_step')} us/burst, "
              f"bursts {p.get('st:step_seconds_count'):.0f}, burst ms {p.get('burst_ms'):.1f} profiled / {u.get('burst_ms'):.1f} unprofiled, "
              f"iteration ms {p.get('iteration_ms'):.2f} / {u.get('iteration_ms'):.2f}, acceptance {p.get('acceptance'):.3f}"
              f"{'' if ok else '  -> dropped (the profile captured no decode kernels)'}")
        if ok:
            good.append(r)
    kern = {}
    for r in good:
        it = r["profiled"]["iterations"] / r["profiled"]["st:step_seconds_count"]
        for k in r["kernels"]:
            e = kern.setdefault(k["kernel"], [0.0, 0.0])
            e[0] += k["us_per_step"] / it / len(good) / 1000
            e[1] += k["calls"] / r["profiled"]["iterations"] / len(good)
    lanes = {}
    for name, (ms, calls) in kern.items():
        entry = lanes.setdefault(lane_of(name), [0.0, 0.0, []])
        entry[0] += ms
        entry[1] += calls
        entry[2].append((ms, calls, name))
    total_ms = sum(v[0] for v in lanes.values())
    total_calls = sum(v[1] for v in lanes.values())
    print(f"\n{len(good)} runs; kernel time {total_ms:.2f} ms and {total_calls:.0f} launches per iteration")
    print("| lane | ms / iteration | share | launches / iteration | biggest kernels (ms) |")
    print("|---|---|---|---|---|")
    for lane, (ms, calls, rows) in sorted(lanes.items(), key=lambda kv: -kv[1][0]):
        rows.sort(reverse=True)
        top = "; ".join(f"`{short(n)}` {m:.2f}" for m, c, n in rows[:3])
        print(f"| {lane} | {ms:.2f} | {100 * ms / total_ms:.1f}% | {calls:.1f} | {top} |")
    if show_all:
        print("\nall kernels >= 0.01 ms")
        for name, (ms, calls) in sorted(kern.items(), key=lambda kv: -kv[1][0]):
            if ms >= 0.01:
                print(f"  {ms:6.2f} ms {calls:7.1f}/it [{lane_of(name)[:12]}] {re.sub(r'\s+', ' ', name)[:110]}")


if __name__ == "__main__":
    main()
