"""Tables from the threads probe record: the C=2 step as one batch against two request threads, one rank.

    python3 summarize.py c2threads-0916-d263c317.json
"""
import json
import sys


def main(path):
    r = json.load(open(path))
    print(f"rank {r['rank']}, {r['layers']} layers, spec_k {r['spec_k']}, lanes {r['lanes']}, moe {r['moe_static']}")
    print(f"scope: {r['scope']}\n")
    print("decode by rows (captured replay, median of samples):")
    print(f"  {'rows':>4s} {'tokens':>6s} {'ms/step':>8s} {'min':>7s} {'U/layer':>8s}")
    for d in r["decode"]:
        print(f"  {d['rows']:4d} {d['tokens']:6d} {d['replay_ms_median']:8.2f} {d['replay_ms_min']:7.2f} {d['unique_experts_mean']:8.1f}")
    t = r["threads"]
    v = t["verdict"]
    rows = {x["arm"]: x for x in t["rows"]}
    print("\nthread arms (16 rows unless solo8; one stream, identity exchanges):")
    print(f"  {'arm':12s} {'ms/step':>8s} {'min':>7s} {'launches':>9s} {'gaps ms':>8s} {'busy ms':>8s}")
    for name in ("joint16", "solo8", "split_attn", "split_both"):
        x = rows[name]
        tl = x["timeline"]
        print(f"  {name:12s} {x['replay_ms_median']:8.2f} {x['replay_ms_min']:7.2f} {v['launches'][name]:9.0f} {tl['idle_ms']:8.2f} {tl['busy_ms']:8.2f}")
    print("\nverdict:")
    for k in ("T1_solo8_ms", "T2_joint16_ms", "s1_attn_penalty_ms", "s1_both_penalty_ms", "s2_dense_penalty_ms", "joint16_gaps_ms"):
        print(f"  {k:22s} {v[k]:8.2f}")
    u = v["unique_experts_per_layer"]
    print(f"  unique experts/layer: batch {u['joint']:.1f}, thread A {u['thread_a']:.1f}, thread B {u['thread_b']:.1f}, "
          f"A+B {u['threads_sum']:.1f} ({u['threads_sum'] / u['joint']:.2f}x the batch's bytes)")
    print("\nwhere the split's time goes -- lane raw ms per step (timeline), split minus joint16:")
    lanes = sorted({l for x in rows.values() for l in x["timeline"]["lanes"]})
    j = rows["joint16"]["timeline"]["lanes"]
    print(f"  {'lane':22s} {'joint16':>8s} {'attn':>8s} {'d':>7s} {'both':>8s} {'d':>7s} {'solo8':>8s}   launches j/attn/both")
    order = sorted(lanes, key=lambda l: -(rows["split_both"]["timeline"]["lanes"].get(l, {}).get("raw_ms", 0)))
    for l in order:
        g = lambda arm: rows[arm]["timeline"]["lanes"].get(l, {})
        jm, am, bm, sm = (g(a).get("raw_ms", 0.0) for a in ("joint16", "split_attn", "split_both", "solo8"))
        print(f"  {l:22s} {jm:8.2f} {am:8.2f} {am - jm:+7.2f} {bm:8.2f} {bm - jm:+7.2f} {sm:8.2f}   "
              f"{g('joint16').get('launches', 0):.0f}/{g('split_attn').get('launches', 0):.0f}/{g('split_both').get('launches', 0):.0f}")
    print("\nlargest kernel deltas, split_attn - joint16 (us per step, launches per step):")
    def by_name(arm):
        return {k["kernel"]: k for k in rows[arm]["kernels"]["kernels"]}
    ja, aa, ba = by_name("joint16"), by_name("split_attn"), by_name("split_both")
    deltas = []
    for name in set(ja) | set(aa):
        j0, a0 = ja.get(name, {}).get("us", 0.0), aa.get(name, {}).get("us", 0.0)
        deltas.append((a0 - j0, name, ja.get(name, {}).get("launches", 0.0), aa.get(name, {}).get("launches", 0.0), j0, a0))
    for d, name, lj, la, j0, a0 in sorted(deltas, reverse=True)[:14]:
        print(f"  {d / 1000:+8.3f} ms  joint {j0 / 1000:7.3f} ({lj:5.1f})  attn {a0 / 1000:7.3f} ({la:5.1f})  {name[:80]}")
    if "isolated" in r and r["isolated"]:
        print("\nkernel classes alone (min of samples), rows 8 and 32:")
        for x in r["isolated"]:
            print(f"  rows={x['rows']:2d} {x['kernel']:30s} {x['us']:8.1f} us {x['mb']:7.2f} MiB {x['gbps']:6.0f} GB/s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c2threads-0916-d263c317.json")
