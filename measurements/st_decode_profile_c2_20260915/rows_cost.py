"""Decode iteration cost by row count, from onepass latency.jsonl (gpu_iteration records)."""
import collections
import glob
import json
import os
import statistics as st
import sys

STAGES = ("forward", "propose", "observe", "sample", "commit", "boundaries")
root = "/home/choiceoh/expert-capture/onepass-runs"
runs = sys.argv[1:]
max_pos = 6000

for run in runs:
    base = f"{root}/{run}"
    by = collections.defaultdict(list)
    host = collections.defaultdict(list)
    for ph in sorted(glob.glob(base + "/measure-c*")):
        f = ph + "/latency.jsonl"
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("phase") != "decode":
                continue
            if r.get("kind") == "gpu_iteration":
                if max(r.get("positions") or [0]) > max_pos:
                    continue
                by[len(r["rows"])].append(r)
            elif r.get("kind") == "host_step":
                host[len(r["rows"])].append(r["duration_us"])
    print("==", run)
    one = None
    for n in sorted(by):
        rs = by[n]
        d = st.median(x["duration_us"] for x in rs)
        stg = {k: st.median(x["stages_us"].get(k, 0) for x in rs) for k in STAGES}
        acc = st.mean(sum(x["accepted"]) / n for x in rs)
        if n == 1:
            one = (d, stg)
        hs = st.median(host[n]) if host[n] else float("nan")
        line = (f"n={n} iters={len(rs):5d} iter_ms={d/1e3:6.2f} fwd={stg['forward']/1e3:6.2f} "
                f"prop={stg['propose']/1e3:5.2f} obs={stg['observe']/1e3:5.2f} acc/row={acc:.2f} "
                f"host_step_ms={hs/1e3:5.2f}")
        if one:
            line += (f" | x{d/one[0]:.2f} fwd x{stg['forward']/one[1]['forward']:.2f} "
                     f"prop x{stg['propose']/one[1]['propose']:.2f} obs x{stg['observe']/one[1]['observe']:.2f}"
                     f" | per-row-token-rate x{n*(1+acc)/(1+st.mean(sum(x['accepted']) for x in by[1]))/(d/one[0]):.2f}")
        print(line)
