#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_nodes.py <trace.gz> [--dump FILE] [--gap LO,HI] [--all-cats]
#                                          [--step N] [--anchor NAME] [--top K]
"""Per-step NODE view of a decode trace, streaming (RSS stays ~tens of MB).

Three things the per-category tools do not give:

  1. the per-kernel table -- count/step (median over clean steps), median
     duration, duration x count, the streams it ran on, and who owns it;
  2. `--dump FILE`: the median step's node sequence in launch order (ts, dur,
     gap on its own stream, stream, name) -- what a "graph node" fusion has to
     look at, because the model's layer order is the same in every step;
  3. `--gap LO,HI`: every event of every category (gpu_memcpy, gpu_memset,
     cuda_runtime, ...) inside [t0+LO, t0+HI] of that step -- what the host was
     doing in a GPU gap (this is how 34차 found the drafter's DtoH sync and
     the profiler's own cudaGraphLaunch inflation).

Clean steps = the steps whose kernel count is the mode (a warm-up or
prefill-touched step has a different count). Steps are cut at the anchor
with the most occurrences (trace_common.STEP_ANCHORS), unless --anchor.
"""
from __future__ import annotations

import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_common import STEP_ANCHORS, category, owner, union_busy_us  # noqa: E402


def iter_events(path: str):
    """traceEvents one at a time (same cutter as census.py: string-aware,
    because kernel names carry `{lambda()#3}`)."""
    with gzip.open(path, "rt") as f:
        head = ""
        while '"traceEvents"' not in head:
            chunk = f.read(1 << 16)
            if not chunk:
                return
            head = head[-32:] + chunk
        tail = head[head.index('"traceEvents"'):]
        buf, depth, in_str, esc, started = [], 0, False, False, False
        while True:
            for c in tail:
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                    if depth == 1:
                        buf, started = [], True
                elif c == "}":
                    depth -= 1
                    if depth == 0 and started:
                        buf.append(c)
                        yield json.loads("".join(buf))
                        buf, started = [], False
                        continue
                if started:
                    buf.append(c)
            tail = f.read(1 << 20)
            if not tail:
                return


def _med(v):
    return statistics.median(v) if v else float("nan")


def collect_steps(path: str, anchor: str | None = None):
    """(anchor, steps) where steps is a list of per-step kernel lists
    [(ts, dur, stream, name), ...] sorted by ts; the partial first and last
    step are dropped. Only kernel events are kept."""
    counts = defaultdict(int)
    steps, cur = [], []
    chosen = anchor
    for e in iter_events(path):
        if e.get("ph") != "X" or "kernel" not in (e.get("cat") or ""):
            continue
        n = e.get("name", "?")
        for a in STEP_ANCHORS:
            if n.startswith(a):
                counts[a] += 1
        cur.append((e.get("ts", 0.0), e.get("dur", 0.0),
                    (e.get("args") or {}).get("stream"), n))
    if chosen is None:
        if not counts:
            raise SystemExit(f"no step anchor found among {STEP_ANCHORS}")
        # ties (a sampler anchor and a prep anchor both fire once a step) go to
        # the earlier STEP_ANCHORS entry, the step-START kernel
        chosen = max(STEP_ANCHORS, key=lambda a: counts.get(a, 0))
    for row in cur:
        if row[3].startswith(chosen):
            steps.append([])
        if steps:
            steps[-1].append(row)
    steps = steps[1:-1]
    for s in steps:
        s.sort()
    return chosen, steps


def analyze(path: str, anchor: str | None = None) -> dict:
    chosen, steps = collect_steps(path, anchor)
    if len(steps) < 2:
        raise SystemExit(f"{path}: need >= 4 anchors, got {len(steps) + 2}")
    cnts = [len(s) for s in steps]
    mode = statistics.mode(cnts)
    clean = [i for i, s in enumerate(steps) if len(s) == mode]
    lens = [s[-1][0] + s[-1][1] - s[0][0] for s in steps]
    busy = {i: union_busy_us([{"ts": t, "dur": d} for t, d, _, _ in steps[i]]) for i in clean}
    cat_t, cat_n = defaultdict(list), defaultdict(list)
    kn, kd, strm = defaultdict(list), defaultdict(list), defaultdict(set)
    for i in clean:
        t, c, kc = defaultdict(float), defaultdict(int), defaultdict(int)
        for _, d, st, n in steps[i]:
            k = category(n)
            t[k] += d
            c[k] += 1
            kc[n] += 1
            kd[n].append(d)
            strm[n].add(st)
        for k in t:
            cat_t[k].append(t[k])
            cat_n[k].append(c[k])
        for n, v in kc.items():
            kn[n].append(v)
    step_len = [lens[i] for i in clean]
    target = _med(step_len)
    median_step = min(clean, key=lambda i: abs(lens[i] - target)) if clean else None
    return {
        "path": path, "anchor": chosen, "steps": len(steps), "mode": mode,
        "clean": len(clean), "counts": sorted(set(cnts)),
        "step_ms": target / 1000,
        "busy_ms": _med([busy[i] for i in clean]) / 1000,
        "idle_ms": _med([lens[i] - busy[i] for i in clean]) / 1000,
        "cats": {k: {"ms": _med(cat_t[k]) / 1000, "cnt": _med(cat_n[k])} for k in cat_t},
        "kernels": {n: {"cnt": _med(kn[n]), "dur": _med(kd[n]), "streams": sorted(
            str(x) for x in strm[n]), "owner": owner(n)} for n in kn},
        "median_step": median_step,
        "_steps": steps,
    }


def report(r: dict, top: int) -> None:
    print(f"# {r['path']}\n# anchor={r['anchor']} steps={r['steps']} clean={r['clean']} "
          f"(kernel-count mode {r['mode']}; counts seen {r['counts'][:10]})")
    print(f"# clean step median {r['step_ms']:.2f} ms, union busy {r['busy_ms']:.2f} ms, "
          f"idle {r['idle_ms']:.2f} ms")
    print("\n## composition (median per clean step)")
    for k in sorted(r["cats"], key=lambda k: -r["cats"][k]["ms"]):
        print(f"  {r['cats'][k]['ms']:7.3f} ms  {r['cats'][k]['cnt']:6.0f}/step  {k}")
    print("\n## per-kernel: count/step  dur_us(median)  sum_us/step  streams  owner  name")
    rows = sorted(((v["cnt"] * v["dur"], v["cnt"], v["dur"], n, v)
                   for n, v in r["kernels"].items()), reverse=True)
    for tot, cnt, dur, n, v in rows[:top]:
        print(f"  {cnt:6.1f}  {dur:8.1f}  {tot:9.1f}  {','.join(v['streams']):8s}  "
              f"{v['owner']:5s}  {n[:140]}")


def dump_step(r: dict, path: str, step: int | None) -> int:
    j = r["median_step"] if step is None else step
    s = r["_steps"][j]
    t0 = s[0][0]
    prev_end: dict = {}
    with open(path, "w") as fh:
        fh.write(f"# step {j} t0={t0:.1f} len {s[-1][0] + s[-1][1] - t0:.1f} us "
                 f"kernels {len(s)}\n# rel_ts dur gap_on_stream stream name\n")
        for ts, d, st, n in s:
            gap = ts - prev_end.get(st, ts)
            prev_end[st] = ts + d
            fh.write(f"{ts - t0:10.1f} {d:8.1f} {gap:8.1f} s{st} {n[:160]}\n")
    return j


def gap_events(trace: str, r: dict, step: int | None, lo: float, hi: float,
               min_dur: float = 0.0) -> list[tuple]:
    """Every event (any category) inside [t0+lo, t0+hi] of the step."""
    j = r["median_step"] if step is None else step
    t0 = r["_steps"][j][0][0]
    rows = []
    for e in iter_events(trace):
        if e.get("ph") != "X":
            continue
        ts, dur = e.get("ts", 0.0), e.get("dur", 0.0)
        if ts < t0 + lo or ts > t0 + hi or dur < min_dur:
            continue
        rows.append((ts - t0, dur, str(e.get("cat", "?")), e.get("pid"), e.get("tid"),
                     (e.get("args") or {}).get("stream"), e.get("name", "?")[:120]))
    rows.sort()
    return rows


def main(argv: list) -> int:
    args = list(argv)
    if not args or args[0].startswith("-"):
        print(__doc__)
        return 1
    trace = args.pop(0)
    dump = gap = anchor = None
    step = None
    top = 60
    while args:
        a = args.pop(0)
        if a == "--dump":
            dump = args.pop(0)
        elif a == "--gap":
            gap = args.pop(0)
        elif a == "--step":
            step = int(args.pop(0))
        elif a == "--anchor":
            anchor = args.pop(0)
        elif a == "--top":
            top = int(args.pop(0))
        else:
            print(f"?? ignored: {a}")
    r = analyze(trace, anchor)
    report(r, top)
    if dump:
        j = dump_step(r, dump, step)
        print(f"\n## node dump of step {j} -> {dump}")
    if gap:
        lo, hi = (float(x) for x in gap.split(","))
        rows = gap_events(trace, r, step, lo, hi)
        print(f"\n## events of every category in [{lo}, {hi}] us of step "
              f"{r['median_step'] if step is None else step} ({len(rows)})")
        for rel, dur, cat, pid, tid, st, n in rows:
            print(f"  {rel:10.1f} {dur:8.1f} {cat[:12]:12s} pid={pid} tid={tid} s{st} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
