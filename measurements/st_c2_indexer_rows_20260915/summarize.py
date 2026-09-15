"""summarize.py EVENTS.jsonl: numerics, B/A/A/B timing table and launch census of engine_decode_select_rows."""
import json
import sys
from collections import defaultdict

events = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print("## numerics")
for e in events:
    if e["event"] == "select_rows_numerics":
        print(f"rows={e['rows']} t={e['tokens']} cap={e['capacity']:>6} n_cand={e['n_cand']:>5} "
              f"exact={all(e['exact'].values())} {e['mismatched_elements']} edge ties/phase={e['edge_tie_queries']}")
print("\n## timing (ms per 11-layer replay; B = per-row control, A = candidate)")
print("| rows | t | capacity | contexts | candidate | cache | B mean | B min | A mean | A min | A/B mean | saved ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for e in events:
    if e["event"] == "timing":
        b = [s["ms"] for s in e["samples"] if s["arm"] == "B"]
        a = [s["ms"] for s in e["samples"] if s["arm"] == "A"]
        bm, am = sum(b) / len(b), sum(a) / len(a)
        print(f"| {e['rows']} | {e['tokens']} | {e['capacity']} | {e['contexts']} | {e['candidate']} | {e['cache']} | "
              f"{bm:.3f} | {min(b):.3f} | {am:.3f} | {min(a):.3f} | {am / bm:.3f} | {bm - am:.3f} |")
    elif e["event"] == "timing_skipped":
        print(f"| {e['rows']} | {e['tokens']} | {e['capacity']} | - | {e['candidate']} | skipped: {e['reason']} |")
print("\n## launches per replay (11 layers)")
census = defaultdict(dict)
for e in events:
    if e["event"] == "launches":
        census[e["capacity"]][e["candidate"]] = e
for capacity, arms in census.items():
    names = list(arms)
    print(f"capacity {capacity}: " + ", ".join(f"{n} {arms[n]['per_replay']:.0f}" for n in names))
    kernels = sorted({k for n in names for k in arms[n]["kernels"]})
    for k in kernels:
        row = [arms[n]["kernels"].get(k, 0) for n in names]
        if len(set(row)) > 1:
            print("   " + " / ".join(f"{v:g}" for v in row) + "   " + k[:110])
for e in events:
    if e["event"] in ("launches_failed", "complete", "timing_skipped"):
        print(json.dumps(e)[:300])
