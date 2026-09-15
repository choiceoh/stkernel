"""Build the four shared-root native extensions in this container and time each (no GPU is touched).

Run twice, in two containers that share one /cache mount: the first compiles, the second must load
the kept builds. Prints one JSON line per module and a summary line.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/repo")
label = sys.argv[1]
root = os.environ.get("ST_NATIVE_BUILD_ROOT")
rows = []
from engine.kernels import bounded_graph, decode_queue, mapped_staging, prefill_topk  # noqa: E402
from engine.kernels.common.native_cache import build_root  # noqa: E402

for name, build in (("mapped-staging", mapped_staging.build), ("prefill-topk", prefill_topk._build),
                    ("bounded-graph", bounded_graph.build), ("decode-queue", decode_queue.build)):
    directory = build_root(name)
    before = sorted(str(p) for p in directory.rglob("*.so")) if directory.exists() else []
    t0 = time.perf_counter()
    ext = build()
    seconds = time.perf_counter() - t0
    after = sorted(str(p) for p in directory.rglob("*.so"))
    objects = sorted(directory.rglob("*.cuda.o"))
    row = dict(run=label, module=name, seconds=round(seconds, 2), root=str(directory),
               loaded=getattr(ext, "__file__", None), kept_before=len(before), kept_after=len(after),
               object_mtime_ns=[p.stat().st_mtime_ns for p in objects])
    rows.append(row)
    print(json.dumps(row), flush=True)
print(json.dumps(dict(run=label, ST_NATIVE_BUILD_ROOT=root, total_seconds=round(sum(r["seconds"] for r in rows), 2))), flush=True)
