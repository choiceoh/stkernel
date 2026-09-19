"""Several single-GPU lanes in one ticket, each in its own process (probe, single-GPU lane; engine/SM121_INTAKE.md).

A ticket pays the queue, the container and the lane host's setup once; the lanes it carries run one after another as
`probes/engine_kernel_check.py --lanes <lane>` children, each with its own deadline, its own JSON beside this one's and
its own log -- so a lane that crashes, hangs or leaves a CUDA context behind (the sparse-MLA lane exits by design when a
kernel stalls) costs only itself. The summary names every lane's exit code, seconds and output.

    python3 probes/engine_kernel_check.py --lanes sm121_batch:sm121_gdn,sm121_fp8_l2 --output /cache/sm121-batch.json
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

#: the lanes a batch may carry, and the deadline each gets (seconds)
LANES = {"sm121_inventory": 600, "sm121_gdn": 1800, "sm121_gdn_diag": 900, "sm121_fp8_l2": 1800, "sm121_sanitizer": 2400,
         "sm121_fp4_gemm": 3600, "sm121_fp4_moe": 3600, "sm121_attention": 5400, "sm121_sparse_mla": 3000}


def run(lanes, output) -> dict:
    unknown = [lane for lane in lanes if lane not in LANES]
    if unknown or not lanes:
        raise SystemExit(f"sm121_batch carries {sorted(LANES)}; asked {lanes}")
    base = Path(output) if output else Path("/tmp/sm121-batch.json")
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.name[:-len(".json")] if base.name.endswith(".json") else base.name
    summary = {"lane": "sm121_batch", "lanes": {}}
    for lane in lanes:
        out = base.with_name(f"{stem}-{lane}.json")
        log = base.with_name(f"{stem}-{lane}.log")
        cmd = [sys.executable, str(ROOT / "probes" / "engine_kernel_check.py"), "--lanes", lane, "--output", str(out)]
        began = time.perf_counter()
        with open(log, "w") as sink:
            try:
                rc = subprocess.run(cmd, cwd=str(ROOT), stdout=sink, stderr=subprocess.STDOUT,
                                    timeout=LANES[lane]).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        summary["lanes"][lane] = {"returncode": rc, "seconds": round(time.perf_counter() - began, 1),
                                  "output": str(out) if out.is_file() else None, "log": str(log)}
        print(json.dumps({lane: summary["lanes"][lane]}), flush=True)
        base.write_text(json.dumps(summary, indent=1) + "\n")
    return summary
