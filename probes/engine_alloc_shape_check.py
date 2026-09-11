#!/usr/bin/env python3
"""Does one large device allocation fail where chunked ones succeed? (GB10, UMA)

The first 45-layer boot died in `torch.empty(55.4 GiB)` on srv2 (2026-09-11
10:42 UTC) with CUDA_ERROR_OUT_OF_MEMORY before loading a byte, while the
same box serves vLLM's 63 GiB every day under `expandable_segments:True`.
This probe settles the allocation SHAPE question in a few minutes on a free
node, with the page cache in the state the boot meets (full of the rank file):

    python3 probes/engine_alloc_shape_check.py --gib 55.4 --fill /home/choiceoh/models/<ranks>/rank0of4.safetensors

Each condition runs in its own process (allocator settings are process-wide
and a failed cudaMalloc is not something to build on):

    plain       one cudaMalloc of --gib
    expandable  one tensor of --gib under expandable_segments (20 MiB physical chunks)
    reclaim     base/arena.prepare_allocation's page-cache reclaim, then plain

Before every condition the page cache is refilled by reading --fill (plain
buffered reads) until MemFree is below --gib, so every condition starts
from the same pressure. Output: one JSON line per condition with MemFree /
MemAvailable before and after, wall seconds and the outcome. Nothing here
is a gate; it is the measurement the boot guard was written without.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GIB = 1 << 30


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        out[key] = int(rest.split()[0]) * 1024
    return {k: out[k] for k in ("MemTotal", "MemFree", "MemAvailable", "Cached")}


def fill_cache(paths, until_free_below: int, chunk: int = 64 << 20) -> int:
    """Read files through the page cache until MemFree drops under the target (or the files run out)."""
    read = 0
    for path in paths:
        with open(path, "rb", buffering=0) as f:
            while True:
                if meminfo()["MemFree"] < until_free_below:
                    return read
                buf = f.read(chunk)
                if not buf:
                    break
                read += len(buf)
    return read


def condition(name: str, nbytes: int) -> dict:
    """Runs inside the child: allocate under one policy and report."""
    from engine.base.arena import GIB as _G, prepare_allocation, expandable_segments  # noqa: F401
    before = meminfo()
    report = {"condition": name, "gib": nbytes / GIB, "before": before}
    t0 = time.perf_counter()
    try:
        import torch
        torch.cuda.init()
        if name == "expandable":
            report["setting_took"] = expandable_segments()
        if name == "reclaim":
            report["admission"] = prepare_allocation(nbytes, [], 16 * GIB, lambda: torch.cuda.mem_get_info()[0])
        t1 = time.perf_counter()
        buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        buf[::1 << 21] = 1                                      # touch every 2 MiB: the mapping is real, not lazy
        torch.cuda.synchronize()
        report.update(ok=True, alloc_seconds=round(time.perf_counter() - t1, 3),
                      segments=[(s["total_size"] >> 20, bool(s.get("is_expandable")))
                                for s in torch.cuda.memory._snapshot()["segments"] if s["total_size"] >= nbytes])
        del buf
    except Exception as exc:                                   # noqa: BLE001 -- the failure IS the measurement
        report.update(ok=False, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    report["seconds"] = round(time.perf_counter() - t0, 3)
    report["after"] = meminfo()
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gib", type=float, default=55.4)
    ap.add_argument("--fill", nargs="*", default=[], help="files to read through the page cache before each condition")
    ap.add_argument("--conditions", default="plain,expandable,reclaim")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    nbytes = int(a.gib * GIB)
    if a.child:
        print(json.dumps(condition(a.child, nbytes)))
        return 0
    for name in a.conditions.split(","):
        if a.fill:
            t0 = time.perf_counter()
            read = fill_cache(a.fill, nbytes)
            print(json.dumps({"fill": read / GIB, "seconds": round(time.perf_counter() - t0, 1), **meminfo()}), flush=True)
        env = dict(os.environ)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)                   # each condition sets its own policy in-process
        proc = subprocess.run([sys.executable, __file__, "--gib", str(a.gib), "--child", name],
                              capture_output=True, text=True, env=env)
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else json.dumps(
            {"condition": name, "ok": False, "error": proc.stderr.strip().splitlines()[-1][:200] if proc.stderr.strip() else f"exit {proc.returncode}"})
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
