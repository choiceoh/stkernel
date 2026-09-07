#!/usr/bin/env python3
"""Refuse an isolated GLM GPU probe without spare UMA host memory.

GB10 device allocations consume host RAM. An idle serving container is not
proof of room for a second CUDA process. Reserve 8 GiB for the probe and
leave at least 10% of RAM (minimum 8 GiB) available for the existing workload.
There is deliberately no CLI/environment override of this safety margin.
"""
import json
from pathlib import Path
import socket
import sys


def assess(text):
    fields = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ("MemTotal:", "MemAvailable:"):
            if len(parts) != 3 or parts[2] != "kB":
                raise ValueError("invalid host memory units")
            fields[parts[0][:-1]] = int(parts[1])
    total, available = fields.get("MemTotal"), fields.get("MemAvailable")
    if total is None or available is None or not 0 <= available <= total or total <= 0:
        raise ValueError("valid MemTotal and MemAvailable counters are required")
    required = 8 * 1024**2 + max(8 * 1024**2, (total + 9) // 10)
    return {"passed": available >= required, "available_kib": available,
            "required_kib": required, "total_kib": total}


def main():
    try:
        result = assess(Path("/proc/meminfo").read_text())
    except (OSError, ValueError) as exc:
        print(f"REFUSED: cannot establish probe memory headroom: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"host": socket.gethostname(), "probe_memory": result}), flush=True)
    if not result["passed"]:
        print("REFUSED: insufficient UMA memory for an additional GPU probe; "
              "use an offline fleet turn before starting serving. Refusing the next GPU container.",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
