#!/usr/bin/env python3
"""Read-only four-node host-memory/disk sampling during this fleet trial."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1])
deadline = time.monotonic() + 3600
read = "date +%s; cat /proc/meminfo; df -B1 /home/choiceoh/glm53-cache | tail -1"
def sample(node):
    argv = ["bash", "-c", read] if node == 2 else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", f"choiceoh@10.10.10.{node}", read]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        lines = result.stdout.splitlines()
        memory = {line.split(":", 1)[0]: int(line.split()[1]) for line in lines if ":" in line and line.split(":", 1)[0] in ("MemFree", "MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback")}
        return {"node": node, "epoch": int(lines[0]), "memory_kib": memory, "disk_available_bytes": int(lines[-1].split()[3]), "returncode": result.returncode}
    except Exception as exc:
        return {"node": node, "epoch": int(time.time()), "error": str(exc)}
with ThreadPoolExecutor(max_workers=4) as pool, (root / "host-resources.jsonl").open("a", buffering=1) as out:
    while time.monotonic() < deadline and not (root / "exit-code").exists():
        for row in pool.map(sample, (1, 2, 3, 4)):
            out.write(json.dumps(row) + "\n")
        time.sleep(10)
