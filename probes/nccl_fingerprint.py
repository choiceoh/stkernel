#!/usr/bin/env python3
"""Extract an NCCL topology fingerprint from a hy4 boot log.

Pulls the init-time ring/channel/NET lines NCCL_DEBUG=INFO emits, normalizes
volatile fields (pids, ports, timestamps), and prints a stable digest plus a
short human summary. Usage: nccl_fingerprint.py <logfile>"""
import hashlib
import re
import sys

KEEP = re.compile(
    r"(Ring \d+|Trees|Channel \d+|NET/IB|Using network|"
    r"comm .* rank \d+ nranks|via NET|P2P|Connected all rings)")
VOLATILE = re.compile(r"(\[\d+\]|pid \d+|0x[0-9a-f]+|:\d{4,5}|\d+\.\d+\.\d+\.\d+<\d+>)")

lines = []
for raw in open(sys.argv[1], errors="replace"):
    if "NCCL INFO" not in raw:
        continue
    if not KEEP.search(raw):
        continue
    line = raw.split("NCCL INFO", 1)[1].strip()
    line = VOLATILE.sub("_", line)
    lines.append(line)

if not lines:
    print("fingerprint: NO-NCCL-INFO (was NCCL_DEBUG=INFO set?)")
    sys.exit(0)

digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
rings = [ln for ln in lines if ln.startswith("Ring")]
nets = [ln for ln in lines if "NET/IB" in ln or "via NET" in ln]
print(f"fingerprint: {digest}  ({len(lines)} topo lines, "
      f"{len(rings)} ring lines, {len(nets)} net lines)")
for ln in rings[:8]:
    print(f"  {ln}")
for ln in nets[:6]:
    print(f"  {ln}")
