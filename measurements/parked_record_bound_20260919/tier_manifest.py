"""What a conversation tier holds, from its manifest and record file sizes alone -- no record is opened.

On a node:  cd ~/glm53-logs/st-tier/rank0 && python3 /path/to/tier_manifest.py
"""
import json
import os
import statistics
import time

index = json.load(open("manifest.json"))
live = {k: m for k, m in index.items() if not m.get("deleting")}
tokens = sorted(int(m.get("tokens", 0)) for m in live.values())
records = sorted(os.path.getsize(m["record"]) for m in live.values() if m.get("record") and os.path.exists(m["record"]))


def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))]


print(time.strftime("%Y-%m-%d %H:%M:%S %z"))
print(f"entries {len(index)}, live {len(live)}, with a record file {len(records)}")
if tokens:
    print(f"KV tokens: sum {sum(tokens)}, mean {statistics.mean(tokens):.0f}, p50 {q(tokens, .5)}, p90 {q(tokens, .9)}, "
          f"max {tokens[-1]}")
if records:
    print(f"record JSON bytes: sum {sum(records)}, mean {statistics.mean(records):.0f}, p50 {q(records, .5)}, "
          f"p90 {q(records, .9)}, max {records[-1]}")
print(f"used {sum(int(m.get('bytes', 0)) for m in live.values()) / 2**30:.2f} GiB; "
      f"slot bytes {sorted({int(m.get('extra', 0)) for m in live.values()})}; "
      f"block_bytes {sorted({int(m.get('block_bytes', 0)) for m in live.values()})}; "
      f"blocks {sum(int(m['blocks']) for m in live.values())}")
print(f"state_format {sorted({m.get('state_format', '') for m in live.values()})}")
