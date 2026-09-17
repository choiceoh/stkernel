"""Split one boot's head.jsonl into per-pass directories head_compare.py reads: pass k is documents
[k * n, (k + 1) * n), renumbered to 0..n-1 so every pass pairs with pass 0 position for position.

    python3 split_passes.py <session dir with head.jsonl> <documents per pass> <passes>
"""
import json
import sys
from pathlib import Path

root, n, passes = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
outs = []
for k in range(passes):
    d = root / f"pass{k}"
    d.mkdir(exist_ok=True)
    outs.append(open(d / "head.jsonl", "w"))
counts = [0] * passes
for line in (root / "head.jsonl").read_text().splitlines():
    r = json.loads(line)
    k = r["doc"] // n
    if k >= passes:
        continue
    r["doc"] -= k * n
    outs[k].write(json.dumps(r) + "\n")
    counts[k] += 1
for f in outs:
    f.close()
print("rows per pass:", counts)
