"""Deneb prompts deduplicated on their final user turn (cron jobs and short messages repeat across sessions)."""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
seen, keep = set(), []
for line in open(src):
    r = json.loads(line)
    k = (r["messages"][-1]["content"] or "").strip()
    if k in seen:
        continue
    seen.add(k)
    keep.append(r)
with open(dst, "w") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"kept": len(keep)}))
