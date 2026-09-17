"""The item universe of the model-dependence study: every first-parent main commit that touched engine/, one item per
pull request (squash `... (#N)` or `Merge pull request #N`) or direct commit, with the message and diffstat a
classifier reads. Read-only git.

    python3 measurements/st_model_dependence_20260917/collect.py OUT_DIR [--end 3398a72b] [--since 2026-09-10] [--batches 8]

Writes OUT_DIR/items.json (hash, date, kind, pr, title) and OUT_DIR/batch_<b>.txt. The record's classification.jsonl
carries the same items in the same order (item = index into items.json).
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def clip(text, n):
    return text if len(text) <= n else text[:n] + f"\n...[clipped {len(text) - n} chars]"


def items(end, since):
    raw = git("log", end, "--first-parent", "--format=%h%x09%ad%x09%s%x09%b%x1e", "--date=short", f"--since={since}",
              "--", "engine")
    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        h, d, s, b = (rec.split("\t", 3) + ["", "", "", ""])[:4]
        m = re.match(r"Merge pull request #(\d+) from (\S+)", s)
        if m:
            title = b.strip().split("\n")[0] if b.strip() else m.group(2)
            out.append(dict(hash=h, date=d, kind="merge", pr=int(m.group(1)), title=title))
            continue
        m = re.search(r"\(#(\d+)\)\s*$", s)
        out.append(dict(hash=h, date=d, kind="squash" if m else "direct", pr=int(m.group(1)) if m else None, title=s))
    return out


def block(i, it):
    h = it["hash"]
    if it["kind"] == "merge":
        subjects = git("log", "--format=  - %s", f"{h}^1..{h}^2")
        body = git("log", "-1", "--format=%b", h)
        msg = f"PR title: {it['title']}\n{body.strip()}\nbranch commits:\n{clip(subjects, 2500)}"
        stat = git("diff", "--stat=160", f"{h}^1", h)
    else:
        msg = git("log", "-1", "--format=%B", h)
        stat = git("show", "--stat=160", "--format=", h)
    lines = stat.rstrip("\n").split("\n")
    if len(lines) > 25:
        lines = lines[:24] + [f"  ...({len(lines) - 25} more files)", lines[-1]]
    ident = f"PR #{it['pr']}" if it["pr"] else "direct"
    return f"=== ITEM {i} | {ident} | {h} | {it['date']} | {it['kind']}\n{clip(msg.strip(), 3500)}\n--- files:\n" + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--end", default="3398a72b")
    ap.add_argument("--since", default="2026-09-10")
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    its = items(a.end, a.since)
    (out / "items.json").write_text(json.dumps(its, ensure_ascii=False, indent=0))
    blocks = [block(i, it) for i, it in enumerate(its)]
    per = -(-len(blocks) // a.batches)
    for b in range(a.batches):
        (out / f"batch_{b}.txt").write_text("\n".join(blocks[b * per:(b + 1) * per]))
    kinds = {k: sum(it["kind"] == k for it in its) for k in ("squash", "merge", "direct")}
    print(f"{len(its)} items {kinds} -> {out}")


if __name__ == "__main__":
    main()
