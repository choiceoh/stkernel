#!/usr/bin/env python3
"""Put this branch's ledger entries after whatever the reference now ends with.

MEASUREMENTS.md conflicts on every rebase, because sections are numbered and several sessions append at once. The
resolution is always the same -- take the reference's file whole, renumber my entries to continue from its last
section, append -- and doing it by hand has gone wrong: one attempt wrote `### 45차 §65### 45차 §63 — ...` into
main. So it is a script, and the script refuses to write a file whose headings are not well formed.

    python3 tools/ledger_rebase.py                 # against origin/main
    python3 tools/ledger_rebase.py --ref HEAD~3 --dry-run

References to a renumbered section elsewhere in the tree (`45차 §63`, `원장 45차 §63`) are moved with it.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = "MEASUREMENTS.md"
HEAD = re.compile(r"^### (\d+)차 §(\d+)(?= )", re.M)          # a well-formed heading: "### 45차 §63 — ..."
ANY = re.compile(r"^### \d+차 §", re.M)


def show(ref: str, path: str) -> str:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"cannot read {path} at {ref}")
    return out.stdout


def sections(text: str) -> "list[tuple[int, int, int]]":
    return [(m.start(), int(m.group(1)), int(m.group(2))) for m in HEAD.finditer(text)]


def check(text: str) -> None:
    """Every heading that starts a section must be well formed. That is the failure this exists to prevent: a
    hand-rolled renumber once wrote `### 45차 §65### 45차 §63 — ...` into main and nothing noticed.

    Repeated numbers are NOT checked: the file already has two §30s, two §31s and two §32s from sessions that
    appended the same evening, and that history is not this script's to rewrite."""
    starts = {m.start() for m in ANY.finditer(text)}
    good = {s for s, _, _ in sections(text)}
    if starts - good:
        bad = sorted(starts - good)[0]
        raise SystemExit(f"malformed ledger heading: {text[bad: bad + 90]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=ROOT, check=False)

    base = show(a.ref, LEDGER)
    mine = (ROOT / LEDGER).read_text()
    check(base)
    check(mine)
    base_n = {n for _, _, n in sections(base)}
    # Mine are the ones whose TITLE the reference does not have. Not "whose number it lacks": numbers collide --
    # two sessions appending the same evening is exactly why this file conflicts in the first place.
    titles = {base[at:].split("\n", 1)[0].split(" — ", 1)[-1] for at, _, _ in sections(base)}
    ours = [(at, era, n) for at, era, n in sections(mine)
            if mine[at:].split("\n", 1)[0].split(" — ", 1)[-1] not in titles]
    if not ours:
        print(f"  nothing of mine to move: every section already exists at {a.ref}")
        return 0
    tail = mine[ours[0][0]:]
    era = ours[0][1]
    nxt = max(n for _, _, n in sections(base)) + 1
    moves = {}
    for _, _, n in ours:
        moves[n] = nxt
        nxt += 1
    for old, new in sorted(moves.items(), reverse=True):                 # descending: no number lands on another
        tail = re.sub(rf"^### {era}차 §{old}(?= )", f"### {era}차 §{new}", tail, flags=re.M)
    out = base.rstrip() + "\n\n" + tail.strip() + "\n"
    check(out)
    landed = {n for _, _, n in sections(out)}
    if moves and not set(moves.values()) <= landed:
        raise SystemExit(f"renumber lost a section: wanted {sorted(moves.values())}")
    if set(moves.values()) & base_n:
        raise SystemExit(f"renumber collided with {a.ref}: {sorted(set(moves.values()) & base_n)}")

    print(f"  {a.ref} ends at §{max(n for _, _, n in sections(base))}; "
          + ", ".join(f"§{o} -> §{n}" for o, n in sorted(moves.items())))
    if a.dry_run:
        return 0
    (ROOT / LEDGER).write_text(out)
    for path in ROOT.rglob("*"):                                         # carry the cross-references with them
        if path.suffix not in (".py", ".md", ".cu", ".h", ".sh") or LEDGER in str(path) or ".git/" in str(path):
            continue
        text = path.read_text(errors="ignore")
        after = text
        for old, new in sorted(moves.items(), reverse=True):
            after = re.sub(rf"({era}차 )§{old}\b", rf"\g<1>§{new}", after)
        if after != text:
            path.write_text(after)
            print(f"  cross-reference: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
